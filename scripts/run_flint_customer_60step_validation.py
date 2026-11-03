#!/usr/bin/env python3
"""Customer-Grade 60-Step Steady-State TorchTitan TPU Validation & Context-Length Ablation for RSI Flint.

Addresses all customer audit requirements on TPU v7x-16 (32 TensorCores / 16 Ironwood chips),
TPU v5p-16 (16 chips), and TPU v5p-32 (32 chips):
  1. Exact First-Principles FLOPs/Token Accounting (`RecursiveSI/flint/config.json`):
     - `hidden_size = 2048`, `vocab_size = 150,144`, `49` layers (`2` dense SwiGLU `inter=6144` +
       `47` MoE layers with `256` experts, `top_k=6`, `moe_inter=512`, `shared_inter=1024`),
       `32` Q heads, `4` KV heads, `head_dim = 128`, `14` full causal NoPE layers + `35` one-sided
       causal sliding-window (`W = 513`) RoPE layers + per-head attention sinks.
     - Active non-embedding parameters: `N_active = 2,515,533,824` (`6 * N_active = 15.0932 GFLOP/tok`).
     - Reports THREE FLOP/MFU conventions side-by-side:
       (a) Exact Causal-Masked MFU (`6 * H * D` lower-triangle on 14 full layers + exact causal ramp on 35 W=513 SWA layers)
       (b) PaLM / TorchTitan Full-Square MFU (`12 * H * D` upstream `quadratic_attention_flops_per_token` convention)
       (c) Parameter-Only `6 * N_active` MFU (`15.0932 GFLOP/tok` constant across all sequence lengths)
  2. Hardware-Normalized Throughput First:
     - Reports `steady_tps_per_chip` (`2 * tps_per_rank` on dual-TensorCore `v7x`, `1 * tps_per_rank` on `v5p`/`v6e`),
       `steady_tps_per_rank`, and `steady_slice_tps` across 50 consecutive post-warmup steps (`steps 11..60` out of `60`).
     - Reports Steady-State Mean, Median (`P50`), and StdDev (no single-step peak cherry-picking).
  3. True Contiguous Unskipped Full-Sequence Causal Attention (`TT_CONTIGUOUS_FULL_SEQ=1`):
     - Eliminates `c4_test` short-article EoD `segment_ids` resets (`document_aware_skipping=True` in `SbdConfig`)
       so every `[mb, seq_len]` row executes 100% unskipped `seq_len` causal attention across `2K, 4K, 8K, 16K, 32K, 64K`
       at strictly constant recipe and constant `65,536` tokens/microbatch/rank (`mb=32..1`), plus direct `c4_test`
       EoD-packed control comparison at `64K`.
  4. Long-Context Extension Suite (`--suite=customer_val60_longctx`: `128K` and `256K`, Flint `max_position_embeddings`):
     - Flint in this TorchTitan TPU stack rejects context/tensor parallelism, so each rank holds one full `seq_len`
       sequence per microbatch (`mb=1` -> `131,072` / `262,144` tokens/rank/microbatch instead of `65,536`).
     - Starts from the identical val60 recipe (Splash-attention output saved in all `49` layers) and, only after an HBM
       OOM, retries with fewer saved layers (`_attn{k}` suffix); recomputed attention FLOPs are never counted in MFU.
     - A node-0 HTTP coordinator keeps every host on the same attempt and rendezvous port, so a failed attempt can never
       strand hosts on stale ports.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
import datetime as dt
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import os
from pathlib import Path
import re
import signal
import socket
import statistics
import subprocess
import sys
import threading
import time
import urllib.parse
import urllib.request

# Exact RecursiveSI/flint/config.json constants
FLINT_ACTIVE_NON_EMBED_PARAMS = 2_515_533_824
FLINT_PARAM_FLOPS_PER_TOKEN = 6.0 * FLINT_ACTIVE_NON_EMBED_PARAMS  # 15,093,202,944 FLOPs/tok
FLINT_NUM_FULL_LAYERS = 14
FLINT_NUM_SLIDING_LAYERS = 35
FLINT_SLIDING_WINDOW = 513
FLINT_Q_HEADS = 32
FLINT_HEAD_DIM = 128
# Per attended (query, key) token pair across H=32, D=128:
# Forward QK^T (2*H*D) + AV (2*H*D) = 4*H*D; Backward dQ, dK, dV = 8*H*D -> 12*H*D = 49,152 FLOPs/pair
FLINT_ATTN_FLOPS_PER_PAIR = 12.0 * FLINT_Q_HEADS * FLINT_HEAD_DIM  # 49,152.0

PEAK_BF16_TFLOPS_PER_RANK = {"v7x": 1153.5, "v5p": 459.0, "v6e": 918.0}
RANKS_PER_CHIP = {"v7x": 2, "v5p": 1, "v6e": 1}

STEP_RE = re.compile(
    r"step:\s*(?P<step>\d+).*?loss:\s*(?P<loss>[0-9.]+).*?tps:\s*(?P<tps>[0-9.,]+)",
    re.IGNORECASE,
)
TFLOPS_RE = re.compile(r"tflops:\s*(?P<tflops>[0-9.]+)", re.IGNORECASE)
MFU_RE = re.compile(r"mfu:\s*(?P<mfu>[0-9.]+)%", re.IGNORECASE)
MEM_RE = re.compile(r"memory:\s*(?P<mem>[0-9.]+)GiB", re.IGNORECASE)


def flint_param_only_flops_per_token() -> float:
    """Constant 6 * N_active non-embedding FLOPs/token (15.0932 GFLOP/tok)."""
    return FLINT_PARAM_FLOPS_PER_TOKEN


def flint_causal_flops_per_token(seq_len: int) -> float:
    """Exact causal-masked FLOPs/token for RecursiveSI/flint (14 full causal + 35 W=513 causal SWA layers)."""
    # 14 full causal layers: query i in [0..S-1] attends to (i + 1) keys -> mean (S + 1) / 2 keys/token
    full_keys_avg = (seq_len + 1.0) / 2.0
    full_flops = FLINT_NUM_FULL_LAYERS * FLINT_ATTN_FLOPS_PER_PAIR * full_keys_avg

    # 35 one-sided causal SWA layers (0 <= q - k < W, W = 513):
    w = FLINT_SLIDING_WINDOW
    if seq_len <= w:
        swa_keys_avg = (seq_len + 1.0) / 2.0
    else:
        # First W tokens attend to 1..W keys (sum = W*(W+1)/2); remaining (S - W) tokens attend to W keys
        swa_keys_avg = w - (w * (w - 1.0)) / (2.0 * seq_len)
    swa_flops = FLINT_NUM_SLIDING_LAYERS * FLINT_ATTN_FLOPS_PER_PAIR * swa_keys_avg
    return FLINT_PARAM_FLOPS_PER_TOKEN + full_flops + swa_flops


def flint_palm_flops_per_token(seq_len: int) -> float:
    """Upstream PaLM / TorchTitan full-square convention (12 * H * D * min(S, W), not halved for causal mask)."""
    full_flops = FLINT_NUM_FULL_LAYERS * FLINT_ATTN_FLOPS_PER_PAIR * float(seq_len)
    swa_flops = FLINT_NUM_SLIDING_LAYERS * FLINT_ATTN_FLOPS_PER_PAIR * float(min(seq_len, FLINT_SLIDING_WINDOW))
    return FLINT_PARAM_FLOPS_PER_TOKEN + full_flops + swa_flops


NATIVE_VAL60_CONFIG_PATCH = '''

# ==============================================================================
# Customer 60-Step Validation Fixed-Recipe Context-Length Configs (2K..64K)
# ==============================================================================
from torchtitan.experiments.tpu.token_dispatcher import (
    TpuRoutedExperts,
    TpuRoutedExpertsConverter,
)

_VAL60_BF16_ROPE = (
    "torchtitan.experiments.tpu.overrides.bf16_rope"
    ".tpu_activation_dtype_cossin_rope"
)


def _val60_set_max_context_length(root_cfg, seq_len: int) -> None:
  seen = set()

  def _walk(obj):
    oid = id(obj)
    if oid in seen or obj is None or isinstance(obj, (int, float, str, bytes, bool)):
      return
    seen.add(oid)
    if hasattr(obj, "max_context_length"):
      try:
        obj.max_context_length = seq_len
      except Exception:
        pass
    if hasattr(obj, "traverse"):
      try:
        for _, child, _, _ in obj.traverse(object):
          if hasattr(child, "max_context_length"):
            try:
              child.max_context_length = seq_len
            except Exception:
              pass
      except Exception:
        pass
    if isinstance(obj, dict):
      for v in obj.values():
        _walk(v)
    elif isinstance(obj, (list, tuple)):
      for v in obj:
        _walk(v)
    elif hasattr(obj, "__dict__"):
      for v in vars(obj).values():
        _walk(v)

  _walk(root_cfg)


def _build_v7x_val60_fixed(seq_len: int) -> TpuGraphTrainer.Config:
  config = flint_40b_v014_tile1024()
  _val60_set_max_context_length(config, seq_len)
  config.model_spec.model = (
      TpuRoutedExpertsConverter.Config().build().convert(config.model_spec.model)
  )
  for _, r, _, _ in config.model_spec.model.traverse(TpuRoutedExperts.Config):
    r.scale_before_down_proj = True
  if _VAL60_BF16_ROPE not in config.override.imports:
    config.override.imports.append(_VAL60_BF16_ROPE)
  config.compile.memory_policy = "tpu_selective"
  config.compile.save_regions = ("moe.routing",)
  config.compile.save_last_n_layers = {"attention.inner_attention": 49}
  config.compile.optimizer_compile = "separate_graph"
  config.compile.compile_grad_norm = True
  config.optimizer.implementation = "fused"
  return config


def _build_v5p_val60_fixed(seq_len: int) -> TpuGraphTrainer.Config:
  config = flint_40b_v5p_n40()
  _val60_set_max_context_length(config, seq_len)
  config.model_spec.model = (
      TpuRoutedExpertsConverter.Config().build().convert(config.model_spec.model)
  )
  for _, r, _, _ in config.model_spec.model.traverse(TpuRoutedExperts.Config):
    r.scale_before_down_proj = True
  for _, ex, _, _ in config.model_spec.model.traverse(TpuGroupedExperts.Config):
    ex.use_pallas_grouped_mm = True
    ex.use_pallas_paired_grouped_mm = True
    ex.pallas_grouped_mm_tile_m = 512
    ex.pallas_grouped_mm_tgmm_tiling = (256, 1024, 0)
    ex.pallas_grouped_mm_w2_tgmm_tiling = (256, 512, 0)
    ex.pallas_grouped_mm_w2_transposed_tgmm_tiling = (256, 0, 1024)
    ex.use_kernel_native_weight_layout = True
  if _VAL60_BF16_ROPE not in config.override.imports:
    config.override.imports.append(_VAL60_BF16_ROPE)
  config.compile.memory_policy = "tpu_selective"
  config.compile.save_regions = ("moe.routing",)
  config.compile.save_last_n_layers = {"attention.inner_attention": 49}
  config.compile.enable_fsdp_ag_rs_overlap = True
  config.compile.enable_fsdp_dense_region_overlap = True
  config.compile.optimizer_compile = "separate_graph"
  config.compile.compile_grad_norm = True
  config.optimizer.implementation = "fused"
  return config


def flint_40b_val60_v7x_s2048() -> TpuGraphTrainer.Config:
  return _build_v7x_val60_fixed(2048)

def flint_40b_val60_v7x_s4096() -> TpuGraphTrainer.Config:
  return _build_v7x_val60_fixed(4096)

def flint_40b_val60_v7x_s8192() -> TpuGraphTrainer.Config:
  return _build_v7x_val60_fixed(8192)

def flint_40b_val60_v7x_s16384() -> TpuGraphTrainer.Config:
  return _build_v7x_val60_fixed(16384)

def flint_40b_val60_v7x_s32768() -> TpuGraphTrainer.Config:
  return _build_v7x_val60_fixed(32768)

def flint_40b_val60_v7x_s65536() -> TpuGraphTrainer.Config:
  return _build_v7x_val60_fixed(65536)


def flint_40b_val60_v5p_s2048() -> TpuGraphTrainer.Config:
  return _build_v5p_val60_fixed(2048)

def flint_40b_val60_v5p_s4096() -> TpuGraphTrainer.Config:
  return _build_v5p_val60_fixed(4096)

def flint_40b_val60_v5p_s8192() -> TpuGraphTrainer.Config:
  return _build_v5p_val60_fixed(8192)

def flint_40b_val60_v5p_s16384() -> TpuGraphTrainer.Config:
  return _build_v5p_val60_fixed(16384)

def flint_40b_val60_v5p_s32768() -> TpuGraphTrainer.Config:
  return _build_v5p_val60_fixed(32768)

def flint_40b_val60_v5p_s65536() -> TpuGraphTrainer.Config:
  return _build_v5p_val60_fixed(65536)


__all__.extend([
    "flint_40b_val60_v7x_s2048",
    "flint_40b_val60_v7x_s4096",
    "flint_40b_val60_v7x_s8192",
    "flint_40b_val60_v7x_s16384",
    "flint_40b_val60_v7x_s32768",
    "flint_40b_val60_v7x_s65536",
    "flint_40b_val60_v5p_s2048",
    "flint_40b_val60_v5p_s4096",
    "flint_40b_val60_v5p_s8192",
    "flint_40b_val60_v5p_s16384",
    "flint_40b_val60_v5p_s32768",
    "flint_40b_val60_v5p_s65536",
])
'''


# ==============================================================================
# Long-Context Extension Suite (128K / 256K True Unskipped Full-Seq, mb=1)
# ==============================================================================
LONGCTX_SUITE = "customer_val60_longctx"
# Every attention-save count a long-context attempt may use (one registry config each).
LONGCTX_ATTN_SAVE_CHOICES = (49, 32, 16, 0)
# (group tag, seq_len, attention-save counts tried in order -- next one only after an HBM OOM, per-attempt timeout s)
# 128K starts from the exact 2K..64K val60 recipe (49 saved layers). At 256K a single sequence is 4x the 64K
# per-rank token budget, so its Splash outputs alone (2 GiB/layer -> 98 GiB for 49 layers) exceed one v5p chip's
# 95 GiB HBM; it therefore recomputes attention in every layer from the start.
LONGCTX_GROUPS = (
    ("12_long_s131072", 131072, (49, 32, 16, 0), 3 * 3600),
    ("13_long_s262144", 262144, (0,), 6 * 3600),
)
OOM_MARKERS = (
    "RESOURCE_EXHAUSTED",
    "Ran out of memory in memory space",
    "Out of memory",
    "Error allocating device buffer",
)
HBM_EXCEEDED_RE = re.compile(r"[Ee]xceeded hbm capacity by\s*(?P<val>[0-9.]+)\s*(?P<unit>[KMGT])")
ALLOC_FAILED_RE = re.compile(
    r"[Aa]ttempting to allocate\s*(?P<want>[0-9.]+)\s*(?P<wunit>[KMGT])i?B?\.?\s*"
    r"That was not possible\.\s*There are\s*(?P<free>[0-9.]+)\s*(?P<funit>[KMGT])i?B?\s*free"
)
_GIB_PER_UNIT = {"K": 1.0 / (1 << 20), "M": 1.0 / (1 << 10), "G": 1.0, "T": 1024.0}


def attn_saved_gib_per_layer(tokens_per_rank: int) -> float:
    """HBM one layer holds when its Splash output is saved for backward: T x 32 heads x 128 x 2 B (bf16)."""
    return tokens_per_rank * FLINT_Q_HEADS * FLINT_HEAD_DIM * 2 / float(1 << 30)


def parse_hbm_overflow_gib(line: str) -> float | None:
    """GiB missing according to an XLA:TPU compile-time or runtime HBM OOM message, if `line` reports it."""
    m = HBM_EXCEEDED_RE.search(line)
    if m:
        return float(m.group("val")) * _GIB_PER_UNIT[m.group("unit")]
    m = ALLOC_FAILED_RE.search(line)
    if m:
        want = float(m.group("want")) * _GIB_PER_UNIT[m.group("wunit")]
        free = float(m.group("free")) * _GIB_PER_UNIT[m.group("funit")]
        return max(want - free, 0.0)
    return None


def longctx_config_patch() -> str:
    """Registry patch with explicit `flint_40b_val60_{chip}_s{seq}_attn{k}` long-context configs."""
    out = [
        "",
        "",
        "# ==============================================================================",
        "# Customer 60-Step Validation Long-Context Extension Configs (128K / 256K, mb=1)",
        "# ==============================================================================",
        "def _build_val60_long(chip: str, seq_len: int, n_attn_saved: int) -> TpuGraphTrainer.Config:",
        '  """val60 fixed recipe at `seq_len`, keeping the Splash output of the last `n_attn_saved` layers',
        '  for backward (49 = the 2K..64K val60 recipe, 0 = recompute attention in every layer)."""',
        '  builder = _build_v7x_val60_fixed if chip == "v7x" else _build_v5p_val60_fixed',
        "  config = builder(seq_len)",
        "  if n_attn_saved > 0:",
        '    config.compile.save_last_n_layers = {"attention.inner_attention": n_attn_saved}',
        "  else:",
        "    config.compile.save_last_n_layers = {}",
        "  return config",
    ]
    names = []
    for chip in ("v7x", "v5p"):
        for _, seq_len, _, _ in LONGCTX_GROUPS:
            for k in LONGCTX_ATTN_SAVE_CHOICES:
                name = f"flint_40b_val60_{chip}_s{seq_len}_attn{k}"
                names.append(name)
                out += [
                    "",
                    "",
                    f"def {name}() -> TpuGraphTrainer.Config:",
                    f"  return _build_val60_long({chip!r}, {seq_len}, {k})",
                ]
    out += ["", "", "__all__.extend(["]
    out += [f"    {n!r}," for n in names]
    out += ["])", ""]
    return "\n".join(out)


def apply_runtime_patches() -> None:
    nat_reg = Path("/torchtitan-native/torchtitan/experiments/tpu/examples/flint/config_registry.py")
    if nat_reg.exists():
        txt = nat_reg.read_text()
        if "flint_40b_val60_v7x_s2048" not in txt:
            nat_reg.write_text(txt + NATIVE_VAL60_CONFIG_PATCH)
            print("[flint-val60] Patched Native config_registry.py with fixed-recipe 2K..64K configs", flush=True)
        txt = nat_reg.read_text()
        if "def _build_val60_long(" not in txt:
            nat_reg.write_text(txt + longctx_config_patch())
            print("[flint-val60] Patched Native config_registry.py with 128K/256K long-context configs", flush=True)
        txt = nat_reg.read_text()
        if "def flint_40b_engabl_v014_sbdp_rope(" not in txt:
            nat_reg.write_text(txt + ENG_ABLATION_CONFIG_PATCH)
            print("[flint-val60] Patched Native config_registry.py with eng-parity 2K ablation config", flush=True)

    packing_py = Path("/torchtitan-native/torchtitan/components/data/packing.py")
    if packing_py.exists():
        ptxt = packing_py.read_text()
        target = "    positions = token_indices - segment_starts"
        replacement = (
            "    import os as _tt_pack_os\n"
            "    if _tt_pack_os.environ.get('TT_CONTIGUOUS_FULL_SEQ', '0') == '1':\n"
            "        positions = token_indices % max_context_length\n"
            "    else:\n"
            "        positions = token_indices - segment_starts"
        )
        if "TT_CONTIGUOUS_FULL_SEQ" not in ptxt and target in ptxt:
            packing_py.write_text(ptxt.replace(target, replacement, 1))
            print("[flint-val60] Patched packing.py with TT_CONTIGUOUS_FULL_SEQ=1 unbroken sequence support", flush=True)


def filter_supported_xla_flags(candidates: list[str]) -> str:
    try:
        import libtpu

        so_dir = Path(libtpu.__file__).parent
        so_files = list(so_dir.glob("*.so"))
        blob = b"".join(p.read_bytes() for p in so_files) if so_files else b""
    except Exception:
        blob = b""
    if not blob:
        return " ".join(candidates)
    kept = [f for f in candidates if f.lstrip("-").split("=", 1)[0].encode("utf-8") in blob]
    return " ".join(kept)


def wait_for_master_dns(master_addr: str, timeout_s: int = 300) -> str:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            return socket.gethostbyname(master_addr)
        except socket.gaierror:
            time.sleep(3)
    return master_addr


def chip_xla_flags_and_prefix(chip_type: str) -> tuple[str, str]:
    """LIBTPU_INIT_ARGS and registry prefix shared by every val60 suite (2K..64K and 128K/256K) on `chip_type`."""
    if chip_type == "v7x":
        xla_flags = filter_supported_xla_flags([
            "--xla_tpu_enable_sparse_core_collective_aggregator=true",
            "--xla_tpu_enable_latency_hiding_layer_scheduler=true",
            "--xla_tpu_use_single_sparse_core_for_reduce_scatter_offload=false",
            "--xla_tpu_enable_sparse_core_collective_offload_nd_reduce_scatter=true",
            "--xla_tpu_enable_sparse_core_hierarchical_reduce_scatter=true",
            "--xla_tpu_dvfs_p_state=7",
            "--xla_enable_async_all_gather=true",
            "--xla_enable_async_all_reduce=true",
            "--xla_tpu_enable_async_all_to_all=true",
            "--xla_tpu_enable_ag_backward_pipelining=true",
            "--xla_tpu_enable_ici_ag_pipelining=true",
            "--xla_tpu_overlap_compute_collective_tc=true",
            "--xla_max_concurrent_async_all_gathers=4",
            "--xla_latency_hiding_scheduler_rerun=2",
            "--xla_enable_transpose_trace=false",
        ])
        return xla_flags, "flint_40b_val60_v7x"
    xla_flags = filter_supported_xla_flags([
        "--xla_tpu_scoped_vmem_limit_kib=81920",
        "--xla_tpu_enable_offloading_copy_to_sparsecore=false",
        "--xla_tpu_copy_insertion_use_region_analysis=true",
        "--xla_tpu_copy_insertion_use_region_analysis_limit=1000",
        "--xla_tpu_prefer_async_allgather_to_allreduce=true",
        "--xla_enable_async_all_gather=true",
        "--xla_enable_async_all_reduce=true",
        "--xla_tpu_enable_async_all_to_all=true",
        "--xla_tpu_enable_ag_backward_pipelining=true",
        "--xla_tpu_enable_ici_ag_pipelining=true",
        "--xla_tpu_overlap_compute_collective_tc=true",
        "--xla_max_concurrent_async_all_gathers=4",
        "--xla_latency_hiding_scheduler_rerun=2",
        "--xla_enable_transpose_trace=false",
    ])
    return xla_flags, "flint_40b_val60_v5p"


def build_experiments(chip_type: str, world_size: int, steps: int = 60) -> list[dict]:
    xla_flags, cfg_prefix = chip_xla_flags_and_prefix(chip_type)

    def _exp(
        name: str,
        lane: str,
        seq_len: int,
        tok_per_rank: int,
        ga: int,
        contiguous_full_seq: bool = True,
        run_steps: int = steps,
    ) -> dict:
        mb = tok_per_rank // seq_len
        return {
            "name": name,
            "lane": lane,
            "seq_len": seq_len,
            "mb": mb,
            "tok_per_rank": tok_per_rank,
            "ga": ga,
            "contiguous_full_seq": contiguous_full_seq,
            "workdir": "/torchtitan-native",
            "env": {
                "LIBTPU_INIT_ARGS": xla_flags,
                "TT_CONTIGUOUS_FULL_SEQ": "1" if contiguous_full_seq else "0",
            },
            "args": [
                "--module=torchtitan.experiments.tpu.examples.flint",
                f"--config={cfg_prefix}_s{seq_len}",
                f"--training.max_context_length={seq_len}",
                f"--training.steps={run_steps}",
                "--training.mixed_precision_reduce=bfloat16",
                "--metrics.log_freq=1",
                "--debug.seed=42",
                "--tpu_config.mfu_benchmark_start_step=11",
                f"--training.num_tokens_per_microbatch_per_dp_rank={tok_per_rank}",
                f"--training.num_tokens_per_train_step={ga * world_size * tok_per_rank}",
            ],
        }

    # Priority order: 2K, 16K, 32K, 64K first, then 4K, 8K, then 98K/ga4/EoD-control comparisons
    exps = [
        _exp(
            f"{chip_type}_val60_01_iso65k_s2048_mb32_ga1_fullseq",
            "Iso-65.5K/rank Fixed-Recipe Context Ablation: 2K (mb=32, ga=1, True Unskipped Full-Seq, 60 steps)",
            seq_len=2048,
            tok_per_rank=65536,
            ga=1,
            contiguous_full_seq=True,
        ),
        _exp(
            f"{chip_type}_val60_02_iso65k_s16384_mb4_ga1_fullseq",
            "Iso-65.5K/rank Fixed-Recipe Context Ablation: 16K (mb=4, ga=1, True Unskipped Full-Seq, 60 steps)",
            seq_len=16384,
            tok_per_rank=65536,
            ga=1,
            contiguous_full_seq=True,
        ),
        _exp(
            f"{chip_type}_val60_03_iso65k_s32768_mb2_ga1_fullseq",
            "Iso-65.5K/rank Fixed-Recipe Context Ablation: 32K (mb=2, ga=1, True Unskipped Full-Seq, 60 steps)",
            seq_len=32768,
            tok_per_rank=65536,
            ga=1,
            contiguous_full_seq=True,
        ),
        _exp(
            f"{chip_type}_val60_04_iso65k_s65536_mb1_ga1_fullseq",
            "Iso-65.5K/rank Fixed-Recipe Context Ablation: 64K (mb=1, ga=1, True Unskipped Full-Seq, 60 steps)",
            seq_len=65536,
            tok_per_rank=65536,
            ga=1,
            contiguous_full_seq=True,
        ),
        _exp(
            f"{chip_type}_val60_05_iso65k_s4096_mb16_ga1_fullseq",
            "Iso-65.5K/rank Fixed-Recipe Context Ablation: 4K (mb=16, ga=1, True Unskipped Full-Seq, 60 steps)",
            seq_len=4096,
            tok_per_rank=65536,
            ga=1,
            contiguous_full_seq=True,
        ),
        _exp(
            f"{chip_type}_val60_06_iso65k_s8192_mb8_ga1_fullseq",
            "Iso-65.5K/rank Fixed-Recipe Context Ablation: 8K (mb=8, ga=1, True Unskipped Full-Seq, 60 steps)",
            seq_len=8192,
            tok_per_rank=65536,
            ga=1,
            contiguous_full_seq=True,
        ),
        _exp(
            f"{chip_type}_val60_07_ctrl_s65536_mb1_ga1_c4eod",
            "64K Short-Doc EoD Packing Control (c4_test document_aware_skipping=True vs val60_04 Full-Seq, 60 steps)",
            seq_len=65536,
            tok_per_rank=65536,
            ga=1,
            contiguous_full_seq=False,
        ),
        _exp(
            f"{chip_type}_val60_08_opt_s65536_mb1_ga4_fullseq",
            "64K High-GA Production Batch (mb=1, ga=4, True Unskipped Full-Seq, 60 steps)",
            seq_len=65536,
            tok_per_rank=65536,
            ga=4,
            contiguous_full_seq=True,
        ),
    ]
    if chip_type == "v7x":
        exps.extend([
            _exp(
                "v7x_val60_09_opt_s2048_mb48_ga2_fullseq",
                "v7x 2K Peak-Batch 98.3K/rank (mb=48, ga=2, True Unskipped Full-Seq, 60 steps)",
                seq_len=2048,
                tok_per_rank=98304,
                ga=2,
                contiguous_full_seq=True,
            ),
            _exp(
                "v7x_val60_10_opt_s16384_mb6_ga2_fullseq",
                "v7x 16K Peak-Batch 98.3K/rank (mb=6, ga=2, True Unskipped Full-Seq, 60 steps)",
                seq_len=16384,
                tok_per_rank=98304,
                ga=2,
                contiguous_full_seq=True,
            ),
            _exp(
                "v7x_val60_11_opt_s32768_mb3_ga2_fullseq",
                "v7x 32K Peak-Batch 98.3K/rank (mb=3, ga=2, True Unskipped Full-Seq, 60 steps)",
                seq_len=32768,
                tok_per_rank=98304,
                ga=2,
                contiguous_full_seq=True,
            ),
        ])
    return exps


def _kill_proc_group(proc: subprocess.Popen) -> None:
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass
    subprocess.run(["pkill", "-9", "-f", "torchtitan.experiments.tpu.train"], check=False)


def run_with_oom_watchdog(
    cmd: list[str],
    workdir: str,
    env: dict[str, str],
    node_rank: int,
    per_run_timeout: int,
    idle_timeout_s: float = 1800.0,
    abort_fn: Callable[[], bool] | None = None,
    oom_info: dict | None = None,
) -> tuple[int, list[str]]:
    """Runs `cmd`, killing it on timeout, inactivity, fatal markers, or (if given) when `abort_fn()` is True.

    When `oom_info` is given, it is filled on every host with `oom` (bool) and `overflow_gib` (largest HBM
    shortfall reported by XLA:TPU, or None).
    """
    log_chunks: list[str] = []
    proc = subprocess.Popen(
        cmd,
        cwd=workdir,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        start_new_session=True,
    )
    assert proc.stdout is not None
    state = {"last_line": time.time(), "fatal_ts": None}
    fatal_markers = (
        "RESOURCE_EXHAUSTED",
        "RuntimeProgramAllocationFailure",
        "Ran out of memory in memory space",
        "Out of memory",
        "ChildFailedError",
        "local_exit_code: 1",
    )

    def _watchdog() -> None:
        t_start = time.time()
        last_abort_check = 0.0
        while proc.poll() is None:
            now = time.time()
            if now - t_start > per_run_timeout or now - state["last_line"] > idle_timeout_s:
                _kill_proc_group(proc)
                return
            if state["fatal_ts"] is not None and now - state["fatal_ts"] > 10.0:
                _kill_proc_group(proc)
                return
            if abort_fn is not None and now - last_abort_check >= 6.0:
                last_abort_check = now
                try:
                    should_abort = abort_fn()
                except Exception:
                    should_abort = False
                if should_abort:
                    print("[flint-val60] Coordinator abort received; stopping this attempt on this host", flush=True)
                    _kill_proc_group(proc)
                    return
            time.sleep(2.0)

    threading.Thread(target=_watchdog, daemon=True).start()
    for line in proc.stdout:
        state["last_line"] = time.time()
        if state["fatal_ts"] is None and any(m in line for m in fatal_markers):
            state["fatal_ts"] = time.time()
        if oom_info is not None:
            if any(m in line for m in OOM_MARKERS):
                oom_info["oom"] = True
            if "hbm" in line or "allocate" in line:
                gib = parse_hbm_overflow_gib(line)
                if gib is not None:
                    oom_info["oom"] = True
                    oom_info["overflow_gib"] = max(gib, oom_info.get("overflow_gib") or 0.0)
        sys.stdout.write(line)
        sys.stdout.flush()
        if node_rank == 0:
            log_chunks.append(line)
    rc = proc.wait(timeout=30)
    subprocess.run(["pkill", "-9", "-f", "torchtitan.experiments.tpu.train"], check=False)
    return rc, log_chunks


def parse_metrics(log_text: str, chip_type: str, seq_len: int, world_size: int) -> dict:
    causal_flops = flint_causal_flops_per_token(seq_len)
    palm_flops = flint_palm_flops_per_token(seq_len)
    param_flops = flint_param_only_flops_per_token()
    peak_tf = PEAK_BF16_TFLOPS_PER_RANK.get(chip_type, 459.0) * 1e12
    ranks_per_chip = RANKS_PER_CHIP.get(chip_type, 1)

    steps = []
    seen_steps: set[int] = set()
    for line in log_text.splitlines():
        m = STEP_RE.search(line)
        if not m:
            continue
        step = int(m.group("step"))
        if step in seen_steps:
            continue
        seen_steps.add(step)
        loss = float(m.group("loss"))
        tps = float(m.group("tps").replace(",", ""))
        tf_m = TFLOPS_RE.search(line)
        mfu_m = MFU_RE.search(line)
        mem_m = MEM_RE.search(line)
        causal_mfu = 100.0 * causal_flops * tps / peak_tf
        palm_mfu = 100.0 * palm_flops * tps / peak_tf
        param_mfu = 100.0 * param_flops * tps / peak_tf
        logged_mfu = float(mfu_m.group("mfu")) if mfu_m else palm_mfu
        logged_tf = float(tf_m.group("tflops")) if tf_m else round(palm_flops * tps / 1e12, 2)
        mem_gib = float(mem_m.group("mem")) if mem_m else None
        steps.append(
            {
                "step": step,
                "loss": loss,
                "tps_per_rank": round(tps, 1),
                "tps_per_chip": round(tps * ranks_per_chip, 1),
                "causal_mfu_pct": round(causal_mfu, 2),
                "palm_mfu_pct": round(palm_mfu, 2),
                "param_only_mfu_pct": round(param_mfu, 2),
                "logged_tflops": logged_tf,
                "logged_mfu_pct": round(logged_mfu, 2),
                "mem_gib": mem_gib,
            }
        )

    steps.sort(key=lambda s: s["step"])
    steady = [s for s in steps if s["step"] >= 11]
    if not steady and len(steps) >= 4:
        steady = steps[len(steps) // 2 :]
    if not steady:
        return {
            "seq_len": seq_len,
            "causal_gflops_per_tok": round(causal_flops / 1e9, 4),
            "palm_gflops_per_tok": round(palm_flops / 1e9, 4),
            "param_only_gflops_per_tok": round(param_flops / 1e9, 4),
            "num_logged_steps": len(steps),
            "num_steady_steps": 0,
            "steady_tps_per_rank": 0.0,
            "steady_tps_per_chip": 0.0,
            "steady_slice_tps": 0.0,
            "steady_causal_mfu_pct": 0.0,
            "steady_palm_mfu_pct": 0.0,
            "steady_param_only_mfu_pct": 0.0,
        }

    tps_vals = [s["tps_per_rank"] for s in steady]
    avg_tps = sum(tps_vals) / len(tps_vals)
    p50_tps = statistics.median(tps_vals)
    std_tps = statistics.stdev(tps_vals) if len(tps_vals) > 1 else 0.0
    avg_logged_mfu = sum(s["logged_mfu_pct"] for s in steady) / len(steady)

    # Engineering-team measurement windows on the same run (google-pytorch/torchtitan PR #122: median of
    # steps 3..8; PR #123: mean of steps 2..10), MFU with the TorchTitan counter (= PaLM column convention).
    eng_windows: dict = {}
    w122 = [s["tps_per_rank"] for s in steps if 3 <= s["step"] <= 8]
    if w122:
        med122 = statistics.median(w122)
        eng_windows["eng_pr122_tps_per_rank"] = round(med122, 1)
        eng_windows["eng_pr122_palm_mfu_pct"] = round(100.0 * palm_flops * med122 / peak_tf, 2)
    w123 = [s["tps_per_rank"] for s in steps if 2 <= s["step"] <= 10]
    if w123:
        mean123 = sum(w123) / len(w123)
        eng_windows["eng_pr123_tps_per_rank"] = round(mean123, 1)
        eng_windows["eng_pr123_palm_mfu_pct"] = round(100.0 * palm_flops * mean123 / peak_tf, 2)

    return {
        "seq_len": seq_len,
        "causal_gflops_per_tok": round(causal_flops / 1e9, 4),
        "palm_gflops_per_tok": round(palm_flops / 1e9, 4),
        "param_only_gflops_per_tok": round(param_flops / 1e9, 4),
        "num_logged_steps": len(steps),
        "num_steady_steps": len(steady),
        "steady_window": f"{steady[0]['step']}..{steady[-1]['step']}",
        "first_loss": steps[0]["loss"],
        "final_loss": steps[-1]["loss"],
        "steady_tps_per_rank": round(avg_tps, 1),
        "steady_p50_tps_per_rank": round(p50_tps, 1),
        "steady_std_tps_per_rank": round(std_tps, 1),
        "steady_tps_per_chip": round(avg_tps * ranks_per_chip, 1),
        "steady_p50_tps_per_chip": round(p50_tps * ranks_per_chip, 1),
        "steady_slice_tps": round(avg_tps * world_size, 1),
        "steady_causal_mfu_pct": round(100.0 * causal_flops * avg_tps / peak_tf, 2),
        "steady_palm_mfu_pct": round(100.0 * palm_flops * avg_tps / peak_tf, 2),
        "steady_param_only_mfu_pct": round(100.0 * param_flops * avg_tps / peak_tf, 2),
        "steady_logged_mfu_pct": round(avg_logged_mfu, 2),
        "peak_mem_gib": max((s["mem_gib"] for s in steps if s["mem_gib"] is not None), default=None),
        **eng_windows,
        "steps": steps,
    }


def write_summary(
    out_dir: Path,
    bench_dir: Path,
    suite: str,
    slice_name: str,
    chip_type: str,
    world_size: int,
    results: list[dict],
    notes: list[str] | None = None,
) -> None:
    ranks_per_chip = RANKS_PER_CHIP.get(chip_type, 1)
    num_chips = world_size // ranks_per_chip
    payload = {
        "updated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "suite": suite,
        "slice_name": slice_name,
        "chip_type": chip_type,
        "num_chips": num_chips,
        "world_size_ranks": world_size,
        "ranks_per_chip": ranks_per_chip,
        "peak_bf16_tflops_per_rank": PEAK_BF16_TFLOPS_PER_RANK.get(chip_type),
        "peak_bf16_tflops_per_chip": PEAK_BF16_TFLOPS_PER_RANK.get(chip_type, 459.0) * ranks_per_chip,
        "notes": list(notes or []),
        "results": results,
    }
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2))
    eng_cols = suite == ENG_PARITY_SUITE
    header = (
        "| Experiment | Seq | Seq Mode | Steady Steps | **tok/s/chip** | tok/s/rank (±std) | Slice tok/s | Causal GF/tok "
        "| **Steady Causal MFU** | PaLM GF/tok | Steady PaLM MFU | `6*N_act` MFU | HBM (GiB) | Loss (`1 -> 60`) | Status |"
    )
    sep = "|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|"
    if eng_cols:
        header = header.replace(
            " | HBM (GiB)", " | PR #122-window PaLM MFU (median 3..8) | PR #123-window PaLM MFU (mean 2..10) | HBM (GiB)"
        )
        sep = "|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|"
    md_lines = [
        f"# RSI Flint (`RecursiveSI/flint` 40B-A2.5B MoE) 60-Step Customer Validation (`{suite.upper()}`) — `{slice_name}`",
        "",
        f"- **Updated (UTC):** `{payload['updated_at_utc']}`",
        f"- **Hardware:** `{num_chips}` × `{chip_type}` chips (`{world_size}` TensorCore ranks, `{payload['peak_bf16_tflops_per_chip']}` BF16 TFLOP/s/chip)",
        "- **Methodology:** `60` optimizer steps per run (`log_freq=1`), Steady-State Mean over **50 post-warmup steps (`11..60`)**; True Unskipped Contiguous `seq_len` Causal Attention (`TT_CONTIGUOUS_FULL_SEQ=1`) unless marked `c4eod`.",
        *[f"- {note}" for note in (notes or [])],
        "",
        header,
        sep,
    ]
    for r in results:
        m = r.get("metrics", {})
        mode = "Full-Seq" if r.get("contiguous_full_seq", True) else "C4-EoD"
        loss_str = f"{m.get('first_loss', '-')} -> {m.get('final_loss', '-')}" if m.get("num_logged_steps") else "-"
        eng_part = (
            f"{m.get('eng_pr122_palm_mfu_pct', 0.0):.2f}% | {m.get('eng_pr123_palm_mfu_pct', 0.0):.2f}% | "
            if eng_cols
            else ""
        )
        md_lines.append(
            f"| `{r['name']}` | `{r['seq_len']}` | `{mode}` | `{m.get('num_steady_steps', 0)}` (`{m.get('steady_window', '-')}`) | "
            f"**{m.get('steady_tps_per_chip', 0.0):,.1f}** | {m.get('steady_tps_per_rank', 0.0):,.1f} (±{m.get('steady_std_tps_per_rank', 0.0):.1f}) | "
            f"{m.get('steady_slice_tps', 0.0):,.0f} | `{m.get('causal_gflops_per_tok', '-')}` | "
            f"**{m.get('steady_causal_mfu_pct', 0.0):.2f}%** | `{m.get('palm_gflops_per_tok', '-')}` | "
            f"{m.get('steady_palm_mfu_pct', 0.0):.2f}% | {m.get('steady_param_only_mfu_pct', 0.0):.2f}% | "
            f"{eng_part}{m.get('peak_mem_gib') or '-'} | {loss_str} | `{r['status']}` |"
        )
    md_text = "\n".join(md_lines) + "\n"
    (out_dir / "SUMMARY.md").write_text(md_text)
    if bench_dir.exists():
        (bench_dir / f"flint_{slice_name}_{suite}_summary.json").write_text(json.dumps(payload, indent=2))
        (bench_dir / f"flint_{slice_name}_{suite}_SUMMARY.md").write_text(md_text)


# ==============================================================================
# Long-context suite runtime (multi-host lockstep coordinator + OOM-only fallback)
# ==============================================================================
LONGCTX_NOTES = (
    "**Long-context extension (`128K` / `256K`):** Flint in this TorchTitan TPU stack rejects context and tensor "
    "parallelism, so each rank trains on one full `seq_len` sequence per microbatch (`mb=1` -> `131,072` / `262,144` "
    "tokens/rank/microbatch vs a constant `65,536` in the `2K..64K` rows); global batch = ranks x `seq_len` tokens/step.",
    "`_attn{k}` = layers whose Splash-attention output is kept for backward (`49` = the `2K..64K` recipe). A smaller "
    "`k` is used only after an HBM OOM at the larger one; recomputed attention FLOPs are **not** counted in any MFU column.",
)


class LongCtxCoordinator:
    """Node-0 HTTP key/value store: every host runs the same attempt on the same rendezvous port and takes the
    same OOM-fallback decision, so one failed attempt can never strand other hosts on stale ports."""

    def __init__(self, node_rank: int, nnodes: int, master_ip: str, port: int) -> None:
        self.node_rank = node_rank
        self.nnodes = nnodes
        self._kv: dict[str, str] = {}
        self._lock = threading.Lock()
        if node_rank == 0:
            self._serve(port)
        self.url = f"http://{master_ip}:{port}"

    def _serve(self, port: int) -> None:
        coord = self

        class _Handler(BaseHTTPRequestHandler):
            def log_message(self, fmt, *args) -> None:  # keep pod logs clean
                return

            def _reply(self, payload: dict) -> None:
                raw = json.dumps(payload).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def do_GET(self) -> None:
                parsed = urllib.parse.urlparse(self.path)
                qs = urllib.parse.parse_qs(parsed.query)
                if parsed.path == "/get" and "key" in qs:
                    self._reply({"val": coord.get(qs["key"][0])})
                    return
                self.send_response(404)
                self.end_headers()

            def do_POST(self) -> None:
                parsed = urllib.parse.urlparse(self.path)
                qs = urllib.parse.parse_qs(parsed.query, keep_blank_values=True)
                if parsed.path == "/set" and "key" in qs and "val" in qs:
                    coord.set(qs["key"][0], qs["val"][0])
                    self._reply({"ok": True})
                    return
                self.send_response(404)
                self.end_headers()

        server = ThreadingHTTPServer(("0.0.0.0", port), _Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        print(f"[flint-val60] Long-context coordinator listening on :{port}", flush=True)

    def set(self, key: str, val: str, timeout_s: float = 600.0) -> None:
        if self.node_rank == 0:
            with self._lock:
                self._kv[key] = val
            return
        query = urllib.parse.urlencode({"key": key, "val": val})
        deadline = time.time() + timeout_s
        while True:
            try:
                req = urllib.request.Request(f"{self.url}/set?{query}", data=b"", method="POST")
                with urllib.request.urlopen(req, timeout=10.0) as resp:
                    resp.read()
                return
            except Exception:
                if time.time() > deadline:
                    raise
                time.sleep(2.0)

    def get(self, key: str) -> str:
        if self.node_rank == 0:
            with self._lock:
                return self._kv.get(key, "")
        query = urllib.parse.urlencode({"key": key})
        try:
            with urllib.request.urlopen(f"{self.url}/get?{query}", timeout=10.0) as resp:
                return json.loads(resp.read().decode("utf-8")).get("val", "")
        except Exception:
            return ""

    def wait(self, key: str, timeout_s: float) -> str:
        deadline = time.time() + timeout_s
        while True:
            val = self.get(key)
            if val:
                return val
            if time.time() > deadline:
                raise TimeoutError(f"Timed out after {timeout_s:.0f}s waiting for coordinator key {key!r}")
            time.sleep(3.0)

    def barrier(self, name: str, timeout_s: float) -> None:
        self.set(f"{name}/r{self.node_rank}", "1")
        for r in range(self.nnodes):
            self.wait(f"{name}/r{r}", timeout_s)


def next_attn_index_after_oom(ladder: tuple[int, ...], idx: int, overflow_gib: float | None, seq_len: int) -> int:
    """Index of the next attention-save count to try after an HBM OOM at `ladder[idx]` (-1: nothing left)."""
    remaining = list(range(idx + 1, len(ladder)))
    if not remaining:
        return -1
    if not overflow_gib or overflow_gib <= 0:
        return remaining[0]
    per_layer = attn_saved_gib_per_layer(seq_len)
    # Drop enough saved layers to cover the reported shortfall plus >= 1 GiB of headroom.
    max_k = ladder[idx] - math.ceil((overflow_gib + 1.0) / per_layer)
    for j in remaining:
        if ladder[j] <= max_k:
            return j
    return remaining[-1]  # Predicted short even there; still measure the smallest setting once.


def build_longctx_attempt(
    chip_type: str,
    cfg_prefix: str,
    xla_flags: str,
    tag: str,
    seq_len: int,
    n_attn: int,
    world_size: int,
    steps: int,
) -> dict:
    total_layers = FLINT_NUM_FULL_LAYERS + FLINT_NUM_SLIDING_LAYERS
    if n_attn >= total_layers:
        policy = f"val60 recipe: Splash output saved in all {total_layers} layers"
    elif n_attn > 0:
        policy = f"HBM fallback: Splash output saved in last {n_attn}/{total_layers} layers, recomputed in the rest"
    else:
        policy = f"HBM fallback: Splash attention recomputed in backward in all {total_layers} layers"
    return {
        "name": f"{chip_type}_val60_{tag}_mb1_ga1_fullseq_attn{n_attn}",
        "group": tag,
        "lane": (
            f"Long-Context Extension {seq_len // 1024}K (mb=1 -> {seq_len:,} tok/rank, ga=1, "
            f"True Unskipped Full-Seq, {steps} steps; {policy})"
        ),
        "seq_len": seq_len,
        "mb": 1,
        "tok_per_rank": seq_len,
        "ga": 1,
        "contiguous_full_seq": True,
        "attn_saved_layers": n_attn,
        "workdir": "/torchtitan-native",
        "env": {"LIBTPU_INIT_ARGS": xla_flags, "TT_CONTIGUOUS_FULL_SEQ": "1"},
        "args": [
            "--module=torchtitan.experiments.tpu.examples.flint",
            f"--config={cfg_prefix}_s{seq_len}_attn{n_attn}",
            f"--training.max_context_length={seq_len}",
            f"--training.steps={steps}",
            "--training.mixed_precision_reduce=bfloat16",
            "--metrics.log_freq=1",
            "--debug.seed=42",
            "--tpu_config.mfu_benchmark_start_step=11",
            f"--training.num_tokens_per_microbatch_per_dp_rank={seq_len}",
            f"--training.num_tokens_per_train_step={world_size * seq_len}",
        ],
    }


def load_previous_longctx_passes(output_dir: Path, slice_name: str) -> dict[str, dict]:
    """PASS rows from earlier runs of this suite on this slice, so a Spot-preempted Job resumes instead of redoing them."""
    passed: dict[str, dict] = {}
    for summary in sorted(output_dir.glob(f"{slice_name}_{LONGCTX_SUITE}_*/summary.json")):
        try:
            payload = json.loads(summary.read_text())
        except Exception:
            continue
        for row in payload.get("results", []):
            if row.get("status") == "PASS" and row.get("group"):
                passed[row["group"]] = {**row, "resumed_from": row.get("resumed_from") or summary.parent.name}
    return passed


def wait_for_hosts_done(
    coord: LongCtxCoordinator, key: str, nnodes: int, abort_after_s: float = 300.0, timeout_s: float = 900.0
) -> None:
    """Node 0: wait until every host finished attempt `key`, aborting hosts still stuck in collectives."""
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        pending = [r for r in range(1, nnodes) if not coord.get(f"{key}/done/r{r}")]
        if not pending:
            return
        if time.time() - t0 > abort_after_s and not coord.get(f"{key}/abort"):
            print(f"[flint-val60] Hosts {pending} still running {key}; sending abort", flush=True)
            coord.set(f"{key}/abort", "1")
        time.sleep(3.0)
    print(f"[flint-val60] WARNING: not every host reported finishing {key} within {timeout_s:.0f}s", flush=True)


def run_longctx_suite(
    args: argparse.Namespace, node_rank: int, world_size: int, master_ip: str, out_dir: Path, bench_dir: Path
) -> int:
    for _, _, ladder, _ in LONGCTX_GROUPS:
        assert set(ladder) <= set(LONGCTX_ATTN_SAVE_CHOICES), ladder
    coord = LongCtxCoordinator(node_rank, args.nnodes, master_ip, args.coord_port)
    xla_flags, cfg_prefix = chip_xla_flags_and_prefix(args.chip_type)
    notes = list(LONGCTX_NOTES)
    results: list[dict] = []

    if node_rank == 0:
        previous = load_previous_longctx_passes(Path(args.output_dir), args.slice_name)
        results.extend(previous[tag] for tag, *_ in LONGCTX_GROUPS if tag in previous)
        if results:
            write_summary(out_dir, bench_dir, args.suite, args.slice_name, args.chip_type, world_size, results, notes)
        coord.set("resume_skip", ",".join(sorted(previous)) or "none")
    skip_raw = coord.wait("resume_skip", timeout_s=1800)
    skip = set() if skip_raw == "none" else set(skip_raw.split(","))

    attempt_no = 0
    for tag, seq_len, ladder, attempt_timeout in LONGCTX_GROUPS:
        if tag in skip:
            print(f"[flint-val60] {tag}: PASS already recorded by an earlier run of this suite; skipping", flush=True)
            continue
        idx = 0
        retries_left = 1
        while idx < len(ladder):
            n_attn = ladder[idx]
            attempt_no += 1
            key = f"a{attempt_no:02d}_{tag}_attn{n_attn}"
            exp = build_longctx_attempt(
                args.chip_type, cfg_prefix, xla_flags, tag, seq_len, n_attn, world_size, args.steps
            )
            time.sleep(10)  # Let the previous attempt release the TPU devices on every host.
            coord.barrier(f"{key}/start", timeout_s=3600)
            port = args.base_port + 200 + attempt_no * 5
            cmd = [
                "torchrun",
                f"--nnodes={args.nnodes}",
                f"--nproc_per_node={args.nproc_per_node}",
                "--rdzv_backend=static",
                f"--rdzv_endpoint={args.master_addr}:{port}",
                f"--node_rank={node_rank}",
                "-m",
                "torchtitan.experiments.tpu.train",
                *exp["args"],
            ]
            env = os.environ.copy()
            env["PYTHONPATH"] = exp["workdir"]
            env.update(exp["env"])
            print(f"\n=== [longctx attempt {attempt_no}] {exp['name']} ({exp['lane']}) ===", flush=True)
            if node_rank == 0:
                def abort_fn(key: str = key) -> bool:
                    return any(coord.get(f"{key}/done/r{r}") not in ("", "0") for r in range(1, args.nnodes))
            else:
                def abort_fn(key: str = key) -> bool:
                    return bool(coord.get(f"{key}/abort"))
            oom_info: dict = {"oom": False, "overflow_gib": None}
            t0 = time.time()
            try:
                rc, log_chunks = run_with_oom_watchdog(
                    cmd,
                    exp["workdir"],
                    env,
                    node_rank,
                    max(args.per_run_timeout, attempt_timeout),
                    idle_timeout_s=1800.0,
                    abort_fn=abort_fn,
                    oom_info=oom_info,
                )
            except Exception as exc:
                rc, log_chunks = -1, [f"[flint-val60] Exception: {exc}\n"]
            dur = time.time() - t0
            if oom_info["oom"]:
                coord.set(f"{key}/oom/r{node_rank}", f"{oom_info['overflow_gib'] or 0.0:.3f}")
            coord.set(f"{key}/done/r{node_rank}", str(rc))

            if node_rank == 0:
                if rc != 0:
                    coord.set(f"{key}/abort", "1")
                wait_for_hosts_done(coord, key, args.nnodes)
                raw_log = "".join(log_chunks)
                (out_dir / f"{exp['name']}.log").write_text(raw_log)
                metrics = parse_metrics(raw_log, args.chip_type, seq_len=seq_len, world_size=world_size)
                any_oom = bool(oom_info["oom"])
                overflow = float(oom_info["overflow_gib"] or 0.0)
                for r in range(1, args.nnodes):
                    reported = coord.get(f"{key}/oom/r{r}")
                    if reported:
                        any_oom = True
                        overflow = max(overflow, float(reported))
                if rc == 0 and metrics.get("num_steady_steps", 0) >= 20:
                    decision, status = "PASS", "PASS"
                elif any_oom:
                    nxt = next_attn_index_after_oom(ladder, idx, overflow or None, seq_len)
                    decision = f"OOM:{nxt}"
                    status = f"OOM(rc={rc})" + (f" -> retry attn{ladder[nxt]}" if nxt >= 0 else " -> no smaller setting")
                else:
                    decision, status = "FAIL", f"FAIL(rc={rc})"
                results.append(
                    {
                        "name": exp["name"],
                        "group": tag,
                        "lane": exp["lane"],
                        "seq_len": seq_len,
                        "mb": 1,
                        "tok_per_rank": seq_len,
                        "ga": 1,
                        "contiguous_full_seq": True,
                        "attn_saved_layers": n_attn,
                        "status": status,
                        "decision": decision,
                        "hbm_overflow_gib": round(overflow, 3) if any_oom else None,
                        "return_code": rc,
                        "duration_s": round(dur, 1),
                        "command": cmd,
                        "env_overrides": exp["env"],
                        "metrics": metrics,
                    }
                )
                write_summary(out_dir, bench_dir, args.suite, args.slice_name, args.chip_type, world_size, results, notes)
                print(
                    f"[flint-val60] SAVED {exp['name']}: status={status} "
                    f"tps/chip={metrics.get('steady_tps_per_chip', 0)} "
                    f"causal_mfu={metrics.get('steady_causal_mfu_pct', 0)}% "
                    f"steady_steps={metrics.get('num_steady_steps', 0)}",
                    flush=True,
                )
                coord.set(key, decision)
            decision = coord.wait(key, timeout_s=4 * 3600)
            if decision == "PASS":
                break
            if decision.startswith("OOM:"):
                idx = int(decision.split(":", 1)[1])
                if idx < 0:
                    break
                retries_left = 1
                continue
            if retries_left > 0:  # Non-OOM failure (e.g. a transient rendezvous error): retry the same setting once.
                retries_left -= 1
                continue
            break

    try:
        coord.barrier("longctx/all_done", timeout_s=3600)
    except TimeoutError:
        if node_rank == 0:
            raise
    if node_rank == 0:
        time.sleep(60)  # Keep the coordinator up while the other hosts finish their last polls.
    return 0


# ==============================================================================
# Engineering-parity 2K suite (apples-to-apples vs google-pytorch/torchtitan PR #122 / PR #123)
# ==============================================================================
ENG_PARITY_SUITE = "eng_parity_2k"
ENG_HF_MODEL_PATH = "/models/models/RecursiveSI/flint"
# PR #123 best arm (r3) libtpu flags, verbatim from the PR description.
ENG_PR123_LIBTPU_FLAGS = (
    "xla_tpu_enable_sparse_core_collective_aggregator=true "
    "xla_tpu_enable_latency_hiding_layer_scheduler=true "
    "xla_tpu_use_single_sparse_core_for_reduce_scatter_offload=false "
    "xla_tpu_enable_sparse_core_collective_offload_nd_reduce_scatter=true "
    "xla_tpu_enable_sparse_core_hierarchical_reduce_scatter=true "
    "xla_tpu_dvfs_p_state=7"
)
ENG_PARITY_NOTES = (
    "**Apples-to-apples with google-pytorch/torchtitan PR #122 (native lane, 16.23% MFU) and PR #123 (HF backend, "
    "16.29% MFU):** same slice shape (`tpu7x-2x2x2` = v7x-8, `16` TensorCore ranks, SimpleFSDP over 16), same "
    "`seq_len=2048`, same global batch `512 x 2048 = 1,048,576` tok/step (`mb=32`, `ga=1`) unless a row says "
    "otherwise, same `c4_test` EoD packing as upstream (`C4-EoD`) unless marked `Full-Seq`, and the same MFU "
    "definition (TorchTitan counter `17.385` GFLOP/token = **PaLM GF/tok** column, `1,153.5` BF16 TFLOP/s per "
    "TensorCore).",
    "Each run is `60` steps; the steady column averages steps `11..60`. The two PR-window columns re-score the same "
    "run with each PR's own window (PR #122: median of steps `3..8`; PR #123: mean of steps `2..10`).",
    "`eng_*` rows reproduce the PR recipes verbatim with this repo's image; `abl_*` rows add one group of changes at "
    "a time; `ours_*` rows are the full recipe of this repo (`ours_05` also changes the batch, so it is not iso-batch).",
)

ENG_ABLATION_CONFIG_PATCH = '''


# ==============================================================================
# Engineering-parity (PR #122 / #123) 2K ablation config: v014 + SBDP + BF16 RoPE only
# ==============================================================================
def flint_40b_engabl_v014_sbdp_rope() -> TpuGraphTrainer.Config:
  """flint_40b_v014_tile1024 + scale_before_down_proj + BF16 RoPE (no routing save, eager optimizer)."""
  config = flint_40b_v014_tile1024()
  _val60_set_max_context_length(config, 2048)
  config.model_spec.model = (
      TpuRoutedExpertsConverter.Config().build().convert(config.model_spec.model)
  )
  for _, r, _, _ in config.model_spec.model.traverse(TpuRoutedExperts.Config):
    r.scale_before_down_proj = True
  if _VAL60_BF16_ROPE not in config.override.imports:
    config.override.imports.append(_VAL60_BF16_ROPE)
  return config


__all__.extend(["flint_40b_engabl_v014_sbdp_rope"])
'''


def build_eng_parity_experiments(world_size: int, steps: int) -> list[dict]:
    """2K runs on v7x-8 that line up with PR #122 / PR #123, ordered by priority."""
    our_flags, _ = chip_xla_flags_and_prefix("v7x")
    seq = 2048

    def _native(name: str, lane: str, config: str, tok_per_rank: int, ga: int, full_seq: bool, our_stack: bool) -> dict:
        env = {"TT_CONTIGUOUS_FULL_SEQ": "1" if full_seq else "0"}
        args = [
            "--module=torchtitan.experiments.tpu.examples.flint",
            f"--config={config}",
            f"--training.max_context_length={seq}",
            f"--training.steps={steps}",
        ]
        if our_stack:
            # Our collective/async flag set on top of the config's own libtpu flags, plus BF16 gradient reduce-scatter.
            env["LIBTPU_INIT_ARGS"] = our_flags
            args.append("--training.mixed_precision_reduce=bfloat16")
        args += [
            "--metrics.log_freq=1",
            "--debug.seed=42",
            "--tpu_config.mfu_benchmark_start_step=11",
            f"--training.num_tokens_per_microbatch_per_dp_rank={tok_per_rank}",
            f"--training.num_tokens_per_train_step={ga * world_size * tok_per_rank}",
        ]
        return {
            "name": name,
            "lane": lane,
            "seq_len": seq,
            "mb": tok_per_rank // seq,
            "tok_per_rank": tok_per_rank,
            "ga": ga,
            "contiguous_full_seq": full_seq,
            "workdir": "/torchtitan-native",
            "env": env,
            "args": args,
        }

    pr123 = {
        "name": "v7x8_eng_04_pr123_hf_attnac_r3_c4eod_mb32_ga1",
        "lane": (
            "PR #123 verbatim (HF backend r3: hf_moe_auto_max_attnac, TT_HF_GMM tile_m 1024, CE b-block 1024, "
            "Splash 512 blocks, hier-SC RS + dvfs flags)"
        ),
        "seq_len": seq,
        "mb": 32,
        "tok_per_rank": 65536,
        "ga": 1,
        "contiguous_full_seq": False,
        "workdir": "/torchtitan-hf",
        "env": {"TT_HF_GMM": "pallas_grouped_mm_tile_m=1024,pallas_grouped_mm_tgmm_tiling=512:512:512"},
        "args": [
            "--module=torchtitan.experiments.tpu.transformers_modeling_backend",
            "--config=hf_moe_auto_max_attnac",
            f"--hf-model={ENG_HF_MODEL_PATH}",
            f"--training.max_context_length={seq}",
            f"--training.steps={steps}",
            "--metrics.log_freq=1",
            "--debug.seed=42",
            "--loss.b_block_size=1024",
            "--training.num_tokens_per_microbatch_per_dp_rank=65536",
            f"--training.num_tokens_per_train_step={world_size * 65536}",
            *[
                f"--splash_attention_kernel.{k}=512"
                for k in (
                    "sa_block_q",
                    "sa_block_kv",
                    "sa_block_kv_compute",
                    "sa_block_q_dkv",
                    "sa_block_kv_dkv",
                    "sa_block_kv_dkv_compute",
                )
            ],
            f"--extra-libtpu-init-flags={ENG_PR123_LIBTPU_FLAGS}",
        ],
    }
    return [
        _native(
            "v7x8_eng_01_pr122_v011_n40_c4eod_mb32_ga1",
            "PR #122 verbatim (flint_40b_v011_n40: GraphTrainer split, Pallas GMM tile_m 512, n40 selective remat, "
            "FP32 reduce, eager AdamW, config libtpu flags)",
            "flint_40b_v011_n40", 65536, 1, full_seq=False, our_stack=False,
        ),
        _native(
            "v7x8_ours_02_val60_recipe_c4eod_mb32_ga1",
            "This repo's full 2K recipe at the PR batch (v014 + SBDP + BF16 RoPE + routing save + compiled/fused "
            "AdamW + compiled grad-norm + BF16 reduce + async collective flags)",
            "flint_40b_val60_v7x_s2048", 65536, 1, full_seq=False, our_stack=True,
        ),
        _native(
            "v7x8_ours_03_val60_recipe_fullseq_mb32_ga1",
            "ours_02 with true unskipped full-sequence attention (no EoD block skipping)",
            "flint_40b_val60_v7x_s2048", 65536, 1, full_seq=True, our_stack=True,
        ),
        pr123,
        _native(
            "v7x8_ours_05_val60_recipe_c4eod_mb48_ga2",
            "ours_02 with a larger batch (mb=48 -> 98,304 tok/rank, ga=2 -> 3.15M tok/step); NOT iso-batch",
            "flint_40b_val60_v7x_s2048", 98304, 2, full_seq=False, our_stack=True,
        ),
        _native(
            "v7x8_ours_05b_val60_recipe_c4eod_mb40_ga2",
            "ours_02 with a larger batch that fits 16-way FSDP HBM (mb=40 -> 81,920 tok/rank, ga=2 -> 2.62M tok/step); "
            "NOT iso-batch",
            "flint_40b_val60_v7x_s2048", 81920, 2, full_seq=False, our_stack=True,
        ),
        _native(
            "v7x8_abl_06_v011n40_bf16red_ourflags_c4eod_mb32_ga1",
            "Ablation 1: PR #122 config + BF16 reduce-scatter + async collective flag set",
            "flint_40b_v011_n40", 65536, 1, full_seq=False, our_stack=True,
        ),
        _native(
            "v7x8_abl_07_v014_bf16red_ourflags_c4eod_mb32_ga1",
            "Ablation 2: + hierarchical SparseCore RS, Splash saved in all 49 layers, Pallas GMM tile_m 1024 "
            "(flint_40b_v014_tile1024)",
            "flint_40b_v014_tile1024", 65536, 1, full_seq=False, our_stack=True,
        ),
        _native(
            "v7x8_abl_08_v014_sbdp_rope_bf16red_ourflags_c4eod_mb32_ga1",
            "Ablation 3: + scale_before_down_proj + BF16 RoPE (ours_02 then adds routing save + compiled/fused "
            "optimizer)",
            "flint_40b_engabl_v014_sbdp_rope", 65536, 1, full_seq=False, our_stack=True,
        ),
    ]


def run_lockstep_suite(
    args: argparse.Namespace,
    exps: list[dict],
    node_rank: int,
    world_size: int,
    master_ip: str,
    out_dir: Path,
    bench_dir: Path,
    notes: list[str] | None = None,
) -> int:
    """Runs `exps` with every host in lockstep (same attempt, same fresh rendezvous port, shared abort), retrying a
    failed run once, so a crash on one host cannot desynchronize the hosts for the remaining runs."""
    coord = LongCtxCoordinator(node_rank, args.nnodes, master_ip, args.coord_port)
    results: list[dict] = []
    attempt_no = 0
    for idx, exp in enumerate(exps):
        retries_left = 1
        while True:
            attempt_no += 1
            key = f"s{attempt_no:02d}_{exp['name']}"
            time.sleep(10)  # Let the previous attempt release the TPU devices on every host.
            coord.barrier(f"{key}/start", timeout_s=3600)
            port = args.base_port + 100 + attempt_no * 5
            cmd = [
                "torchrun",
                f"--nnodes={args.nnodes}",
                f"--nproc_per_node={args.nproc_per_node}",
                "--rdzv_backend=static",
                f"--rdzv_endpoint={args.master_addr}:{port}",
                f"--node_rank={node_rank}",
                "-m",
                "torchtitan.experiments.tpu.train",
                *exp["args"],
            ]
            env = os.environ.copy()
            env["PYTHONPATH"] = exp["workdir"]
            env.update(exp["env"])
            print(f"\n=== [{idx + 1}/{len(exps)} attempt {attempt_no}] {exp['name']} ({exp['lane']}) ===", flush=True)
            if node_rank == 0:
                def abort_fn(key: str = key) -> bool:
                    return any(coord.get(f"{key}/done/r{r}") not in ("", "0") for r in range(1, args.nnodes))
            else:
                def abort_fn(key: str = key) -> bool:
                    return bool(coord.get(f"{key}/abort"))
            t0 = time.time()
            try:
                rc, log_chunks = run_with_oom_watchdog(
                    cmd, exp["workdir"], env, node_rank, args.per_run_timeout, idle_timeout_s=1800.0, abort_fn=abort_fn
                )
            except Exception as exc:
                rc, log_chunks = -1, [f"[flint-val60] Exception: {exc}\n"]
            dur = time.time() - t0
            coord.set(f"{key}/done/r{node_rank}", str(rc))

            if node_rank == 0:
                if rc != 0:
                    coord.set(f"{key}/abort", "1")
                wait_for_hosts_done(coord, key, args.nnodes)
                raw_log = "".join(log_chunks)
                (out_dir / f"{exp['name']}.log").write_text(raw_log)
                metrics = parse_metrics(raw_log, args.chip_type, seq_len=exp["seq_len"], world_size=world_size)
                passed = rc == 0 and metrics.get("num_steady_steps", 0) >= 20
                will_retry = (not passed) and retries_left > 0
                if not passed:
                    (out_dir / f"{exp['name']}.attempt{attempt_no}.log").write_text(raw_log)
                status = "PASS" if passed else f"FAIL(rc={rc})" + (" -> retrying once" if will_retry else "")
                row = {
                    "name": exp["name"],
                    "lane": exp["lane"],
                    "seq_len": exp["seq_len"],
                    "mb": exp["mb"],
                    "tok_per_rank": exp["tok_per_rank"],
                    "ga": exp["ga"],
                    "contiguous_full_seq": exp["contiguous_full_seq"],
                    "status": status,
                    "attempt": attempt_no,
                    "return_code": rc,
                    "duration_s": round(dur, 1),
                    "command": cmd,
                    "env_overrides": exp["env"],
                    "metrics": metrics,
                }
                results = [r for r in results if r["name"] != exp["name"]] + [row]
                write_summary(out_dir, bench_dir, args.suite, args.slice_name, args.chip_type, world_size, results, notes)
                print(
                    f"[flint-val60] SAVED {exp['name']}: status={status} "
                    f"tps/chip={metrics.get('steady_tps_per_chip', 0)} "
                    f"causal_mfu={metrics.get('steady_causal_mfu_pct', 0)}% "
                    f"palm_mfu={metrics.get('steady_palm_mfu_pct', 0)}% "
                    f"steady_steps={metrics.get('num_steady_steps', 0)}",
                    flush=True,
                )
                coord.set(key, "PASS" if passed else ("RETRY" if will_retry else "FAIL"))
            decision = coord.wait(key, timeout_s=4 * 3600)
            if decision == "RETRY":
                retries_left -= 1
                continue
            break

    try:
        coord.barrier("lockstep/all_done", timeout_s=3600)
    except TimeoutError:
        if node_rank == 0:
            raise
    if node_rank == 0:
        time.sleep(60)  # Keep the coordinator up while the other hosts finish their last polls.
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", default="customer_val60")
    parser.add_argument("--chip-type", choices=["v7x", "v5p", "v6e"], required=True)
    parser.add_argument("--nnodes", type=int, required=True)
    parser.add_argument("--nproc-per-node", type=int, required=True)
    parser.add_argument("--master-addr", type=str, required=True)
    parser.add_argument("--slice-name", type=str, required=True)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--only-filter", type=str, default="")
    parser.add_argument("--base-port", type=int, default=29910)
    parser.add_argument("--coord-port", type=int, default=29899)
    parser.add_argument("--per-run-timeout", type=int, default=5400)
    parser.add_argument("--output-dir", type=str, default="/models/flint-results")
    parser.add_argument("--lockstep", action="store_true", help="Run the selected suite with the multi-host lockstep runner.")
    parser.add_argument("--note", action="append", default=[], help="Extra methodology note for SUMMARY.md (repeatable).")
    args = parser.parse_args()

    apply_runtime_patches()
    node_rank = int(os.environ.get("JOB_COMPLETION_INDEX", os.environ.get("NODE_RANK", "0")))
    world_size = args.nnodes * args.nproc_per_node
    master_ip = wait_for_master_dns(args.master_addr)

    ts = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.output_dir) / f"{args.slice_name}_{args.suite}_{ts}"
    bench_dir = Path("/models/benchmark-results")
    if node_rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
        bench_dir.mkdir(parents=True, exist_ok=True)

    if args.suite == LONGCTX_SUITE:
        return run_longctx_suite(args, node_rank, world_size, master_ip, out_dir, bench_dir)

    if args.suite == ENG_PARITY_SUITE:
        if args.chip_type != "v7x":
            raise SystemExit(f"--suite={ENG_PARITY_SUITE} reproduces v7x-8 PR results; use --chip-type=v7x")
        exps = build_eng_parity_experiments(world_size, steps=args.steps)
    else:
        exps = build_experiments(args.chip_type, world_size, steps=args.steps)
    if args.only_filter:
        tokens = [t.strip() for t in args.only_filter.split(",") if t.strip()]
        exps = [e for e in exps if any(t in e["name"] for t in tokens)]

    if args.suite == ENG_PARITY_SUITE or args.lockstep:
        notes = (list(ENG_PARITY_NOTES) if args.suite == ENG_PARITY_SUITE else []) + list(args.note)
        return run_lockstep_suite(args, exps, node_rank, world_size, master_ip, out_dir, bench_dir, notes)

    results = []
    for idx, exp in enumerate(exps):
        port = args.base_port + idx * 5
        cmd = [
            "torchrun",
            f"--nnodes={args.nnodes}",
            f"--nproc_per_node={args.nproc_per_node}",
            "--rdzv_backend=static",
            f"--rdzv_endpoint={args.master_addr}:{port}",
            f"--node_rank={node_rank}",
            "-m",
            "torchtitan.experiments.tpu.train",
            *exp["args"],
        ]
        env = os.environ.copy()
        env["PYTHONPATH"] = exp["workdir"]
        env.update(exp["env"])
        print(f"\n=== [{idx + 1}/{len(exps)}] {exp['name']} ({exp['lane']}) ===", flush=True)
        t0 = time.time()
        try:
            rc, log_chunks = run_with_oom_watchdog(
                cmd, exp["workdir"], env, node_rank, args.per_run_timeout, idle_timeout_s=1800.0
            )
        except Exception as exc:
            rc = -1
            log_chunks = [f"[flint-val60] Exception: {exc}\n"]
        dur = time.time() - t0
        if node_rank == 0:
            raw_log = "".join(log_chunks)
            (out_dir / f"{exp['name']}.log").write_text(raw_log)
            metrics = parse_metrics(raw_log, args.chip_type, seq_len=exp["seq_len"], world_size=world_size)
            status = "PASS" if (rc == 0 and metrics.get("num_steady_steps", 0) >= 20) else f"FAIL(rc={rc})"
            results.append(
                {
                    "name": exp["name"],
                    "lane": exp["lane"],
                    "seq_len": exp["seq_len"],
                    "mb": exp["mb"],
                    "tok_per_rank": exp["tok_per_rank"],
                    "ga": exp["ga"],
                    "contiguous_full_seq": exp["contiguous_full_seq"],
                    "status": status,
                    "return_code": rc,
                    "duration_s": round(dur, 1),
                    "command": cmd,
                    "env_overrides": exp["env"],
                    "metrics": metrics,
                }
            )
            write_summary(out_dir, bench_dir, args.suite, args.slice_name, args.chip_type, world_size, results)
            print(
                f"[flint-val60] SAVED {exp['name']}: status={status} "
                f"tps/chip={metrics.get('steady_tps_per_chip', 0)} "
                f"causal_mfu={metrics.get('steady_causal_mfu_pct', 0)}% "
                f"palm_mfu={metrics.get('steady_palm_mfu_pct', 0)}% "
                f"steady_steps={metrics.get('num_steady_steps', 0)}",
                flush=True,
            )
        time.sleep(15 if rc != 0 else 8)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
