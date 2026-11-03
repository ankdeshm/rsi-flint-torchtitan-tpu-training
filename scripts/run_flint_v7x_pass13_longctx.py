#!/usr/bin/env python3
"""Pass 13 TPU v7x-16 (2x2x4, 16 Ironwood chips / 32 TensorCores) Long-Context (4K / 8K / 16K) TorchTitan MFU Sweep.

Extends our top TorchTitan 2K Ironwood configurations (`v7x_p12g_02` @ 20.80% Steady / 20.81% Peak MFU =
441,664 / 441,952 tok/s on Native PR#122 TpuGraphTrainer, and `v7x_p12_04` @ 18.58% MFU = 394,592 tok/s on HF PR#123
TransformersBackend) across 4,096 / 8,192 / 16,384 context lengths at constant tokens/microbatch/rank:
  - Constant 98,304 tok/rank (24x4096, 12x8192, 6x16384) & 131,072 tok/rank (32x4096, 16x8192, 8x16384)
    preserves identical ~23-26 GiB HBM footprint (0 O(S^2) buffers via Pallas SplashAttention sa=1024) and identical
    79.6 GB FSDP AllGather / BF16 ReduceScatter bytes while scaling MXU arithmetic intensity across the 14 full
    causal attention layers (strictly capping the 35 sliding-window W=513 layers at 513).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import threading
import time

# Exact RecursiveSI/flint (49 layers: 35 SWA W=513 + 14 Full Causal NoPE, H=32, D=128) FLOPs/token:
N_ACTIVE_NON_ATTN_FLOPS_PER_TOKEN = 15_093_202_944.0  # 6 * N_active (N_active = 2,515,533,824)
N_FULL_LAYERS = 14
N_SWA_LAYERS = 35
SWA_WINDOW = 513
N_HEADS = 32
HEAD_DIM = 128
PEAK_BF16_TFLOPS_PER_RANK = {"v7x": 1153.5, "v5p": 459.0, "v6e": 918.0}
RANKS_PER_CHIP = {"v7x": 2, "v5p": 1, "v6e": 1}


STEP_RE = re.compile(
    r"step:\s*(?P<step>\d+).*?loss:\s*(?P<loss>[0-9.]+).*?tps:\s*(?P<tps>[0-9.,]+)",
    re.IGNORECASE,
)
TFLOPS_RE = re.compile(r"tflops:\s*(?P<tflops>[0-9.]+)", re.IGNORECASE)
MFU_RE = re.compile(r"mfu:\s*(?P<mfu>[0-9.]+)%", re.IGNORECASE)
MEM_RE = re.compile(r"memory:\s*(?P<mem>[0-9.]+)GiB", re.IGNORECASE)


def flint_causal_flops_per_token(seq_len: int) -> float:
    """True hardware-executed causal-masked fwd+bwd FLOPs/token (6*H*D*(S+1) on 14 full layers; exact causal SWA on 35 W=513 layers)."""
    s = max(int(seq_len), 1)
    full_attn = N_FULL_LAYERS * 6.0 * N_HEADS * HEAD_DIM * (s + 1.0)
    w_eff = min(s, SWA_WINDOW)
    swa_pairs_per_tok = (w_eff * (2 * s - w_eff + 1)) / (2.0 * s)
    swa_attn = N_SWA_LAYERS * 12.0 * N_HEADS * HEAD_DIM * swa_pairs_per_tok
    return N_ACTIVE_NON_ATTN_FLOPS_PER_TOKEN + full_attn + swa_attn


def flint_palm_flops_per_token(seq_len: int) -> float:
    """Upstream TorchTitan / PaLM full-square convention (12*H*D*min(S,W) -> 2x overcounts causal full-attention triangle)."""
    if seq_len <= 2048:
        return 17.385e9
    return 17.385e9 + (14 * 12 * 32 * 128) * (seq_len - 2048)


def flint_flops_per_token(seq_len: int) -> float:
    return flint_causal_flops_per_token(seq_len)


NATIVE_P13_LONGCTX_PATCH = '''

# ==============================================================================
# Pass 13 Flint v7x-16 Long-Context (4K / 8K / 16K) TorchTitan Native Configs
# ==============================================================================
from torchtitan.experiments.tpu.token_dispatcher import (
    TpuRoutedExperts,
    TpuRoutedExpertsConverter,
)

_P13_BF16_ROPE = (
    "torchtitan.experiments.tpu.overrides.bf16_rope"
    ".tpu_activation_dtype_cossin_rope"
)


def _p13_set_max_context_length(root_cfg, seq_len: int) -> None:
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


def _build_p13_longctx_ep1_sbdp_fusedopt(seq_len: int, save_attn_layers: int) -> TpuGraphTrainer.Config:
  config = flint_40b_v014_tile1024()
  _p13_set_max_context_length(config, seq_len)
  config.model_spec.model = (
      TpuRoutedExpertsConverter.Config().build().convert(config.model_spec.model)
  )
  for _, r, _, _ in config.model_spec.model.traverse(TpuRoutedExperts.Config):
    r.scale_before_down_proj = True
  if _P13_BF16_ROPE not in config.override.imports:
    config.override.imports.append(_P13_BF16_ROPE)
  config.compile.memory_policy = "tpu_selective"
  config.compile.save_regions = ("moe.routing",)
  config.compile.save_last_n_layers = {"attention.inner_attention": save_attn_layers}
  config.compile.optimizer_compile = "separate_graph"
  config.compile.compile_grad_norm = True
  config.optimizer.implementation = "fused"
  return config


def flint_40b_p13_s4096_sbdp_fusedopt_n49() -> TpuGraphTrainer.Config:
  return _build_p13_longctx_ep1_sbdp_fusedopt(4096, 49)


def flint_40b_p13_s8192_sbdp_fusedopt_n49() -> TpuGraphTrainer.Config:
  return _build_p13_longctx_ep1_sbdp_fusedopt(8192, 49)


def flint_40b_p13_s16384_sbdp_fusedopt_n49() -> TpuGraphTrainer.Config:
  return _build_p13_longctx_ep1_sbdp_fusedopt(16384, 49)


def flint_40b_p13_s16384_sbdp_fusedopt_n36() -> TpuGraphTrainer.Config:
  return _build_p13_longctx_ep1_sbdp_fusedopt(16384, 36)


__all__.extend([
    "flint_40b_p13_s4096_sbdp_fusedopt_n49",
    "flint_40b_p13_s8192_sbdp_fusedopt_n49",
    "flint_40b_p13_s16384_sbdp_fusedopt_n49",
    "flint_40b_p13_s16384_sbdp_fusedopt_n36",
])
'''

HF_HYBRID_AC_PATCH_V3 = '''

import os as _tt_os
import torch._dynamo

torch._dynamo.config.cache_size_limit = 256
torch._dynamo.config.accumulated_cache_size_limit = 1024


class _LayerSplashSaveOps(set):
  def __init__(self, save_attn: bool):
    super().__init__()
    self._save_attn = save_attn

  def __contains__(self, op) -> bool:
    if not self._save_attn:
      return False
    return _PALLAS_ATTENTION_FWD_MARKER in _op_name(op)


def _hybrid_attn_only_wrap_v3(self, module, *, base_fqn=None):
  self.config.preserve_rng_state = False
  n_full = int(_tt_os.environ.get("TT_FULL_AC_FIRST_N", "0"))
  self._cur_layer_save_attn = True
  if n_full > 0 and base_fqn:
    parts = [p for p in base_fqn.split(".") if p.isdigit()]
    if parts and int(parts[0]) < n_full:
      self._cur_layer_save_attn = False
  return SelectiveAC._wrap_block(self, module, base_fqn=base_fqn)


def _hybrid_get_save_ops_v3(self) -> set:
  return _LayerSplashSaveOps(getattr(self, "_cur_layer_save_attn", True))


AttentionOnlyAC._wrap_block = _hybrid_attn_only_wrap_v3
AttentionOnlyAC.get_save_ops = _hybrid_get_save_ops_v3
'''

HF_SBDP_PATCH = '''

_orig_substitute_tpu_experts = substitute_tpu_experts


def _p9_substitute_tpu_experts(model):
  import os
  _orig_substitute_tpu_experts(model)
  if os.environ.get("TT_HF_SBDP", "0") != "1":
    return
  from torchtitan.experiments.tpu.token_dispatcher import (
      TpuRoutedExperts,
      TpuRoutedExpertsConverter,
  )
  rc = TpuRoutedExpertsConverter(TpuRoutedExpertsConverter.Config())
  n_sbdp = 0
  for layer in model.layers.values():
    cfg = getattr(layer, "_native_moe_config", None)
    if cfg is not None:
      cfg = rc.convert(cfg)
      for _, r, _, _ in cfg.traverse(TpuRoutedExperts.Config):
        r.scale_before_down_proj = True
        n_sbdp += 1
      layer._native_moe_config = cfg
  if n_sbdp:
    logger.info("Enabled scale_before_down_proj on %d HF MoE layer(s).", n_sbdp)


substitute_tpu_experts = _p9_substitute_tpu_experts
'''


def probe_compile_opt_args(workdir: str) -> list[str]:
    code = (
        "from torchtitan.experiments.tpu.tpu_job_config import TPUTrainerConfig; "
        "c = TPUTrainerConfig(); out = []; "
        "hasattr(c, 'compile') and hasattr(c.compile, 'compile_grad_norm') and out.append('--compile.compile_grad_norm'); "
        "hasattr(c, 'compile') and hasattr(c.compile, 'optimizer_compile') and out.append('--compile.optimizer_compile=separate_graph'); "
        "hasattr(c, 'tpu_config') and hasattr(c.tpu_config, 'compile_grad_norm') and out.append('--tpu_config.compile_grad_norm'); "
        "hasattr(c, 'tpu_config') and hasattr(c.tpu_config, 'compile_optimizer') and out.append('--tpu_config.compile_optimizer'); "
        "print(' '.join(out))"
    )
    try:
        env = os.environ.copy()
        env["PYTHONPATH"] = workdir
        env["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
        res = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=20)
        if res.returncode == 0 and res.stdout.strip():
            return res.stdout.strip().split()
    except Exception:
        pass
    return []


def apply_runtime_patches() -> None:
    nat_reg = Path("/torchtitan-native/torchtitan/experiments/tpu/examples/flint/config_registry.py")
    if nat_reg.exists():
        txt = nat_reg.read_text()
        if "flint_40b_p13_s4096_sbdp_fusedopt_n49" not in txt:
            nat_reg.write_text(txt + NATIVE_P13_LONGCTX_PATCH)
            print("[flint-v7x-p13] Patched Native config_registry.py with Pass 13 4K/8K/16K configs", flush=True)

    hf_ac = Path("/torchtitan-hf/torchtitan/experiments/tpu/activation_checkpoint.py")
    if hf_ac.exists():
        txt = hf_ac.read_text()
        if "_hybrid_attn_only_wrap_v3" not in txt:
            hf_ac.write_text(txt + HF_HYBRID_AC_PATCH_V3)
            print("[flint-v7x-p13] Patched HF activation_checkpoint.py with HybridAC v3", flush=True)

    hf_mod = Path("/torchtitan-hf/torchtitan/experiments/tpu/transformers_modeling_backend/model.py")
    if hf_mod.exists():
        txt = hf_mod.read_text()
        if "_p9_substitute_tpu_experts" not in txt:
            hf_mod.write_text(txt + HF_SBDP_PATCH)
            hf_par = Path("/torchtitan-hf/torchtitan/experiments/tpu/transformers_modeling_backend/parallelize.py")
            if hf_par.exists():
                ptxt = hf_par.read_text()
                if "from .model import substitute_tpu_experts" in ptxt:
                    hf_par.write_text(
                        ptxt.replace(
                            "from .model import substitute_tpu_experts",
                            "from . import model as _hf_model_mod\n"
                            "def substitute_tpu_experts(m): return _hf_model_mod.substitute_tpu_experts(m)",
                        )
                    )
            print("[flint-v7x-p13] Patched HF model.py & parallelize.py with TT_HF_SBDP=1 hook", flush=True)


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
    kept, dropped = [], []
    for flag in candidates:
        name = flag.lstrip("-").split("=", 1)[0].encode("utf-8")
        if name in blob:
            kept.append(flag)
        else:
            dropped.append(flag)
    print(f"[flint-v7x-p13] Validated XLA flags: kept={len(kept)} dropped={dropped}", flush=True)
    return " ".join(kept)


def wait_for_master_dns(master_addr: str, timeout_s: int = 300) -> str:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            return socket.gethostbyname(master_addr)
        except socket.gaierror:
            time.sleep(3)
    return master_addr


def build_v7x_pass13_longctx_experiments(
    world_size: int,
    hf_model_path: str = "/models/models/RecursiveSI/flint",
) -> list[dict]:
    tok_98k = 48 * 2048   # 98,304 tok/rank = 24*4096 = 12*8192 = 6*16384
    tok_131k = 64 * 2048  # 131,072 tok/rank = 32*4096 = 16*8192 = 8*16384

    p9_scout_flags = filter_supported_xla_flags([
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

    v7x_hf_flags = filter_supported_xla_flags([
        "--xla_tpu_dvfs_p_state=7",
        "--xla_tpu_enable_offloading_copy_to_sparsecore=false",
        "--xla_tpu_copy_insertion_use_region_analysis=true",
        "--xla_tpu_copy_insertion_use_region_analysis_limit=1000",
        "--xla_tpu_enable_sparse_core_collective_aggregator=true",
        "--xla_tpu_enable_latency_hiding_layer_scheduler=true",
        "--xla_tpu_use_single_sparse_core_for_reduce_scatter_offload=false",
        "--xla_tpu_enable_sparse_core_collective_offload_nd_reduce_scatter=true",
        "--xla_tpu_enable_sparse_core_hierarchical_reduce_scatter=true",
        "--xla_tpu_prefer_async_allgather_to_allreduce=true",
        "--xla_enable_async_all_gather=true",
        "--xla_enable_async_all_reduce=true",
        "--xla_tpu_enable_async_all_to_all=true",
        "--xla_tpu_enable_ag_backward_pipelining=true",
        "--xla_tpu_enable_ici_ag_pipelining=true",
        "--xla_tpu_overlap_compute_collective_tc=true",
        "--xla_tpu_layer_scheduler_min_compute_latency_for_backward_pass=25000000",
        "--xla_max_concurrent_async_all_gathers=4",
        "--xla_latency_hiding_scheduler_rerun=2",
        "--xla_enable_transpose_trace=false",
    ])

    hf_comp_opt = probe_compile_opt_args("/torchtitan-hf")
    hf_splash_1024 = [
        "--splash_attention_kernel.sa_block_q=1024",
        "--splash_attention_kernel.sa_block_kv=1024",
        "--splash_attention_kernel.sa_block_kv_compute=512",
        "--splash_attention_kernel.sa_block_q_dkv=1024",
        "--splash_attention_kernel.sa_block_kv_dkv=1024",
        "--splash_attention_kernel.sa_block_kv_dkv_compute=512",
    ]
    paired_gmm_1024_safe = (
        "use_pallas_paired_grouped_mm=true,"
        "pallas_grouped_mm_tile_m=1024,"
        "pallas_grouped_mm_tgmm_tiling=512:2048:0,"
        "pallas_grouped_mm_w2_tgmm_tiling=512:512:0,"
        "pallas_grouped_mm_w2_transposed_tgmm_tiling=512:0:2048"
    )

    def _nat_longctx(name: str, lane: str, cfg: str, seq_len: int, tok_per_rank: int, ga: int, steps: int = 15) -> dict:
        return {
            "name": name,
            "lane": lane,
            "seq_len": seq_len,
            "workdir": "/torchtitan-native",
            "env": {"LIBTPU_INIT_ARGS": p9_scout_flags},
            "args": [
                "--module=torchtitan.experiments.tpu.examples.flint",
                f"--config={cfg}",
                f"--training.max_context_length={seq_len}",
                f"--training.steps={steps}",
                "--training.mixed_precision_reduce=bfloat16",
                "--metrics.log_freq=5",
                "--debug.seed=42",
                "--tpu_config.mfu_benchmark_start_step=5",
                f"--training.num_tokens_per_microbatch_per_dp_rank={tok_per_rank}",
                f"--training.num_tokens_per_train_step={ga * world_size * tok_per_rank}",
            ],
        }

    def _hf_longctx(
        name: str,
        lane: str,
        seq_len: int,
        tok_per_rank: int,
        ga: int,
        n_full_ac: int,
        steps: int = 15,
        extra_args: list[str] | None = None,
    ) -> dict:
        return {
            "name": name,
            "lane": lane,
            "seq_len": seq_len,
            "workdir": "/torchtitan-hf",
            "env": {
                "TT_HF_GMM": paired_gmm_1024_safe,
                "TT_HF_SBDP": "1",
                "TT_FULL_AC_FIRST_N": str(n_full_ac),
                "TORCH_TPU_DEFER_AND_FUSE": "1",
                "LIBTPU_INIT_ARGS": v7x_hf_flags,
            },
            "args": [
                "--module=torchtitan.experiments.tpu.transformers_modeling_backend",
                "--config=hf_moe_auto_max_attnac",
                f"--hf-model={hf_model_path}",
                f"--training.max_context_length={seq_len}",
                f"--training.steps={steps}",
                "--training.mixed_precision_reduce=bfloat16",
                "--optimizer.implementation=fused",
                "--lr_scheduler.warmup_steps=0",
                "--lr_scheduler.decay_ratio=0.0",
                "--metrics.log_freq=5",
                "--debug.seed=42",
                "--loss.b_block_size=1024",
                *hf_splash_1024,
                *hf_comp_opt,
                *(extra_args or []),
                f"--training.num_tokens_per_microbatch_per_dp_rank={tok_per_rank}",
                f"--training.num_tokens_per_train_step={ga * world_size * tok_per_rank}",
            ],
        }

    return [
        # 1. Native PR#122 (Pass 12G 20.80% baseline) @ seq=4096 (mb=24 seqs = 98,304 tok/rank, 8xGA = 25.17M tok/step)
        _nat_longctx(
            "v7x_p13_01_native_s4096_mb24_ga8_sbdp_bf16rope_fusedopt_n49",
            "v7x-16 Native PR#122 LongCtx 4K (seq=4096, mb=24 = 98.3K tok/rank, 8xGA = 25.17M tok/step, SBDP + BF16-RoPE + PureFusedOpt + n49)",
            "flint_40b_p13_s4096_sbdp_fusedopt_n49",
            seq_len=4096,
            tok_per_rank=tok_98k,
            ga=8,
        ),
        # 2. Native PR#122 (Pass 12G baseline) @ seq=8192 (mb=12 seqs = 98,304 tok/rank, 8xGA = 25.17M tok/step)
        _nat_longctx(
            "v7x_p13_02_native_s8192_mb12_ga8_sbdp_bf16rope_fusedopt_n49",
            "v7x-16 Native PR#122 LongCtx 8K (seq=8192, mb=12 = 98.3K tok/rank, 8xGA = 25.17M tok/step, SBDP + BF16-RoPE + PureFusedOpt + n49)",
            "flint_40b_p13_s8192_sbdp_fusedopt_n49",
            seq_len=8192,
            tok_per_rank=tok_98k,
            ga=8,
        ),
        # 3. Native PR#122 (Pass 12G baseline) @ seq=16384 (mb=6 seqs = 98,304 tok/rank, 8xGA = 25.17M tok/step)
        _nat_longctx(
            "v7x_p13_03_native_s16384_mb6_ga8_sbdp_bf16rope_fusedopt_n49",
            "v7x-16 Native PR#122 LongCtx 16K (seq=16384, mb=6 = 98.3K tok/rank, 8xGA = 25.17M tok/step, SBDP + BF16-RoPE + PureFusedOpt + n49)",
            "flint_40b_p13_s16384_sbdp_fusedopt_n49",
            seq_len=16384,
            tok_per_rank=tok_98k,
            ga=8,
        ),
        # 4. Native PR#122 @ seq=16384 with 131,072 tok/rank (mb=8 seqs = 3,072 tok/exp, n36, 4xGA = 16.78M tok/step)
        _nat_longctx(
            "v7x_p13_04_native_s16384_mb8_ga4_sbdp_bf16rope_fusedopt_n36",
            "v7x-16 Native PR#122 LongCtx 16K HighTok (seq=16384, mb=8 = 131.1K tok/rank, 4xGA = 16.78M tok/step, SBDP + BF16-RoPE + PureFusedOpt + n36)",
            "flint_40b_p13_s16384_sbdp_fusedopt_n36",
            seq_len=16384,
            tok_per_rank=tok_131k,
            ga=4,
        ),
        # 5. HF PR#123 (Pass 12 18.58% baseline) @ seq=4096 (mb=32 seqs = 131,072 tok/rank, 2xGA = 8.39M tok/step)
        _hf_longctx(
            "v7x_p13_05_hf_pr123_s4096_mb32_ga2_sbdp_fusedopt_sa1024_tile1024_ac12",
            "v7x-16 HF PR#123 LongCtx 4K (seq=4096, mb=32 = 131.1K tok/rank, 2xGA = 8.39M tok/step, SBDP + PureFusedOpt + sa=1024 + tile1024 + HybridAC12)",
            seq_len=4096,
            tok_per_rank=tok_131k,
            ga=2,
            n_full_ac=12,
        ),
        # 6. HF PR#123 (Pass 12 baseline) @ seq=8192 (mb=16 seqs = 131,072 tok/rank, 2xGA = 8.39M tok/step)
        _hf_longctx(
            "v7x_p13_06_hf_pr123_s8192_mb16_ga2_sbdp_fusedopt_sa1024_tile1024_ac12",
            "v7x-16 HF PR#123 LongCtx 8K (seq=8192, mb=16 = 131.1K tok/rank, 2xGA = 8.39M tok/step, SBDP + PureFusedOpt + sa=1024 + tile1024 + HybridAC12)",
            seq_len=8192,
            tok_per_rank=tok_131k,
            ga=2,
            n_full_ac=12,
        ),
        # 7. HF PR#123 (Pass 12 baseline) @ seq=16384 (mb=8 seqs = 131,072 tok/rank, 2xGA = 8.39M tok/step)
        _hf_longctx(
            "v7x_p13_07_hf_pr123_s16384_mb8_ga2_sbdp_fusedopt_sa1024_tile1024_ac16",
            "v7x-16 HF PR#123 LongCtx 16K (seq=16384, mb=8 = 131.1K tok/rank, 2xGA = 8.39M tok/step, SBDP + PureFusedOpt + sa=1024 + tile1024 + HybridAC16)",
            seq_len=16384,
            tok_per_rank=tok_131k,
            ga=2,
            n_full_ac=16,
        ),
    ]


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
    idle_timeout_s: float = 1200.0,
) -> tuple[int, list[str]]:
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
        while proc.poll() is None:
            now = time.time()
            if now - t_start > per_run_timeout:
                print(f"[watchdog] per_run_timeout ({per_run_timeout}s) reached; killing process group", flush=True)
                _kill_proc_group(proc)
                return
            if now - state["last_line"] > idle_timeout_s:
                print(f"[watchdog] idle_timeout ({idle_timeout_s}s) reached; killing process group", flush=True)
                _kill_proc_group(proc)
                return
            if state["fatal_ts"] is not None and now - state["fatal_ts"] > 10.0:
                print("[watchdog] fatal OOM/VMEM/error detected >10s ago; killing process group", flush=True)
                _kill_proc_group(proc)
                return
            time.sleep(2.0)

    wt = threading.Thread(target=_watchdog, daemon=True)
    wt.start()
    for line in proc.stdout:
        state["last_line"] = time.time()
        if state["fatal_ts"] is None and any(m in line for m in fatal_markers):
            state["fatal_ts"] = time.time()
        sys.stdout.write(line)
        sys.stdout.flush()
        if node_rank == 0:
            log_chunks.append(line)
    rc = proc.wait(timeout=30)
    subprocess.run(["pkill", "-9", "-f", "torchtitan.experiments.tpu.train"], check=False)
    return rc, log_chunks


def parse_metrics(log_text: str, chip_type: str, seq_len: int) -> dict:
    flops_per_tok = flint_flops_per_token(seq_len)
    peak_tf = PEAK_BF16_TFLOPS_PER_RANK.get(chip_type, 1153.5) * 1e12
    steps = []
    for line in log_text.splitlines():
        m = STEP_RE.search(line)
        if not m:
            continue
        step = int(m.group("step"))
        if steps and steps[-1]["step"] == step:
            continue
        loss = float(m.group("loss"))
        tps = float(m.group("tps").replace(",", ""))
        tf_m = TFLOPS_RE.search(line)
        mfu_m = MFU_RE.search(line)
        mem_m = MEM_RE.search(line)
        calc_mfu = 100.0 * flops_per_tok * tps / peak_tf
        logged_mfu = float(mfu_m.group("mfu")) if mfu_m else calc_mfu
        logged_tf = float(tf_m.group("tflops")) if tf_m else round(flops_per_tok * tps / 1e12, 2)
        mem_gib = float(mem_m.group("mem")) if mem_m else None
        steps.append(
            {
                "step": step,
                "loss": loss,
                "tps_per_rank": round(tps, 1),
                "logged_tflops": logged_tf,
                "logged_mfu_pct": round(logged_mfu, 2),
                "mfu_pct": round(calc_mfu, 2),
                "mem_gib": mem_gib,
            }
        )
    if not steps:
        return {
            "seq_len": seq_len,
            "flops_per_token_g": round(flops_per_tok / 1e9, 4),
            "num_logged_steps": 0,
            "steady_tps": 0.0,
            "steady_mfu_pct": 0.0,
        }
    max_step = max(s["step"] for s in steps)
    if max_step >= 15 and any(s["step"] >= 10 for s in steps):
        steady = [s for s in steps if s["step"] >= 10]
    else:
        steady = [s for s in steps if s["step"] >= 6] or steps[-2:]
    avg_tps = sum(s["tps_per_rank"] for s in steady) / len(steady)
    max_tps = max(s["tps_per_rank"] for s in steady)
    avg_logged_mfu = sum(s["logged_mfu_pct"] for s in steady) / len(steady)
    max_logged_mfu = max(s["logged_mfu_pct"] for s in steady)
    return {
        "seq_len": seq_len,
        "flops_per_token_g": round(flops_per_tok / 1e9, 4),
        "num_logged_steps": len(steps),
        "first_loss": steps[0]["loss"],
        "final_loss": steps[-1]["loss"],
        "steady_tps": round(avg_tps, 1),
        "peak_tps": round(max_tps, 1),
        "steady_mfu_pct": round(100.0 * flops_per_tok * avg_tps / peak_tf, 2),
        "peak_mfu_pct": round(100.0 * flops_per_tok * max_tps / peak_tf, 2),
        "steady_logged_mfu_pct": round(avg_logged_mfu, 2),
        "peak_logged_mfu_pct": round(max_logged_mfu, 2),
        "peak_mem_gib": max((s["mem_gib"] for s in steps if s["mem_gib"] is not None), default=None),
        "steps": steps,
    }


def write_summary(
    out_dir: Path, bench_dir: Path, suite: str, slice_name: str, chip_type: str, world_size: int, results: list[dict]
) -> None:
    payload = {
        "updated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "suite": suite,
        "slice_name": slice_name,
        "chip_type": chip_type,
        "world_size_ranks": world_size,
        "peak_bf16_tflops_per_rank": PEAK_BF16_TFLOPS_PER_RANK.get(chip_type),
        "results": results,
    }
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2))
    md_lines = [
        f"# Flint (`RecursiveSI/flint` 40B MoE) TorchTitan Long-Context (`4K / 8K / 16K`) MFU Sweep (`{suite.upper()}`) — `{slice_name}`",
        "",
        f"- **Updated (UTC):** `{payload['updated_at_utc']}`",
        f"- **Chip / World Size:** `{chip_type}` (`{world_size}` TensorCore ranks, `{PEAK_BF16_TFLOPS_PER_RANK.get(chip_type)}` BF16 TFLOP/s/rank)",
        "",
        "| Experiment | Seq Len | GFLOP/tok | Status | Steady tok/s/rank | Cluster tok/s | Steady MFU (%) | Peak MFU (%) | Peak HBM (GiB) | Loss (start -> end) | Wall Time (s) |",
        "|---|---:|---:|---|---:|---:|---:|---:|---:|---|---:|",
    ]
    for r in results:
        m = r.get("metrics", {})
        loss_str = f"{m.get('first_loss', '-')} -> {m.get('final_loss', '-')}" if m.get("num_logged_steps") else "-"
        st_tps = m.get("steady_tps", 0.0)
        md_lines.append(
            f"| `{r['name']}` | `{m.get('seq_len', '-')}` | `{m.get('flops_per_token_g', '-')}` | `{r['status']}` | "
            f"**{st_tps:,.1f}** | {st_tps * world_size:,.0f} | **{m.get('steady_mfu_pct', 0.0):.2f}%** | "
            f"{m.get('peak_mfu_pct', 0.0):.2f}% | {m.get('peak_mem_gib') or '-'} | {loss_str} | {r['duration_s']:.1f} |"
        )
    md_text = "\n".join(md_lines) + "\n"
    (out_dir / "SUMMARY.md").write_text(md_text)
    if bench_dir.exists():
        (bench_dir / f"flint_{slice_name}_{suite}_summary.json").write_text(json.dumps(payload, indent=2))
        (bench_dir / f"flint_{slice_name}_{suite}_SUMMARY.md").write_text(md_text)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", default="pass13_longctx")
    parser.add_argument("--chip-type", choices=["v7x", "v5p", "v6e"], default="v7x")
    parser.add_argument("--nnodes", type=int, required=True)
    parser.add_argument("--nproc-per-node", type=int, default=8)
    parser.add_argument("--master-addr", type=str, required=True)
    parser.add_argument("--slice-name", type=str, required=True)
    parser.add_argument("--hf-model", type=str, default="/models/models/RecursiveSI/flint")
    parser.add_argument("--base-port", type=int, default=29840)
    parser.add_argument("--per-run-timeout", type=int, default=2400)
    parser.add_argument("--output-dir", type=str, default="/models/flint-results")
    args = parser.parse_args()

    apply_runtime_patches()
    node_rank = int(os.environ.get("JOB_COMPLETION_INDEX", os.environ.get("NODE_RANK", "0")))
    world_size = args.nnodes * args.nproc_per_node
    wait_for_master_dns(args.master_addr)

    ts = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.output_dir) / f"{args.slice_name}_{args.suite}_{ts}"
    bench_dir = Path("/models/benchmark-results")
    if node_rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
        bench_dir.mkdir(parents=True, exist_ok=True)

    exps = build_v7x_pass13_longctx_experiments(world_size, args.hf_model)
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
                cmd, exp["workdir"], env, node_rank, args.per_run_timeout, idle_timeout_s=1500.0
            )
        except Exception as exc:
            rc = -1
            log_chunks = [f"[flint-v7x-p13] Exception: {exc}\n"]
        dur = time.time() - t0
        if node_rank == 0:
            raw_log = "".join(log_chunks)
            (out_dir / f"{exp['name']}.log").write_text(raw_log)
            metrics = parse_metrics(raw_log, args.chip_type, seq_len=exp["seq_len"])
            status = "PASS" if (rc == 0 and metrics.get("num_logged_steps", 0) > 0) else f"FAIL(rc={rc})"
            results.append(
                {
                    "name": exp["name"],
                    "lane": exp["lane"],
                    "seq_len": exp["seq_len"],
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
                f"[flint-v7x-{args.suite}] SAVED {exp['name']}: status={status} "
                f"steady_tps={metrics.get('steady_tps', 0)} steady_mfu={metrics.get('steady_mfu_pct', 0)}% "
                f"(logged_mfu={metrics.get('steady_logged_mfu_pct', 0)}%)",
                flush=True,
            )
        time.sleep(20 if rc != 0 else 10)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
