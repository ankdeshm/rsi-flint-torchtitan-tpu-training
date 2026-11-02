#!/usr/bin/env python3
"""Pass 10 TPU v5p-32 (2x4x4) & v5p-16 (2x2x4) Long-Context (4K / 8K / 16K) TorchTitan MFU Sweep.

Extends our top TorchTitan 2K v5p configurations (`v5p_p9_02` @ 25.76% Steady / 25.81% Peak MFU on v5p-16
and 24.88% MFU = 210,189 tok/s on v5p-32 for HF PR#123 TransformersBackend, plus `v5p_p9_06` @ 23.89% / 24.00% MFU
for Native PR#122 TpuGraphTrainer) across 4,096 / 8,192 / 16,384 context lengths:
  - Constant 131,072 tok/rank (32x4096, 16x8192, 8x16384), 196,608 tok/rank (12x16384), and 98,304 tok/rank (Native)
    preserves flat ~22-28 GiB HBM while raising MXU arithmetic intensity across the 14 full causal attention layers
    (strictly capping the 35 sliding-window W=513 layers at 513).
  - Also tests 2D HSDP (DP_rep=2 x FSDP=16) on v5p-32 (2x4x4) at 8K & 16K to align FSDP=16 onto 4x4 2D Torus planes.
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

BASE_FLOPS_PER_TOKEN_2K = 17.385e9
FULL_ATTN_FLOPS_PER_TOK_PER_CTX = 14 * 12 * 32 * 128  # 688,128 FLOPs/token per +1 seq_len above 2048
PEAK_BF16_TFLOPS_PER_RANK = {"v7x": 1153.5, "v5p": 459.0, "v6e": 918.0}

STEP_RE = re.compile(
    r"step:\s*(?P<step>\d+).*?loss:\s*(?P<loss>[0-9.]+).*?tps:\s*(?P<tps>[0-9.,]+)",
    re.IGNORECASE,
)
TFLOPS_RE = re.compile(r"tflops:\s*(?P<tflops>[0-9.]+)", re.IGNORECASE)
MFU_RE = re.compile(r"mfu:\s*(?P<mfu>[0-9.]+)%", re.IGNORECASE)
MEM_RE = re.compile(r"memory:\s*(?P<mem>[0-9.]+)GiB", re.IGNORECASE)


def flint_flops_per_token(seq_len: int) -> float:
    if seq_len <= 2048:
        return BASE_FLOPS_PER_TOKEN_2K
    return BASE_FLOPS_PER_TOKEN_2K + FULL_ATTN_FLOPS_PER_TOK_PER_CTX * (seq_len - 2048)


NATIVE_V5P_P10_LONGCTX_PATCH = '''

# ==============================================================================
# Pass 10 Flint v5p Long-Context (4K / 8K / 16K) TorchTitan Native Configs
# ==============================================================================
from torchtitan.experiments.tpu.token_dispatcher import (
    TpuRoutedExperts,
    TpuRoutedExpertsConverter,
)

_P10_BF16_ROPE = (
    "torchtitan.experiments.tpu.overrides.bf16_rope"
    ".tpu_activation_dtype_cossin_rope"
)


def _p10_set_max_context_length(root_cfg, seq_len: int) -> None:
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


def _build_v5p_p10_longctx_sbdp_tile256(seq_len: int, save_attn_layers: int = 40) -> TpuGraphTrainer.Config:
  config = flint_40b_v5p_n40()
  _p10_set_max_context_length(config, seq_len)
  config.model_spec.model = (
      TpuRoutedExpertsConverter.Config().build().convert(config.model_spec.model)
  )
  for _, r, _, _ in config.model_spec.model.traverse(TpuRoutedExperts.Config):
    r.scale_before_down_proj = True
  for _, ex, _, _ in config.model_spec.model.traverse(TpuGroupedExperts.Config):
    ex.use_pallas_grouped_mm = True
    ex.use_pallas_paired_grouped_mm = True
    ex.pallas_grouped_mm_tile_m = 256
    ex.pallas_grouped_mm_tgmm_tiling = (256, 1024, 0)
    ex.pallas_grouped_mm_w2_tgmm_tiling = (256, 512, 0)
    ex.pallas_grouped_mm_w2_transposed_tgmm_tiling = (256, 0, 1024)
    ex.use_kernel_native_weight_layout = True
  if _P10_BF16_ROPE not in config.override.imports:
    config.override.imports.append(_P10_BF16_ROPE)
  config.compile.save_last_n_layers = {"attention.inner_attention": save_attn_layers}
  config.compile.optimizer_compile = "separate_graph"
  config.compile.compile_grad_norm = True
  config.optimizer.implementation = "fused"
  return config


def flint_40b_v5p_p10_s4096_sbdp_tile256_n40() -> TpuGraphTrainer.Config:
  return _build_v5p_p10_longctx_sbdp_tile256(4096, 40)


def flint_40b_v5p_p10_s8192_sbdp_tile256_n40() -> TpuGraphTrainer.Config:
  return _build_v5p_p10_longctx_sbdp_tile256(8192, 40)


def flint_40b_v5p_p10_s16384_sbdp_tile256_n40() -> TpuGraphTrainer.Config:
  return _build_v5p_p10_longctx_sbdp_tile256(16384, 40)


__all__.extend([
    "flint_40b_v5p_p10_s4096_sbdp_tile256_n40",
    "flint_40b_v5p_p10_s8192_sbdp_tile256_n40",
    "flint_40b_v5p_p10_s16384_sbdp_tile256_n40",
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


def apply_runtime_patches() -> None:
    nat_reg = Path("/torchtitan-native/torchtitan/experiments/tpu/examples/flint/config_registry.py")
    if nat_reg.exists():
        txt = nat_reg.read_text()
        if "flint_40b_v5p_p10_s4096_sbdp_tile256_n40" not in txt:
            nat_reg.write_text(txt + NATIVE_V5P_P10_LONGCTX_PATCH)
            print("[flint-v5p-p10] Patched Native config_registry.py with Pass 10 4K/8K/16K configs", flush=True)

    hf_ac = Path("/torchtitan-hf/torchtitan/experiments/tpu/activation_checkpoint.py")
    if hf_ac.exists():
        txt = hf_ac.read_text()
        if "_hybrid_attn_only_wrap_v3" not in txt:
            hf_ac.write_text(txt + HF_HYBRID_AC_PATCH_V3)
            print("[flint-v5p-p10] Patched HF activation_checkpoint.py with HybridAC v3", flush=True)

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
            print("[flint-v5p-p10] Patched HF model.py & parallelize.py with TT_HF_SBDP=1 hook", flush=True)


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
    print(f"[flint-v5p-p10] Validated XLA flags: kept={len(kept)} dropped={dropped}", flush=True)
    return " ".join(kept)


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


def wait_for_master_dns(master_addr: str, timeout_s: int = 300) -> str:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            return socket.gethostbyname(master_addr)
        except socket.gaierror:
            time.sleep(3)
    return master_addr


def build_v5p_pass10_longctx_experiments(world_size: int, hf_model_path: str, steps: int) -> list[dict]:
    tok_98k = 48 * 2048    # 98,304 tok/rank = 24*4096 = 12*8192 = 6*16384
    tok_131k = 64 * 2048   # 131,072 tok/rank = 32*4096 = 16*8192 = 8*16384
    tok_196k = 96 * 2048   # 196,608 tok/rank = 48*4096 = 24*8192 = 12*16384

    v5p_p9_flags = filter_supported_xla_flags([
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

    hf_comp_opt = probe_compile_opt_args("/torchtitan-hf")
    hf_splash_1024 = [
        "--splash_attention_kernel.sa_block_q=1024",
        "--splash_attention_kernel.sa_block_kv=1024",
        "--splash_attention_kernel.sa_block_kv_compute=512",
        "--splash_attention_kernel.sa_block_q_dkv=1024",
        "--splash_attention_kernel.sa_block_kv_dkv=1024",
        "--splash_attention_kernel.sa_block_kv_dkv_compute=512",
    ]
    gmm_env_256 = (
        "use_pallas_paired_grouped_mm=true,"
        "pallas_grouped_mm_tile_m=256,"
        "pallas_grouped_mm_tgmm_tiling=256:1024:0,"
        "pallas_grouped_mm_w2_tgmm_tiling=256:512:0,"
        "pallas_grouped_mm_w2_transposed_tgmm_tiling=256:0:1024"
    )

    def _hf_longctx(
        name: str,
        lane: str,
        seq_len: int,
        tok_per_rank: int,
        ga: int,
        n_full_ac: int,
        extra_args: list[str] | None = None,
    ) -> dict:
        return {
            "name": name,
            "lane": lane,
            "seq_len": seq_len,
            "workdir": "/torchtitan-hf",
            "env": {
                "TT_HF_GMM": gmm_env_256,
                "TT_HF_SBDP": "1",
                "TT_FULL_AC_FIRST_N": str(n_full_ac),
                "TORCH_TPU_DEFER_AND_FUSE": "1",
                "LIBTPU_INIT_ARGS": v5p_p9_flags,
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
                "--metrics.log_freq=1",
                "--debug.seed=42",
                "--loss.b_block_size=1024",
                *hf_splash_1024,
                *hf_comp_opt,
                *(extra_args or []),
                f"--training.num_tokens_per_microbatch_per_dp_rank={tok_per_rank}",
                f"--training.num_tokens_per_train_step={ga * world_size * tok_per_rank}",
            ],
        }

    def _nat_longctx(name: str, lane: str, cfg: str, seq_len: int, tok_per_rank: int, ga: int) -> dict:
        return {
            "name": name,
            "lane": lane,
            "seq_len": seq_len,
            "workdir": "/torchtitan-native",
            "env": {"LIBTPU_INIT_ARGS": v5p_p9_flags},
            "args": [
                "--module=torchtitan.experiments.tpu.examples.flint",
                f"--config={cfg}",
                f"--training.max_context_length={seq_len}",
                f"--training.steps={steps}",
                "--training.mixed_precision_reduce=bfloat16",
                "--metrics.log_freq=1",
                "--debug.seed=42",
                "--tpu_config.mfu_benchmark_start_step=4",
                f"--training.num_tokens_per_microbatch_per_dp_rank={tok_per_rank}",
                f"--training.num_tokens_per_train_step={ga * world_size * tok_per_rank}",
            ],
        }

    prefix = f"v5p{world_size}_p10"
    exps = [
        # 1. Direct 4K extension of v5p_p9_02 (25.76% v5p-16 / 24.88% v5p-32 baseline): seq=4096, mb=32 (131.1K tok/rank)
        _hf_longctx(
            f"{prefix}_01_hf_pr123_s4096_mb32_1x_sbdp_fusedopt_sa1024_ac16",
            f"v5p-{world_size} HF PR#123 LongCtx 4K (seq=4096, mb=32 = 131.1K tok/rank, 1x, SBDP + FusedOpt + sa=1024 + HybridAC16)",
            seq_len=4096,
            tok_per_rank=tok_131k,
            ga=1,
            n_full_ac=16,
        ),
        # 2. Direct 8K extension of v5p_p9_02: seq=8192, mb=16 (131.1K tok/rank)
        _hf_longctx(
            f"{prefix}_02_hf_pr123_s8192_mb16_1x_sbdp_fusedopt_sa1024_ac16",
            f"v5p-{world_size} HF PR#123 LongCtx 8K (seq=8192, mb=16 = 131.1K tok/rank, 1x, SBDP + FusedOpt + sa=1024 + HybridAC16)",
            seq_len=8192,
            tok_per_rank=tok_131k,
            ga=1,
            n_full_ac=16,
        ),
        # 3. Direct 16K extension of v5p_p9_02: seq=16384, mb=8 (131.1K tok/rank)
        _hf_longctx(
            f"{prefix}_03_hf_pr123_s16384_mb8_1x_sbdp_fusedopt_sa1024_ac18",
            f"v5p-{world_size} HF PR#123 LongCtx 16K (seq=16384, mb=8 = 131.1K tok/rank, 1x, SBDP + FusedOpt + sa=1024 + HybridAC18)",
            seq_len=16384,
            tok_per_rank=tok_131k,
            ga=1,
            n_full_ac=18,
        ),
        # 4. Higher-token 16K: seq=16384, mb=12 (196.6K tok/rank = 4,608 tok/exp!), HybridAC36
        _hf_longctx(
            f"{prefix}_04_hf_pr123_s16384_mb12_1x_sbdp_fusedopt_sa1024_ac36",
            f"v5p-{world_size} HF PR#123 LongCtx 16K HighTok (seq=16384, mb=12 = 196.6K tok/rank, 1x, SBDP + FusedOpt + sa=1024 + HybridAC36)",
            seq_len=16384,
            tok_per_rank=tok_196k,
            ga=1,
            n_full_ac=36,
        ),
    ]
    if world_size == 32:
        # 5a. On v5p-32 (2x4x4): 2D HSDP (DP_rep=2 x FSDP=16) @ seq=16384 (maps FSDP=16 onto 4x4 2D Torus plane!)
        exps.append(
            _hf_longctx(
                f"{prefix}_05_hf_pr123_hsdp2x16_s16384_mb8_ga2_sbdp_fusedopt_sa1024_ac28",
                "v5p-32 HF PR#123 LongCtx 16K 2D HSDP (DP_rep=2 x FSDP=16, seq=16384, mb=8 = 131.1K tok/rank, 2xGA = 8.39M tok/step)",
                seq_len=16384,
                tok_per_rank=tok_131k,
                ga=2,
                n_full_ac=28,
                extra_args=[
                    "--parallelism.data_parallel_replicate_degree=2",
                    "--parallelism.data_parallel_shard_degree=16",
                ],
            )
        )
    else:
        # 5b. On v5p-16 (2x2x4): 2x GradAccum @ seq=16384 (mb=8 = 131.1K tok/rank, 2xGA = 4.19M tok/step)
        exps.append(
            _hf_longctx(
                f"{prefix}_05_hf_pr123_s16384_mb8_ga2_sbdp_fusedopt_sa1024_ac24",
                "v5p-16 HF PR#123 LongCtx 16K 2xGA (seq=16384, mb=8 = 131.1K tok/rank, 2xGA = 4.19M tok/step, HybridAC24)",
                seq_len=16384,
                tok_per_rank=tok_131k,
                ga=2,
                n_full_ac=24,
            )
        )

    # 6 & 7. Native PR#122 TpuGraphTrainer (tile256 + SBDP + BF16-RoPE + FusedOpt) @ 8K & 16K
    exps.extend([
        _nat_longctx(
            f"{prefix}_06_native_pr122_s8192_mb12_1x_sbdp_bf16rope_fusedopt",
            f"v5p-{world_size} Native PR#122 LongCtx 8K (seq=8192, mb=12 = 98.3K tok/rank, 1x, SBDP + BF16-RoPE + FusedOpt + tile256)",
            "flint_40b_v5p_p10_s8192_sbdp_tile256_n40",
            seq_len=8192,
            tok_per_rank=tok_98k,
            ga=1,
        ),
        _nat_longctx(
            f"{prefix}_07_native_pr122_s16384_mb6_1x_sbdp_bf16rope_fusedopt",
            f"v5p-{world_size} Native PR#122 LongCtx 16K (seq=16384, mb=6 = 98.3K tok/rank, 1x, SBDP + BF16-RoPE + FusedOpt + tile256)",
            "flint_40b_v5p_p10_s16384_sbdp_tile256_n40",
            seq_len=16384,
            tok_per_rank=tok_98k,
            ga=1,
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
    peak_tf = PEAK_BF16_TFLOPS_PER_RANK.get(chip_type, 459.0) * 1e12
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
    steady = [s for s in steps if s["step"] >= 6] or [s for s in steps if s["step"] >= 4] or steps[-2:]
    if not steady:
        return {
            "seq_len": seq_len,
            "flops_per_token_g": round(flops_per_tok / 1e9, 4),
            "num_logged_steps": 0,
            "steady_tps": 0.0,
            "steady_mfu_pct": 0.0,
        }
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
        f"- **Chip / World Size:** `{chip_type}` (`{world_size}` chips/ranks, `{PEAK_BF16_TFLOPS_PER_RANK.get(chip_type)}` BF16 TFLOP/s/rank)",
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
    parser.add_argument("--suite", default="pass10_longctx")
    parser.add_argument("--chip-type", choices=["v7x", "v5p", "v6e"], default="v5p")
    parser.add_argument("--nnodes", type=int, required=True)
    parser.add_argument("--nproc-per-node", type=int, default=4)
    parser.add_argument("--master-addr", type=str, required=True)
    parser.add_argument("--slice-name", type=str, required=True)
    parser.add_argument("--hf-model", type=str, default="/models/models/RecursiveSI/flint")
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--base-port", type=int, default=29880)
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

    exps = build_v5p_pass10_longctx_experiments(world_size, args.hf_model, args.steps)
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
            log_chunks = [f"[flint-v5p-p10] Exception: {exc}\n"]
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
                f"[flint-v5p-{args.suite}] SAVED {exp['name']}: status={status} "
                f"steady_tps={metrics.get('steady_tps', 0)} steady_mfu={metrics.get('steady_mfu_pct', 0)}% "
                f"(logged_mfu={metrics.get('steady_logged_mfu_pct', 0)}%)",
                flush=True,
            )
        time.sleep(10)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
