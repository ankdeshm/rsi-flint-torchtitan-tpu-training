#!/usr/bin/env python3
"""Pass 10B TPU v7x-16 (2x2x4, 16 Ironwood chips / 32 TensorCores) MFU Breakthrough Sweep for Flint.

Incorporates exact VMEM & HBM calibration from Pass 10A + Intra-Host Expert Parallelism (EP=8) + ZeRO-2:
  1. Physical Ironwood TensorCore VMEM is 64 MiB (`xla_tpu_scoped_vmem_limit_kib=65536`) and `--loss.b_block_size=1024`
     (fixes `lce_bwd_bt_2048` 69.84 MiB VMEM overflow and `n43` 0.01 GiB bottom-of-HBM reservation miss).
  2. Un-padded Pallas GroupedMM (`tile_m=256`, 0% expert padding vs 33.3% at `tile_m=1024`) +
     `TpuRoutedExpertsConverter(scale_before_down_proj=True)` (eliminating 47-layer backward `W2` recompute) +
     `BF16 RoPE` + `RegionAnalysis` across BOTH Native PR#122 (`n40`) and HF PR#123 (`b_block=1024`).
  3. Intra-Host Expert Parallelism (`EP=8` across the 8 local TensorCores inside each 4-chip dual-core host) +
     `ZeRO-2` (`fsdp_reshard_after_forward="never"`):
     - Cuts gathered expert BF16 weights per rank from 75.7 GB -> 9.46 GB (`~13.9 GB` total persistent params!).
     - Eliminates 100% of cross-host backward weight AllGather AND 2nd-microbatch forward AllGather under 2xGA!
     - Keeps token dispatch/combine `AllToAll` 100% intra-host while boosting tokens/expert 8x (`12,288` @ mb=32, `18,432` @ mb=48)!
  4. Intra-Host `EP=8` + Host Activation Offload (`tpu_selective_and_offload` + `prefetch=1`) @ `mb=96` (`36,864` tok/expert!).
  5. Process-group (`start_new_session=True` + `os.killpg`) OOM/VMEM watchdog so failed runs cleanly exit in <12s.
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

FLINT_FLOPS_PER_TOKEN = 17.385e9
PEAK_BF16_TFLOPS_PER_RANK = {"v7x": 1153.5, "v5p": 459.0, "v6e": 918.0}

STEP_RE = re.compile(
    r"step:\s*(?P<step>\d+).*?loss:\s*(?P<loss>[0-9.]+).*?tps:\s*(?P<tps>[0-9.,]+)",
    re.IGNORECASE,
)
MFU_RE = re.compile(r"mfu:\s*(?P<mfu>[0-9.]+)%", re.IGNORECASE)
MEM_RE = re.compile(r"memory:\s*(?P<mem>[0-9.]+)GiB", re.IGNORECASE)

NATIVE_P10B_PATCH = '''

# ==============================================================================
# Pass 10B Flint v7x-16 MFU Breakthrough Configs (Calibrated VMEM + EP=8 ZeRO-2)
# ==============================================================================
from torchtitan.experiments.tpu.token_dispatcher import (
    TpuRoutedExperts,
    TpuRoutedExpertsConverter,
)

_P10B_BF16_ROPE = (
    "torchtitan.experiments.tpu.overrides.bf16_rope"
    ".tpu_activation_dtype_cossin_rope"
)


def _apply_flint_sbdp_and_tiles_p10b(config, *, tile_m: int = 256) -> None:
  config.model_spec.model = (
      TpuRoutedExpertsConverter.Config().build().convert(config.model_spec.model)
  )
  for _, r, _, _ in config.model_spec.model.traverse(TpuRoutedExperts.Config):
    r.scale_before_down_proj = True
  for _, ex, _, _ in config.model_spec.model.traverse(TpuGroupedExperts.Config):
    ex.use_pallas_grouped_mm = True
    ex.use_pallas_paired_grouped_mm = True
    ex.pallas_grouped_mm_tile_m = tile_m
    ex.pallas_grouped_mm_tgmm_tiling = (512, 2048, 0)
    ex.pallas_grouped_mm_w2_tgmm_tiling = (tile_m, 512, 0)
    ex.pallas_grouped_mm_w2_transposed_tgmm_tiling = (tile_m, 0, 2048)
    ex.use_kernel_native_weight_layout = True
  if _P10B_BF16_ROPE not in config.override.imports:
    config.override.imports.append(_P10B_BF16_ROPE)


def _apply_flint_ep_tiles_p10b(config, *, ep_degree: int = 8, tile_m: int = 512) -> None:
  config.parallelism.expert_parallel_degree = ep_degree
  for _, ex, _, _ in config.model_spec.model.traverse(TpuGroupedExperts.Config):
    ex.use_pallas_grouped_mm = True
    ex.use_pallas_paired_grouped_mm = False
    ex.pallas_grouped_mm_tile_m = tile_m
    ex.pallas_grouped_mm_tgmm_tiling = (512, 512, 512)
    ex.use_kernel_native_weight_layout = True
  if _P10B_BF16_ROPE not in config.override.imports:
    config.override.imports.append(_P10B_BF16_ROPE)


def flint_40b_p10b_tile256_sbdp_n40() -> TpuGraphTrainer.Config:
  config = flint_40b_v012_hier_rs()
  _apply_flint_sbdp_and_tiles_p10b(config, tile_m=256)
  config.compile.memory_policy = "tpu_selective"
  config.compile.save_regions = ("moe.routing",)
  config.compile.save_last_n_layers = {"attention.inner_attention": 40}
  config.compile.optimizer_compile = "separate_graph"
  config.compile.compile_grad_norm = True
  config.optimizer.implementation = "fused"
  return config


def flint_40b_p10b_ep8_zero2_n36() -> TpuGraphTrainer.Config:
  config = flint_40b_v012_hier_rs()
  _apply_flint_ep_tiles_p10b(config, ep_degree=8, tile_m=512)
  config.parallelism.fsdp_reshard_after_forward = "never"
  config.compile.memory_policy = "tpu_selective"
  config.compile.save_regions = ("moe.routing",)
  config.compile.save_last_n_layers = {"attention.inner_attention": 36}
  config.compile.optimizer_compile = "separate_graph"
  config.compile.compile_grad_norm = True
  config.optimizer.implementation = "fused"
  return config


def flint_40b_p10b_ep8_offload() -> TpuGraphTrainer.Config:
  config = flint_40b_v012_hier_rs()
  _apply_flint_ep_tiles_p10b(config, ep_degree=8, tile_m=512)
  config.compile.memory_policy = "tpu_selective_and_offload"
  config.compile.save_regions = ("moe.routing",)
  config.compile.save_last_n_layers = {"attention.inner_attention": 32}
  config.compile.graph_split.forward_layers_per_executable = 1
  config.compile.graph_split.backward_partition_mode = "by_layer"
  config.compile.graph_split.backward_layers_per_executable = 1
  config.compile.graph_split.cpu_offload_prefetch_executables = 1
  config.compile.optimizer_compile = "separate_graph"
  config.compile.compile_grad_norm = True
  config.optimizer.implementation = "fused"
  return config


__all__.extend([
    "flint_40b_p10b_tile256_sbdp_n40",
    "flint_40b_p10b_ep8_zero2_n36",
    "flint_40b_p10b_ep8_offload",
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


def _p10_substitute_tpu_experts(model):
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


substitute_tpu_experts = _p10_substitute_tpu_experts
'''


def apply_runtime_patches() -> None:
    nat_reg = Path("/torchtitan-native/torchtitan/experiments/tpu/examples/flint/config_registry.py")
    if nat_reg.exists():
        txt = nat_reg.read_text()
        if "flint_40b_p10b_tile256_sbdp_n40" not in txt:
            nat_reg.write_text(txt + NATIVE_P10B_PATCH)
            print("[flint-v7x-p10b] Patched Native config_registry.py with Pass 10B configs", flush=True)

    hf_ac = Path("/torchtitan-hf/torchtitan/experiments/tpu/activation_checkpoint.py")
    if hf_ac.exists():
        txt = hf_ac.read_text()
        if "_hybrid_attn_only_wrap_v3" not in txt:
            hf_ac.write_text(txt + HF_HYBRID_AC_PATCH_V3)
            print("[flint-v7x-p10b] Patched HF activation_checkpoint.py with HybridAC v3", flush=True)

    hf_mod = Path("/torchtitan-hf/torchtitan/experiments/tpu/transformers_modeling_backend/model.py")
    if hf_mod.exists():
        txt = hf_mod.read_text()
        if "_p10_substitute_tpu_experts" not in txt:
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
            print("[flint-v7x-p10b] Patched HF model.py & parallelize.py with TT_HF_SBDP=1 hook", flush=True)


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
    print(f"[flint-v7x-p10b] Validated XLA flags: kept={len(kept)} dropped={dropped}", flush=True)
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


def build_v7x_pass10b_experiments(world_size: int, hf_model_path: str, steps: int) -> list[dict]:
    mb32 = 32 * 2048    # 65,536 tok/rank  -> @EP=8: 12,288 tok/expert!
    mb48 = 48 * 2048    # 98,304 tok/rank  -> @EP=1: 2,304 tok/exp (9x256); @EP=8: 18,432 tok/exp!
    mb64 = 64 * 2048    # 131,072 tok/rank -> @EP=1: 3,072 tok/exp (12x256)
    mb96 = 96 * 2048    # 196,608 tok/rank -> @EP=8: 36,864 tok/expert!

    # Calibrated 64 MiB physical Ironwood TensorCore VMEM + Region Analysis + Hierarchical SparseCore RS + Async AllToAll
    calibrated_v7x_flags = filter_supported_xla_flags([
        "--xla_tpu_dvfs_p_state=7",
        "--xla_tpu_scoped_vmem_limit_kib=65536",
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
        "--splash_attention_kernel.sa_block_kv_dkv_compute=1024",
    ]
    paired_gmm_256 = (
        "use_pallas_paired_grouped_mm=true,"
        "pallas_grouped_mm_tile_m=256,"
        "pallas_grouped_mm_tgmm_tiling=512:2048:0,"
        "pallas_grouped_mm_w2_tgmm_tiling=256:512:0,"
        "pallas_grouped_mm_w2_transposed_tgmm_tiling=256:0:2048"
    )
    hf_base = [
        "--module=torchtitan.experiments.tpu.transformers_modeling_backend",
        "--config=hf_moe_auto_max_attnac",
        f"--hf-model={hf_model_path}",
        "--training.max_context_length=2048",
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
    ]

    return [
        # 1. Native PR#122: Calibrated 64MiB VMEM + n40 + tile_m=256 (0% pad) + scale_before_down_proj + BF16 RoPE + 2xGA (mb=48)
        {
            "name": "v7x_p10b_01_native_tile256_sbdp_bf16rope_n40_mb48_ga2",
            "lane": "v7x-16 Native PR#122 (tile_m=256 [0% pad] + scale_before_down_proj + BF16-RoPE + RegionAnalysis 64MiB VMEM, n40, mb=48 2xGA)",
            "workdir": "/torchtitan-native",
            "env": {"LIBTPU_INIT_ARGS": calibrated_v7x_flags},
            "args": [
                "--module=torchtitan.experiments.tpu.examples.flint",
                "--config=flint_40b_p10b_tile256_sbdp_n40",
                f"--training.steps={steps}",
                "--training.mixed_precision_reduce=bfloat16",
                "--metrics.log_freq=1",
                "--debug.seed=42",
                "--tpu_config.mfu_benchmark_start_step=3",
                f"--training.num_tokens_per_microbatch_per_dp_rank={mb48}",
                f"--training.num_tokens_per_train_step={2 * world_size * mb48}",
            ],
        },
        # 2. HF PR#123: Calibrated 64MiB VMEM + CE b=1024 + tile_m=256 + scale_before_down_proj + sa=1024 + 2xGA (mb=48)
        {
            "name": "v7x_p10b_02_hf_pr123_tile256_sbdp_sa1024_b1024_mb48_ga2",
            "lane": "v7x-16 HF PR#123 (tile_m=256 + scale_before_down_proj + sa=1024 + CE b=1024 + RegionAnalysis 64MiB VMEM, mb=48 2xGA)",
            "workdir": "/torchtitan-hf",
            "env": {
                "TT_HF_GMM": paired_gmm_256,
                "TT_HF_SBDP": "1",
                "TT_FULL_AC_FIRST_N": "0",
                "TORCH_TPU_DEFER_AND_FUSE": "1",
                "LIBTPU_INIT_ARGS": calibrated_v7x_flags,
            },
            "args": [
                *hf_base,
                f"--training.num_tokens_per_microbatch_per_dp_rank={mb48}",
                f"--training.num_tokens_per_train_step={2 * world_size * mb48}",
            ],
        },
        # 3. Intra-Host EP=8 + ZeRO-2 (`fsdp_reshard_after_forward="never"`) @ mb=32, 2xGA (12,288 tok/expert, 0 cross-host AllToAll, 0 bwd AG!)
        {
            "name": "v7x_p10b_03_ep8_zero2_tile512_mb32_ga2",
            "lane": "v7x-16 Intra-Host EP=8 + ZeRO-2 (`reshard=never`, 0 cross-host AllToAll, 0 bwd AG, 12,288 tok/exp, mb=32 2xGA)",
            "workdir": "/torchtitan-native",
            "env": {"LIBTPU_INIT_ARGS": calibrated_v7x_flags},
            "args": [
                "--module=torchtitan.experiments.tpu.examples.flint",
                "--config=flint_40b_p10b_ep8_zero2_n36",
                f"--training.steps={steps}",
                "--training.mixed_precision_reduce=bfloat16",
                "--metrics.log_freq=1",
                "--debug.seed=42",
                "--tpu_config.mfu_benchmark_start_step=3",
                f"--training.num_tokens_per_microbatch_per_dp_rank={mb32}",
                f"--training.num_tokens_per_train_step={2 * world_size * mb32}",
            ],
        },
        # 4. Intra-Host EP=8 + ZeRO-2 @ mb=48, 2xGA (18,432 tok/expert, 6.29M tok/step!)
        {
            "name": "v7x_p10b_04_ep8_zero2_tile512_mb48_ga2",
            "lane": "v7x-16 Intra-Host EP=8 + ZeRO-2 (`reshard=never`, 18,432 tok/exp, mb=48 = 6.29M tok/step 2xGA)",
            "workdir": "/torchtitan-native",
            "env": {"LIBTPU_INIT_ARGS": calibrated_v7x_flags},
            "args": [
                "--module=torchtitan.experiments.tpu.examples.flint",
                "--config=flint_40b_p10b_ep8_zero2_n36",
                f"--training.steps={steps}",
                "--training.mixed_precision_reduce=bfloat16",
                "--metrics.log_freq=1",
                "--debug.seed=42",
                "--tpu_config.mfu_benchmark_start_step=3",
                f"--training.num_tokens_per_microbatch_per_dp_rank={mb48}",
                f"--training.num_tokens_per_train_step={2 * world_size * mb48}",
            ],
        },
        # 5. Intra-Host EP=8 + Host Activation Offload (`tpu_selective_and_offload` + prefetch=1) @ mb=96, 2xGA (36,864 tok/expert!)
        {
            "name": "v7x_p10b_05_ep8_offload_mb96_ga2",
            "lane": "v7x-16 Intra-Host EP=8 + Host-Offload (`tpu_selective_and_offload` + prefetch=1, 36,864 tok/exp, mb=96 2xGA)",
            "workdir": "/torchtitan-native",
            "env": {"LIBTPU_INIT_ARGS": calibrated_v7x_flags},
            "args": [
                "--module=torchtitan.experiments.tpu.examples.flint",
                "--config=flint_40b_p10b_ep8_offload",
                f"--training.steps={steps}",
                "--training.mixed_precision_reduce=bfloat16",
                "--metrics.log_freq=1",
                "--debug.seed=42",
                "--tpu_config.mfu_benchmark_start_step=3",
                f"--training.num_tokens_per_microbatch_per_dp_rank={mb96}",
                f"--training.num_tokens_per_train_step={2 * world_size * mb96}",
            ],
        },
        # 6. HF PR#123 @ mb=64 (131,072 tok/rank) + HybridAC N=24 + CE b=1024 + tile_m=256 + scale_before_down_proj + 2xGA
        {
            "name": "v7x_p10b_06_hf_pr123_tile256_sbdp_hybridac24_b1024_mb64_ga2",
            "lane": "v7x-16 HF PR#123 (mb=64 = 4.19M tok/mb + HybridAC N=24 + CE b=1024 + tile_m=256 + scale_before_down_proj + 2xGA)",
            "workdir": "/torchtitan-hf",
            "env": {
                "TT_HF_GMM": paired_gmm_256,
                "TT_HF_SBDP": "1",
                "TT_FULL_AC_FIRST_N": "24",
                "TORCH_TPU_DEFER_AND_FUSE": "1",
                "LIBTPU_INIT_ARGS": calibrated_v7x_flags,
            },
            "args": [
                *hf_base,
                f"--training.num_tokens_per_microbatch_per_dp_rank={mb64}",
                f"--training.num_tokens_per_train_step={2 * world_size * mb64}",
            ],
        },
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
    idle_timeout_s: float = 900.0,
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


def parse_metrics(log_text: str, chip_type: str) -> dict:
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
        mfu_m = MFU_RE.search(line)
        mem_m = MEM_RE.search(line)
        peak_tf = PEAK_BF16_TFLOPS_PER_RANK.get(chip_type, 1153.5) * 1e12
        calc_mfu = 100.0 * FLINT_FLOPS_PER_TOKEN * tps / peak_tf
        logged_mfu = float(mfu_m.group("mfu")) if mfu_m else calc_mfu
        mem_gib = float(mem_m.group("mem")) if mem_m else None
        steps.append(
            {
                "step": step,
                "loss": loss,
                "tps_per_rank": tps,
                "logged_mfu_pct": round(logged_mfu, 2),
                "mfu_pct": round(calc_mfu, 2),
                "mem_gib": mem_gib,
            }
        )
    steady = [s for s in steps if s["step"] >= 4] or steps[-5:]
    if not steady:
        return {"num_logged_steps": 0, "steady_tps": 0.0, "steady_mfu_pct": 0.0}
    avg_tps = sum(s["tps_per_rank"] for s in steady) / len(steady)
    max_tps = max(s["tps_per_rank"] for s in steady)
    peak_tf = PEAK_BF16_TFLOPS_PER_RANK.get(chip_type, 1153.5) * 1e12
    return {
        "num_logged_steps": len(steps),
        "first_loss": steps[0]["loss"],
        "final_loss": steps[-1]["loss"],
        "steady_tps": round(avg_tps, 1),
        "peak_tps": round(max_tps, 1),
        "steady_mfu_pct": round(100.0 * FLINT_FLOPS_PER_TOKEN * avg_tps / peak_tf, 2),
        "peak_mfu_pct": round(100.0 * FLINT_FLOPS_PER_TOKEN * max_tps / peak_tf, 2),
        "peak_mem_gib": max((s["mem_gib"] for s in steps if s["mem_gib"] is not None), default=None),
        "steps": steps,
    }


def write_summary(
    out_dir: Path, bench_dir: Path, slice_name: str, chip_type: str, world_size: int, results: list[dict]
) -> None:
    payload = {
        "updated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "slice_name": slice_name,
        "chip_type": chip_type,
        "world_size_ranks": world_size,
        "flops_per_token": FLINT_FLOPS_PER_TOKEN,
        "peak_bf16_tflops_per_rank": PEAK_BF16_TFLOPS_PER_RANK.get(chip_type),
        "results": results,
    }
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2))
    md_lines = [
        f"# Flint (`RecursiveSI/flint` 40B MoE) TPU v7x Pass 10B (`tile_m=256` + `SBDP` + `EP=8 ZeRO-2` + `Offload`) — `{slice_name}`",
        "",
        f"- **Updated (UTC):** `{payload['updated_at_utc']}`",
        f"- **Chip / World Size:** `{chip_type}` (`{world_size}` TensorCore ranks, `{PEAK_BF16_TFLOPS_PER_RANK.get(chip_type)}` BF16 TFLOP/s/rank)",
        "",
        "| Experiment | Lane | Status | Steady tok/s/rank | Steady MFU (%) | Peak MFU (%) | Peak HBM (GiB) | Loss (start -> end) | Wall Time (s) |",
        "|---|---|---|---:|---:|---:|---:|---|---:|",
    ]
    for r in results:
        m = r.get("metrics", {})
        loss_str = f"{m.get('first_loss', '-')} -> {m.get('final_loss', '-')}" if m.get("num_logged_steps") else "-"
        md_lines.append(
            f"| `{r['name']}` | {r['lane']} | `{r['status']}` | "
            f"**{m.get('steady_tps', 0.0):,.1f}** | **{m.get('steady_mfu_pct', 0.0):.2f}%** | "
            f"{m.get('peak_mfu_pct', 0.0):.2f}% | {m.get('peak_mem_gib') or '-'} | {loss_str} | {r['duration_s']:.1f} |"
        )
    md_text = "\n".join(md_lines) + "\n"
    (out_dir / "SUMMARY.md").write_text(md_text)
    if bench_dir.exists():
        (bench_dir / f"flint_{slice_name}_pass10b_summary.json").write_text(json.dumps(payload, indent=2))
        (bench_dir / f"flint_{slice_name}_pass10b_SUMMARY.md").write_text(md_text)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chip-type", choices=["v7x", "v5p", "v6e"], default="v7x")
    parser.add_argument("--nnodes", type=int, required=True)
    parser.add_argument("--nproc-per-node", type=int, default=8)
    parser.add_argument("--master-addr", type=str, required=True)
    parser.add_argument("--slice-name", type=str, required=True)
    parser.add_argument("--hf-model", type=str, default="/models/models/RecursiveSI/flint")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--base-port", type=int, default=29641)
    parser.add_argument("--per-run-timeout", type=int, default=2100)
    parser.add_argument("--output-dir", type=str, default="/models/flint-results")
    args = parser.parse_args()

    apply_runtime_patches()
    node_rank = int(os.environ.get("JOB_COMPLETION_INDEX", os.environ.get("NODE_RANK", "0")))
    world_size = args.nnodes * args.nproc_per_node
    wait_for_master_dns(args.master_addr)

    ts = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.output_dir) / f"{args.slice_name}_pass10b_{ts}"
    bench_dir = Path("/models/benchmark-results")
    if node_rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
        bench_dir.mkdir(parents=True, exist_ok=True)

    exps = build_v7x_pass10b_experiments(world_size, args.hf_model, args.steps)
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
                cmd, exp["workdir"], env, node_rank, args.per_run_timeout, idle_timeout_s=900.0
            )
        except Exception as exc:
            rc = -1
            log_chunks = [f"[flint-v7x-p10b] Exception: {exc}\n"]
        dur = time.time() - t0
        if node_rank == 0:
            raw_log = "".join(log_chunks)
            (out_dir / f"{exp['name']}.log").write_text(raw_log)
            metrics = parse_metrics(raw_log, args.chip_type)
            status = "PASS" if (rc == 0 and metrics.get("num_logged_steps", 0) > 0) else f"FAIL(rc={rc})"
            results.append(
                {
                    "name": exp["name"],
                    "lane": exp["lane"],
                    "status": status,
                    "return_code": rc,
                    "duration_s": round(dur, 1),
                    "command": cmd,
                    "env_overrides": exp["env"],
                    "metrics": metrics,
                }
            )
            write_summary(out_dir, bench_dir, args.slice_name, args.chip_type, world_size, results)
            print(
                f"[flint-v7x-p10b] SAVED {exp['name']}: status={status} "
                f"steady_tps={metrics.get('steady_tps', 0)} steady_mfu={metrics.get('steady_mfu_pct', 0)}%",
                flush=True,
            )
        time.sleep(8)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
