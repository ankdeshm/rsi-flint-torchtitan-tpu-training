#!/usr/bin/env python3
"""Pass 10C & Pass 10D TPU v7x-16 (2x2x4, 16 Ironwood chips / 32 TensorCores) MFU Breakthrough Sweeps.

Executes across BOTH v7x-16 slices in parallel (`--suite=pass10c` on `tpu-v7x-16-pool` and
`--suite=pass10d` on `tpu-v7x-16-pool`), combining ALL root-cause discoveries from Pass 10A/10B
and TorchTitan's Qwen3-30B-A3B / Qwen3-235B Ironwood recipes:
  1. Expert Parallelism (`EP=8` intra-host, `EP=16` half-slice, `EP=32` full-slice) + `ZeRO-2`
     (`fsdp_reshard_after_forward="never"`, `compile_grad_norm=False`):
     - Slashes per-rank expert weight HBM streaming by 8x..32x (`75.7 GB` -> `2.37 GB` = `50.3 MB/layer` at EP=32!).
     - Eliminates 100% of expert weight `AllGather` and 100% of expert gradient `ReduceScatter` at EP=32!
     - Boosts tokens per expert 8x..32x (`12,288` -> `98,304` tok/expert).
  2. Whole-Step XLA Program (`graph_split.enabled=False` + `enable_remat_barriers=True`) vs Per-Layer `GraphSplit`:
     - Compiles all 49 layers (forward + Pallas CrossEntropy + backward) into ONE single XLA executable
       (eliminating 98 per-layer host dispatch barriers and enabling global cross-layer collective hiding!).
     - Also tests `optimizer_compile="in_graph"` (fusing AdamW directly into the whole-step XLA executable).
  3. Direct `config.libtpu_init_flags` injection (not just `os.environ["LIBTPU_INIT_ARGS"]`, which `TpuGraphTrainer`
     overwrites from `config.libtpu_init_flags`!) with calibrated 64 MiB Ironwood VMEM (`65536` KiB),
     Region Analysis (`limit=1000`), Async AllToAll, and 25M backward layer-scheduler latency threshold.
  4. Async multi-step host pipelining (`metrics.log_freq=5` over `15` steps, matching `qwen3_moe_30b_fsdp_only_full_ac_benchmark`
     so host loss logging does not stall PJRT step-to-step pipelining, while still logging steps 1, 5, 10, 15).
  5. `ScannedTpuFlintModel` (`use_scan_layers=True`, `moe.load_balance_coeff=None`, `full_and_scan_offload`, `_Q235_SCAN_FLAGS`)
     and `EP=16` Full Host Offload (`save_last_n_layers={}`, `prefetch=2`) @ `mb=96`.
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

NATIVE_P10CD_PATCH = '''

# ==============================================================================
# Pass 10C / 10D Flint v7x-16 Breakthrough Configs (EP=8/16/32 ZeRO-2 + WholeStep + Scan)
# ==============================================================================
from torchtitan.experiments.tpu.token_dispatcher import (
    TpuRoutedExperts,
    TpuRoutedExpertsConverter,
)
from torchtitan.models.common.moe import MoE

_P10CD_BF16_ROPE = (
    "torchtitan.experiments.tpu.overrides.bf16_rope"
    ".tpu_activation_dtype_cossin_rope"
)
_P10CD_DISCARDED_COUNTER = (
    "torchtitan.experiments.tpu.overrides.discarded_token_counter"
    ".tpu_discarded_token_counter"
)

_P10CD_V7X_BASE_FLAGS = {
    "xla_tpu_dvfs_p_state": 7,
    "xla_tpu_scoped_vmem_limit_kib": 65536,
    "xla_tpu_enable_offloading_copy_to_sparsecore": False,
    "xla_tpu_copy_insertion_use_region_analysis": True,
    "xla_tpu_copy_insertion_use_region_analysis_limit": 1000,
    "xla_tpu_prefer_async_allgather_to_allreduce": True,
    "xla_enable_async_all_gather": True,
    "xla_enable_async_all_reduce": True,
    "xla_tpu_enable_async_all_to_all": True,
    "xla_tpu_overlap_compute_collective_tc": True,
    "xla_tpu_layer_scheduler_min_compute_latency_for_backward_pass": 25000000,
}

_P10CD_SCAN_EXTRA_FLAGS = tpu_flags.merge_flags(
    _P10CD_V7X_BASE_FLAGS,
    tpu_flags.HIERARCHICAL_REDUCE_SCATTER,
    tpu_flags.SELECTIVE_ICI_RS_PIPELINING,
    tpu_flags.ICI_AG_PIPELINING,
    tpu_flags.SPARSECORE_ASYNC_ALL_GATHER,
    tpu_flags.SPARSECORE_LATENCY_HIDING_SCHEDULER,
    {
        "xla_tpu_ici_rs_pipelining_threshold_kib": 32768,
    },
)


def _configure_ep_experts_p10cd(config, *, ep_degree: int, tile_m: int = 512) -> None:
  config.parallelism.expert_parallel_degree = ep_degree
  for _, ex, _, _ in config.model_spec.model.traverse(TpuGroupedExperts.Config):
    ex.use_pallas_grouped_mm = True
    ex.use_pallas_paired_grouped_mm = False
    ex.pallas_grouped_mm_tile_m = tile_m
    ex.pallas_grouped_mm_tgmm_tiling = (tile_m, 512, 512)
    ex.use_kernel_native_weight_layout = True
  if _P10CD_BF16_ROPE not in config.override.imports:
    config.override.imports.append(_P10CD_BF16_ROPE)
  config.libtpu_init_flags = tpu_flags.merge_flags(
      config.libtpu_init_flags,
      tpu_flags.SPARSECORE_LATENCY_HIDING_SCHEDULER,
      tpu_flags.DUAL_SPARSECORE_REDUCE_SCATTER,
      _HIER_SPARSECORE_RS_FLAGS,
      _P10CD_V7X_BASE_FLAGS,
  )


def _build_wholestep_ep_zero2_p10cd(
    *,
    ep_degree: int,
    save_attn_layers: int,
    opt_compile: str = "separate_graph",
) -> TpuGraphTrainer.Config:
  config = flint_40b_v005_gt_rb()
  _configure_ep_experts_p10cd(config, ep_degree=ep_degree, tile_m=512)
  config.parallelism.fsdp_reshard_after_forward = "never"
  config.compile.graph_split.enabled = False
  config.compile.enable_remat_barriers = True
  config.compile.memory_policy = "tpu_selective"
  config.compile.save_regions = ("moe.routing",)
  config.compile.save_last_n_layers = {"attention.inner_attention": save_attn_layers}
  config.compile.optimizer_compile = opt_compile
  config.compile.compile_grad_norm = False
  config.optimizer.implementation = "fused"
  return config


def _build_split_ep_zero2_p10cd(
    *,
    ep_degree: int,
    save_attn_layers: int,
) -> TpuGraphTrainer.Config:
  config = flint_40b_v012_hier_rs()
  _configure_ep_experts_p10cd(config, ep_degree=ep_degree, tile_m=512)
  config.parallelism.fsdp_reshard_after_forward = "never"
  config.compile.graph_split.enabled = True
  config.compile.graph_split.forward_layers_per_executable = 1
  config.compile.graph_split.backward_partition_mode = "by_layer"
  config.compile.graph_split.backward_layers_per_executable = 1
  config.compile.memory_policy = "tpu_selective"
  config.compile.save_regions = ("moe.routing",)
  config.compile.save_last_n_layers = {"attention.inner_attention": save_attn_layers}
  config.compile.optimizer_compile = "separate_graph"
  config.compile.compile_grad_norm = False
  config.optimizer.implementation = "fused"
  return config


# --- Pass 10C Configs ---
def flint_40b_p10c_wholestep_ep32_zero2_n49() -> TpuGraphTrainer.Config:
  return _build_wholestep_ep_zero2_p10cd(ep_degree=32, save_attn_layers=49)


def flint_40b_p10c_wholestep_ep16_zero2_n40() -> TpuGraphTrainer.Config:
  return _build_wholestep_ep_zero2_p10cd(ep_degree=16, save_attn_layers=40)


def flint_40b_p10c_split_ep32_zero2_n40() -> TpuGraphTrainer.Config:
  return _build_split_ep_zero2_p10cd(ep_degree=32, save_attn_layers=40)


def flint_40b_p10c_split_ep16_zero2_n36() -> TpuGraphTrainer.Config:
  return _build_split_ep_zero2_p10cd(ep_degree=16, save_attn_layers=36)


def flint_40b_p10c_split_ep8_zero2_n36() -> TpuGraphTrainer.Config:
  return _build_split_ep_zero2_p10cd(ep_degree=8, save_attn_layers=36)


# --- Pass 10D Configs ---
def flint_40b_p10d_wholestep_ep32_zero2_ingraph_n36() -> TpuGraphTrainer.Config:
  return _build_wholestep_ep_zero2_p10cd(
      ep_degree=32, save_attn_layers=36, opt_compile="in_graph"
  )


def flint_40b_p10d_wholestep_ep8_zero2_n32() -> TpuGraphTrainer.Config:
  return _build_wholestep_ep_zero2_p10cd(ep_degree=8, save_attn_layers=32)


def flint_40b_p10d_split_ep32_zero2_n28() -> TpuGraphTrainer.Config:
  return _build_split_ep_zero2_p10cd(ep_degree=32, save_attn_layers=28)


def flint_40b_p10d_scanned_q235_sbdp() -> TpuGraphTrainer.Config:
  config = flint_40b_v009()
  for _, m, _, _ in config.model_spec.model.traverse(MoE.Config):
    m.load_balance_coeff = None
  config.model_spec.model = (
      TpuRoutedExpertsConverter.Config().build().convert(config.model_spec.model)
  )
  for _, r, _, _ in config.model_spec.model.traverse(TpuRoutedExperts.Config):
    r.scale_before_down_proj = True
  for _, ex, _, _ in config.model_spec.model.traverse(TpuGroupedExperts.Config):
    ex.use_pallas_grouped_mm = True
    ex.use_pallas_paired_grouped_mm = True
    ex.pallas_grouped_mm_tile_m = 256
    ex.pallas_grouped_mm_tgmm_tiling = (512, 2048, 0)
    ex.pallas_grouped_mm_w2_tgmm_tiling = (256, 512, 0)
    ex.pallas_grouped_mm_w2_transposed_tgmm_tiling = (256, 0, 2048)
    ex.use_kernel_native_weight_layout = True
  for imp in (_P10CD_BF16_ROPE, _P10CD_DISCARDED_COUNTER):
    if imp not in config.override.imports:
      config.override.imports.append(imp)
  config.compile.graph_split.enabled = False
  config.compile.use_scan_layers = True
  config.compile.memory_policy = "full_and_scan_offload"
  config.compile.save_regions = ()
  config.compile.save_last_n_layers = {}
  config.compile.fsdp_allreduce_slice_parameter_patterns = ()
  config.compile.optimizer_compile = "separate_graph"
  config.compile.compile_grad_norm = True
  config.optimizer.implementation = "fused"
  if hasattr(config.metrics, "enable_moe_routing"):
    config.metrics.enable_moe_routing = False
  config.libtpu_init_flags = tpu_flags.merge_flags(
      config.libtpu_init_flags, _P10CD_SCAN_EXTRA_FLAGS
  )
  return config


def flint_40b_p10d_ep16_fulloffload() -> TpuGraphTrainer.Config:
  config = flint_40b_v012_hier_rs()
  _configure_ep_experts_p10cd(config, ep_degree=16, tile_m=512)
  config.parallelism.fsdp_reshard_after_forward = "always"
  config.compile.memory_policy = "tpu_selective_and_offload"
  config.compile.save_regions = ("moe.routing",)
  config.compile.save_last_n_layers = {}
  config.compile.graph_split.enabled = True
  config.compile.graph_split.forward_layers_per_executable = 1
  config.compile.graph_split.backward_partition_mode = "by_layer"
  config.compile.graph_split.backward_layers_per_executable = 1
  config.compile.graph_split.cpu_offload_prefetch_executables = 2
  config.compile.optimizer_compile = "separate_graph"
  config.compile.compile_grad_norm = False
  config.optimizer.implementation = "fused"
  return config


__all__.extend([
    "flint_40b_p10c_wholestep_ep32_zero2_n49",
    "flint_40b_p10c_wholestep_ep16_zero2_n40",
    "flint_40b_p10c_split_ep32_zero2_n40",
    "flint_40b_p10c_split_ep16_zero2_n36",
    "flint_40b_p10c_split_ep8_zero2_n36",
    "flint_40b_p10d_wholestep_ep32_zero2_ingraph_n36",
    "flint_40b_p10d_wholestep_ep8_zero2_n32",
    "flint_40b_p10d_split_ep32_zero2_n28",
    "flint_40b_p10d_scanned_q235_sbdp",
    "flint_40b_p10d_ep16_fulloffload",
])
'''


def apply_runtime_patches() -> None:
    nat_reg = Path("/torchtitan-native/torchtitan/experiments/tpu/examples/flint/config_registry.py")
    if nat_reg.exists():
        txt = nat_reg.read_text()
        if "flint_40b_p10c_wholestep_ep32_zero2_n49" not in txt:
            nat_reg.write_text(txt + NATIVE_P10CD_PATCH)
            print("[flint-v7x-p10cd] Patched Native config_registry.py with Pass 10C/10D configs", flush=True)


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
    print(f"[flint-v7x-p10cd] Validated XLA flags: kept={len(kept)} dropped={dropped}", flush=True)
    return " ".join(kept)


def wait_for_master_dns(master_addr: str, timeout_s: int = 300) -> str:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            return socket.gethostbyname(master_addr)
        except socket.gaierror:
            time.sleep(3)
    return master_addr


def build_v7x_pass10cd_experiments(suite: str, world_size: int, steps: int) -> list[dict]:
    mb32 = 32 * 2048    # 65,536 tok/rank  -> @EP=32: 49,152 tok/exp | @EP=16: 24,576 tok/exp | @EP=8: 12,288 tok/exp
    mb48 = 48 * 2048    # 98,304 tok/rank  -> @EP=32: 73,728 tok/exp | @EP=16: 36,864 tok/exp | @EP=8: 18,432 tok/exp
    mb64 = 64 * 2048    # 131,072 tok/rank -> @EP=32: 98,304 tok/exp
    mb96 = 96 * 2048    # 196,608 tok/rank -> @EP=16: 73,728 tok/exp

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
        "--xla_tpu_layer_scheduler_min_compute_latency_for_backward_pass=25000000",
        "--xla_max_concurrent_async_all_gathers=4",
        "--xla_latency_hiding_scheduler_rerun=2",
        "--xla_enable_transpose_trace=false",
    ])

    def _nat_exp(name: str, lane: str, cfg: str, mb: int, ga: int, log_freq: int = 5) -> dict:
        return {
            "name": name,
            "lane": lane,
            "workdir": "/torchtitan-native",
            "env": {"LIBTPU_INIT_ARGS": calibrated_v7x_flags},
            "args": [
                "--module=torchtitan.experiments.tpu.examples.flint",
                f"--config={cfg}",
                f"--training.steps={steps}",
                "--training.mixed_precision_reduce=bfloat16",
                f"--metrics.log_freq={log_freq}",
                "--debug.seed=42",
                "--tpu_config.mfu_benchmark_start_step=5",
                f"--training.num_tokens_per_microbatch_per_dp_rank={mb}",
                f"--training.num_tokens_per_train_step={ga * world_size * mb}",
            ],
        }

    if suite == "pass10c":
        return [
            _nat_exp(
                "v7x_p10c_01_wholestep_ep32_zero2_rematbar_mb32_1x",
                "v7x-16 WholeStep Graph + EP=32 + ZeRO-2 (0 expert AG/RS, 50.3MB/layer, 49,152 tok/exp, n49, mb=32 1x, log_freq=5)",
                "flint_40b_p10c_wholestep_ep32_zero2_n49",
                mb32,
                1,
            ),
            _nat_exp(
                "v7x_p10c_02_wholestep_ep16_zero2_rematbar_mb32_ga2",
                "v7x-16 WholeStep Graph + EP=16 + ZeRO-2 (100.6MB/layer, 24,576 tok/exp, n40, mb=32 2xGA, log_freq=5)",
                "flint_40b_p10c_wholestep_ep16_zero2_n40",
                mb32,
                2,
            ),
            _nat_exp(
                "v7x_p10c_03_split_ep32_zero2_n40_mb48_ga2",
                "v7x-16 GraphSplit + EP=32 + ZeRO-2 (0 expert AG/RS, 73,728 tok/exp, n40, mb=48 2xGA, log_freq=5)",
                "flint_40b_p10c_split_ep32_zero2_n40",
                mb48,
                2,
            ),
            _nat_exp(
                "v7x_p10c_04_split_ep16_zero2_n36_mb48_ga2",
                "v7x-16 GraphSplit + EP=16 + ZeRO-2 (36,864 tok/exp, n36, mb=48 2xGA, log_freq=5)",
                "flint_40b_p10c_split_ep16_zero2_n36",
                mb48,
                2,
            ),
            _nat_exp(
                "v7x_p10c_05_split_ep8_zero2_n36_mb48_ga2",
                "v7x-16 GraphSplit + Intra-Host EP=8 + ZeRO-2 (100% intra-host AllToAll, 18,432 tok/exp, n36, mb=48 2xGA)",
                "flint_40b_p10c_split_ep8_zero2_n36",
                mb48,
                2,
            ),
        ]

    return [
        _nat_exp(
            "v7x_p10d_01_wholestep_ep32_zero2_ingraph_opt_mb48_1x",
            "v7x-16 WholeStep Graph + EP=32 + ZeRO-2 + in_graph AdamW (73,728 tok/exp, n36, mb=48 1x, log_freq=5)",
            "flint_40b_p10d_wholestep_ep32_zero2_ingraph_n36",
            mb48,
            1,
        ),
        _nat_exp(
            "v7x_p10d_02_wholestep_ep8_zero2_rematbar_mb32_ga2",
            "v7x-16 WholeStep Graph + Intra-Host EP=8 + ZeRO-2 (100% intra-host AllToAll, 12,288 tok/exp, n32, mb=32 2xGA)",
            "flint_40b_p10d_wholestep_ep8_zero2_n32",
            mb32,
            2,
        ),
        _nat_exp(
            "v7x_p10d_03_split_ep32_zero2_n28_mb64_1x",
            "v7x-16 GraphSplit + EP=32 + ZeRO-2 (98,304 tok/exp, n28, mb=64 = 4.19M tok/step 1x, log_freq=5)",
            "flint_40b_p10d_split_ep32_zero2_n28",
            mb64,
            1,
        ),
        _nat_exp(
            "v7x_p10d_04_scanned_layers_q235_flags_mb64_1x",
            "v7x-16 ScannedTpuFlintModel (use_scan_layers + full_and_scan_offload + Q235 scan flags + SBDP, mb=64 1x)",
            "flint_40b_p10d_scanned_q235_sbdp",
            mb64,
            1,
        ),
        _nat_exp(
            "v7x_p10d_05_ep16_fulloffload_mb96_1x",
            "v7x-16 GraphSplit + EP=16 + Full Host Offload (save_last_n=0, prefetch=2, 73,728 tok/exp, mb=96 = 6.29M tok/step 1x)",
            "flint_40b_p10d_ep16_fulloffload",
            mb96,
            1,
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
    # Exclude step 1..5 (compile/warmup window when log_freq=5) if step >= 6 exists
    steady = [s for s in steps if s["step"] >= 6] or [s for s in steps if s["step"] >= 4] or steps[-2:]
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
    out_dir: Path, bench_dir: Path, suite: str, slice_name: str, chip_type: str, world_size: int, results: list[dict]
) -> None:
    payload = {
        "updated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "suite": suite,
        "slice_name": slice_name,
        "chip_type": chip_type,
        "world_size_ranks": world_size,
        "flops_per_token": FLINT_FLOPS_PER_TOKEN,
        "peak_bf16_tflops_per_rank": PEAK_BF16_TFLOPS_PER_RANK.get(chip_type),
        "results": results,
    }
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2))
    md_lines = [
        f"# Flint (`RecursiveSI/flint` 40B MoE) TPU v7x `{suite.upper()}` (`EP=8/16/32 ZeRO-2` + `WholeStep` + `Scan`) — `{slice_name}`",
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
        (bench_dir / f"flint_{slice_name}_{suite}_summary.json").write_text(json.dumps(payload, indent=2))
        (bench_dir / f"flint_{slice_name}_{suite}_SUMMARY.md").write_text(md_text)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=["pass10c", "pass10d"], default="pass10c")
    parser.add_argument("--chip-type", choices=["v7x", "v5p", "v6e"], default="v7x")
    parser.add_argument("--nnodes", type=int, required=True)
    parser.add_argument("--nproc-per-node", type=int, default=8)
    parser.add_argument("--master-addr", type=str, required=True)
    parser.add_argument("--slice-name", type=str, required=True)
    parser.add_argument("--hf-model", type=str, default="/models/models/RecursiveSI/flint")
    parser.add_argument("--steps", type=int, default=15)
    parser.add_argument("--base-port", type=int, default=29661)
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

    exps = build_v7x_pass10cd_experiments(args.suite, world_size, args.steps)
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
                cmd, exp["workdir"], env, node_rank, args.per_run_timeout, idle_timeout_s=1200.0
            )
        except Exception as exc:
            rc = -1
            log_chunks = [f"[flint-v7x-p10cd] Exception: {exc}\n"]
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
            write_summary(out_dir, bench_dir, args.suite, args.slice_name, args.chip_type, world_size, results)
            print(
                f"[flint-v7x-{args.suite}] SAVED {exp['name']}: status={status} "
                f"steady_tps={metrics.get('steady_tps', 0)} steady_mfu={metrics.get('steady_mfu_pct', 0)}%",
                flush=True,
            )
        time.sleep(8)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
