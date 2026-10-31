#!/usr/bin/env python3
"""Pass 9 TPU v7x-16 (2x2x4, 16 Ironwood chips / 32 TensorCores) MFU Frontier for Flint (Priority #1).

Combines all Pass 7 v7x-16 breakthroughs (`18.36%` Native PR#122 & `18.02%` HF PR#123) with
the Pass 8 v5p-32 record-setting breakthroughs (`24.87%` Steady / `24.95%` Peak MFU):
  1. `--training.mixed_precision_reduce=bfloat16` (halves 32-rank FSDP ReduceScatter payload 159 GB -> 79.6 GB)
  2. `Llama-4-Scout` Async AllGather + AG Backward Pipelining + ICI AG Pipelining + `v7x` Hierarchical SparseCore RS
  3. Native `TpuGraphTrainer` (`n45` 1x step & `n43` 2x GradAccum + all-49 `moe.routing` + `tile_m=1024` + CompiledOpt)
  4. HF Modeling Backend PR #123 (`RecursiveSI/flint` weights, Paired GMM `tile_m=1024`, `b=1024`, `DEFER_AND_FUSE=1`, Probed CompiledOpt)
  5. Per-layer closure-safe `HybridAC` (`HSDP 2x16` 8-chip / 16-TensorCore FSDP-16 shards)
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import time

FLINT_FLOPS_PER_TOKEN = 17.385e9
PEAK_BF16_TFLOPS_PER_RANK = {"v7x": 1153.5, "v5p": 459.0, "v6e": 918.0}

STEP_RE = re.compile(
    r"step:\s*(?P<step>\d+).*?loss:\s*(?P<loss>[0-9.]+).*?tps:\s*(?P<tps>[0-9.,]+)",
    re.IGNORECASE,
)
MFU_RE = re.compile(r"mfu:\s*(?P<mfu>[0-9.]+)%", re.IGNORECASE)
MEM_RE = re.compile(r"memory:\s*(?P<mem>[0-9.]+)GiB", re.IGNORECASE)

NATIVE_P9_PATCH = '''

# ==============================================================================
# Pass 9 Flint v7x-16 MFU Configs
# ==============================================================================

def flint_40b_p9_n45_moe_route_comp_opt() -> TpuGraphTrainer.Config:
  config = flint_40b_v014_tile1024()
  config.compile.memory_policy = "tpu_selective"
  config.compile.save_regions = ("moe.routing",)
  config.compile.save_last_n_layers = {"attention.inner_attention": 45}
  config.compile.optimizer_compile = "separate_graph"
  config.compile.compile_grad_norm = True
  return config


def flint_40b_p9_n43_moe_route_comp_opt() -> TpuGraphTrainer.Config:
  config = flint_40b_v014_tile1024()
  config.compile.memory_policy = "tpu_selective"
  config.compile.save_regions = ("moe.routing",)
  config.compile.save_last_n_layers = {"attention.inner_attention": 43}
  config.compile.optimizer_compile = "separate_graph"
  config.compile.compile_grad_norm = True
  return config


def flint_40b_p9_n22_moe_route_comp_opt() -> TpuGraphTrainer.Config:
  config = flint_40b_v014_tile1024()
  config.compile.memory_policy = "tpu_selective"
  config.compile.save_regions = ("moe.routing",)
  config.compile.save_last_n_layers = {"attention.inner_attention": 22}
  config.compile.optimizer_compile = "separate_graph"
  config.compile.compile_grad_norm = True
  return config


__all__.extend([
    "flint_40b_p9_n45_moe_route_comp_opt",
    "flint_40b_p9_n43_moe_route_comp_opt",
    "flint_40b_p9_n22_moe_route_comp_opt",
])
'''

HF_HYBRID_AC_PATCH_V3 = '''

# ==============================================================================
# Per-Layer Compiler-Native HybridAC (Pass 9B — SelectiveAC._wrap_block)
# ==============================================================================
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


def apply_runtime_patches() -> None:
    nat_reg = Path("/torchtitan-native/torchtitan/experiments/tpu/examples/flint/config_registry.py")
    if nat_reg.exists():
        txt = nat_reg.read_text()
        if "flint_40b_p9_n45_moe_route_comp_opt" not in txt:
            nat_reg.write_text(txt + NATIVE_P9_PATCH)
            print("[flint-v7x-pass9] Patched Native config_registry.py with Pass 9 configs", flush=True)

    hf_ac = Path("/torchtitan-hf/torchtitan/experiments/tpu/activation_checkpoint.py")
    if hf_ac.exists():
        txt = hf_ac.read_text()
        if "_hybrid_attn_only_wrap_v3" not in txt:
            hf_ac.write_text(txt + HF_HYBRID_AC_PATCH_V3)
            print("[flint-v7x-pass9] Patched HF activation_checkpoint.py with per-layer HybridAC v3", flush=True)


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
    print(f"[flint-v7x-pass9] Validated XLA flags: kept={len(kept)} dropped={dropped}", flush=True)
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
            flags = res.stdout.strip().split()
            print(f"[flint-v7x-pass9] {workdir} compiled-opt flags: {flags}", flush=True)
            return flags
    except Exception as exc:
        print(f"[flint-v7x-pass9] probe_compile_opt_args({workdir}) warn: {exc}", flush=True)
    return []


def wait_for_master_dns(master_addr: str, timeout_s: int = 300) -> str:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            return socket.gethostbyname(master_addr)
        except socket.gaierror:
            time.sleep(3)
    return master_addr


def build_v7x_pass9_experiments(world_size: int, hf_model_path: str, steps: int) -> list[dict]:
    mb48 = 48 * 2048  # 98,304 tok/rank -> 3.15M tok/microbatch across 32 TensorCores
    mb56 = 56 * 2048  # 114,688 tok/rank -> 3.67M tok/microbatch across 32 TensorCores

    v7x_scout_candidates = [
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
    ]
    v7x_scout_xla = filter_supported_xla_flags(v7x_scout_candidates)
    hf_comp_opt = probe_compile_opt_args("/torchtitan-hf")

    hf_splash_args = [
        "--splash_attention_kernel.sa_block_q=512",
        "--splash_attention_kernel.sa_block_kv=512",
        "--splash_attention_kernel.sa_block_kv_compute=512",
        "--splash_attention_kernel.sa_block_q_dkv=512",
        "--splash_attention_kernel.sa_block_kv_dkv=512",
        "--splash_attention_kernel.sa_block_kv_dkv_compute=512",
    ]
    paired_gmm_str = (
        "use_pallas_paired_grouped_mm=true,"
        "pallas_grouped_mm_tile_m=1024,"
        "pallas_grouped_mm_tgmm_tiling=512:2048:0,"
        "pallas_grouped_mm_w2_tgmm_tiling=512:512:0,"
        "pallas_grouped_mm_w2_transposed_tgmm_tiling=512:0:2048"
    )
    hf_base = [
        "--module=torchtitan.experiments.tpu.transformers_modeling_backend",
        "--config=hf_moe_auto_max_attnac",
        f"--hf-model={hf_model_path}",
        "--training.max_context_length=2048",
        f"--training.steps={steps}",
        "--training.mixed_precision_reduce=bfloat16",
        "--lr_scheduler.warmup_steps=0",
        "--lr_scheduler.decay_ratio=0.0",
        "--metrics.log_freq=1",
        "--debug.seed=42",
        "--loss.b_block_size=1024",
        *hf_splash_args,
        *hf_comp_opt,
    ]

    return [
        # 1. Native PR#122 (n45 + All49 MoE Routing + tile_m=1024 + CompiledOpt) + BF16 Reduce + v7x HierSC/Scout XLA (1x step)
        {
            "name": "v7x_p9_01_native_pr122_bf16red_scout_n45_mb48_1x",
            "lane": "v7x-16 Native TpuGraphTrainer PR#122 (n45 + All49 MoERouting + BF16 Reduce + Scout/HierSC XLA, mb=48 = 3.15M tok/step)",
            "workdir": "/torchtitan-native",
            "env": {"LIBTPU_INIT_ARGS": v7x_scout_xla},
            "args": [
                "--module=torchtitan.experiments.tpu.examples.flint",
                "--config=flint_40b_p9_n45_moe_route_comp_opt",
                f"--training.steps={steps}",
                "--training.mixed_precision_reduce=bfloat16",
                "--metrics.log_freq=1",
                "--debug.seed=42",
                "--tpu_config.mfu_benchmark_start_step=3",
                f"--training.num_tokens_per_microbatch_per_dp_rank={mb48}",
                f"--training.num_tokens_per_train_step={world_size * mb48}",
            ],
        },
        # 2. Native PR#122 (n43 + All49 MoE Routing + 2x GradAccum) + BF16 Reduce + v7x HierSC/Scout XLA (6.29M tok/step)
        {
            "name": "v7x_p9_02_native_pr122_bf16red_scout_n43_mb48_ga2",
            "lane": "v7x-16 Native TpuGraphTrainer PR#122 (2x GradAccum @ mb=48 = 6.29M tok/step + n43 + BF16 Reduce + Scout/HierSC XLA)",
            "workdir": "/torchtitan-native",
            "env": {"LIBTPU_INIT_ARGS": v7x_scout_xla},
            "args": [
                "--module=torchtitan.experiments.tpu.examples.flint",
                "--config=flint_40b_p9_n43_moe_route_comp_opt",
                f"--training.steps={steps}",
                "--training.mixed_precision_reduce=bfloat16",
                "--metrics.log_freq=1",
                "--debug.seed=42",
                "--tpu_config.mfu_benchmark_start_step=3",
                f"--training.num_tokens_per_microbatch_per_dp_rank={mb48}",
                f"--training.num_tokens_per_train_step={2 * world_size * mb48}",
            ],
        },
        # 3. HF PR#123 (RecursiveSI/flint weights + Paired GMM tile_m=1024) + BF16 Reduce + Scout/HierSC XLA + CompiledOpt (1x step)
        {
            "name": "v7x_p9_03_hf_pr123_bf16red_scout_compopt_mb48_1x",
            "lane": "v7x-16 FSDP-32 HF PR#123 (Paired GMM tile_m=1024 + BF16 Reduce + Scout/HierSC XLA + CompiledOpt, mb=48 = 3.15M tok/step)",
            "workdir": "/torchtitan-hf",
            "env": {
                "TT_HF_GMM": paired_gmm_str,
                "TT_FULL_AC_FIRST_N": "0",
                "TORCH_TPU_DEFER_AND_FUSE": "1",
                "LIBTPU_INIT_ARGS": v7x_scout_xla,
            },
            "args": [
                *hf_base,
                f"--training.num_tokens_per_microbatch_per_dp_rank={mb48}",
                f"--training.num_tokens_per_train_step={world_size * mb48}",
            ],
        },
        # 4. Native PR#122 (2x GradAccum @ mb=56 = 7.34M tok/step + n22 + BF16 Reduce + Scout/HierSC XLA)
        {
            "name": "v7x_p9_04_native_pr122_bf16red_scout_n22_mb56_ga2",
            "lane": "v7x-16 Native TpuGraphTrainer PR#122 (2x GradAccum @ mb=56 = 7.34M tok/step + n22 + BF16 Reduce + Scout/HierSC XLA)",
            "workdir": "/torchtitan-native",
            "env": {"LIBTPU_INIT_ARGS": v7x_scout_xla},
            "args": [
                "--module=torchtitan.experiments.tpu.examples.flint",
                "--config=flint_40b_p9_n22_moe_route_comp_opt",
                f"--training.steps={steps}",
                "--training.mixed_precision_reduce=bfloat16",
                "--metrics.log_freq=1",
                "--debug.seed=42",
                "--tpu_config.mfu_benchmark_start_step=3",
                f"--training.num_tokens_per_microbatch_per_dp_rank={mb56}",
                f"--training.num_tokens_per_train_step={2 * world_size * mb56}",
            ],
        },
        # 5. HF PR#123 HSDP 2x FSDP-16 (8-Chip / 16-TC Shards) + Closure-Safe HybridAC v3 N=28 + BF16 Reduce (1x step)
        {
            "name": "v7x_p9_05_hf_pr123_hsdp2x16_bf16red_hybridac28_mb48_1x",
            "lane": "v7x-16 HSDP 2x FSDP-16 (8-Chip / 16-TC Shards, HybridAC v3 N=28 + BF16 Reduce + Scout/HierSC XLA, mb=48 = 3.15M tok/step)",
            "workdir": "/torchtitan-hf",
            "env": {
                "TT_HF_GMM": paired_gmm_str,
                "TT_FULL_AC_FIRST_N": "28",
                "TORCH_TPU_DEFER_AND_FUSE": "1",
                "LIBTPU_INIT_ARGS": v7x_scout_xla,
            },
            "args": [
                *hf_base,
                "--parallelism.data_parallel_replicate_degree=2",
                "--parallelism.data_parallel_shard_degree=16",
                f"--training.num_tokens_per_microbatch_per_dp_rank={mb48}",
                f"--training.num_tokens_per_train_step={world_size * mb48}",
            ],
        },
    ]


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
                "logged_mfu_pct": logged_mfu,
                "calc_mfu_pct": round(calc_mfu, 3),
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
        f"# Flint (`RecursiveSI/flint` 40B MoE) TPU v7x Pass 9 (`BF16 Reduce` + `Scout/HierSC XLA`) — `{slice_name}`",
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
        tag = "pass9b" if "--only-hf" in sys.argv else "pass9"
        (bench_dir / f"flint_{slice_name}_{tag}_summary.json").write_text(json.dumps(payload, indent=2))
        (bench_dir / f"flint_{slice_name}_{tag}_SUMMARY.md").write_text(md_text)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chip-type", choices=["v7x", "v5p", "v6e"], default="v7x")
    parser.add_argument("--nnodes", type=int, required=True)
    parser.add_argument("--nproc-per-node", type=int, default=8)
    parser.add_argument("--master-addr", type=str, required=True)
    parser.add_argument("--slice-name", type=str, required=True)
    parser.add_argument("--hf-model", type=str, default="/models/models/RecursiveSI/flint")
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--base-port", type=int, default=29991)
    parser.add_argument("--per-run-timeout", type=int, default=1800)
    parser.add_argument("--output-dir", type=str, default="/models/flint-results")
    parser.add_argument("--only-hf", action="store_true", help="Run only the HF PR#123 experiments (p9_03, p9_05)")
    args = parser.parse_args()

    apply_runtime_patches()
    node_rank = int(os.environ.get("JOB_COMPLETION_INDEX", os.environ.get("NODE_RANK", "0")))
    world_size = args.nnodes * args.nproc_per_node
    wait_for_master_dns(args.master_addr)

    ts = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    tag = "pass9b" if args.only_hf else "pass9"
    out_dir = Path(args.output_dir) / f"{args.slice_name}_{tag}_{ts}"
    bench_dir = Path("/models/benchmark-results")
    if node_rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
        bench_dir.mkdir(parents=True, exist_ok=True)

    exps = build_v7x_pass9_experiments(world_size, args.hf_model, args.steps)
    if args.only_hf:
        exps = [e for e in exps if e["workdir"] == "/torchtitan-hf"]
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
        log_chunks: list[str] = []
        rc = -1
        try:
            proc = subprocess.Popen(
                cmd, cwd=exp["workdir"], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
            )
            assert proc.stdout is not None
            deadline = t0 + args.per_run_timeout
            for line in proc.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                if node_rank == 0:
                    log_chunks.append(line)
                if time.time() > deadline:
                    proc.kill()
                    break
            rc = proc.wait(timeout=30)
        except Exception as exc:
            log_chunks.append(f"[flint-v7x-pass9] Exception: {exc}\n")
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
                f"[flint-v7x-pass9] SAVED {exp['name']}: status={status} "
                f"steady_tps={metrics.get('steady_tps', 0)} steady_mfu={metrics.get('steady_mfu_pct', 0)}%",
                flush=True,
            )
        time.sleep(8)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
