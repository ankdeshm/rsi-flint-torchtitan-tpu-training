#!/usr/bin/env python3
"""Pass 7 Next-Wave MFU Frontier on TPU v7x (pushing past 18.21% Native & 17.72% HF Backend).

New MFU Levers over Pass 4 (`18.15%` Steady / `18.21%` Peak) & Pass 5 (`17.68%` Steady / `17.72%` Peak):
  1. `p7_01_native_grad_accum_2x_mb48_n43`: 2x gradient accumulation (`2 * 98,304 = 196,608` tok/rank,
     `6.29M` tok/step on 32 ranks) — amortizes FSDP ReduceScatter, CompiledGradNorm, and CompiledAdamW by 2x!
  2. `p7_02_native_n45_all49_moe_routing_mb48`: Pushes saved SplashAttention layers from `n43` -> `n45` WITH
     `save_regions=("moe.routing",)` (`n49` missed 96 GiB by 1.93 GiB; `4` recomputed layers frees `~3.0 GiB`).
  3. `p7_03_hf_pr123_paired_gmm_b1024_mb48`: Scales PR #123 HF Backend + Paired W1/W3 Pallas GMM (`tile_m=1024`,
     `tgmm_tiling=(512,2048,0)`, `b=1024`) from `mb=32` (`17.68%` MFU) -> `mb=48` (`98,304` tok/rank, 1.5x expert density)!
  4. `p7_04_hf_pr123_paired_gmm_grad_accum_2x_mb48`: Combines PR #123 Paired W1/W3 GMM at `mb=48` with 2x
     gradient accumulation (`196,608` tok/rank) to amortize optimizer + FSDP collectives.
"""

import argparse
import datetime as dt
import json
import os
import re
import socket
import subprocess
import sys
import time
from pathlib import Path

FLINT_FLOPS_PER_TOKEN = 17.385e9
PEAK_BF16_TFLOPS_PER_RANK = {"v7x": 1153.5, "v5p": 459.0, "v6e": 918.0}

STEP_RE = re.compile(
    r"step:\s*(?P<step>\d+).*?loss:\s*(?P<loss>[0-9.]+).*?tps:\s*(?P<tps>[0-9.,]+)",
    re.IGNORECASE,
)
MFU_RE = re.compile(r"mfu:\s*(?P<mfu>[0-9.]+)%", re.IGNORECASE)
MEM_RE = re.compile(r"memory:\s*(?P<mem>[0-9.]+)GiB", re.IGNORECASE)

BASE_PATCH = '''

# ==============================================================================
# Pass 7 RSI Flint Next-Wave v7x MFU Configs
# ==============================================================================

def flint_40b_p7_n43_moe_route_comp_opt() -> TpuGraphTrainer.Config:
  config = flint_40b_v014_tile1024()
  config.compile.memory_policy = "tpu_selective"
  config.compile.save_regions = ("moe.routing",)
  config.compile.save_last_n_layers = {"attention.inner_attention": 43}
  config.compile.optimizer_compile = "separate_graph"
  config.compile.compile_grad_norm = True
  return config


def flint_40b_p7_n45_moe_route_comp_opt() -> TpuGraphTrainer.Config:
  config = flint_40b_v014_tile1024()
  config.compile.memory_policy = "tpu_selective"
  config.compile.save_regions = ("moe.routing",)
  config.compile.save_last_n_layers = {"attention.inner_attention": 45}
  config.compile.optimizer_compile = "separate_graph"
  config.compile.compile_grad_norm = True
  return config


__all__.extend([
    "flint_40b_p7_n43_moe_route_comp_opt",
    "flint_40b_p7_n45_moe_route_comp_opt",
])
'''


def apply_runtime_patches() -> None:
    nat_reg = Path("/torchtitan-native/torchtitan/experiments/tpu/examples/flint/config_registry.py")
    if nat_reg.exists():
        txt = nat_reg.read_text()
        if "flint_40b_p7_n45_moe_route_comp_opt" not in txt:
            nat_reg.write_text(txt + BASE_PATCH)
            print("[flint-pass7] Patched Native config_registry.py with Pass 7 configs", flush=True)


def wait_for_master_dns(master_addr: str, timeout_s: int = 300) -> str:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            return socket.gethostbyname(master_addr)
        except socket.gaierror:
            time.sleep(3)
    return master_addr


def build_pass7_experiments(world_size: int, hf_model_path: str, steps: int) -> list[dict]:
    mb48 = 48 * 2048  # 98,304 tok/rank
    hf_splash_args = [
        "--splash_attention_kernel.sa_block_q=512",
        "--splash_attention_kernel.sa_block_kv=512",
        "--splash_attention_kernel.sa_block_kv_compute=512",
        "--splash_attention_kernel.sa_block_q_dkv=512",
        "--splash_attention_kernel.sa_block_kv_dkv=512",
        "--splash_attention_kernel.sa_block_kv_dkv_compute=512",
    ]
    v7x_hier_sc_flags = (
        "xla_tpu_enable_sparse_core_collective_aggregator=true "
        "xla_tpu_enable_latency_hiding_layer_scheduler=true "
        "xla_tpu_use_single_sparse_core_for_reduce_scatter_offload=false "
        "xla_tpu_enable_sparse_core_collective_offload_nd_reduce_scatter=true "
        "xla_tpu_enable_sparse_core_hierarchical_reduce_scatter=true "
        "xla_tpu_dvfs_p_state=7"
    )
    paired_gmm_env = {
        "TT_HF_GMM": (
            "use_pallas_paired_grouped_mm=true,"
            "pallas_grouped_mm_tile_m=1024,"
            "pallas_grouped_mm_tgmm_tiling=512:2048:0,"
            "pallas_grouped_mm_w2_tgmm_tiling=512:512:0,"
            "pallas_grouped_mm_w2_transposed_tgmm_tiling=512:0:2048"
        )
    }

    return [
        {
            "name": "p7_01_native_grad_accum_2x_mb48_n43",
            "lane": "Pass7 Native (2x GradAccum @ mb=48 = 196K tok/rank, n43 + All49 MoERouting + CompiledOpt)",
            "workdir": "/torchtitan-native",
            "env": {},
            "args": [
                "--module=torchtitan.experiments.tpu.examples.flint",
                "--config=flint_40b_p7_n43_moe_route_comp_opt",
                f"--training.steps={steps}",
                "--metrics.log_freq=1",
                "--debug.seed=42",
                "--tpu_config.mfu_benchmark_start_step=3",
                f"--training.num_tokens_per_microbatch_per_dp_rank={mb48}",
                f"--training.num_tokens_per_train_step={world_size * mb48 * 2}",
            ],
        },
        {
            "name": "p7_02_native_n45_all49_moe_routing_mb48",
            "lane": "Pass7 Native (n45 Attn + All49 MoERouting + CompiledOpt, mb=48 / 98K tok/rank)",
            "workdir": "/torchtitan-native",
            "env": {},
            "args": [
                "--module=torchtitan.experiments.tpu.examples.flint",
                "--config=flint_40b_p7_n45_moe_route_comp_opt",
                f"--training.steps={steps}",
                "--metrics.log_freq=1",
                "--debug.seed=42",
                "--tpu_config.mfu_benchmark_start_step=3",
                f"--training.num_tokens_per_microbatch_per_dp_rank={mb48}",
                f"--training.num_tokens_per_train_step={world_size * mb48}",
            ],
        },
        {
            "name": "p7_03_hf_pr123_paired_gmm_b1024_mb48",
            "lane": "Pass7 HF Backend PR#123 (Paired W1/W3 GMM tile_m=1024 + b=1024, mb=48 / 98K tok/rank)",
            "workdir": "/torchtitan-hf",
            "env": paired_gmm_env,
            "args": [
                "--module=torchtitan.experiments.tpu.transformers_modeling_backend",
                "--config=hf_moe_auto_max_attnac",
                f"--hf-model={hf_model_path}",
                "--training.max_context_length=2048",
                f"--training.steps={steps}",
                "--metrics.log_freq=1",
                "--debug.seed=42",
                "--loss.b_block_size=1024",
                f"--training.num_tokens_per_microbatch_per_dp_rank={mb48}",
                f"--training.num_tokens_per_train_step={world_size * mb48}",
                *hf_splash_args,
                f"--extra-libtpu-init-flags={v7x_hier_sc_flags}",
            ],
        },
        {
            "name": "p7_04_hf_pr123_paired_gmm_grad_accum_2x_mb48",
            "lane": "Pass7 HF Backend PR#123 (2x GradAccum @ mb=48 + Paired W1/W3 GMM tile_m=1024)",
            "workdir": "/torchtitan-hf",
            "env": paired_gmm_env,
            "args": [
                "--module=torchtitan.experiments.tpu.transformers_modeling_backend",
                "--config=hf_moe_auto_max_attnac",
                f"--hf-model={hf_model_path}",
                "--training.max_context_length=2048",
                f"--training.steps={steps}",
                "--metrics.log_freq=1",
                "--debug.seed=42",
                "--loss.b_block_size=1024",
                f"--training.num_tokens_per_microbatch_per_dp_rank={mb48}",
                f"--training.num_tokens_per_train_step={world_size * mb48 * 2}",
                *hf_splash_args,
                f"--extra-libtpu-init-flags={v7x_hier_sc_flags}",
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
        loss = float(m.group("loss"))
        tps = float(m.group("tps").replace(",", ""))
        mfu_m = MFU_RE.search(line)
        mem_m = MEM_RE.search(line)
        peak_tf = PEAK_BF16_TFLOPS_PER_RANK.get(chip_type, 1153.5) * 1e12
        calc_mfu = 100.0 * FLINT_FLOPS_PER_TOKEN * tps / peak_tf
        logged_mfu = float(mfu_m.group("mfu")) if mfu_m else calc_mfu
        mem_gib = float(mem_m.group("mem")) if mem_m else None
        steps.append({
            "step": step,
            "loss": loss,
            "tps_per_rank": tps,
            "logged_mfu_pct": logged_mfu,
            "calc_mfu_pct": round(calc_mfu, 3),
            "mem_gib": mem_gib,
        })
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chip-type", choices=["v7x", "v5p", "v6e"], default="v7x")
    parser.add_argument("--nnodes", type=int, required=True)
    parser.add_argument("--nproc-per-node", type=int, default=8)
    parser.add_argument("--master-addr", type=str, required=True)
    parser.add_argument("--slice-name", type=str, required=True)
    parser.add_argument("--hf-model-path", type=str, default="/models/models/RecursiveSI/flint")
    parser.add_argument("--steps", type=int, default=15)
    parser.add_argument("--base-port", type=int, default=29971)
    parser.add_argument("--per-run-timeout", type=int, default=2700)
    parser.add_argument("--output-dir", type=str, default="/models/flint-results")
    args = parser.parse_args()

    apply_runtime_patches()
    node_rank = int(os.environ.get("JOB_COMPLETION_INDEX", os.environ.get("NODE_RANK", "0")))
    world_size = args.nnodes * args.nproc_per_node
    wait_for_master_dns(args.master_addr)

    ts = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.output_dir) / f"{args.slice_name}_pass7_{ts}"
    if node_rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)

    exps = build_pass7_experiments(world_size, args.hf_model_path, args.steps)
    results = []
    for idx, exp in enumerate(exps):
        port = args.base_port + idx
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
            log_chunks.append(f"[flint-pass7] Exception: {exc}\n")
        dur = time.time() - t0
        if node_rank == 0:
            raw_log = "".join(log_chunks)
            (out_dir / f"{exp['name']}.log").write_text(raw_log)
            metrics = parse_metrics(raw_log, args.chip_type)
            status = "PASS" if (rc == 0 and metrics.get("num_logged_steps", 0) > 0) else f"FAIL(rc={rc})"
            results.append({
                "name": exp["name"],
                "lane": exp["lane"],
                "status": status,
                "return_code": rc,
                "duration_s": round(dur, 1),
                "command": cmd,
                "env_overrides": exp["env"],
                "metrics": metrics,
            })
            (out_dir / "summary.json").write_text(json.dumps({"results": results}, indent=2))
        time.sleep(12)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
