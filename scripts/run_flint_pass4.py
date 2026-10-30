#!/usr/bin/env python3
"""Pass 4 Peak-MFU Frontier + Real `RecursiveSI/flint` Checkpoint Fine-Tuning on v7x-16 (32 ranks).

Builds directly on Pass 3's 18.11% Steady / 18.18% Peak MFU breakthrough (`p3_05`: mb=48, n28):
  - At `mb=48` (98,304 tok/rank = 3.15M tok/step), `n49` missed fitting in 96 GiB HBM by only 1.93 GiB,
    while `p3_05` dropped all the way to `n28` (recomputing 21 SplashAttention layers) and still hit 18.18% MFU!
  - `p4_01` saves 43 SplashAttention layers (`n43`) + all-49 `moe.routing` (freeing 4.50 GiB vs `n49`, >1 GiB safety
    margin over the 3.43 GiB total delta), eliminating 15 backward SplashAttention recomputations at `mb=48`!
  - `p4_02` saves 45 SplashAttention layers (`n45`, without `moe.routing`, freeing 3.00 GiB > 1.93 GiB deficit).
  - `p4_03` scales microbatch to `mb=56` (114,688 tok/rank = 3.67M tok/step) with `n22` SplashAttention + all-49 `moe.routing`.
  - `p4_04` scales microbatch to `mb=64` (131,072 tok/rank = 4.19M tok/step) with `n14` SplashAttention + all-49 `moe.routing`
    (building on `p3_06` proving `mb=64 + fullremat` fits comfortably in 96 GiB HBM).
  - `p4_05` loads the REAL `RecursiveSI/flint` 39.8B HuggingFace checkpoint (`/models/models/RecursiveSI/flint`) +
    real Flint tokenizer via `FlintStateDictAdapter` (`checkpoint.initial_load_in_hf=True`) and runs a 30-step
    end-to-end C4 fine-tuning validation on the native >18.2%+ MFU stack!
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
PEAK_BF16_TFLOPS_PER_RANK = {
    "v7x": 1153.5,
    "v5p": 459.0,
    "v6e": 918.0,
}

STEP_RE = re.compile(
    r"step:\s*(?P<step>\d+).*?loss:\s*(?P<loss>[0-9.]+).*?"
    r"tps:\s*(?P<tps>[0-9.,]+)",
    re.IGNORECASE,
)
MFU_RE = re.compile(r"mfu:\s*(?P<mfu>[0-9.]+)%", re.IGNORECASE)
MEM_RE = re.compile(r"memory:\s*(?P<mem>[0-9.]+)GiB", re.IGNORECASE)

BASE_PATCH = '''

# ==============================================================================
# Pass 4 RSI Flint Native Peak-MFU + Real Checkpoint Configs
# ==============================================================================

def flint_40b_p4_n43_moe_route_comp_opt() -> TpuGraphTrainer.Config:
  """p4_01 (for mb=48): Save moe.routing in all 49 layers + attention in last 43 layers."""
  config = flint_40b_v014_tile1024()
  config.compile.memory_policy = "tpu_selective"
  config.compile.save_regions = ("moe.routing",)
  config.compile.save_last_n_layers = {"attention.inner_attention": 43}
  config.compile.optimizer_compile = "separate_graph"
  config.compile.compile_grad_norm = True
  return config


def flint_40b_p4_n45_comp_opt() -> TpuGraphTrainer.Config:
  """p4_02 (for mb=48): Save attention in last 45 layers + compiled opt/grad_norm."""
  config = flint_40b_v014_tile1024()
  config.compile.memory_policy = "tpu_selective"
  config.compile.save_regions = ()
  config.compile.save_last_n_layers = {"attention.inner_attention": 45}
  config.compile.optimizer_compile = "separate_graph"
  config.compile.compile_grad_norm = True
  return config


def flint_40b_p4_n22_moe_route_comp_opt() -> TpuGraphTrainer.Config:
  """p4_03 (for mb=56): Save moe.routing in all 49 layers + attention in last 22 layers."""
  config = flint_40b_v014_tile1024()
  config.compile.memory_policy = "tpu_selective"
  config.compile.save_regions = ("moe.routing",)
  config.compile.save_last_n_layers = {"attention.inner_attention": 22}
  config.compile.optimizer_compile = "separate_graph"
  config.compile.compile_grad_norm = True
  return config


def flint_40b_p4_n14_moe_route_comp_opt() -> TpuGraphTrainer.Config:
  """p4_04 (for mb=64): Save moe.routing in all 49 layers + attention in last 14 layers."""
  config = flint_40b_v014_tile1024()
  config.compile.memory_policy = "tpu_selective"
  config.compile.save_regions = ("moe.routing",)
  config.compile.save_last_n_layers = {"attention.inner_attention": 14}
  config.compile.optimizer_compile = "separate_graph"
  config.compile.compile_grad_norm = True
  return config


def flint_40b_p4_real_hf_ckpt() -> TpuGraphTrainer.Config:
  """p4_05 (for mb=48, 30 steps): Real RecursiveSI/flint HF checkpoint + tokenizer via FlintStateDictAdapter."""
  config = flint_40b_p4_n43_moe_route_comp_opt()
  config.compile.save_last_n_layers = {"attention.inner_attention": 38}
  return config


__all__.extend([
    "flint_40b_p4_n43_moe_route_comp_opt",
    "flint_40b_p4_n45_comp_opt",
    "flint_40b_p4_n22_moe_route_comp_opt",
    "flint_40b_p4_n14_moe_route_comp_opt",
    "flint_40b_p4_real_hf_ckpt",
])
'''


def apply_runtime_patches() -> None:
    nat_reg = Path("/torchtitan-native/torchtitan/experiments/tpu/examples/flint/config_registry.py")
    if nat_reg.exists():
        txt = nat_reg.read_text()
        if "flint_40b_p4_n43_moe_route_comp_opt" not in txt:
            nat_reg.write_text(txt + BASE_PATCH)
            print("[flint-pass4] Patched Native config_registry.py with Pass 4 configs", flush=True)


def wait_for_master_dns(master_addr: str, timeout_s: int = 300) -> str:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            return socket.gethostbyname(master_addr)
        except socket.gaierror:
            time.sleep(3)
    return master_addr


def build_pass4_experiments(chip_type: str, world_size: int, hf_model_path: str, steps: int) -> list[dict]:
    mb48 = 48 * 2048   # 98,304 tok/rank -> 3,145,728 tok/step
    mb56 = 56 * 2048   # 114,688 tok/rank -> 3,670,016 tok/step
    mb64 = 64 * 2048   # 131,072 tok/rank -> 4,194,304 tok/step

    return [
        {
            "name": "p4_01_native_n43_tile1024_moe_routing_comp_opt_mb48",
            "lane": "Pass4 Native (n43 Attn + All49 MoERouting + CompiledOpt, mb=48 / 98K tok/rank)",
            "workdir": "/torchtitan-native",
            "env": {},
            "args": [
                "--module=torchtitan.experiments.tpu.examples.flint",
                "--config=flint_40b_p4_n43_moe_route_comp_opt",
                f"--training.steps={steps}",
                "--metrics.log_freq=1",
                "--debug.seed=42",
                "--tpu_config.mfu_benchmark_start_step=3",
                f"--training.num_tokens_per_microbatch_per_dp_rank={mb48}",
                f"--training.num_tokens_per_train_step={world_size * mb48}",
            ],
        },
        {
            "name": "p4_02_native_n45_tile1024_comp_opt_mb48",
            "lane": "Pass4 Native (n45 Attn + CompiledOpt, mb=48 / 98K tok/rank)",
            "workdir": "/torchtitan-native",
            "env": {},
            "args": [
                "--module=torchtitan.experiments.tpu.examples.flint",
                "--config=flint_40b_p4_n45_comp_opt",
                f"--training.steps={steps}",
                "--metrics.log_freq=1",
                "--debug.seed=42",
                "--tpu_config.mfu_benchmark_start_step=3",
                f"--training.num_tokens_per_microbatch_per_dp_rank={mb48}",
                f"--training.num_tokens_per_train_step={world_size * mb48}",
            ],
        },
        {
            "name": "p4_03_native_n22_tile1024_moe_routing_comp_opt_mb56",
            "lane": "Pass4 Native (n22 Attn + All49 MoERouting + CompiledOpt, mb=56 / 115K tok/rank)",
            "workdir": "/torchtitan-native",
            "env": {},
            "args": [
                "--module=torchtitan.experiments.tpu.examples.flint",
                "--config=flint_40b_p4_n22_moe_route_comp_opt",
                f"--training.steps={steps}",
                "--metrics.log_freq=1",
                "--debug.seed=42",
                "--tpu_config.mfu_benchmark_start_step=3",
                f"--training.num_tokens_per_microbatch_per_dp_rank={mb56}",
                f"--training.num_tokens_per_train_step={world_size * mb56}",
            ],
        },
        {
            "name": "p4_04_native_n14_tile1024_moe_routing_comp_opt_mb64",
            "lane": "Pass4 Native (n14 Attn + All49 MoERouting + CompiledOpt, mb=64 / 131K tok/rank)",
            "workdir": "/torchtitan-native",
            "env": {},
            "args": [
                "--module=torchtitan.experiments.tpu.examples.flint",
                "--config=flint_40b_p4_n14_moe_route_comp_opt",
                f"--training.steps={steps}",
                "--metrics.log_freq=1",
                "--debug.seed=42",
                "--tpu_config.mfu_benchmark_start_step=3",
                f"--training.num_tokens_per_microbatch_per_dp_rank={mb64}",
                f"--training.num_tokens_per_train_step={world_size * mb64}",
            ],
        },
        {
            "name": "p4_05_native_real_flint_hf_checkpoint_30steps_mb48",
            "lane": "Pass4 Native Real RecursiveSI/flint HF Weights + Tokenizer (30 steps, n38+MoERouting, mb=48)",
            "workdir": "/torchtitan-native",
            "env": {},
            "args": [
                "--module=torchtitan.experiments.tpu.examples.flint",
                "--config=flint_40b_p4_real_hf_ckpt",
                f"--hf_assets_path={hf_model_path}",
                "--checkpoint.enable=True",
                "--checkpoint.initial_load_in_hf=True",
                f"--checkpoint.initial_load_path={hf_model_path}",
                "--checkpoint.initial_load_model_only=True",
                "--checkpoint.interval=1000",
                "--training.steps=30",
                "--metrics.log_freq=1",
                "--debug.seed=42",
                "--tpu_config.mfu_benchmark_start_step=3",
                f"--training.num_tokens_per_microbatch_per_dp_rank={mb48}",
                f"--training.num_tokens_per_train_step={world_size * mb48}",
            ],
        },
        {
            "name": "p4_06_hf_pr123_paired_gmm_b1024_mb32",
            "lane": "Pass4 HF Backend (AttnOnlyAC + Paired W1/W3 Pallas GMM + b=1024, mb=32)",
            "workdir": "/torchtitan-hf",
            "env": {
                "TT_HF_GMM": (
                    "use_pallas_paired_grouped_mm=true,"
                    "pallas_grouped_mm_tile_m=1024,"
                    "pallas_grouped_mm_tgmm_tiling=512:2048:0,"
                    "pallas_grouped_mm_w2_tgmm_tiling=512:512:0,"
                    "pallas_grouped_mm_w2_transposed_tgmm_tiling=512:0:2048"
                ),
            },
            "args": [
                "--module=torchtitan.experiments.tpu.transformers_modeling_backend",
                "--config=hf_moe_auto_max_attnac",
                f"--hf-model={hf_model_path}",
                "--training.max_context_length=2048",
                f"--training.steps={steps}",
                "--metrics.log_freq=1",
                "--debug.seed=42",
                "--loss.b_block_size=1024",
                f"--training.num_tokens_per_microbatch_per_dp_rank={mb32 := 32 * 2048}",
                f"--training.num_tokens_per_train_step={world_size * mb32}",
                "--splash_attention_kernel.sa_block_q=512",
                "--splash_attention_kernel.sa_block_kv=512",
                "--splash_attention_kernel.sa_block_kv_compute=512",
                "--splash_attention_kernel.sa_block_q_dkv=512",
                "--splash_attention_kernel.sa_block_kv_dkv=512",
                "--splash_attention_kernel.sa_block_kv_dkv_compute=512",
                "--extra-libtpu-init-flags="
                "xla_tpu_enable_sparse_core_collective_aggregator=true "
                "xla_tpu_enable_latency_hiding_layer_scheduler=true "
                "xla_tpu_use_single_sparse_core_for_reduce_scatter_offload=false "
                "xla_tpu_enable_sparse_core_collective_offload_nd_reduce_scatter=true "
                "xla_tpu_enable_sparse_core_hierarchical_reduce_scatter=true "
                "xla_tpu_dvfs_p_state=7",
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
    avg_mfu = 100.0 * FLINT_FLOPS_PER_TOKEN * avg_tps / peak_tf
    max_mfu = 100.0 * FLINT_FLOPS_PER_TOKEN * max_tps / peak_tf
    peak_mem = max((s["mem_gib"] for s in steps if s["mem_gib"] is not None), default=None)
    return {
        "num_logged_steps": len(steps),
        "first_loss": steps[0]["loss"],
        "final_loss": steps[-1]["loss"],
        "steady_tps": round(avg_tps, 1),
        "peak_tps": round(max_tps, 1),
        "steady_mfu_pct": round(avg_mfu, 2),
        "peak_mfu_pct": round(max_mfu, 2),
        "peak_mem_gib": peak_mem,
        "steps": steps,
    }


def write_summary(out_dir: Path, bench_dir: Path, slice_name: str, chip_type: str, world_size: int, results: list[dict]) -> None:
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
        f"# RSI Flint (`RecursiveSI/flint` 40B MoE) Pass 4 Peak-MFU + Real Checkpoint — `{slice_name}`",
        "",
        f"- **Updated (UTC):** `{payload['updated_at_utc']}`",
        f"- **Chip / World Size:** `{chip_type}` (`{world_size}` ranks, `{PEAK_BF16_TFLOPS_PER_RANK.get(chip_type)}` BF16 TFLOP/s/rank, `17.385` GFLOP/token)",
        "",
        "| Experiment | Lane | Status | Steady tok/s/rank | Cluster tok/s | Steady MFU (%) | Peak MFU (%) | Loss (start -> end) | Wall Time (s) |",
        "|---|---|---|---:|---:|---:|---:|---|---:|",
    ]
    for r in results:
        m = r.get("metrics", {})
        loss_str = (
            f"{m.get('first_loss', '-')} -> {m.get('final_loss', '-')}"
            if m.get("num_logged_steps")
            else "-"
        )
        tps = m.get("steady_tps", 0.0)
        md_lines.append(
            f"| `{r['name']}` | {r['lane']} | `{r['status']}` | "
            f"**{tps:,.1f}** | {tps * world_size:,.0f} | **{m.get('steady_mfu_pct', 0.0):.2f}%** | "
            f"{m.get('peak_mfu_pct', 0.0):.2f}% | {loss_str} | {r['duration_s']:.1f} |"
        )
    md_text = "\n".join(md_lines) + "\n"
    (out_dir / "SUMMARY.md").write_text(md_text)

    if bench_dir.exists():
        (bench_dir / f"flint_{slice_name}_pass4_summary.json").write_text(json.dumps(payload, indent=2))
        (bench_dir / f"flint_{slice_name}_pass4_SUMMARY.md").write_text(md_text)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chip-type", choices=["v7x", "v5p", "v6e"], required=True)
    parser.add_argument("--nnodes", type=int, required=True)
    parser.add_argument("--nproc-per-node", type=int, required=True)
    parser.add_argument("--master-addr", type=str, required=True)
    parser.add_argument("--slice-name", type=str, required=True)
    parser.add_argument("--hf-model", type=str, default="/models/models/RecursiveSI/flint")
    parser.add_argument("--steps", type=int, default=15)
    parser.add_argument("--base-port", type=int, default=29801)
    parser.add_argument("--per-run-timeout", type=int, default=2700)
    parser.add_argument("--output-dir", type=str, default="/models/flint-results")
    args = parser.parse_args()

    apply_runtime_patches()

    node_rank = int(os.environ.get("JOB_COMPLETION_INDEX", os.environ.get("NODE_RANK", "0")))
    world_size = args.nnodes * args.nproc_per_node

    print(f"[flint-pass4] node_rank={node_rank}/{args.nnodes} world_size={world_size} slice={args.slice_name}", flush=True)
    master_ip = wait_for_master_dns(args.master_addr)
    print(f"[flint-pass4] Resolved {args.master_addr} -> {master_ip}", flush=True)

    ts = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.output_dir) / f"{args.slice_name}_pass4_{ts}"
    bench_dir = Path("/models/benchmark-results")
    if node_rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
        bench_dir.mkdir(parents=True, exist_ok=True)

    exps = build_pass4_experiments(args.chip_type, world_size, args.hf_model, args.steps)
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

        print(f"\n=== [{idx + 1}/{len(exps)}] {exp['name']} ({exp['lane']}) port={port} ===", flush=True)
        print("CMD:", " ".join(cmd), flush=True)

        t0 = time.time()
        log_chunks: list[str] = []
        rc = -1
        try:
            proc = subprocess.Popen(
                cmd,
                cwd=exp["workdir"],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert proc.stdout is not None
            deadline = t0 + args.per_run_timeout
            for line in proc.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                if node_rank == 0:
                    log_chunks.append(line)
                if time.time() > deadline:
                    print(f"[flint-pass4] TIMEOUT ({args.per_run_timeout}s) on {exp['name']}, killing...", flush=True)
                    proc.kill()
                    break
            rc = proc.wait(timeout=30)
        except Exception as exc:
            msg = f"[flint-pass4] Exception running {exp['name']}: {exc}\n"
            print(msg, flush=True)
            log_chunks.append(msg)

        dur = time.time() - t0
        if node_rank == 0:
            raw_log = "".join(log_chunks)
            (out_dir / f"{exp['name']}.log").write_text(raw_log)
            metrics = parse_metrics(raw_log, args.chip_type)
            status = "PASS" if (rc == 0 and metrics.get("num_logged_steps", 0) > 0) else f"FAIL(rc={rc})"
            rec = {
                "name": exp["name"],
                "lane": exp["lane"],
                "status": status,
                "return_code": rc,
                "duration_s": round(dur, 1),
                "command": cmd,
                "env_overrides": exp["env"],
                "metrics": metrics,
            }
            results.append(rec)
            write_summary(out_dir, bench_dir, args.slice_name, args.chip_type, world_size, results)
            print(
                f"[flint-pass4] SAVED {exp['name']}: status={status} "
                f"steady_tps={metrics.get('steady_tps', 0)} steady_mfu={metrics.get('steady_mfu_pct', 0)}%",
                flush=True,
            )

        time.sleep(12)

    print("[flint-pass4] All Pass 4 Flint experiments finished.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
