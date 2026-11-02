#!/usr/bin/env python3
"""Automated multi-host TorchTitan + TorchTPU GA (0.2.0) Flint training & MFU hillclimb runner.

Runs both:
  1. PR #123 HF Transformers Modeling Backend (`hf_moe_auto_max_attnac`) on staged `RecursiveSI/flint`
  2. PR #122 Native TorchTPU GraphTrainer (`flint_40b_v011_n40` + hillclimb configs `v012..v015` / `v5p` / `v6e`)

Persists raw logs, per-step metrics JSON, and Markdown summary table to `/models/flint-results/`
and `/models/benchmark-results/` after EVERY configuration so Spot preemptions never lose data.
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

FLINT_FLOPS_PER_TOKEN = 17.385e9  # 6 * active_params + attention FLOPs per PR #122/#123
PEAK_BF16_TFLOPS_PER_RANK = {
    "v7x": 1153.5,  # per TensorCore (2 TensorCores/chip, 8 ranks/host)
    "v5p": 459.0,   # per Megacore chip (4 ranks/host)
    "v6e": 918.0,   # per Trillium chip (4 ranks/host)
}

STEP_RE = re.compile(
    r"step:\s*(?P<step>\d+).*?loss:\s*(?P<loss>[0-9.]+).*?"
    r"tps:\s*(?P<tps>[0-9.,]+)",
    re.IGNORECASE,
)
MFU_RE = re.compile(r"mfu:\s*(?P<mfu>[0-9.]+)%", re.IGNORECASE)
MEM_RE = re.compile(r"memory:\s*(?P<mem>[0-9.]+)GiB", re.IGNORECASE)
BENCH_MFU_RE = re.compile(
    r"(?:Benchmark|Steady[- ]state|Average).*?(?:tps|tok/s).*?(?P<tps>[0-9.,]+)",
    re.IGNORECASE,
)


def wait_for_master_dns(master_addr: str, timeout_s: int = 300) -> str:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            ip = socket.gethostbyname(master_addr)
            return ip
        except socket.gaierror:
            time.sleep(3)
    return master_addr


def build_experiments(chip_type: str, world_size: int, hf_model_path: str, steps: int) -> list[dict]:
    mb32_tokens = 65536
    train_step_tokens_mb32 = world_size * mb32_tokens

    hf_Splash_args = [
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

    exps: list[dict] = []

    if chip_type == "v7x":
        exps.append({
            "name": "01_hf_backend_pr123_flint_40b",
            "lane": "PR#123 HF Backend",
            "workdir": "/torchtitan-hf",
            "env": {
                "TT_HF_GMM": "pallas_grouped_mm_tile_m=1024,pallas_grouped_mm_tgmm_tiling=512:512:512",
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
                f"--training.num_tokens_per_microbatch_per_dp_rank={mb32_tokens}",
                f"--training.num_tokens_per_train_step={train_step_tokens_mb32}",
                *hf_Splash_args,
                f"--extra-libtpu-init-flags={v7x_hier_sc_flags}",
            ],
        })
        exps.append({
            "name": "02_native_pr122_v011_n40_baseline",
            "lane": "PR#122 Native Flint",
            "workdir": "/torchtitan-native",
            "env": {},
            "args": [
                "--module=torchtitan.experiments.tpu.examples.flint",
                "--config=flint_40b_v011_n40",
                f"--training.steps={steps}",
                "--metrics.log_freq=1",
                "--debug.seed=42",
                "--tpu_config.mfu_benchmark_start_step=3",
                f"--training.num_tokens_per_microbatch_per_dp_rank={mb32_tokens}",
                f"--training.num_tokens_per_train_step={train_step_tokens_mb32}",
            ],
        })
        exps.append({
            "name": "03_native_v012_hier_sparsecore_rs",
            "lane": "PR#122 Hillclimb 1 (Hier ND RS)",
            "workdir": "/torchtitan-native",
            "env": {},
            "args": [
                "--module=torchtitan.experiments.tpu.examples.flint",
                "--config=flint_40b_v012_hier_rs",
                f"--training.steps={steps}",
                "--metrics.log_freq=1",
                "--debug.seed=42",
                "--tpu_config.mfu_benchmark_start_step=3",
                f"--training.num_tokens_per_microbatch_per_dp_rank={mb32_tokens}",
                f"--training.num_tokens_per_train_step={train_step_tokens_mb32}",
            ],
        })
        exps.append({
            "name": "04_native_v013_save_all_49_layers",
            "lane": "PR#122 Hillclimb 2 (Save All 49 Attn + Hier RS)",
            "workdir": "/torchtitan-native",
            "env": {},
            "args": [
                "--module=torchtitan.experiments.tpu.examples.flint",
                "--config=flint_40b_v013_n49",
                f"--training.steps={steps}",
                "--metrics.log_freq=1",
                "--debug.seed=42",
                "--tpu_config.mfu_benchmark_start_step=3",
                f"--training.num_tokens_per_microbatch_per_dp_rank={mb32_tokens}",
                f"--training.num_tokens_per_train_step={train_step_tokens_mb32}",
            ],
        })
        exps.append({
            "name": "05_native_v014_n49_tile_m_1024",
            "lane": "PR#122 Hillclimb 3 (n49 + tile_m=1024 + Hier RS)",
            "workdir": "/torchtitan-native",
            "env": {},
            "args": [
                "--module=torchtitan.experiments.tpu.examples.flint",
                "--config=flint_40b_v014_tile1024",
                f"--training.steps={steps}",
                "--metrics.log_freq=1",
                "--debug.seed=42",
                "--tpu_config.mfu_benchmark_start_step=3",
                f"--training.num_tokens_per_microbatch_per_dp_rank={mb32_tokens}",
                f"--training.num_tokens_per_train_step={train_step_tokens_mb32}",
            ],
        })
        if world_size >= 32:
            mb48_tokens = 48 * 2048  # 98,304 tokens/rank enabled by FSDP32 HBM headroom
            exps.append({
                "name": "06_native_v014_n49_tile1024_mb48",
                "lane": "PR#122 Hillclimb 4 (FSDP32 mb=48 seqs / 98K tok/rank)",
                "workdir": "/torchtitan-native",
                "env": {},
                "args": [
                    "--module=torchtitan.experiments.tpu.examples.flint",
                    "--config=flint_40b_v014_tile1024",
                    f"--training.steps={steps}",
                    "--metrics.log_freq=1",
                    "--debug.seed=42",
                    "--tpu_config.mfu_benchmark_start_step=3",
                    f"--training.num_tokens_per_microbatch_per_dp_rank={mb48_tokens}",
                    f"--training.num_tokens_per_train_step={world_size * mb48_tokens}",
                ],
            })
        exps.append({
            "name": "07_native_v015_compiled_optimizers",
            "lane": "PR#122 Hillclimb 5 (n49 + Compiled Optimizer)",
            "workdir": "/torchtitan-native",
            "env": {},
            "args": [
                "--module=torchtitan.experiments.tpu.examples.flint",
                "--config=flint_40b_v015_compiled_opt",
                f"--training.steps={steps}",
                "--metrics.log_freq=1",
                "--debug.seed=42",
                "--tpu_config.mfu_benchmark_start_step=3",
                f"--training.num_tokens_per_microbatch_per_dp_rank={mb32_tokens}",
                f"--training.num_tokens_per_train_step={train_step_tokens_mb32}",
            ],
        })
    elif chip_type == "v5p":
        exps.append({
            "name": "01_hf_backend_pr123_flint_v5p",
            "lane": "PR#123 HF Backend (v5p)",
            "workdir": "/torchtitan-hf",
            "env": {
                "TT_HF_GMM": "pallas_grouped_mm_tile_m=1024,pallas_grouped_mm_tgmm_tiling=512:512:512",
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
                f"--training.num_tokens_per_microbatch_per_dp_rank={mb32_tokens}",
                f"--training.num_tokens_per_train_step={train_step_tokens_mb32}",
                *hf_Splash_args,
            ],
        })
        exps.append({
            "name": "02_native_pr122_v5p_n40",
            "lane": "PR#122 Native Flint (v5p n40)",
            "workdir": "/torchtitan-native",
            "env": {},
            "args": [
                "--module=torchtitan.experiments.tpu.examples.flint",
                "--config=flint_40b_v5p_n40",
                f"--training.steps={steps}",
                "--metrics.log_freq=1",
                "--debug.seed=42",
                "--tpu_config.mfu_benchmark_start_step=3",
                f"--training.num_tokens_per_microbatch_per_dp_rank={mb32_tokens}",
                f"--training.num_tokens_per_train_step={train_step_tokens_mb32}",
            ],
        })
        exps.append({
            "name": "03_native_pr122_v5p_n49_tile1024",
            "lane": "PR#122 Hillclimb (v5p n49 + tile_m=1024)",
            "workdir": "/torchtitan-native",
            "env": {},
            "args": [
                "--module=torchtitan.experiments.tpu.examples.flint",
                "--config=flint_40b_v5p_n49",
                f"--training.steps={steps}",
                "--metrics.log_freq=1",
                "--debug.seed=42",
                "--tpu_config.mfu_benchmark_start_step=3",
                f"--training.num_tokens_per_microbatch_per_dp_rank={mb32_tokens}",
                f"--training.num_tokens_per_train_step={train_step_tokens_mb32}",
            ],
        })
    else:  # v6e (32 GiB HBM per chip)
        mb8_tokens = 8 * 2048  # 16,384 tokens/rank
        exps.append({
            "name": "01_native_pr122_v6e_fullremat_mb8",
            "lane": "PR#122 Native Flint (v6e 32GiB FullRemat mb=8)",
            "workdir": "/torchtitan-native",
            "env": {},
            "args": [
                "--module=torchtitan.experiments.tpu.examples.flint",
                "--config=flint_40b_v6e_fullremat",
                f"--training.steps={steps}",
                "--metrics.log_freq=1",
                "--debug.seed=42",
                "--tpu_config.mfu_benchmark_start_step=3",
                f"--training.num_tokens_per_microbatch_per_dp_rank={mb8_tokens}",
                f"--training.num_tokens_per_train_step={world_size * mb8_tokens}",
            ],
        })
        exps.append({
            "name": "02_hf_backend_pr123_flint_v6e_mb8",
            "lane": "PR#123 HF Backend (v6e mb=8)",
            "workdir": "/torchtitan-hf",
            "env": {
                "TT_HF_GMM": "pallas_grouped_mm_tile_m=512,pallas_grouped_mm_tgmm_tiling=512:512:512",
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
                f"--training.num_tokens_per_microbatch_per_dp_rank={mb8_tokens}",
                f"--training.num_tokens_per_train_step={world_size * mb8_tokens}",
                *hf_Splash_args,
            ],
        })

    return exps


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
        f"# Flint (`RecursiveSI/flint` 40B MoE) TorchTPU 0.2.0 GA Training Results — `{slice_name}`",
        "",
        f"- **Updated (UTC):** `{payload['updated_at_utc']}`",
        f"- **Chip / World Size:** `{chip_type}` (`{world_size}` ranks, `{PEAK_BF16_TFLOPS_PER_RANK.get(chip_type)}` BF16 TFLOP/s/rank, `17.385` GFLOP/token)",
        "",
        "| Experiment | Lane | Status | Steady tok/s/rank | Steady MFU (%) | Peak MFU (%) | Peak HBM (GiB) | Loss (start -> end) | Wall Time (s) |",
        "|---|---|---|---:|---:|---:|---:|---|---:|",
    ]
    for r in results:
        m = r.get("metrics", {})
        loss_str = (
            f"{m.get('first_loss', '-') } -> {m.get('final_loss', '-')}"
            if m.get("num_logged_steps")
            else "-"
        )
        md_lines.append(
            f"| `{r['name']}` | {r['lane']} | `{r['status']}` | "
            f"**{m.get('steady_tps', 0.0):,.1f}** | **{m.get('steady_mfu_pct', 0.0):.2f}%** | "
            f"{m.get('peak_mfu_pct', 0.0):.2f}% | {m.get('peak_mem_gib') or '-'} | "
            f"{loss_str} | {r['duration_s']:.1f} |"
        )
    md_text = "\n".join(md_lines) + "\n"
    (out_dir / "SUMMARY.md").write_text(md_text)

    if bench_dir.exists():
        (bench_dir / f"flint_{slice_name}_summary.json").write_text(json.dumps(payload, indent=2))
        (bench_dir / f"flint_{slice_name}_SUMMARY.md").write_text(md_text)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chip-type", choices=["v7x", "v5p", "v6e"], required=True)
    parser.add_argument("--nnodes", type=int, required=True)
    parser.add_argument("--nproc-per-node", type=int, required=True)
    parser.add_argument("--master-addr", type=str, required=True)
    parser.add_argument("--slice-name", type=str, required=True)
    parser.add_argument("--hf-model", type=str, default="/models/models/RecursiveSI/flint")
    parser.add_argument("--steps", type=int, default=15)
    parser.add_argument("--base-port", type=int, default=29501)
    parser.add_argument("--per-run-timeout", type=int, default=2700)
    parser.add_argument("--output-dir", type=str, default="/models/flint-results")
    args = parser.parse_args()

    node_rank = int(os.environ.get("JOB_COMPLETION_INDEX", os.environ.get("NODE_RANK", "0")))
    world_size = args.nnodes * args.nproc_per_node

    print(f"[flint-sweep] node_rank={node_rank}/{args.nnodes} world_size={world_size} slice={args.slice_name} master={args.master_addr}", flush=True)
    master_ip = wait_for_master_dns(args.master_addr)
    print(f"[flint-sweep] Resolved {args.master_addr} -> {master_ip}", flush=True)

    ts = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.output_dir) / f"{args.slice_name}_{ts}"
    bench_dir = Path("/models/benchmark-results")
    if node_rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
        bench_dir.mkdir(parents=True, exist_ok=True)

    exps = build_experiments(args.chip_type, world_size, args.hf_model, args.steps)
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
                    print(f"[flint-sweep] TIMEOUT ({args.per_run_timeout}s) on {exp['name']}, killing...", flush=True)
                    proc.kill()
                    break
            rc = proc.wait(timeout=30)
        except Exception as exc:
            msg = f"[flint-sweep] Exception running {exp['name']}: {exc}\n"
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
                f"[flint-sweep] SAVED {exp['name']}: status={status} "
                f"steady_tps={metrics.get('steady_tps', 0)} steady_mfu={metrics.get('steady_mfu_pct', 0)}%",
                flush=True,
            )

        # Give TPU device lock & sockets 12s to release cleanly between torchrun invocations
        time.sleep(12)

    print("[flint-sweep] All Flint experiments finished.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
