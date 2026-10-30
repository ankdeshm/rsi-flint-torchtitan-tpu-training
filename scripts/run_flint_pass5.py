#!/usr/bin/env python3
"""Pass 5 Final Peak-MFU Hillclimb (Native + PR #123 HF Backend Paired-GMM) on v7x-16 (32 ranks).

Experiments:
  - p5_01: Native n49 (ALL 49 SplashAttention + ALL 49 moe.routing) + CompiledOpt/GradNorm at mb=48.
  - p5_02: Native n43 + ALL 49 moe.routing + CompiledOpt/GradNorm + backward tgmm_tiling M=1024 at mb=48.
  - p5_03: Native n49 (ALL 49 SplashAttention + ALL 49 moe.routing) + backward tgmm_tiling M=1024 at mb=44.
  - p5_04: PR #123 HF Transformers Modeling Backend (real RecursiveSI/flint weights) + Paired W1/W3 Pallas GMM
           (tile_m=1024) + --loss.b_block_size=1024 (VMEM-safe) at mb=32.
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
# Pass 5 RSI Flint Final Native Peak-MFU Configs
# ==============================================================================

def _set_bwd_tgmm_1024(config) -> None:
  for _, experts, _, _ in config.model_spec.model.traverse(
      TpuGroupedExperts.Config
  ):
    experts.pallas_grouped_mm_tile_m = 1024
    experts.pallas_grouped_mm_tgmm_tiling = (1024, 2048, 0)
    experts.pallas_grouped_mm_w2_tgmm_tiling = (1024, 512, 0)
    experts.pallas_grouped_mm_w2_transposed_tgmm_tiling = (1024, 0, 2048)


def flint_40b_p5_n49_moe_route_comp_opt() -> TpuGraphTrainer.Config:
  """p5_01: Save ALL 49 attention.inner_attention + ALL 49 moe.routing + compiled opt/grad_norm."""
  config = flint_40b_v014_tile1024()
  config.compile.memory_policy = "tpu_selective"
  config.compile.save_regions = ("moe.routing",)
  config.compile.save_last_n_layers = {"attention.inner_attention": 49}
  config.compile.optimizer_compile = "separate_graph"
  config.compile.compile_grad_norm = True
  return config


def flint_40b_p5_n43_moe_route_comp_opt_tgmm1024() -> TpuGraphTrainer.Config:
  """p5_02: n43 Attn + ALL 49 moe.routing + compiled opt/grad_norm + backward tgmm M=1024."""
  config = flint_40b_v014_tile1024()
  _set_bwd_tgmm_1024(config)
  config.compile.memory_policy = "tpu_selective"
  config.compile.save_regions = ("moe.routing",)
  config.compile.save_last_n_layers = {"attention.inner_attention": 43}
  config.compile.optimizer_compile = "separate_graph"
  config.compile.compile_grad_norm = True
  return config


def flint_40b_p5_n49_moe_route_comp_opt_tgmm1024() -> TpuGraphTrainer.Config:
  """p5_03: ALL 49 Attn + ALL 49 moe.routing + compiled opt/grad_norm + backward tgmm M=1024 (for mb=44)."""
  config = flint_40b_p5_n49_moe_route_comp_opt()
  _set_bwd_tgmm_1024(config)
  return config


__all__.extend([
    "flint_40b_p5_n49_moe_route_comp_opt",
    "flint_40b_p5_n43_moe_route_comp_opt_tgmm1024",
    "flint_40b_p5_n49_moe_route_comp_opt_tgmm1024",
])
'''


def apply_runtime_patches() -> None:
    nat_reg = Path("/torchtitan-native/torchtitan/experiments/tpu/examples/flint/config_registry.py")
    if nat_reg.exists():
        txt = nat_reg.read_text()
        if "flint_40b_p5_n49_moe_route_comp_opt" not in txt:
            nat_reg.write_text(txt + BASE_PATCH)
            print("[flint-pass5] Patched Native config_registry.py with Pass 5 configs", flush=True)


def wait_for_master_dns(master_addr: str, timeout_s: int = 300) -> str:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            return socket.gethostbyname(master_addr)
        except socket.gaierror:
            time.sleep(3)
    return master_addr


def build_pass5_experiments(chip_type: str, world_size: int, hf_model_path: str, steps: int) -> list[dict]:
    mb32 = 32 * 2048   # 65,536 tok/rank  -> 2,097,152 tok/step
    mb44 = 44 * 2048   # 90,112 tok/rank  -> 2,883,584 tok/step
    mb48 = 48 * 2048   # 98,304 tok/rank  -> 3,145,728 tok/step

    return [
        {
            "name": "p5_01_native_n49_moe_routing_comp_opt_mb48",
            "lane": "Pass5 Native (ALL49 Attn + ALL49 MoERouting + CompiledOpt, mb=48 / 98K tok/rank)",
            "workdir": "/torchtitan-native",
            "env": {},
            "args": [
                "--module=torchtitan.experiments.tpu.examples.flint",
                "--config=flint_40b_p5_n49_moe_route_comp_opt",
                f"--training.steps={steps}",
                "--metrics.log_freq=1",
                "--debug.seed=42",
                "--tpu_config.mfu_benchmark_start_step=3",
                f"--training.num_tokens_per_microbatch_per_dp_rank={mb48}",
                f"--training.num_tokens_per_train_step={world_size * mb48}",
            ],
        },
        {
            "name": "p5_02_native_n43_moe_routing_comp_opt_tgmm1024_mb48",
            "lane": "Pass5 Native (n43 Attn + ALL49 MoERouting + CompiledOpt + BwdTGMM1024, mb=48)",
            "workdir": "/torchtitan-native",
            "env": {},
            "args": [
                "--module=torchtitan.experiments.tpu.examples.flint",
                "--config=flint_40b_p5_n43_moe_route_comp_opt_tgmm1024",
                f"--training.steps={steps}",
                "--metrics.log_freq=1",
                "--debug.seed=42",
                "--tpu_config.mfu_benchmark_start_step=3",
                f"--training.num_tokens_per_microbatch_per_dp_rank={mb48}",
                f"--training.num_tokens_per_train_step={world_size * mb48}",
            ],
        },
        {
            "name": "p5_03_native_n49_moe_routing_comp_opt_tgmm1024_mb44",
            "lane": "Pass5 Native (ALL49 Attn + ALL49 MoERouting + CompiledOpt + BwdTGMM1024, mb=44)",
            "workdir": "/torchtitan-native",
            "env": {},
            "args": [
                "--module=torchtitan.experiments.tpu.examples.flint",
                "--config=flint_40b_p5_n49_moe_route_comp_opt_tgmm1024",
                f"--training.steps={steps}",
                "--metrics.log_freq=1",
                "--debug.seed=42",
                "--tpu_config.mfu_benchmark_start_step=3",
                f"--training.num_tokens_per_microbatch_per_dp_rank={mb44}",
                f"--training.num_tokens_per_train_step={world_size * mb44}",
            ],
        },
        {
            "name": "p5_04_hf_pr123_paired_gmm_b1024_mb32",
            "lane": "Pass5 HF Backend PR #123 (Real Flint + AttnOnlyAC + Paired W1/W3 Pallas GMM + b=1024, mb=32)",
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
                f"--training.num_tokens_per_microbatch_per_dp_rank={mb32}",
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
        f"# RSI Flint (`RecursiveSI/flint` 40B MoE) Pass 5 Final Hillclimb — `{slice_name}`",
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
        (bench_dir / f"flint_{slice_name}_pass5_summary.json").write_text(json.dumps(payload, indent=2))
        (bench_dir / f"flint_{slice_name}_pass5_SUMMARY.md").write_text(md_text)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chip-type", choices=["v7x", "v5p", "v6e"], required=True)
    parser.add_argument("--nnodes", type=int, required=True)
    parser.add_argument("--nproc-per-node", type=int, required=True)
    parser.add_argument("--master-addr", type=str, required=True)
    parser.add_argument("--slice-name", type=str, required=True)
    parser.add_argument("--hf-model", type=str, default="/models/models/RecursiveSI/flint")
    parser.add_argument("--steps", type=int, default=15)
    parser.add_argument("--base-port", type=int, default=29901)
    parser.add_argument("--per-run-timeout", type=int, default=2700)
    parser.add_argument("--output-dir", type=str, default="/models/flint-results")
    args = parser.parse_args()

    apply_runtime_patches()

    node_rank = int(os.environ.get("JOB_COMPLETION_INDEX", os.environ.get("NODE_RANK", "0")))
    world_size = args.nnodes * args.nproc_per_node

    print(f"[flint-pass5] node_rank={node_rank}/{args.nnodes} world_size={world_size} slice={args.slice_name}", flush=True)
    master_ip = wait_for_master_dns(args.master_addr)
    print(f"[flint-pass5] Resolved {args.master_addr} -> {master_ip}", flush=True)

    ts = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.output_dir) / f"{args.slice_name}_pass5_{ts}"
    bench_dir = Path("/models/benchmark-results")
    if node_rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
        bench_dir.mkdir(parents=True, exist_ok=True)

    exps = build_pass5_experiments(args.chip_type, world_size, args.hf_model, args.steps)
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
                    print(f"[flint-pass5] TIMEOUT ({args.per_run_timeout}s) on {exp['name']}, killing...", flush=True)
                    proc.kill()
                    break
            rc = proc.wait(timeout=30)
        except Exception as exc:
            msg = f"[flint-pass5] Exception running {exp['name']}: {exc}\n"
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
                f"[flint-pass5] SAVED {exp['name']}: status={status} "
                f"steady_tps={metrics.get('steady_tps', 0)} steady_mfu={metrics.get('steady_mfu_pct', 0)}%",
                flush=True,
            )

        time.sleep(12)

    print("[flint-pass5] All Pass 5 Flint experiments finished.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
