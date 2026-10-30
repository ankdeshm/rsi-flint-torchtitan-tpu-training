#!/usr/bin/env python3
"""Pass 2 Aggressive MFU Hillclimb for Flint (`RecursiveSI/flint` 40B MoE) on v7x-16 (32 ranks).

Exploits the massive HBM headroom discovered in Pass 1 on FSDP32 (`14.3-15.7 GiB / 96 GiB = ~15% HBM`):
  1. Eliminates activation recomputation (`NoAC` on PR#123 HF Backend; `memory_policy="default"` on PR#122 Native GraphTrainer).
  2. Doubles/triples per-rank microbatch (`mb=64` = 131,072 tok/rank & `mb=96` = 196,608 tok/rank) to saturate 256-expert Pallas GroupedMM MXU tiles and amortize FSDP32 AllGather/ReduceScatter by 2x-3x.
  3. Tunes fused Pallas CrossEntropy (`b_block_size=2048`), saves `moe.routing` + `attention.inner_attention`, compiles AdamW + global grad_norm (`optimizer_compile='separate_graph'`, `compile_grad_norm=True`), and widens GraphSplitter scheduling window (`2 layers/executable`).
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

HF_PATCH = '''

# ==============================================================================
# Pass 2 Flint MFU Hillclimb Configs (NoAC + Compiled Optimizer)
# ==============================================================================

def hf_moe_auto_max_noac() -> TPUTransformersBackendConfig:
  """Zero activation checkpointing (enabled by FSDP32 80+ GiB HBM headroom on v7x-16)."""
  config = hf_moe_auto_max()
  config.activation_checkpoint = None
  return config


def hf_moe_auto_max_noac_comp_opt() -> TPUTransformersBackendConfig:
  """Zero AC + compiled optimizer step."""
  config = hf_moe_auto_max_noac()
  if "compiled_optimizers" not in config.compile.components:
    config.compile.components = list(config.compile.components) + ["compiled_optimizers"]
  return config
'''

NATIVE_PATCH = '''

# ==============================================================================
# Pass 2 Flint MFU Hillclimb Configs (Zero Remat + Compiled Opt + 2-Layer Split)
# ==============================================================================

def flint_40b_p2_opt_moe_route() -> TpuGraphTrainer.Config:
  """Save both Splash attention AND MoE routing across all 49 layers + compiled opt/grad_norm + 2-layer split."""
  config = flint_40b_v014_tile1024()
  config.compile.memory_policy = "tpu_selective"
  config.compile.save_last_n_layers = {}
  config.compile.save_regions = ("attention.inner_attention", "moe.routing")
  config.compile.optimizer_compile = "separate_graph"
  config.compile.compile_grad_norm = True
  config.compile.graph_split.forward_layers_per_executable = 2
  config.compile.graph_split.backward_layers_per_executable = 2
  return config


def flint_40b_p2_no_remat() -> TpuGraphTrainer.Config:
  """Zero backward rematerialization (memory_policy='default') + compiled opt/grad_norm + 2-layer split."""
  config = flint_40b_v014_tile1024()
  config.compile.memory_policy = "default"
  config.compile.save_last_n_layers = {}
  config.compile.save_regions = ()
  config.compile.optimizer_compile = "separate_graph"
  config.compile.compile_grad_norm = True
  config.compile.graph_split.forward_layers_per_executable = 2
  config.compile.graph_split.backward_layers_per_executable = 2
  return config


__all__.extend([
    "flint_40b_p2_opt_moe_route",
    "flint_40b_p2_no_remat",
])
'''


def apply_runtime_patches() -> None:
    hf_reg = Path("/torchtitan-hf/torchtitan/experiments/tpu/transformers_modeling_backend/config_registry.py")
    if hf_reg.exists():
        txt = hf_reg.read_text()
        if "hf_moe_auto_max_noac" not in txt:
            hf_reg.write_text(txt + HF_PATCH)
            print("[flint-pass2] Patched HF config_registry.py", flush=True)

    nat_reg = Path("/torchtitan-native/torchtitan/experiments/tpu/examples/flint/config_registry.py")
    if nat_reg.exists():
        txt = nat_reg.read_text()
        if "flint_40b_p2_no_remat" not in txt:
            nat_reg.write_text(txt + NATIVE_PATCH)
            print("[flint-pass2] Patched Native config_registry.py", flush=True)


def wait_for_master_dns(master_addr: str, timeout_s: int = 300) -> str:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            return socket.gethostbyname(master_addr)
        except socket.gaierror:
            time.sleep(3)
    return master_addr


def build_pass2_experiments(chip_type: str, world_size: int, hf_model_path: str, steps: int) -> list[dict]:
    mb32 = 32 * 2048   # 65,536 tok/rank
    mb64 = 64 * 2048   # 131,072 tok/rank
    mb96 = 96 * 2048   # 196,608 tok/rank

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
    extra_flags = [f"--extra-libtpu-init-flags={v7x_hier_sc_flags}"] if chip_type == "v7x" else []

    return [
        {
            "name": "p2_01_hf_no_ac_mb32_b2048",
            "lane": "Pass2 HF Backend (NoAC + b=2048 + mb=32)",
            "workdir": "/torchtitan-hf",
            "env": {"TT_HF_GMM": "pallas_grouped_mm_tile_m=1024,pallas_grouped_mm_tgmm_tiling=512:512:512"},
            "args": [
                "--module=torchtitan.experiments.tpu.transformers_modeling_backend",
                "--config=hf_moe_auto_max_noac",
                f"--hf-model={hf_model_path}",
                "--training.max_context_length=2048",
                f"--training.steps={steps}",
                "--metrics.log_freq=1",
                "--debug.seed=42",
                "--loss.b_block_size=2048",
                f"--training.num_tokens_per_microbatch_per_dp_rank={mb32}",
                f"--training.num_tokens_per_train_step={world_size * mb32}",
                *hf_splash_args,
                *extra_flags,
            ],
        },
        {
            "name": "p2_02_hf_selective_ac_mb64_b2048",
            "lane": "Pass2 HF Backend (SelectiveAC + b=2048 + mb=64 / 131K tok/rank)",
            "workdir": "/torchtitan-hf",
            "env": {"TT_HF_GMM": "pallas_grouped_mm_tile_m=1024,pallas_grouped_mm_tgmm_tiling=512:512:512"},
            "args": [
                "--module=torchtitan.experiments.tpu.transformers_modeling_backend",
                "--config=hf_moe_auto_max",
                f"--hf-model={hf_model_path}",
                "--training.max_context_length=2048",
                f"--training.steps={steps}",
                "--metrics.log_freq=1",
                "--debug.seed=42",
                "--loss.b_block_size=2048",
                f"--training.num_tokens_per_microbatch_per_dp_rank={mb64}",
                f"--training.num_tokens_per_train_step={world_size * mb64}",
                *hf_splash_args,
                *extra_flags,
            ],
        },
        {
            "name": "p2_03_hf_no_ac_mb64_b2048",
            "lane": "Pass2 HF Backend (NoAC + b=2048 + mb=64 / 131K tok/rank)",
            "workdir": "/torchtitan-hf",
            "env": {"TT_HF_GMM": "pallas_grouped_mm_tile_m=1024,pallas_grouped_mm_tgmm_tiling=512:512:512"},
            "args": [
                "--module=torchtitan.experiments.tpu.transformers_modeling_backend",
                "--config=hf_moe_auto_max_noac",
                f"--hf-model={hf_model_path}",
                "--training.max_context_length=2048",
                f"--training.steps={steps}",
                "--metrics.log_freq=1",
                "--debug.seed=42",
                "--loss.b_block_size=2048",
                f"--training.num_tokens_per_microbatch_per_dp_rank={mb64}",
                f"--training.num_tokens_per_train_step={world_size * mb64}",
                *hf_splash_args,
                *extra_flags,
            ],
        },
        {
            "name": "p2_04_hf_attnac_mb96_b2048",
            "lane": "Pass2 HF Backend (AttnOnlyAC + b=2048 + mb=96 / 196K tok/rank)",
            "workdir": "/torchtitan-hf",
            "env": {"TT_HF_GMM": "pallas_grouped_mm_tile_m=1024,pallas_grouped_mm_tgmm_tiling=512:512:512"},
            "args": [
                "--module=torchtitan.experiments.tpu.transformers_modeling_backend",
                "--config=hf_moe_auto_max_attnac",
                f"--hf-model={hf_model_path}",
                "--training.max_context_length=2048",
                f"--training.steps={steps}",
                "--metrics.log_freq=1",
                "--debug.seed=42",
                "--loss.b_block_size=2048",
                f"--training.num_tokens_per_microbatch_per_dp_rank={mb96}",
                f"--training.num_tokens_per_train_step={world_size * mb96}",
                *hf_splash_args,
                *extra_flags,
            ],
        },
        {
            "name": "p2_05_native_no_remat_mb32_grp2",
            "lane": "Pass2 Native GraphTrainer (Zero Remat + CompiledOpt + 2-Layer Split, mb=32)",
            "workdir": "/torchtitan-native",
            "env": {},
            "args": [
                "--module=torchtitan.experiments.tpu.examples.flint",
                "--config=flint_40b_p2_no_remat",
                f"--training.steps={steps}",
                "--metrics.log_freq=1",
                "--debug.seed=42",
                "--tpu_config.mfu_benchmark_start_step=3",
                f"--training.num_tokens_per_microbatch_per_dp_rank={mb32}",
                f"--training.num_tokens_per_train_step={world_size * mb32}",
            ],
        },
        {
            "name": "p2_06_native_opt_moe_route_mb64_grp2",
            "lane": "Pass2 Native GraphTrainer (Save Attn+MoERouting + CompiledOpt + 2-Layer Split, mb=64)",
            "workdir": "/torchtitan-native",
            "env": {},
            "args": [
                "--module=torchtitan.experiments.tpu.examples.flint",
                "--config=flint_40b_p2_opt_moe_route",
                f"--training.steps={steps}",
                "--metrics.log_freq=1",
                "--debug.seed=42",
                "--tpu_config.mfu_benchmark_start_step=3",
                f"--training.num_tokens_per_microbatch_per_dp_rank={mb64}",
                f"--training.num_tokens_per_train_step={world_size * mb64}",
            ],
        },
        {
            "name": "p2_07_native_no_remat_mb64_grp2",
            "lane": "Pass2 Native GraphTrainer (Zero Remat + CompiledOpt + 2-Layer Split, mb=64)",
            "workdir": "/torchtitan-native",
            "env": {},
            "args": [
                "--module=torchtitan.experiments.tpu.examples.flint",
                "--config=flint_40b_p2_no_remat",
                f"--training.steps={steps}",
                "--metrics.log_freq=1",
                "--debug.seed=42",
                "--tpu_config.mfu_benchmark_start_step=3",
                f"--training.num_tokens_per_microbatch_per_dp_rank={mb64}",
                f"--training.num_tokens_per_train_step={world_size * mb64}",
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
        f"# Flint (`RecursiveSI/flint` 40B MoE) Pass 2 MFU Hillclimb — `{slice_name}`",
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
            f"{m.get('first_loss', '-')} -> {m.get('final_loss', '-')}"
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
        (bench_dir / f"flint_{slice_name}_pass2_summary.json").write_text(json.dumps(payload, indent=2))
        (bench_dir / f"flint_{slice_name}_pass2_SUMMARY.md").write_text(md_text)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chip-type", choices=["v7x", "v5p", "v6e"], required=True)
    parser.add_argument("--nnodes", type=int, required=True)
    parser.add_argument("--nproc-per-node", type=int, required=True)
    parser.add_argument("--master-addr", type=str, required=True)
    parser.add_argument("--slice-name", type=str, required=True)
    parser.add_argument("--hf-model", type=str, default="/models/models/RecursiveSI/flint")
    parser.add_argument("--steps", type=int, default=15)
    parser.add_argument("--base-port", type=int, default=29601)
    parser.add_argument("--per-run-timeout", type=int, default=2700)
    parser.add_argument("--output-dir", type=str, default="/models/flint-results")
    args = parser.parse_args()

    apply_runtime_patches()

    node_rank = int(os.environ.get("JOB_COMPLETION_INDEX", os.environ.get("NODE_RANK", "0")))
    world_size = args.nnodes * args.nproc_per_node

    print(f"[flint-pass2] node_rank={node_rank}/{args.nnodes} world_size={world_size} slice={args.slice_name}", flush=True)
    master_ip = wait_for_master_dns(args.master_addr)
    print(f"[flint-pass2] Resolved {args.master_addr} -> {master_ip}", flush=True)

    ts = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.output_dir) / f"{args.slice_name}_pass2_{ts}"
    bench_dir = Path("/models/benchmark-results")
    if node_rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
        bench_dir.mkdir(parents=True, exist_ok=True)

    exps = build_pass2_experiments(args.chip_type, world_size, args.hf_model, args.steps)
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
                    print(f"[flint-pass2] TIMEOUT ({args.per_run_timeout}s) on {exp['name']}, killing...", flush=True)
                    proc.kill()
                    break
            rc = proc.wait(timeout=30)
        except Exception as exc:
            msg = f"[flint-pass2] Exception running {exp['name']}: {exc}\n"
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
                f"[flint-pass2] SAVED {exp['name']}: status={status} "
                f"steady_tps={metrics.get('steady_tps', 0)} steady_mfu={metrics.get('steady_mfu_pct', 0)}%",
                flush=True,
            )

        time.sleep(12)

    print("[flint-pass2] All Pass 2 Flint experiments finished.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
