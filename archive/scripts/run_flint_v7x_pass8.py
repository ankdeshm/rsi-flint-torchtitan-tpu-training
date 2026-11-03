#!/usr/bin/env python3
"""Pass 8 TPU v7x (16 chips / 32 TensorCores & 8 chips / 16 TensorCores) MFU Sweep for Flint.

Compiler & Memory Optimization (`TT_FULL_AC_FIRST_N` Hybrid `AttentionOnlyAC` + `FullAC`):
  - Pure `AttentionOnlyAC` (`hf_moe_auto_max_attnac`) saves `bf16[T, 2048] + bf16[T, 4096] + f32[32, T]`
    (`12,416 * T` bytes/layer) across all 49 layers, hitting the 95 GiB HBM wall right above `mb=48` (`55.7 GiB` saved).
  - Wrapping the first `N` layers with `FullAC` (`ptd_checkpoint_wrapper`, `4,096 * T` bytes/layer) and the
    remaining `49 - N` layers with `AttentionOnlyAC` frees `8,320 * T` bytes per full-AC layer (~1.02 GiB/layer at `mb=64`)
    at a tiny `<0.2%` compute cost per layer (`~9 ms`/layer on v7x)!
  - This unlocks BOTH:
    1. Large-microbatch `32-TensorCore` (`FSDP-32`) scaling (`mb=64`, `mb=80`, `mb=96` = `2.0x–3.0x` expert density!)
    2. `16-TensorCore` (`8-chip` FSDP shard) `HSDP (dp_replicate=2, dp_shard=16)` and pure `FSDP-16` at `mb=48` & `mb=64`
       without OOMing from FSDP-16's 2x larger persistent weight/optimizer shard (`~34 GiB` vs `~17 GiB`)!
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

FLINT_FLOPS_PER_TOKEN = 17.385e9  # Upstream TorchTitan/PaLM 2K convention; exact causal-masked 2K = 16.5704e9, 6*N_active = 15.0932e9
PEAK_BF16_TFLOPS_PER_RANK = {"v7x": 1153.5, "v5p": 459.0, "v6e": 918.0}

STEP_RE = re.compile(
    r"step:\s*(?P<step>\d+).*?loss:\s*(?P<loss>[0-9.]+).*?tps:\s*(?P<tps>[0-9.,]+)",
    re.IGNORECASE,
)
MFU_RE = re.compile(r"mfu:\s*(?P<mfu>[0-9.]+)%", re.IGNORECASE)
MEM_RE = re.compile(r"memory:\s*(?P<mem>[0-9.]+)GiB", re.IGNORECASE)

HF_HYBRID_AC_PATCH = '''

# ==============================================================================
# Hybrid Partial FullAC + AttentionOnlyAC Hook (controlled via TT_FULL_AC_FIRST_N)
# ==============================================================================
import os as _tt_os
from torchtitan.experiments.tpu.utils import ptd_checkpoint_wrapper_with_early_stop as _tt_ptd_ckpt_wrapper

_orig_attn_only_wrap = AttentionOnlyAC._wrap_block

def _hybrid_attn_only_wrap(self, module, *, base_fqn=None):
  n_full = int(_tt_os.environ.get("TT_FULL_AC_FIRST_N", "0"))
  if n_full > 0 and base_fqn and base_fqn.startswith("layers."):
    try:
      layer_idx = int(base_fqn.split(".")[1])
      if layer_idx < n_full:
        return _tt_ptd_ckpt_wrapper(
            module,
            preserve_rng_state=self.config.preserve_rng_state,
            determinism_check=self.config.determinism_check,
            early_stop=False,
            debug=self.config.debug,
        )
    except ValueError:
      pass
  return _orig_attn_only_wrap(self, module, base_fqn=base_fqn)

AttentionOnlyAC._wrap_block = _hybrid_attn_only_wrap
'''


def apply_runtime_patches() -> None:
    hf_ac = Path("/torchtitan-hf/torchtitan/experiments/tpu/activation_checkpoint.py")
    if hf_ac.exists():
        txt = hf_ac.read_text()
        if "TT_FULL_AC_FIRST_N" not in txt:
            hf_ac.write_text(txt + HF_HYBRID_AC_PATCH)
            print("[flint-v7x-pass8] Patched HF activation_checkpoint.py with TT_FULL_AC_FIRST_N hybrid AC", flush=True)


def wait_for_master_dns(master_addr: str, timeout_s: int = 300) -> str:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            return socket.gethostbyname(master_addr)
        except socket.gaierror:
            time.sleep(3)
    return master_addr


def build_v7x_pass8_experiments(world_size: int, hf_model_path: str, steps: int) -> list[dict]:
    mb48 = 48 * 2048  # 98,304 tok/rank
    mb64 = 64 * 2048  # 131,072 tok/rank -> 4.19M tok/step on 32 ranks (2.10M on 16 ranks)
    mb80 = 80 * 2048  # 163,840 tok/rank -> 5.24M tok/step on 32 ranks
    mb96 = 96 * 2048  # 196,608 tok/rank -> 6.29M tok/step on 32 ranks

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
    paired_gmm_str = (
        "use_pallas_paired_grouped_mm=true,"
        "pallas_grouped_mm_tile_m=1024,"
        "pallas_grouped_mm_tgmm_tiling=512:2048:0,"
        "pallas_grouped_mm_w2_tgmm_tiling=512:512:0,"
        "pallas_grouped_mm_w2_transposed_tgmm_tiling=512:0:2048"
    )

    if world_size == 16:
        # True 8-chip (2x2x2, 16 TensorCores) FSDP-16 topology (matching PR #122 2x2x2 topology)
        return [
            {
                "name": "v7x8_p8_01_hf_pr123_fsdp16_hybridac28_mb48",
                "lane": "v7x-8 (8 chips / 16 TensorCores) FSDP-16 HF PR#123 (HybridAC full=28, mb=48 = 1.57M tok/step)",
                "workdir": "/torchtitan-hf",
                "env": {"TT_HF_GMM": paired_gmm_str, "TT_FULL_AC_FIRST_N": "28"},
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
                "name": "v7x8_p8_02_hf_pr123_fsdp16_fullac49_mb64",
                "lane": "v7x-8 (8 chips / 16 TensorCores) FSDP-16 HF PR#123 (FullAC-49, mb=64 = 2.10M tok/step)",
                "workdir": "/torchtitan-hf",
                "env": {"TT_HF_GMM": paired_gmm_str, "TT_FULL_AC_FIRST_N": "49"},
                "args": [
                    "--module=torchtitan.experiments.tpu.transformers_modeling_backend",
                    "--config=hf_moe_auto_max_attnac",
                    f"--hf-model={hf_model_path}",
                    "--training.max_context_length=2048",
                    f"--training.steps={steps}",
                    "--metrics.log_freq=1",
                    "--debug.seed=42",
                    "--loss.b_block_size=1024",
                    f"--training.num_tokens_per_microbatch_per_dp_rank={mb64}",
                    f"--training.num_tokens_per_train_step={world_size * mb64}",
                    *hf_splash_args,
                    f"--extra-libtpu-init-flags={v7x_hier_sc_flags}",
                ],
            },
        ]

    # 16-chip (2x2x4, 32 TensorCores): FSDP-32 Large-MB (mb=64/80/96) + HSDP 2x FSDP-16 (16 TensorCores/shard)
    return [
        {
            "name": "v7x_p8_01_hf_pr123_paired_tile1024_hybridac18_mb64",
            "lane": "v7x-16 (32 TensorCores) FSDP-32 HF PR#123 (Paired GMM tile_m=1024 + HybridAC full=18, mb=64 = 4.19M tok/step)",
            "workdir": "/torchtitan-hf",
            "env": {"TT_HF_GMM": paired_gmm_str, "TT_FULL_AC_FIRST_N": "18"},
            "args": [
                "--module=torchtitan.experiments.tpu.transformers_modeling_backend",
                "--config=hf_moe_auto_max_attnac",
                f"--hf-model={hf_model_path}",
                "--training.max_context_length=2048",
                f"--training.steps={steps}",
                "--metrics.log_freq=1",
                "--debug.seed=42",
                "--loss.b_block_size=1024",
                f"--training.num_tokens_per_microbatch_per_dp_rank={mb64}",
                f"--training.num_tokens_per_train_step={world_size * mb64}",
                *hf_splash_args,
                f"--extra-libtpu-init-flags={v7x_hier_sc_flags}",
            ],
        },
        {
            "name": "v7x_p8_02_hf_pr123_paired_tile1024_hybridac36_mb80",
            "lane": "v7x-16 (32 TensorCores) FSDP-32 HF PR#123 (Paired GMM tile_m=1024 + HybridAC full=36, mb=80 = 5.24M tok/step)",
            "workdir": "/torchtitan-hf",
            "env": {"TT_HF_GMM": paired_gmm_str, "TT_FULL_AC_FIRST_N": "36"},
            "args": [
                "--module=torchtitan.experiments.tpu.transformers_modeling_backend",
                "--config=hf_moe_auto_max_attnac",
                f"--hf-model={hf_model_path}",
                "--training.max_context_length=2048",
                f"--training.steps={steps}",
                "--metrics.log_freq=1",
                "--debug.seed=42",
                "--loss.b_block_size=1024",
                f"--training.num_tokens_per_microbatch_per_dp_rank={mb80}",
                f"--training.num_tokens_per_train_step={world_size * mb80}",
                *hf_splash_args,
                f"--extra-libtpu-init-flags={v7x_hier_sc_flags}",
            ],
        },
        {
            "name": "v7x_p8_03_hf_pr123_paired_tile1024_fullac49_mb96",
            "lane": "v7x-16 (32 TensorCores) FSDP-32 HF PR#123 (Paired GMM tile_m=1024 + FullAC-49, mb=96 = 6.29M tok/step)",
            "workdir": "/torchtitan-hf",
            "env": {"TT_HF_GMM": paired_gmm_str, "TT_FULL_AC_FIRST_N": "49"},
            "args": [
                "--module=torchtitan.experiments.tpu.transformers_modeling_backend",
                "--config=hf_moe_auto_max_attnac",
                f"--hf-model={hf_model_path}",
                "--training.max_context_length=2048",
                f"--training.steps={steps}",
                "--metrics.log_freq=1",
                "--debug.seed=42",
                "--loss.b_block_size=1024",
                f"--training.num_tokens_per_microbatch_per_dp_rank={mb96}",
                f"--training.num_tokens_per_train_step={world_size * mb96}",
                *hf_splash_args,
                f"--extra-libtpu-init-flags={v7x_hier_sc_flags}",
            ],
        },
        {
            "name": "v7x_p8_04_hf_pr123_hsdp_2x16_hybridac28_mb48",
            "lane": "v7x-16 HSDP 2x FSDP-16 (16-TensorCore / 8-Chip FSDP Shards, HybridAC full=28, mb=48 = 3.15M tok/step)",
            "workdir": "/torchtitan-hf",
            "env": {"TT_HF_GMM": paired_gmm_str, "TT_FULL_AC_FIRST_N": "28"},
            "args": [
                "--module=torchtitan.experiments.tpu.transformers_modeling_backend",
                "--config=hf_moe_auto_max_attnac",
                f"--hf-model={hf_model_path}",
                "--training.max_context_length=2048",
                f"--training.steps={steps}",
                "--metrics.log_freq=1",
                "--debug.seed=42",
                "--loss.b_block_size=1024",
                "--parallelism.data_parallel_replicate_degree=2",
                "--parallelism.data_parallel_shard_degree=16",
                f"--training.num_tokens_per_microbatch_per_dp_rank={mb48}",
                f"--training.num_tokens_per_train_step={world_size * mb48}",
                *hf_splash_args,
                f"--extra-libtpu-init-flags={v7x_hier_sc_flags}",
            ],
        },
        {
            "name": "v7x_p8_05_hf_pr123_hsdp_2x16_fullac49_mb64",
            "lane": "v7x-16 HSDP 2x FSDP-16 (16-TensorCore / 8-Chip FSDP Shards, FullAC-49, mb=64 = 4.19M tok/step)",
            "workdir": "/torchtitan-hf",
            "env": {"TT_HF_GMM": paired_gmm_str, "TT_FULL_AC_FIRST_N": "49"},
            "args": [
                "--module=torchtitan.experiments.tpu.transformers_modeling_backend",
                "--config=hf_moe_auto_max_attnac",
                f"--hf-model={hf_model_path}",
                "--training.max_context_length=2048",
                f"--training.steps={steps}",
                "--metrics.log_freq=1",
                "--debug.seed=42",
                "--loss.b_block_size=1024",
                "--parallelism.data_parallel_replicate_degree=2",
                "--parallelism.data_parallel_shard_degree=16",
                f"--training.num_tokens_per_microbatch_per_dp_rank={mb64}",
                f"--training.num_tokens_per_train_step={world_size * mb64}",
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
        f"# Flint (`RecursiveSI/flint` 40B MoE) TPU v7x Pass 8 Results — `{slice_name}`",
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
        (bench_dir / f"flint_{slice_name}_pass8_summary.json").write_text(json.dumps(payload, indent=2))
        (bench_dir / f"flint_{slice_name}_pass8_SUMMARY.md").write_text(md_text)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chip-type", choices=["v7x", "v5p", "v6e"], default="v7x")
    parser.add_argument("--nnodes", type=int, required=True)
    parser.add_argument("--nproc-per-node", type=int, default=8)
    parser.add_argument("--master-addr", type=str, required=True)
    parser.add_argument("--slice-name", type=str, required=True)
    parser.add_argument("--hf-model", type=str, default="/models/models/RecursiveSI/flint")
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--base-port", type=int, default=29981)
    parser.add_argument("--per-run-timeout", type=int, default=2400)
    parser.add_argument("--output-dir", type=str, default="/models/flint-results")
    args = parser.parse_args()

    apply_runtime_patches()
    node_rank = int(os.environ.get("JOB_COMPLETION_INDEX", os.environ.get("NODE_RANK", "0")))
    world_size = args.nnodes * args.nproc_per_node
    wait_for_master_dns(args.master_addr)

    ts = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.output_dir) / f"{args.slice_name}_pass8_{ts}"
    bench_dir = Path("/models/benchmark-results")
    if node_rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
        bench_dir.mkdir(parents=True, exist_ok=True)

    exps = build_v7x_pass8_experiments(world_size, args.hf_model, args.steps)
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
            log_chunks.append(f"[flint-v7x-pass8] Exception: {exc}\n")
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
            write_summary(out_dir, bench_dir, args.slice_name, args.chip_type, world_size, results)
            print(
                f"[flint-v7x-pass8] SAVED {exp['name']}: status={status} "
                f"steady_tps={metrics.get('steady_tps', 0)} steady_mfu={metrics.get('steady_mfu_pct', 0)}%",
                flush=True,
            )
        time.sleep(12)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
