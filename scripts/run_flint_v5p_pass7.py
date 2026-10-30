#!/usr/bin/env python3
"""Pass 7 TPU v5p-32 (2x4x4, 32 ranks) 41–45%+ MFU Frontier for Flint.

Compiler-Native HybridAC v2 + Compiled Optimizer / GradNorm + DEFER_AND_FUSE:
  - Keeps 100% of layers inside `SelectiveAC._wrap_block` (`tag_activation_checkpoint` HOP) using
    `_ConfigurableSplashSaveOps(save_attn: bool)`, `preserve_rng_state=False`, and `cache_size_limit=256`.
  - Enables `TORCH_TPU_DEFER_AND_FUSE=1`, `--tpu_config.compile_optimizer`, and `--tpu_config.compile_grad_norm`
    across `FSDP-32` (`mb=48`, `mb=64`, `mb=80`) and `HSDP 2x16` (`FSDP-16` shards @ `mb=48` and `mb=64`).
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

HF_HYBRID_AC_PATCH_V2 = '''

# ==============================================================================
# Compiler-Native Hybrid Partial FullAC + AttentionOnlyAC Hook (Pass 9 / Pass 7)
# ==============================================================================
import os as _tt_os
import torch._dynamo

torch._dynamo.config.cache_size_limit = 256
torch._dynamo.config.accumulated_cache_size_limit = 1024


class _ConfigurableSplashSaveOps(set):
  def __init__(self, save_attn: bool):
    super().__init__()
    self._save_attn = save_attn

  def __contains__(self, op) -> bool:
    if not self._save_attn:
      return False
    return _PALLAS_ATTENTION_FWD_MARKER in _op_name(op)


def _hybrid_attn_only_wrap_v2(self, module, *, base_fqn=None):
  self.config.preserve_rng_state = False
  n_full = int(_tt_os.environ.get("TT_FULL_AC_FIRST_N", "0"))
  self._cur_layer_save_attn = True
  if n_full > 0 and base_fqn and base_fqn.startswith("layers."):
    try:
      if int(base_fqn.split(".")[1]) < n_full:
        self._cur_layer_save_attn = False
    except ValueError:
      pass
  return SelectiveAC._wrap_block(self, module, base_fqn=base_fqn)


def _hybrid_get_save_ops_v2(self) -> set:
  return _ConfigurableSplashSaveOps(getattr(self, "_cur_layer_save_attn", True))


AttentionOnlyAC._wrap_block = _hybrid_attn_only_wrap_v2
AttentionOnlyAC.get_save_ops = _hybrid_get_save_ops_v2
'''


def apply_runtime_patches() -> None:
    hf_ac = Path("/torchtitan-hf/torchtitan/experiments/tpu/activation_checkpoint.py")
    if hf_ac.exists():
        txt = hf_ac.read_text()
        if "_ConfigurableSplashSaveOps" not in txt:
            hf_ac.write_text(txt + HF_HYBRID_AC_PATCH_V2)
            print("[flint-v5p-pass7] Patched HF activation_checkpoint.py with compiler-native HybridAC v2 + cache_size_limit=256", flush=True)


def wait_for_master_dns(master_addr: str, timeout_s: int = 300) -> str:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            return socket.gethostbyname(master_addr)
        except socket.gaierror:
            time.sleep(3)
    return master_addr


def build_v5p_pass7_experiments(world_size: int, hf_model_path: str, steps: int) -> list[dict]:
    mb48 = 48 * 2048  # 98,304 tok/rank -> 3.15M tok/step
    mb64 = 64 * 2048  # 131,072 tok/rank -> 4.19M tok/step
    mb80 = 80 * 2048  # 163,840 tok/rank -> 5.24M tok/step

    hf_splash = [
        "--splash_attention_kernel.sa_block_q=512",
        "--splash_attention_kernel.sa_block_kv=512",
        "--splash_attention_kernel.sa_block_kv_compute=512",
        "--splash_attention_kernel.sa_block_q_dkv=512",
        "--splash_attention_kernel.sa_block_kv_dkv=512",
        "--splash_attention_kernel.sa_block_kv_dkv_compute=512",
    ]
    gmm_env = (
        "use_pallas_paired_grouped_mm=true,"
        "pallas_grouped_mm_tile_m=256,"
        "pallas_grouped_mm_tgmm_tiling=256:1024:0,"
        "pallas_grouped_mm_w2_tgmm_tiling=256:512:0,"
        "pallas_grouped_mm_w2_transposed_tgmm_tiling=256:0:1024"
    )
    base_args = [
        "--module=torchtitan.experiments.tpu.transformers_modeling_backend",
        "--config=hf_moe_auto_max_attnac",
        f"--hf-model={hf_model_path}",
        "--training.max_context_length=2048",
        f"--training.steps={steps}",
        "--metrics.log_freq=1",
        "--debug.seed=42",
        "--loss.b_block_size=1024",
        *hf_splash,
    ]

    return [
        {
            "name": "v5p_p7_01_hf_pr123_fsdp32_attnac_compopt_mb48",
            "lane": "v5p-32 FSDP-32 HF PR#123 (AttnOnlyAC + DEFER_AND_FUSE + Compiled AdamW/GradNorm, mb=48 = 3.15M tok/step)",
            "workdir": "/torchtitan-hf",
            "env": {"TT_HF_GMM": gmm_env, "TT_FULL_AC_FIRST_N": "0", "TORCH_TPU_DEFER_AND_FUSE": "1"},
            "args": [
                *base_args,
                "--tpu_config.compile_optimizer",
                "--tpu_config.compile_grad_norm",
                f"--training.num_tokens_per_microbatch_per_dp_rank={mb48}",
                f"--training.num_tokens_per_train_step={world_size * mb48}",
            ],
        },
        {
            "name": "v5p_p7_02_hf_pr123_fsdp32_hybridac22_defer_mb64",
            "lane": "v5p-32 FSDP-32 HF PR#123 (Compiler-Native HybridAC N=22 + DEFER_AND_FUSE, mb=64 = 4.19M tok/step)",
            "workdir": "/torchtitan-hf",
            "env": {"TT_HF_GMM": gmm_env, "TT_FULL_AC_FIRST_N": "22", "TORCH_TPU_DEFER_AND_FUSE": "1"},
            "args": [
                *base_args,
                f"--training.num_tokens_per_microbatch_per_dp_rank={mb64}",
                f"--training.num_tokens_per_train_step={world_size * mb64}",
            ],
        },
        {
            "name": "v5p_p7_03_hf_pr123_fsdp32_hybridac24_compopt_mb64",
            "lane": "v5p-32 FSDP-32 HF PR#123 (HybridAC N=24 + DEFER_AND_FUSE + Compiled AdamW/GradNorm, mb=64 = 4.19M tok/step)",
            "workdir": "/torchtitan-hf",
            "env": {"TT_HF_GMM": gmm_env, "TT_FULL_AC_FIRST_N": "24", "TORCH_TPU_DEFER_AND_FUSE": "1"},
            "args": [
                *base_args,
                "--tpu_config.compile_optimizer",
                "--tpu_config.compile_grad_norm",
                f"--training.num_tokens_per_microbatch_per_dp_rank={mb64}",
                f"--training.num_tokens_per_train_step={world_size * mb64}",
            ],
        },
        {
            "name": "v5p_p7_04_hf_pr123_fsdp32_hybridac40_compopt_mb80",
            "lane": "v5p-32 FSDP-32 HF PR#123 (HybridAC N=40 + DEFER_AND_FUSE + Compiled AdamW/GradNorm, mb=80 = 5.24M tok/step)",
            "workdir": "/torchtitan-hf",
            "env": {"TT_HF_GMM": gmm_env, "TT_FULL_AC_FIRST_N": "40", "TORCH_TPU_DEFER_AND_FUSE": "1"},
            "args": [
                *base_args,
                "--tpu_config.compile_optimizer",
                "--tpu_config.compile_grad_norm",
                f"--training.num_tokens_per_microbatch_per_dp_rank={mb80}",
                f"--training.num_tokens_per_train_step={world_size * mb80}",
            ],
        },
        {
            "name": "v5p_p7_05_hf_pr123_hsdp_2x16_hybridac28_compopt_mb48",
            "lane": "v5p-32 HSDP 2x FSDP-16 (16-Chip Shards, HybridAC N=28 + Compiled Opt, mb=48 = 1.57M/shard)",
            "workdir": "/torchtitan-hf",
            "env": {"TT_HF_GMM": gmm_env, "TT_FULL_AC_FIRST_N": "28", "TORCH_TPU_DEFER_AND_FUSE": "1"},
            "args": [
                *base_args,
                "--tpu_config.compile_optimizer",
                "--tpu_config.compile_grad_norm",
                "--parallelism.data_parallel_replicate_degree=2",
                "--parallelism.data_parallel_shard_degree=16",
                f"--training.num_tokens_per_microbatch_per_dp_rank={mb48}",
                f"--training.num_tokens_per_train_step={world_size * mb48}",
            ],
        },
        {
            "name": "v5p_p7_06_hf_pr123_hsdp_2x16_fullac49_compopt_mb64",
            "lane": "v5p-32 HSDP 2x FSDP-16 (16-Chip Shards, FullAC N=49 + Compiled Opt, mb=64 = 2.10M/shard)",
            "workdir": "/torchtitan-hf",
            "env": {"TT_HF_GMM": gmm_env, "TT_FULL_AC_FIRST_N": "49", "TORCH_TPU_DEFER_AND_FUSE": "1"},
            "args": [
                *base_args,
                "--tpu_config.compile_optimizer",
                "--tpu_config.compile_grad_norm",
                "--parallelism.data_parallel_replicate_degree=2",
                "--parallelism.data_parallel_shard_degree=16",
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
        peak_tf = PEAK_BF16_TFLOPS_PER_RANK.get(chip_type, 459.0) * 1e12
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
    peak_tf = PEAK_BF16_TFLOPS_PER_RANK.get(chip_type, 459.0) * 1e12
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
        f"# Flint (`RecursiveSI/flint` 40B MoE) v5p-32 Pass 7 Results — `{slice_name}`",
        "",
        f"- **Updated (UTC):** `{payload['updated_at_utc']}`",
        f"- **Chip / World Size:** `{chip_type}` (`{world_size}` ranks, `{PEAK_BF16_TFLOPS_PER_RANK.get(chip_type)}` BF16 TFLOP/s/rank)",
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
        (bench_dir / f"flint_{slice_name}_summary.json").write_text(json.dumps(payload, indent=2))
        (bench_dir / f"flint_{slice_name}_SUMMARY.md").write_text(md_text)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chip-type", choices=["v7x", "v5p", "v6e"], default="v5p")
    parser.add_argument("--nnodes", type=int, required=True)
    parser.add_argument("--nproc-per-node", type=int, default=4)
    parser.add_argument("--master-addr", type=str, required=True)
    parser.add_argument("--slice-name", type=str, required=True)
    parser.add_argument("--hf-model", type=str, default="/models/models/RecursiveSI/flint")
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--base-port", type=int, default=29991)
    parser.add_argument("--per-run-timeout", type=int, default=1800)
    parser.add_argument("--output-dir", type=str, default="/models/flint-results")
    args = parser.parse_args()

    apply_runtime_patches()
    node_rank = int(os.environ.get("JOB_COMPLETION_INDEX", os.environ.get("NODE_RANK", "0")))
    world_size = args.nnodes * args.nproc_per_node
    wait_for_master_dns(args.master_addr)

    ts = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.output_dir) / f"{args.slice_name}_{ts}"
    bench_dir = Path("/models/benchmark-results")
    if node_rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
        bench_dir.mkdir(parents=True, exist_ok=True)

    exps = build_v5p_pass7_experiments(world_size, args.hf_model, args.steps)
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
            log_chunks.append(f"[flint-v5p-pass7] Exception: {exc}\n")
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
                f"[flint-v5p-pass7] SAVED {exp['name']}: status={status} "
                f"steady_tps={metrics.get('steady_tps', 0)} steady_mfu={metrics.get('steady_mfu_pct', 0)}%",
                flush=True,
            )
        time.sleep(10)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
