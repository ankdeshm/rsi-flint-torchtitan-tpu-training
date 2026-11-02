#!/usr/bin/env python3
"""Pass 11 TPU v5p-32 (2x4x4, 32 v5p chips) 40%-55%+ MFU Frontier Sweep for Flint (40B-A2.5B MoE).

Pushes TorchTitan (`RecursiveSI/flint` 40B-A2.5B MoE) past Pass 10's `35.33%` Steady / `36.20%` Peak MFU
(`190,451 tok/s` @ `16K`) on `v5p-32` by combining five next-wave optimizations:
  1. True Long-Context `32K` (`seq=32,768`, `38.524` GFLOP/tok) and `64K` (`seq=65,536`, `61.073` GFLOP/tok)
     Native `TpuGraphTrainer` with recursive `Configurable.Config.traverse(object)` `max_context_length` propagation
     across all RoPE caches and flat `O(1)` HBM across the 35 sliding-window (`W=513`) + 14 full causal layers.
  2. Native `TpuGraphTrainer` compiler overlap passes (`enable_fsdp_ag_rs_overlap=True`,
     `enable_fsdp_dense_region_overlap=True`) at `16K`, `32K`, and `64K`.
  3. Widened Pallas Megablocks `tile_m=512` (vs `256` in Pass 10) for higher token-per-expert density at `98K-131K` tok/rank.
  4. `by_layer` (`backward_layers_per_executable=1`) backward partitioning to unlock `131,072` tok/rank (`mb=8` @ `16K`,
     `mb=4` @ `32K`, `mb=2` @ `64K`) without bottom-of-HBM executable reservation cliffs.
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

BASE_FLOPS_PER_TOKEN_2K = 17.385e9
FULL_ATTN_FLOPS_PER_TOK_PER_CTX = 14 * 12 * 32 * 128  # 688,128 FLOPs/token per +1 seq_len above 2048
PEAK_BF16_TFLOPS_PER_RANK = {"v7x": 1153.5, "v5p": 459.0, "v6e": 918.0}

STEP_RE = re.compile(
    r"step:\s*(?P<step>\d+).*?loss:\s*(?P<loss>[0-9.]+).*?tps:\s*(?P<tps>[0-9.,]+)",
    re.IGNORECASE,
)
TFLOPS_RE = re.compile(r"tflops:\s*(?P<tflops>[0-9.]+)", re.IGNORECASE)
MFU_RE = re.compile(r"mfu:\s*(?P<mfu>[0-9.]+)%", re.IGNORECASE)
MEM_RE = re.compile(r"memory:\s*(?P<mem>[0-9.]+)GiB", re.IGNORECASE)


def flint_flops_per_token(seq_len: int) -> float:
    if seq_len <= 2048:
        return BASE_FLOPS_PER_TOKEN_2K
    return BASE_FLOPS_PER_TOKEN_2K + FULL_ATTN_FLOPS_PER_TOK_PER_CTX * (seq_len - 2048)


NATIVE_V5P_P11_FRONTIER_PATCH = '''

# ==============================================================================
# Pass 11 Flint v5p-32 40%-55%+ MFU Frontier (16K Overlap/Tile512 + 32K + 64K)
# ==============================================================================
from torchtitan.experiments.tpu.token_dispatcher import (
    TpuRoutedExperts,
    TpuRoutedExpertsConverter,
)

_P11_BF16_ROPE = (
    "torchtitan.experiments.tpu.overrides.bf16_rope"
    ".tpu_activation_dtype_cossin_rope"
)


def _p11_set_max_context_length(root_cfg, seq_len: int) -> None:
  seen = set()

  def _walk(obj):
    oid = id(obj)
    if oid in seen or obj is None or isinstance(obj, (int, float, str, bytes, bool)):
      return
    seen.add(oid)
    if hasattr(obj, "max_context_length"):
      try:
        obj.max_context_length = seq_len
      except Exception:
        pass
    if hasattr(obj, "traverse"):
      try:
        for _, child, _, _ in obj.traverse(object):
          if hasattr(child, "max_context_length"):
            try:
              child.max_context_length = seq_len
            except Exception:
              pass
      except Exception:
        pass
    if isinstance(obj, dict):
      for v in obj.values():
        _walk(v)
    elif isinstance(obj, (list, tuple)):
      for v in obj:
        _walk(v)
    elif hasattr(obj, "__dict__"):
      for v in vars(obj).values():
        _walk(v)

  _walk(root_cfg)


def _build_v5p_p11_frontier(
    seq_len: int,
    save_attn_layers: int = 44,
    tile_m: int = 256,
    bwd_by_layer: bool = False,
    ag_rs_overlap: bool = False,
) -> TpuGraphTrainer.Config:
  config = flint_40b_v5p_n40()
  _p11_set_max_context_length(config, seq_len)
  config.model_spec.model = (
      TpuRoutedExpertsConverter.Config().build().convert(config.model_spec.model)
  )
  for _, r, _, _ in config.model_spec.model.traverse(TpuRoutedExperts.Config):
    r.scale_before_down_proj = True
  for _, ex, _, _ in config.model_spec.model.traverse(TpuGroupedExperts.Config):
    ex.use_pallas_grouped_mm = True
    ex.use_pallas_paired_grouped_mm = True
    ex.pallas_grouped_mm_tile_m = tile_m
    ex.pallas_grouped_mm_tgmm_tiling = (256, 1024, 0)
    ex.pallas_grouped_mm_w2_tgmm_tiling = (256, 512, 0)
    ex.pallas_grouped_mm_w2_transposed_tgmm_tiling = (256, 0, 1024)
    ex.use_kernel_native_weight_layout = True
  if _P11_BF16_ROPE not in config.override.imports:
    config.override.imports.append(_P11_BF16_ROPE)
  config.compile.memory_policy = "tpu_selective"
  config.compile.save_regions = ("moe.routing",)
  config.compile.save_last_n_layers = {"attention.inner_attention": save_attn_layers}
  if bwd_by_layer:
    config.compile.graph_split.enabled = True
    config.compile.graph_split.forward_layers_per_executable = 1
    config.compile.graph_split.backward_partition_mode = "by_layer"
    config.compile.graph_split.backward_layers_per_executable = 1
  if ag_rs_overlap:
    config.compile.enable_fsdp_ag_rs_overlap = True
    config.compile.enable_fsdp_dense_region_overlap = True
  config.compile.optimizer_compile = "separate_graph"
  config.compile.compile_grad_norm = True
  config.optimizer.implementation = "fused"
  return config


def flint_40b_v5p_p11_s32768_tile256_n44() -> TpuGraphTrainer.Config:
  return _build_v5p_p11_frontier(32768, save_attn_layers=44, tile_m=256)


def flint_40b_v5p_p11_s65536_tile256_n44() -> TpuGraphTrainer.Config:
  return _build_v5p_p11_frontier(65536, save_attn_layers=44, tile_m=256)


def flint_40b_v5p_p11_s65536_overlap_n49() -> TpuGraphTrainer.Config:
  return _build_v5p_p11_frontier(65536, save_attn_layers=49, tile_m=512, ag_rs_overlap=True)


def flint_40b_v5p_p11_s32768_bylayer1_n44() -> TpuGraphTrainer.Config:
  return _build_v5p_p11_frontier(32768, save_attn_layers=44, tile_m=512, bwd_by_layer=True)


def flint_40b_v5p_p11_s16384_overlap_tile512_n44() -> TpuGraphTrainer.Config:
  return _build_v5p_p11_frontier(16384, save_attn_layers=44, tile_m=512, ag_rs_overlap=True)


__all__.extend([
    "flint_40b_v5p_p11_s32768_tile256_n44",
    "flint_40b_v5p_p11_s65536_tile256_n44",
    "flint_40b_v5p_p11_s65536_overlap_n49",
    "flint_40b_v5p_p11_s32768_bylayer1_n44",
    "flint_40b_v5p_p11_s16384_overlap_tile512_n44",
])
'''


def apply_runtime_patches() -> None:
    nat_reg = Path("/torchtitan-native/torchtitan/experiments/tpu/examples/flint/config_registry.py")
    if nat_reg.exists():
        txt = nat_reg.read_text()
        if "flint_40b_v5p_p11_s32768_tile256_n44" not in txt:
            nat_reg.write_text(txt + NATIVE_V5P_P11_FRONTIER_PATCH)
            print("[flint-v5p-p11] Patched Native config_registry.py with Pass 11 16K/32K/64K configs", flush=True)


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
    kept = [f for f in candidates if f.lstrip("-").split("=", 1)[0].encode("utf-8") in blob]
    return " ".join(kept)


def wait_for_master_dns(master_addr: str, timeout_s: int = 300) -> str:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            return socket.gethostbyname(master_addr)
        except socket.gaierror:
            time.sleep(3)
    return master_addr


def build_experiments(world_size: int) -> list[dict]:
    v5p_flags = filter_supported_xla_flags([
        "--xla_tpu_scoped_vmem_limit_kib=81920",
        "--xla_tpu_enable_offloading_copy_to_sparsecore=false",
        "--xla_tpu_copy_insertion_use_region_analysis=true",
        "--xla_tpu_copy_insertion_use_region_analysis_limit=1000",
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

    def _nat(name: str, lane: str, cfg: str, seq_len: int, tok_per_rank: int, ga: int, steps: int = 12) -> dict:
        return {
            "name": name,
            "lane": lane,
            "seq_len": seq_len,
            "workdir": "/torchtitan-native",
            "env": {"LIBTPU_INIT_ARGS": v5p_flags},
            "args": [
                "--module=torchtitan.experiments.tpu.examples.flint",
                f"--config={cfg}",
                f"--training.max_context_length={seq_len}",
                f"--training.steps={steps}",
                "--training.mixed_precision_reduce=bfloat16",
                "--metrics.log_freq=2",
                "--debug.seed=42",
                "--tpu_config.mfu_benchmark_start_step=4",
                f"--training.num_tokens_per_microbatch_per_dp_rank={tok_per_rank}",
                f"--training.num_tokens_per_train_step={ga * world_size * tok_per_rank}",
            ],
        }

    return [
        _nat(
            "v5p32_p11_01_native_s32768_mb3_ga4_sbdp_bf16rope_fusedopt_n44",
            "v5p-32 Native PR#122 LongCtx 32K (seq=32768, mb=3 = 98.3K tok/rank, 4xGA = 12.58M tok/step)",
            "flint_40b_v5p_p11_s32768_tile256_n44",
            seq_len=32768,
            tok_per_rank=98304,
            ga=4,
        ),
        _nat(
            "v5p32_p11_02_native_s65536_mb1_ga4_sbdp_bf16rope_fusedopt_n44",
            "v5p-32 Native PR#122 LongCtx 64K (seq=65536, mb=1 = 65.5K tok/rank, 4xGA = 8.39M tok/step)",
            "flint_40b_v5p_p11_s65536_tile256_n44",
            seq_len=65536,
            tok_per_rank=65536,
            ga=4,
        ),
        _nat(
            "v5p32_p11_03_native_s65536_mb1_ga8_overlap_tile512_n49",
            "v5p-32 Native PR#122 LongCtx 64K + AG/RS Overlap + tile512 + n49 (seq=65536, mb=1, 8xGA = 16.78M tok/step)",
            "flint_40b_v5p_p11_s65536_overlap_n49",
            seq_len=65536,
            tok_per_rank=65536,
            ga=8,
        ),
        _nat(
            "v5p32_p11_04_native_s32768_mb4_ga4_bylayer1_tile512_n44",
            "v5p-32 Native PR#122 LongCtx 32K HighTok (seq=32768, mb=4 = 131.1K tok/rank, bwd=by_layer, tile512)",
            "flint_40b_v5p_p11_s32768_bylayer1_n44",
            seq_len=32768,
            tok_per_rank=131072,
            ga=4,
        ),
        _nat(
            "v5p32_p11_05_native_s16384_mb6_ga4_overlap_tile512_n44",
            "v5p-32 Native PR#122 Fixed-16K + AG/RS Overlap + tile512 (seq=16384, mb=6 = 98.3K tok/rank, 4xGA)",
            "flint_40b_v5p_p11_s16384_overlap_tile512_n44",
            seq_len=16384,
            tok_per_rank=98304,
            ga=4,
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
    idle_timeout_s: float = 1500.0,
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
            if now - t_start > per_run_timeout or now - state["last_line"] > idle_timeout_s:
                _kill_proc_group(proc)
                return
            if state["fatal_ts"] is not None and now - state["fatal_ts"] > 10.0:
                _kill_proc_group(proc)
                return
            time.sleep(2.0)

    threading.Thread(target=_watchdog, daemon=True).start()
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


def parse_metrics(log_text: str, chip_type: str, seq_len: int) -> dict:
    flops_per_tok = flint_flops_per_token(seq_len)
    peak_tf = PEAK_BF16_TFLOPS_PER_RANK.get(chip_type, 459.0) * 1e12
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
        tf_m = TFLOPS_RE.search(line)
        mfu_m = MFU_RE.search(line)
        mem_m = MEM_RE.search(line)
        calc_mfu = 100.0 * flops_per_tok * tps / peak_tf
        logged_mfu = float(mfu_m.group("mfu")) if mfu_m else calc_mfu
        logged_tf = float(tf_m.group("tflops")) if tf_m else round(flops_per_tok * tps / 1e12, 2)
        mem_gib = float(mem_m.group("mem")) if mem_m else None
        steps.append(
            {
                "step": step,
                "loss": loss,
                "tps_per_rank": round(tps, 1),
                "logged_tflops": logged_tf,
                "logged_mfu_pct": round(logged_mfu, 2),
                "mfu_pct": round(calc_mfu, 2),
                "mem_gib": mem_gib,
            }
        )
    steady = [s for s in steps if s["step"] >= 6] or [s for s in steps if s["step"] >= 4] or steps[-2:]
    if not steady:
        return {"seq_len": seq_len, "flops_per_token_g": round(flops_per_tok / 1e9, 4), "num_logged_steps": 0, "steady_tps": 0.0, "steady_mfu_pct": 0.0}
    avg_tps = sum(s["tps_per_rank"] for s in steady) / len(steady)
    max_tps = max(s["tps_per_rank"] for s in steady)
    avg_logged_mfu = sum(s["logged_mfu_pct"] for s in steady) / len(steady)
    return {
        "seq_len": seq_len,
        "flops_per_token_g": round(flops_per_tok / 1e9, 4),
        "num_logged_steps": len(steps),
        "first_loss": steps[0]["loss"],
        "final_loss": steps[-1]["loss"],
        "steady_tps": round(avg_tps, 1),
        "peak_tps": round(max_tps, 1),
        "steady_mfu_pct": round(100.0 * flops_per_tok * avg_tps / peak_tf, 2),
        "peak_mfu_pct": round(100.0 * flops_per_tok * max_tps / peak_tf, 2),
        "steady_logged_mfu_pct": round(avg_logged_mfu, 2),
        "peak_mem_gib": max((s["mem_gib"] for s in steps if s["mem_gib"] is not None), default=None),
        "steps": steps,
    }


def write_summary(out_dir: Path, bench_dir: Path, suite: str, slice_name: str, chip_type: str, world_size: int, results: list[dict]) -> None:
    payload = {
        "updated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "suite": suite,
        "slice_name": slice_name,
        "chip_type": chip_type,
        "world_size_ranks": world_size,
        "peak_bf16_tflops_per_rank": PEAK_BF16_TFLOPS_PER_RANK.get(chip_type),
        "results": results,
    }
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2))
    md_lines = [
        f"# Flint (`RecursiveSI/flint` 40B MoE) TorchTitan 40%-55%+ MFU Frontier (`16K / 32K / 64K`) (`{suite.upper()}`) — `{slice_name}`",
        "",
        f"- **Updated (UTC):** `{payload['updated_at_utc']}`",
        f"- **Chip / World Size:** `{chip_type}` (`{world_size}` chips/ranks, `{PEAK_BF16_TFLOPS_PER_RANK.get(chip_type)}` BF16 TFLOP/s/rank)",
        "",
        "| Experiment | Seq Len | GFLOP/tok | Status | Steady tok/s/rank | Cluster tok/s | Steady MFU (%) | Peak MFU (%) | Logged MFU (%) | Peak HBM (GiB) | Loss (start -> end) | Wall Time (s) |",
        "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---|---:|",
    ]
    for r in results:
        m = r.get("metrics", {})
        loss_str = f"{m.get('first_loss', '-')} -> {m.get('final_loss', '-')}" if m.get("num_logged_steps") else "-"
        st_tps = m.get("steady_tps", 0.0)
        md_lines.append(
            f"| `{r['name']}` | `{m.get('seq_len', '-')}` | `{m.get('flops_per_token_g', '-')}` | `{r['status']}` | "
            f"**{st_tps:,.1f}** | {st_tps * world_size:,.0f} | **{m.get('steady_mfu_pct', 0.0):.2f}%** | "
            f"{m.get('peak_mfu_pct', 0.0):.2f}% | {m.get('steady_logged_mfu_pct', 0.0):.2f}% | {m.get('peak_mem_gib') or '-'} | {loss_str} | {r['duration_s']:.1f} |"
        )
    md_text = "\n".join(md_lines) + "\n"
    (out_dir / "SUMMARY.md").write_text(md_text)
    if bench_dir.exists():
        (bench_dir / f"flint_{slice_name}_{suite}_summary.json").write_text(json.dumps(payload, indent=2))
        (bench_dir / f"flint_{slice_name}_{suite}_SUMMARY.md").write_text(md_text)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", default="pass11_mfu_frontier")
    parser.add_argument("--chip-type", choices=["v7x", "v5p", "v6e"], default="v5p")
    parser.add_argument("--nnodes", type=int, default=8)
    parser.add_argument("--nproc-per-node", type=int, default=4)
    parser.add_argument("--master-addr", type=str, required=True)
    parser.add_argument("--slice-name", type=str, default="v5p-32-8n-2x4x4")
    parser.add_argument("--base-port", type=int, default=29920)
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

    exps = build_experiments(world_size)
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
            rc, log_chunks = run_with_oom_watchdog(cmd, exp["workdir"], env, node_rank, args.per_run_timeout)
        except Exception as exc:
            rc = -1
            log_chunks = [f"[flint-v5p-p11] Exception: {exc}\n"]
        dur = time.time() - t0
        if node_rank == 0:
            raw_log = "".join(log_chunks)
            (out_dir / f"{exp['name']}.log").write_text(raw_log)
            metrics = parse_metrics(raw_log, args.chip_type, seq_len=exp["seq_len"])
            status = "PASS" if (rc == 0 and metrics.get("num_logged_steps", 0) > 0) else f"FAIL(rc={rc})"
            results.append(
                {
                    "name": exp["name"],
                    "lane": exp["lane"],
                    "seq_len": exp["seq_len"],
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
                f"[flint-v5p-{args.suite}] SAVED {exp['name']}: status={status} "
                f"steady_tps={metrics.get('steady_tps', 0)} steady_mfu={metrics.get('steady_mfu_pct', 0)}% "
                f"(logged_mfu={metrics.get('steady_logged_mfu_pct', 0)}%)",
                flush=True,
            )
        time.sleep(8)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
