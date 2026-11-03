# RSI Flint (`RecursiveSI/flint` 40B-A2.5B MoE) 60-Step Baseline & Ablation Suite (`BASELINE_2K`) — `v7x-8-2x2x2`

- **Updated (UTC):** `2026-11-03T03:32:50.071554+00:00`
- **Hardware:** `8` × `v7x` chips (`16` TensorCore ranks, `2307.0` BF16 TFLOP/s/chip)
- **Methodology:** `60` optimizer steps per run (`log_freq=1`), Steady-State Mean over **50 post-warmup steps (`11..60`)**; True Unskipped Contiguous `seq_len` Causal Attention (`TT_CONTIGUOUS_FULL_SEQ=1`) unless marked `C4-EoD`.
- **TorchTitan TPU 2K baseline & ablation suite (`google-pytorch/torchtitan` PR #122 Native lane and PR #123 HF backend lane):** fixed `tpu7x-2x2x2` (`v7x-8`, `16` TensorCore ranks, `SimpleFSDP` over 16), `seq_len=2048`, global batch `512 x 2048 = 1,048,576` tok/step (`mb=32`, `ga=1`) unless marked as a batch-scaling probe, and both Exact Causal MFU (`16.5704` GFLOP/tok) and TorchTitan Counter MFU (`17.385` GFLOP/tok, `1,153.5` BF16 TFLOP/s per TensorCore).
- Each run executes `60` optimizer steps; steady-state columns average steps `11..60`. The two PR-window columns also score the same run over the upstream step windows (PR #122: median of steps `3..8`; PR #123: mean of steps `2..10`).
- Base rows reproduce the unmodified PR #122 and PR #123 recipes on this image; step-by-step rows (`step1..step4`) add one optimization group at a time on top of the PR #122 Native baseline.

| Experiment | Seq | Seq Mode | Steady Steps | **tok/s/chip** | tok/s/rank (±std) | Slice tok/s | Causal GF/tok | **Steady Causal MFU** | PaLM GF/tok | Steady PaLM MFU | `6*N_act` MFU | PR #122-window PaLM MFU (median 3..8) | PR #123-window PaLM MFU (mean 2..10) | HBM (GiB) | Loss (`1 -> 60`) | Status |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| `v7x8_09_batch_probe_c4eod_mb40_ga2` | `2048` | `C4-EoD` | `0` (`-`) | **0.0** | 0.0 (±0.0) | 0 | `16.5704` | **0.00%** | `17.385` | 0.00% | 0.00% | 0.00% | 0.00% | - | - -> - | `FAIL(rc=1)` |
