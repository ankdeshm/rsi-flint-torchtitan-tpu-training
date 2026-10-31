# Flint (`RecursiveSI/flint` 40B MoE) TPU v7x Pass 9 (`BF16 Reduce` + `Scout/HierSC XLA`) — `v7x-16-2x2x4`

- **Updated (UTC):** `2026-10-31T05:16:54.033554+00:00`
- **Chip / World Size:** `v7x` (`32` TensorCore ranks, `1153.5` BF16 TFLOP/s/rank)

| Experiment | Lane | Status | Steady tok/s/rank | Steady MFU (%) | Peak MFU (%) | Peak HBM (GiB) | Loss (start -> end) | Wall Time (s) |
|---|---|---|---:|---:|---:|---:|---|---:|
| `v7x_p9_03_hf_pr123_bf16red_scout_compopt_mb48_1x` | v7x-16 FSDP-32 HF PR#123 (Paired GMM tile_m=1024 + BF16 Reduce + Scout/HierSC XLA + CompiledOpt, mb=48 = 3.15M tok/step) | `PASS` | **12,132.8** | **18.29%** | 18.34% | 16.92 | 12.31036 -> 4.50414 | 302.7 |
| `v7x_p9_05_hf_pr123_hsdp2x16_bf16red_hybridac28_mb48_1x` | v7x-16 HSDP 2x FSDP-16 (8-Chip / 16-TC Shards, HybridAC v3 N=28 + BF16 Reduce + Scout/HierSC XLA, mb=48 = 3.15M tok/step) | `PASS` | **11,682.8** | **17.61%** | 17.64% | 21.4 | 12.43409 -> 3.98824 | 301.2 |
