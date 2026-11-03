# Flint (`RecursiveSI/flint` 40B MoE) TPU v7x Pass 8 Results — `v7x-16-2x2x4`

- **Updated (UTC):** `2026-10-30T23:16:49.775399+00:00`
- **Chip / World Size:** `v7x` (`32` TensorCore ranks, `1153.5` BF16 TFLOP/s/rank)

| Experiment | Lane | Status | Steady tok/s/rank | Steady MFU (%) | Peak MFU (%) | Peak HBM (GiB) | Loss (start -> end) | Wall Time (s) |
|---|---|---|---:|---:|---:|---:|---|---:|
| `v7x_p8_01_hf_pr123_paired_tile1024_hybridac18_mb64` | v7x-16 (32 TensorCores) FSDP-32 HF PR#123 (Paired GMM tile_m=1024 + HybridAC full=18, mb=64 = 4.19M tok/step) | `FAIL(rc=1)` | **458.0** | **0.69%** | 0.69% | 27.87 | 12.31039 -> 12.31039 | 365.8 |
| `v7x_p8_02_hf_pr123_paired_tile1024_hybridac36_mb80` | v7x-16 (32 TensorCores) FSDP-32 HF PR#123 (Paired GMM tile_m=1024 + HybridAC full=36, mb=80 = 5.24M tok/step) | `FAIL(rc=1)` | **436.2** | **0.66%** | 0.66% | 34.44 | 12.31053 -> 12.31053 | 426.5 |
| `v7x_p8_03_hf_pr123_paired_tile1024_fullac49_mb96` | v7x-16 (32 TensorCores) FSDP-32 HF PR#123 (Paired GMM tile_m=1024 + FullAC-49, mb=96 = 6.29M tok/step) | `FAIL(rc=1)` | **0.0** | **0.00%** | 0.00% | - | - | 149.5 |
| `v7x_p8_04_hf_pr123_hsdp_2x16_hybridac28_mb48` | v7x-16 HSDP 2x FSDP-16 (16-TensorCore / 8-Chip FSDP Shards, HybridAC full=28, mb=48 = 3.15M tok/step) | `PASS` | **10,391.2** | **15.66%** | 15.71% | 21.8 | 12.43409 -> 4.45818 | 393.6 |
| `v7x_p8_05_hf_pr123_hsdp_2x16_fullac49_mb64` | v7x-16 HSDP 2x FSDP-16 (16-TensorCore / 8-Chip FSDP Shards, FullAC-49, mb=64 = 4.19M tok/step) | `PASS` | **1,176.8** | **1.77%** | 1.79% | 27.95 | 12.43484 -> 4.43833 | 1491.1 |
