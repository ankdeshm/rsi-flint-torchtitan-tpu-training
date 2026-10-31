# Flint (`RecursiveSI/flint` 40B MoE) v5p-32 Pass 6 Results — `v5p-32-2x4x4_pass6`

- **Updated (UTC):** `2026-10-30T23:09:47.749215+00:00`
- **Chip / World Size:** `v5p` (`32` ranks, `459.0` BF16 TFLOP/s/rank)

| Experiment | Lane | Status | Steady tok/s/rank | Steady MFU (%) | Peak MFU (%) | Peak HBM (GiB) | Loss (start -> end) | Wall Time (s) |
|---|---|---|---:|---:|---:|---:|---|---:|
| `v5p_p6_01_hf_pr123_hybridac18_mb64` | v5p-32 Pass6 HF PR#123 (Paired GMM tile_m=256 + HybridAC full=18, mb=64 = 4.19M tok/step) | `FAIL(rc=1)` | **382.0** | **1.45%** | 1.45% | 27.07 | 12.29461 -> 12.29461 | 379.1 |
| `v5p_p6_02_hf_pr123_hybridac36_mb80` | v5p-32 Pass6 HF PR#123 (Paired GMM tile_m=256 + HybridAC full=36, mb=80 = 5.24M tok/step) | `FAIL(rc=1)` | **375.0** | **1.42%** | 1.42% | 33.58 | 12.29422 -> 12.29422 | 477.5 |
| `v5p_p6_03_hf_pr123_fullac49_mb96` | v5p-32 Pass6 HF PR#123 (Paired GMM tile_m=256 + FullAC-49, mb=96 = 6.29M tok/step) | `FAIL(rc=1)` | **0.0** | **0.00%** | 0.00% | - | - | 141.9 |
| `v5p_p6_04_hf_pr123_hsdp_2x16_hybridac28_mb48` | v5p-32 Pass6 HSDP 2x FSDP-16 (dp_replicate=2, dp_shard=16 + HybridAC full=28, mb=48 = 3.15M tok/step) | `PASS` | **5,856.0** | **22.18%** | 22.26% | 21.06 | 12.49627 -> 3.42097 | 484.2 |
| `v5p_p6_05_hf_pr123_hsdp_2x16_fullac49_mb64` | v5p-32 Pass6 HSDP 2x FSDP-16 (dp_replicate=2, dp_shard=16 + FullAC-49, mb=64 = 4.19M tok/step) | `PASS` | **2,104.9** | **7.97%** | 8.03% | 27.07 | 12.49574 -> 3.82188 | 974.7 |
