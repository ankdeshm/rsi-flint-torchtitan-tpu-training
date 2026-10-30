# Flint (`RecursiveSI/flint` 40B MoE) v5p-32 Pass 5 Results — `v5p-32-2x4x4_pass5`

- **Updated (UTC):** `2026-10-30T22:17:21.189031+00:00`
- **Chip / World Size:** `v5p` (`32` ranks, `459.0` BF16 TFLOP/s/rank, `17.385` GFLOP/token)

| Experiment | Lane | Status | Steady tok/s/rank | Steady MFU (%) | Peak MFU (%) | Peak HBM (GiB) | Loss (start -> end) | Wall Time (s) |
|---|---|---|---:|---:|---:|---:|---|---:|
| `v5p_p5_01_hf_pr123_paired_tile256_b512_mb56` | v5p-32 Pass5 HF Backend PR#123 (Paired GMM tile_m=256 + b=512, mb=56 = 3.67M tok/step) | `FAIL(rc=1)` | **0.0** | **0.00%** | 0.00% | - | - | 229.4 |
| `v5p_p5_02_hf_pr123_paired_tile256_b512_mb60` | v5p-32 Pass5 HF Backend PR#123 (Paired GMM tile_m=256 + b=512, mb=60 = 3.93M tok/step) | `FAIL(rc=1)` | **0.0** | **0.00%** | 0.00% | - | - | 228.0 |
| `v5p_p5_03_hf_pr123_paired_tile256_b256_mb62` | v5p-32 Pass5 HF Backend PR#123 (Paired GMM tile_m=256 + b=256, mb=62 = 4.06M tok/step) | `FAIL(rc=1)` | **0.0** | **0.00%** | 0.00% | - | - | 237.3 |
| `v5p_p5_04_hf_pr123_hsdp_2x16_b512_mb40` | v5p-32 Pass5 HSDP 2x FSDP-16 (dp_replicate=2, dp_shard=16, mb=40) | `FAIL(rc=1)` | **460.0** | **1.74%** | 1.74% | 14.34 | 12.49558 -> 12.49558 | 205.3 |
