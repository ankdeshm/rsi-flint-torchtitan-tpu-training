# Flint (`RecursiveSI/flint` 40B MoE) v5p-32 Pass 3 Results — `v5p-32-2x4x4_pass3`

- **Updated (UTC):** `2026-10-30T21:27:42.624734+00:00`
- **Chip / World Size:** `v5p` (`32` ranks, `459.0` BF16 TFLOP/s/rank, `17.385` GFLOP/token)

| Experiment | Lane | Status | Steady tok/s/rank | Steady MFU (%) | Peak MFU (%) | Peak HBM (GiB) | Loss (start -> end) | Wall Time (s) |
|---|---|---|---:|---:|---:|---:|---|---:|
| `v5p_p3_01_hf_pr123_paired_tile256_b1024_mb48` | v5p-32 Pass3 HF Backend PR#123 (Paired GMM tile_m=256 + b=1024, mb=48, 1x step) | `PASS` | **5,930.0** | **22.46%** | 22.50% | 16.91 | 12.29472 -> 4.41089 | 435.1 |
| `v5p_p3_02_hf_pr123_paired_tile256_b1024_mb48_ga2` | v5p-32 Pass3 HF Backend PR#123 (Paired GMM tile_m=256 + b=1024, mb=48, 2x GradAccum) | `FAIL(rc=1)` | **1,714.0** | **6.49%** | 6.49% | 16.91 | 12.29451 -> 12.29451 | 158.9 |
| `v5p_p3_03_native_paired_tile256_n45_mb48_ga2` | v5p-32 Pass3 Native PR#122 (Paired GMM tile_m=256 + n45 + CompiledOpt, mb=48, 2x GradAccum) | `PASS` | **5,578.6** | **21.13%** | 21.18% | 26.87 | 12.32731 -> 4.16976 | 705.9 |
