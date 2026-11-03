# Flint (`RecursiveSI/flint` 40B MoE) v5p-32 Pass 4 Results — `v5p-32-2x4x4_pass4`

- **Updated (UTC):** `2026-10-30T21:53:11.547516+00:00`
- **Chip / World Size:** `v5p` (`32` ranks, `459.0` BF16 TFLOP/s/rank, `17.385` GFLOP/token)

| Experiment | Lane | Status | Steady tok/s/rank | Steady MFU (%) | Peak MFU (%) | Peak HBM (GiB) | Loss (start -> end) | Wall Time (s) |
|---|---|---|---:|---:|---:|---:|---|---:|
| `v5p_p4_01_hf_pr123_paired_tile256_b1024_mb64` | v5p-32 Pass4 HF Backend PR#123 (Paired GMM tile_m=256 + b=1024, mb=64 = 4.19M tok/step) | `FAIL(rc=1)` | **0.0** | **0.00%** | 0.00% | - | - | 261.5 |
| `v5p_p4_02_hf_pr123_paired_tile256_b1024_mb80` | v5p-32 Pass4 HF Backend PR#123 (Paired GMM tile_m=256 + b=1024, mb=80 = 5.24M tok/step) | `FAIL(rc=1)` | **0.0** | **0.00%** | 0.00% | - | - | 320.8 |
| `v5p_p4_03_native_paired_tile256_n30_mb64_ga2` | v5p-32 Pass4 Native PR#122 (Paired GMM tile_m=256 + n30 + CompiledOpt, mb=64, 2x GradAccum = 8.39M tok/step) | `FAIL(rc=1)` | **709.0** | **2.69%** | 2.69% | 37.39 | 12.32786 -> 12.32786 | 408.2 |
