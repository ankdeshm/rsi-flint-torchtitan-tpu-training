# RSI Flint (`RecursiveSI/flint` 40B MoE) v5p-16 Pass 2 Results — `v5p-32-2x4x4_pass2`

- **Updated (UTC):** `2026-10-30T19:50:57.901233+00:00`
- **Chip / World Size:** `v5p` (`32` ranks, `459.0` BF16 TFLOP/s/rank, `17.385` GFLOP/token)

| Experiment | Lane | Status | Steady tok/s/rank | Steady MFU (%) | Peak MFU (%) | Peak HBM (GiB) | Loss (start -> end) | Wall Time (s) |
|---|---|---|---:|---:|---:|---:|---|---:|
| `v5p_p2_01_native_paired_tile256_n49_mb24` | v5p Pass2 Native PR#122 (Paired GMM tile_m=256 + n49 + CompiledOpt, mb=24) | `PASS` | **4,501.8** | **17.05%** | 17.11% | 12.31 | 12.32923 -> 3.67198 | 431.5 |
| `v5p_p2_02_native_paired_tile256_n49_mb32` | v5p Pass2 Native PR#122 (Paired GMM tile_m=256 + n49 + CompiledOpt, mb=32) | `PASS` | **5,016.0** | **19.00%** | 19.07% | 15.86 | 12.32848 -> 3.69982 | 412.7 |
| `v5p_p2_03_native_unpaired_tile512_n49_mb24` | v5p Pass2 Native PR#122 (Unpaired GMM tile_m=512 + n49 + CompiledOpt, mb=24) | `PASS` | **4,480.2** | **16.97%** | 17.07% | 11.18 | 12.32923 -> 3.67184 | 364.8 |
| `v5p_p2_04_hf_pr123_paired_tile256_b1024_mb32` | v5p Pass2 HF Backend PR#123 (Real Flint + Paired GMM tile_m=256 + b=1024, mb=32) | `PASS` | **5,266.9** | **19.95%** | 19.98% | 11.78 | 12.29418 -> 3.60382 | 355.4 |
