# Flint (`RecursiveSI/flint` 40B MoE) v5p-16 Pass 2 Results — `v5p-16-2x2x4_pass2`

- **Updated (UTC):** `2026-10-31T00:57:19.994118+00:00`
- **Chip / World Size:** `v5p` (`16` ranks, `459.0` BF16 TFLOP/s/rank, `17.385` GFLOP/token)

| Experiment | Lane | Status | Steady tok/s/rank | Steady MFU (%) | Peak MFU (%) | Peak HBM (GiB) | Loss (start -> end) | Wall Time (s) |
|---|---|---|---:|---:|---:|---:|---|---:|
| `v5p_p2_01_native_paired_tile256_n49_mb24` | v5p Pass2 Native PR#122 (Paired GMM tile_m=256 + n49 + CompiledOpt, mb=24) | `PASS` | **4,491.6** | **17.01%** | 17.10% | 14.32 | 12.31773 -> 3.39644 | 426.7 |
| `v5p_p2_02_native_paired_tile256_n49_mb32` | v5p Pass2 Native PR#122 (Paired GMM tile_m=256 + n49 + CompiledOpt, mb=32) | `PASS` | **4,964.1** | **18.80%** | 18.93% | 15.31 | 12.31517 -> 3.47345 | 412.7 |
| `v5p_p2_03_native_unpaired_tile512_n49_mb24` | v5p Pass2 Native PR#122 (Unpaired GMM tile_m=512 + n49 + CompiledOpt, mb=24) | `PASS` | **4,466.6** | **16.92%** | 17.06% | 12.82 | 12.31773 -> 3.37785 | 363.7 |
| `v5p_p2_04_hf_pr123_paired_tile256_b1024_mb32` | v5p Pass2 HF Backend PR#123 (Real Flint + Paired GMM tile_m=256 + b=1024, mb=32) | `PASS` | **5,522.2** | **20.92%** | 20.94% | 11.78 | 12.49686 -> 3.44546 | 343.3 |
