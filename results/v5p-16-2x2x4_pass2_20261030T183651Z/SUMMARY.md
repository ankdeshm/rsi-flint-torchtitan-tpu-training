# RSI Flint (`RecursiveSI/flint` 40B MoE) v5p-16 Pass 2 Results — `v5p-16-2x2x4_pass2`

- **Updated (UTC):** `2026-10-30T18:43:31.896795+00:00`
- **Chip / World Size:** `v5p` (`16` ranks, `459.0` BF16 TFLOP/s/rank, `17.385` GFLOP/token)

| Experiment | Lane | Status | Steady tok/s/rank | Steady MFU (%) | Peak MFU (%) | Peak HBM (GiB) | Loss (start -> end) | Wall Time (s) |
|---|---|---|---:|---:|---:|---:|---|---:|
| `v5p_p2_01_native_paired_tile256_n49_mb24` | v5p Pass2 Native PR#122 (Paired GMM tile_m=256 + n49 + CompiledOpt, mb=24) | `PASS` | **4,491.8** | **17.01%** | 17.09% | 14.32 | 12.31773 -> 3.39644 | 400.1 |
