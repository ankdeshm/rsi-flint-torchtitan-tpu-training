# Flint (`RecursiveSI/flint` 40B MoE) TPU v5p `PASS9C` (`mb=96..160` + `StaticEP=4/8/16 ZeRO-2`) — `v5p-16-4n-2x2x4`

- **Updated (UTC):** `2026-10-31T10:02:06.261516+00:00`
- **Chip / World Size:** `v5p` (`16` chips/ranks, `459.0` BF16 TFLOP/s/rank)

| Experiment | Lane | Status | Steady tok/s/rank | Steady MFU (%) | Peak MFU (%) | Peak HBM (GiB) | Loss (start -> end) | Wall Time (s) |
|---|---|---|---:|---:|---:|---:|---|---:|
| `v5p_p9c_01_staticep4_zero2_paired256_mb48_ga2_log1` | v5p-16 Intra-Host StaticEP=4 + ZeRO-2 + PairedGMM-256 (100% intra-host A2A, 9,216 tok/exp, mb=48 2xGA) | `FAIL(rc=-9)` | **0.0** | **0.00%** | 0.00% | - | - | 363.7 |
| `v5p_p9c_02_staticep8_zero2_paired256_mb64_1x_log1` | v5p-16 StaticEP=8 + ZeRO-2 + PairedGMM-256 (24,576 tok/exp, n45, mb=64 1x) | `FAIL(rc=-9)` | **0.0** | **0.00%** | 0.00% | - | - | 328.1 |
| `v5p_p9c_03_staticep16_zero2_paired512_mb48_ga2_log1` | v5p-16 StaticEP=16 + ZeRO-2 + PairedGMM-512 (0 expert AG/RS, 36,864 tok/exp, n45, mb=48 2xGA) | `FAIL(rc=-9)` | **735.0** | **2.78%** | 2.78% | 25.97 | 12.31281 -> 12.31281 | 309.9 |
| `v5p_p9c_04_wholestep_staticep16_zero2_mb32_1x_log1` | v5p-16 WholeStep Single-Graph + StaticEP=16 + ZeRO-2 + PairedGMM-512 (n49, mb=32 1x) | `FAIL(rc=1)` | **69.0** | **0.26%** | 0.26% | 59.03 | 12.3149 -> 12.3149 | 979.9 |
