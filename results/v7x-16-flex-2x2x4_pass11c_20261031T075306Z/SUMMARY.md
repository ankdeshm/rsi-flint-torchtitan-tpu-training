# Flint (`RecursiveSI/flint` 40B MoE) TPU v7x `PASS11C` (`StaticEP=8/16/32 ZeRO-2` + `WholeStep` + `Offload`) — `v7x-16-flex-2x2x4`

- **Updated (UTC):** `2026-10-31T08:15:20.737687+00:00`
- **Chip / World Size:** `v7x` (`32` TensorCore ranks, `1153.5` BF16 TFLOP/s/rank)

| Experiment | Lane | Status | Steady tok/s/rank | Steady MFU (%) | Peak MFU (%) | Peak HBM (GiB) | Loss (start -> end) | Wall Time (s) |
|---|---|---|---:|---:|---:|---:|---|---:|
| `v7x_p11c_01_split_staticep8_zero2_paired256_mb48_ga2` | v7x-16 GraphSplit + Intra-Host StaticEP=8 + ZeRO-2 + PairedGMM-256 (100% intra-host A2A, 18,432 tok/exp, n36, mb=48 2xGA) | `FAIL(rc=-9)` | **717.0** | **1.08%** | 1.08% | 24.71 | 12.3214 -> 12.3214 | 307.3 |
| `v7x_p11c_02_split_staticep16_zero2_paired256_mb48_ga2` | v7x-16 GraphSplit + StaticEP=16 + ZeRO-2 + PairedGMM-256 (36,864 tok/exp, 100.6MB/layer, n40, mb=48 2xGA) | `PASS` | **3,707.0** | **5.59%** | 5.60% | 24.71 | 12.32097 -> 3.69201 | 1016.4 |
