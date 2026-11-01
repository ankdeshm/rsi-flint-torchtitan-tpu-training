# Flint (`RecursiveSI/flint` 40B MoE) TPU v7x `PASS11E` (`StaticEP=8/16/32 ZeRO-2` + `WholeStep` + `Offload`) — `v7x-16-flex-2x2x4`

- **Updated (UTC):** `2026-10-31T10:00:15.032632+00:00`
- **Chip / World Size:** `v7x` (`32` TensorCore ranks, `1153.5` BF16 TFLOP/s/rank)

| Experiment | Lane | Status | Steady tok/s/rank | Steady MFU (%) | Peak MFU (%) | Peak HBM (GiB) | Loss (start -> end) | Wall Time (s) |
|---|---|---|---:|---:|---:|---:|---|---:|
| `v7x_p11e_01_split1_zeroargsort_staticep8_zero3_paired256_mb48_ga2_log1` | v7x-16 Split-1 + Zero-Argsort Intra-Host StaticEP=8 + ZeRO-3 + PairedGMM-256 (0 argsorts, 0 scatter-adds, n36, mb=48 2xGA) | `PASS` | **2,799.6** | **4.22%** | 4.25% | 26.96 | 12.3211 -> 4.19236 | 1026.5 |
