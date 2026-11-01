# Flint (`RecursiveSI/flint` 40B MoE) TorchTitan Long-Context (`4K / 8K / 16K`) MFU Sweep (`PASS13_LONGCTX`) — `v7x-16-flex-2x2x4`

- **Updated (UTC):** `2026-11-01T16:47:16.294287+00:00`
- **Chip / World Size:** `v7x` (`32` TensorCore ranks, `1153.5` BF16 TFLOP/s/rank)

| Experiment | Seq Len | GFLOP/tok | Status | Steady tok/s/rank | Cluster tok/s | Steady MFU (%) | Peak MFU (%) | Peak HBM (GiB) | Loss (start -> end) | Wall Time (s) |
|---|---:|---:|---|---:|---:|---:|---:|---:|---|---:|
| `v7x_p13_01_native_s4096_mb24_ga8_sbdp_bf16rope_fusedopt_n49` | `4096` | `18.7943` | `PASS` | **13,455.5** | 430,576 | **21.92%** | 21.93% | 23.52 | 12.32219 -> 3.7291 | 1073.9 |
| `v7x_p13_02_native_s8192_mb12_ga8_sbdp_bf16rope_fusedopt_n49` | `8192` | `21.6129` | `PASS` | **12,343.5** | 394,992 | **23.13%** | 23.13% | 20.52 | 12.32358 -> 3.74033 | 1143.9 |
| `v7x_p13_03_native_s16384_mb6_ga8_sbdp_bf16rope_fusedopt_n49` | `16384` | `27.25` | `PASS` | **11,494.0** | 367,808 | **27.15%** | 27.20% | 20.52 | 12.32431 -> 3.73677 | 1214.0 |
