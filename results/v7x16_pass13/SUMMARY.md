# Flint (`RecursiveSI/flint` 40B MoE) TorchTitan Long-Context (`4K / 8K / 16K`) MFU Sweep (`PASS13_LONGCTX`) — `v7x-16-flex-2x2x4`

- **Updated (UTC):** `2026-11-01`
- **Chip / World Size:** `v7x` (`32` TensorCore ranks, `1153.5` BF16 TFLOP/s/rank)

| Experiment | Seq Len | GFLOP/tok | Status | Steady tok/s/rank | Cluster tok/s | Steady MFU (%) | Peak MFU (%) | Peak HBM (GiB) | Loss (start -> end) | Wall Time (s) |
|---|---:|---:|---|---:|---:|---:|---:|---:|---|---:|
| `v7x_p13_01_native_s4096_mb24_ga8_sbdp_bf16rope_fusedopt_n49` | `4096` | `18.7943` | `PASS` | **13,455.5** | 430,576 | **21.92%** | 21.93% | 23.52 | 12.32219 -> 3.7291 | 1073.9 |
| `v7x_p13_02_native_s8192_mb12_ga8_sbdp_bf16rope_fusedopt_n49` | `8192` | `21.6129` | `PASS` | **12,343.5** | 394,992 | **23.13%** | 23.13% | 20.52 | 12.32358 -> 3.74033 | 1143.9 |
| `v7x_p13_03_native_s16384_mb6_ga8_sbdp_bf16rope_fusedopt_n49` | `16384` | `27.25` | `PASS` | **11,494.0** | 367,808 | **27.15%** | 27.20% | 20.52 | 12.32431 -> 3.73677 | 1214.0 |
| `v7x_p13_04_native_s16384_mb8_ga4_sbdp_bf16rope_fusedopt_n36` | `16384` | `—` | `FAIL(rc=-9)` | — | — | — | — | OOM (`rc=-9`) | — | 285.9 |
| `v7x_p13_05_hf_pr123_s4096_mb32_ga2_sbdp_fusedopt_sa1024_tile1024_ac12` | `4096` | `18.7943` | `PASS` | **12,018.5** | 384,592 | **19.58%** | 19.59% | 25.5 | 12.30972 -> 4.05178 | 571.8 |
| `v7x_p13_06_hf_pr123_s8192_mb16_ga2_sbdp_fusedopt_sa1024_tile1024_ac12` | `8192` | `21.6129` | `PASS` | **11,517.0** | 368,544 | **21.58%** | 21.61% | 25.5 | 12.30955 -> 3.51351 | 461.5 |
| `v7x_p13_07_hf_pr123_s16384_mb8_ga2_sbdp_fusedopt_sa1024_tile1024_ac16` | `16384` | `27.25` | `PASS` | **10,713.5** | 342,832 | **25.31%** | 25.35% | 25.5 | 12.30953 -> 3.98046 | 485.7 |
