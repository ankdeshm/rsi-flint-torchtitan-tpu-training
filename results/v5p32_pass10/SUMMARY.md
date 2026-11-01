# Flint (`RecursiveSI/flint` 40B MoE) TorchTitan Long-Context (`4K / 8K / 16K`) MFU Sweep (`PASS10_LONGCTX`) — `v5p-32-8n-2x4x4`

- **Updated (UTC):** `2026-11-01T16:47:40.404627+00:00`
- **Chip / World Size:** `v5p` (`32` chips/ranks, `459.0` BF16 TFLOP/s/rank)

| Experiment | Seq Len | GFLOP/tok | Status | Steady tok/s/rank | Cluster tok/s | Steady MFU (%) | Peak MFU (%) | Peak HBM (GiB) | Loss (start -> end) | Wall Time (s) |
|---|---:|---:|---|---:|---:|---:|---:|---:|---|---:|
| `v5p32_p10_01_hf_pr123_s4096_mb32_1x_sbdp_fusedopt_sa1024_ac16` | `4096` | `18.7943` | `PASS` | **6,494.7** | 207,830 | **26.59%** | 26.64% | 22.84 | 12.29401 -> 5.27365 | 561.6 |
| `v5p32_p10_02_hf_pr123_s8192_mb16_1x_sbdp_fusedopt_sa1024_ac16` | `8192` | `21.6129` | `PASS` | **6,291.7** | 201,334 | **29.63%** | 29.84% | 22.84 | 12.29378 -> 5.51056 | 406.3 |
| `v5p32_p10_03_hf_pr123_s16384_mb8_1x_sbdp_fusedopt_sa1024_ac18` | `16384` | `27.25` | `PASS` | **5,951.6** | 190,451 | **35.33%** | 36.20% | 22.84 | 12.29381 -> 5.30979 | 420.0 |
| `v5p32_p10_04_hf_pr123_s16384_mb12_1x_sbdp_fusedopt_sa1024_ac36` | `16384` | `27.25` | `FAIL(rc=1)` | **459.0** | 14,688 | **2.73%** | 2.73% | 37.87 | 12.29372 -> 12.29372 | 461.3 |
| `v5p32_p10_05_hf_pr123_hsdp2x16_s16384_mb8_ga2_sbdp_fusedopt_sa1024_ac28` | `16384` | `27.25` | `FAIL(rc=-9)` | **795.0** | 25,440 | **4.72%** | 4.72% | 25.58 | 12.49596 -> 12.49596 | 389.7 |
| `v5p32_p10_06_native_pr122_s8192_mb12_1x_sbdp_bf16rope_fusedopt` | `8192` | `21.6129` | `PASS` | **6,036.3** | 193,162 | **28.42%** | 28.61% | 29.21 | 12.33347 -> 4.16189 | 587.2 |
| `v5p32_p10_07_native_pr122_s16384_mb6_1x_sbdp_bf16rope_fusedopt` | `16384` | `27.25` | `PASS` | **5,670.7** | 181,462 | **33.67%** | 34.63% | 29.21 | 12.33417 -> 4.15166 | 575.4 |
