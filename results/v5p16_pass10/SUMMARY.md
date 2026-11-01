# Flint (`RecursiveSI/flint` 40B MoE) TorchTitan Long-Context (`4K / 8K / 16K`) MFU Sweep (`PASS10_LONGCTX`) — `v5p-16-4n-2x2x4`

- **Updated (UTC):** `2026-11-01T16:37:52.330739+00:00`
- **Chip / World Size:** `v5p` (`16` chips/ranks, `459.0` BF16 TFLOP/s/rank)

| Experiment | Seq Len | GFLOP/tok | Status | Steady tok/s/rank | Cluster tok/s | Steady MFU (%) | Peak MFU (%) | Peak HBM (GiB) | Loss (start -> end) | Wall Time (s) |
|---|---:|---:|---|---:|---:|---:|---:|---:|---|---:|
| `v5p16_p10_01_hf_pr123_s4096_mb32_1x_sbdp_fusedopt_sa1024_ac16` | `4096` | `18.7943` | `PASS` | **6,339.7** | 101,435 | **25.96%** | 26.00% | 22.84 | 12.49604 -> 3.6804 | 575.9 |
| `v5p16_p10_02_hf_pr123_s8192_mb16_1x_sbdp_fusedopt_sa1024_ac16` | `8192` | `21.6129` | `PASS` | **6,212.4** | 99,398 | **29.25%** | 29.41% | 22.84 | 12.49606 -> 3.61466 | 407.9 |
| `v5p16_p10_03_hf_pr123_s16384_mb8_1x_sbdp_fusedopt_sa1024_ac18` | `16384` | `27.25` | `PASS` | **5,931.0** | 94,896 | **35.21%** | 35.80% | 22.84 | 12.49622 -> 3.55225 | 418.7 |
| `v5p16_p10_04_hf_pr123_s16384_mb12_1x_sbdp_fusedopt_sa1024_ac36` | `16384` | `27.25` | `FAIL(rc=-9)` | **0.0** | 0 | **0.00%** | 0.00% | - | - | 403.6 |
| `v5p16_p10_05_hf_pr123_s16384_mb8_ga2_sbdp_fusedopt_sa1024_ac24` | `16384` | `27.25` | `FAIL(rc=1)` | **1,536.0** | 24,576 | **9.12%** | 9.12% | 22.84 | 12.49581 -> 12.49581 | 220.7 |
| `v5p16_p10_06_native_pr122_s8192_mb12_1x_sbdp_bf16rope_fusedopt` | `8192` | `21.6129` | `FAIL(rc=1)` | **262.0** | 4,192 | **1.23%** | 1.23% | 29.55 | 12.31932 -> 12.31932 | 405.3 |
| `v5p16_p10_07_native_pr122_s16384_mb6_1x_sbdp_bf16rope_fusedopt` | `16384` | `27.25` | `FAIL(rc=1)` | **279.0** | 4,464 | **1.66%** | 1.66% | 29.55 | 12.32027 -> 12.32027 | 382.5 |
