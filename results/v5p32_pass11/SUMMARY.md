# Flint (`RecursiveSI/flint` 40B MoE) TorchTitan 40%-55%+ MFU Frontier (`16K / 32K / 64K`) (`PASS11_MFU_FRONTIER`) — `v5p-32-8n-2x4x4`

- **Updated (UTC):** `2026-11-02T21:11:26.802996+00:00`
- **Chip / World Size:** `v5p` (`32` chips/ranks, `459.0` BF16 TFLOP/s/rank)

| Experiment | Seq Len | GFLOP/tok | Status | Steady tok/s/rank | Cluster tok/s | Steady MFU (%) | Peak MFU (%) | Logged MFU (%) | Peak HBM (GiB) | Loss (start -> end) | Wall Time (s) |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---|---:|
| `v5p32_p11_01_native_s32768_mb3_ga4_sbdp_bf16rope_fusedopt_n44` | `32768` | `38.5243` | `PASS` | **5,439.5** | 174,064 | **45.65%** | 45.89% | 45.66% | 27.52 | 12.33327 -> 4.16861 | 1210.0 |
| `v5p32_p11_02_native_s65536_mb1_ga4_sbdp_bf16rope_fusedopt_n44` | `65536` | `61.0729` | `PASS` | **4,674.5** | 149,584 | **62.20%** | 63.33% | 62.20% | 18.67 | 12.33374 -> 4.17085 | 999.1 |
| `v5p32_p11_03_native_s65536_mb1_ga8_overlap_tile512_n49` | `65536` | `61.0729` | `PASS` | **4,734.5** | 151,504 | **63.00%** | 64.07% | 63.00% | 14.96 | 12.33341 -> 4.20836 | 1611.4 |
| `v5p32_p11_05_native_s16384_mb6_ga4_overlap_tile512_n44` | `16384` | `27.25` | `PASS` | **5,868.8** | 187,802 | **34.84%** | 35.16% | 34.84% | 27.52 | 12.33313 -> 4.17916 | 1135.6 |
