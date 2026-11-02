# Flint (`RecursiveSI/flint` 40B MoE) TorchTitan 40%-55%+ MFU Frontier (`16K / 32K / 64K`) (`PASS11_MFU_FRONTIER`) — `v5p-16-4n-2x2x4`

- **Updated (UTC):** `2026-11-02T20:59:26.726783+00:00`
- **Chip / World Size:** `v5p` (`16` chips/ranks, `459.0` BF16 TFLOP/s/rank)

| Experiment | Seq Len | GFLOP/tok | Status | Steady tok/s/rank | Cluster tok/s | Steady MFU (%) | Peak MFU (%) | Logged MFU (%) | Peak HBM (GiB) | Loss (start -> end) | Wall Time (s) |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---|---:|
| `v5p32_p11_02_native_s65536_mb1_ga4_sbdp_bf16rope_fusedopt_n44` | `65536` | `61.0729` | `PASS` | **4,913.2** | 78,611 | **65.37%** | 66.20% | 65.38% | 18.87 | 12.32201 -> 4.06841 | 970.1 |
| `v5p32_p11_03_native_s65536_mb1_ga8_overlap_tile512_n49` | `65536` | `61.0729` | `PASS` | **4,919.8** | 78,717 | **65.46%** | 65.74% | 65.46% | 18.83 | 12.32187 -> 4.00942 | 1558.5 |
