# Flint (`RecursiveSI/flint` 40B MoE) TorchTitan 40%-55%+ MFU Frontier (`16K / 32K / 64K`) (`PASS11_MFU_FRONTIER`) — `v5p-32-8n-2x4x4`

- **Updated (UTC):** `2026-11-02T20:03:13.776536+00:00`
- **Chip / World Size:** `v5p` (`32` chips/ranks, `459.0` BF16 TFLOP/s/rank)

| Experiment | Seq Len | GFLOP/tok | Status | Steady tok/s/rank | Cluster tok/s | Steady MFU (%) | Peak MFU (%) | Logged MFU (%) | Peak HBM (GiB) | Loss (start -> end) | Wall Time (s) |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---|---:|
| `v5p32_p11_01_native_s32768_mb3_ga4_sbdp_bf16rope_fusedopt_n44` | `32768` | `38.5243` | `PASS` | **5,439.5** | 174,064 | **45.65%** | 45.89% | 45.66% | 27.52 | 12.33327 -> 4.16861 | 1210.0 |
