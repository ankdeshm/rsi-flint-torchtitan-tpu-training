# Flint (`RecursiveSI/flint` 40B MoE) TPU v5p Pass 9 (`SBDP` + `FusedOpt` + `sa=1024` + `EP=4/8 ZeRO-2`) — `v5p-32-2x4x4`

- **Updated (UTC):** `2026-10-31T07:19:54.449219+00:00`
- **Chip / World Size:** `v5p` (`32` chips/ranks, `459.0` BF16 TFLOP/s/rank)

| Experiment | Lane | Status | Steady tok/s/rank | Steady MFU (%) | Peak MFU (%) | Peak HBM (GiB) | Loss (start -> end) | Wall Time (s) |
|---|---|---|---:|---:|---:|---:|---|---:|
| `v5p_p9_01_hf_pr123_sbdp_fusedopt_sa1024_mb48_1x` | v5p-32 HF PR#123 + SBDP + FusedOpt + sa=1024 + RegionAnalysis + BF16Red (mb=48, 1x) | `PASS` | **6,374.9** | **24.15%** | 24.20% | 14.74 | 12.29473 -> 5.22544 | 422.3 |
| `v5p_p9_02_hf_pr123_sbdp_fusedopt_sa1024_hybridac16_mb64_1x` | v5p-32 HF PR#123 + SBDP + FusedOpt + sa=1024 + HybridAC N=16 (mb=64, 1x) | `PASS` | **6,568.4** | **24.88%** | 24.91% | 22.84 | 12.29464 -> 6.29845 | 464.9 |
| `v5p_p9_03_ep4_zero2_mb32_ga2` | v5p-32 Intra-Host EP=4 + ZeRO-2 (`reshard=never`, 100% intra-host AllToAll, 0 bwd AG, mb=32 2xGA) | `FAIL(rc=1)` | **0.0** | **0.00%** | 0.00% | - | - | 13.8 |
| `v5p_p9_04_ep4_zero2_mb48_ga2` | v5p-32 Intra-Host EP=4 + ZeRO-2 (`reshard=never`, 9,216 tok/exp, mb=48 2xGA) | `FAIL(rc=1)` | **0.0** | **0.00%** | 0.00% | - | - | 13.9 |
| `v5p_p9_05_ep8_zero2_mb48_ga2` | v5p-32 EP=8 + ZeRO-2 (`reshard=never`, 18,432 tok/exp, mb=48 2xGA) | `FAIL(rc=1)` | **0.0** | **0.00%** | 0.00% | - | - | 14.3 |
| `v5p_p9_06_native_sbdp_bf16rope_fusedopt_mb48_1x` | v5p-32 Native PR#122 + SBDP + BF16-RoPE + FusedOpt + RegionAnalysis (mb=48, 1x) | `PASS` | **6,308.1** | **23.89%** | 24.00% | 28.49 | 12.32876 -> 4.50644 | 546.7 |
