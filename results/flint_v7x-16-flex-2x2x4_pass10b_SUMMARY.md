# Flint (`RecursiveSI/flint` 40B MoE) TPU v7x Pass 10B (`tile_m=256` + `SBDP` + `EP=8 ZeRO-2` + `Offload`) — `v7x-16-flex-2x2x4`

- **Updated (UTC):** `2026-10-31T07:06:32.296898+00:00`
- **Chip / World Size:** `v7x` (`32` TensorCore ranks, `1153.5` BF16 TFLOP/s/rank)

| Experiment | Lane | Status | Steady tok/s/rank | Steady MFU (%) | Peak MFU (%) | Peak HBM (GiB) | Loss (start -> end) | Wall Time (s) |
|---|---|---|---:|---:|---:|---:|---|---:|
| `v7x_p10b_01_native_tile256_sbdp_bf16rope_n40_mb48_ga2` | v7x-16 Native PR#122 (tile_m=256 [0% pad] + scale_before_down_proj + BF16-RoPE + RegionAnalysis 64MiB VMEM, n40, mb=48 2xGA) | `PASS` | **12,185.4** | **18.37%** | 18.40% | 28.03 | 12.32012 -> 4.46847 | 450.0 |
| `v7x_p10b_02_hf_pr123_tile256_sbdp_sa1024_b1024_mb48_ga2` | v7x-16 HF PR#123 (tile_m=256 + scale_before_down_proj + sa=1024 + CE b=1024 + RegionAnalysis 64MiB VMEM, mb=48 2xGA) | `FAIL(rc=1)` | **1,022.0** | **1.54%** | 1.54% | 21.35 | 12.3108 -> 12.3108 | 214.3 |
| `v7x_p10b_03_ep8_zero2_tile512_mb32_ga2` | v7x-16 Intra-Host EP=8 + ZeRO-2 (`reshard=never`, 0 cross-host AllToAll, 0 bwd AG, 12,288 tok/exp, mb=32 2xGA) | `FAIL(rc=1)` | **0.0** | **0.00%** | 0.00% | - | - | 15.0 |
| `v7x_p10b_04_ep8_zero2_tile512_mb48_ga2` | v7x-16 Intra-Host EP=8 + ZeRO-2 (`reshard=never`, 18,432 tok/exp, mb=48 = 6.29M tok/step 2xGA) | `FAIL(rc=1)` | **0.0** | **0.00%** | 0.00% | - | - | 11.4 |
| `v7x_p10b_05_ep8_offload_mb96_ga2` | v7x-16 Intra-Host EP=8 + Host-Offload (`tpu_selective_and_offload` + prefetch=1, 36,864 tok/exp, mb=96 2xGA) | `FAIL(rc=1)` | **0.0** | **0.00%** | 0.00% | - | - | 11.1 |
| `v7x_p10b_06_hf_pr123_tile256_sbdp_hybridac24_b1024_mb64_ga2` | v7x-16 HF PR#123 (mb=64 = 4.19M tok/mb + HybridAC N=24 + CE b=1024 + tile_m=256 + scale_before_down_proj + 2xGA) | `PASS` | **11,648.4** | **17.56%** | 17.58% | 25.55 | 12.31044 -> 5.33351 | 419.9 |
