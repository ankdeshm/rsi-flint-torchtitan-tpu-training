# Flint (`RecursiveSI/flint` 40B MoE) TPU v7x Pass 10A (`tile_m=256` + `SBDP` + `Q235` + Offload + Scan) — `v7x-16-2x2x4`

- **Updated (UTC):** `2026-10-31T07:05:45.403296+00:00`
- **Chip / World Size:** `v7x` (`32` TensorCore ranks, `1153.5` BF16 TFLOP/s/rank)

| Experiment | Lane | Status | Steady tok/s/rank | Steady MFU (%) | Peak MFU (%) | Peak HBM (GiB) | Loss (start -> end) | Wall Time (s) |
|---|---|---|---:|---:|---:|---:|---|---:|
| `v7x_p10a_01_native_tile256_sbdp_q235_bf16rope_mb48_ga2` | v7x-16 Native PR#122 (tile_m=256 [0% pad] + scale_before_down_proj + BF16-RoPE + Q235 128MiB/RegionAnalysis, mb=48 2xGA) | `FAIL(rc=1)` | **711.0** | **1.07%** | 1.07% | 28.03 | 12.32012 -> 12.32012 | 303.6 |
| `v7x_p10a_02_hf_pr123_tile256_sbdp_sa1024_b2048_q235_mb48_ga2` | v7x-16 HF PR#123 (tile_m=256 + scale_before_down_proj + sa=1024 + CE b=2048 + Q235 128MiB/RegionAnalysis, mb=48 2xGA) | `FAIL(rc=-9)` | **0.0** | **0.00%** | 0.00% | - | - | 564.3 |
| `v7x_p10a_03_native_offload_mb96_tile256_sbdp_q235_ga2` | v7x-16 Native PR#122 Host-Offload (`tpu_selective_and_offload` + prefetch=1, mb=96 = 6.29M tok/mb, 4608 tok/exp, 2xGA) | `FAIL(rc=-9)` | **0.0** | **0.00%** | 0.00% | - | - | 313.8 |
| `v7x_p10a_04_native_offload_mb128_tile512_sbdp_q235_1x` | v7x-16 Native PR#122 Host-Offload (`tpu_selective_and_offload`, mb=128 = 8.39M tok/mb, 6144 tok/exp, tile_m=512, 1x) | `FAIL(rc=-9)` | **0.0** | **0.00%** | 0.00% | - | - | 361.2 |
| `v7x_p10a_05_native_scan_layers_q235_mb64_1x` | v7x-16 Native PR#122 `ScannedTpuFlintModel` (`use_scan_layers=True` + `full_and_scan_offload` + Q235 Scan Flags, mb=64) | `FAIL(rc=1)` | **0.0** | **0.00%** | 0.00% | - | - | 19.8 |
| `v7x_p10a_06_hf_pr123_tile256_sbdp_hybridac24_mb64_ga2` | v7x-16 HF PR#123 (mb=64 = 4.19M tok/mb + HybridAC N=24 + tile_m=256 + scale_before_down_proj + 2xGA) | `FAIL(rc=-9)` | **0.0** | **0.00%** | 0.00% | - | - | 570.1 |
