# Flint (`RecursiveSI/flint` 40B MoE) TPU v7x `PASS11F` (`StaticEP=8/16/32 ZeRO-2` + `WholeStep` + `Offload`) — `v7x-16-flex-2x2x4`

- **Updated (UTC):** `2026-10-31T10:53:36.181981+00:00`
- **Chip / World Size:** `v7x` (`32` TensorCore ranks, `1153.5` BF16 TFLOP/s/rank)

| Experiment | Lane | Status | Steady tok/s/rank | Steady MFU (%) | Peak MFU (%) | Peak HBM (GiB) | Loss (start -> end) | Wall Time (s) |
|---|---|---|---:|---:|---:|---:|---|---:|
| `v7x_p11f_01_hf_sbdp_sa1024_hybridac16_tile512_mb64_1x_log1` | v7x-16 HF PR#123 + SBDP + ChunkedOpt + sa=1024 + HybridAC16 + tile512 (mb=64 = 3,072 tok/exp, 1x) | `PASS` | **12,199.9** | **18.39%** | 18.41% | 25.55 | 12.31041 -> 5.12855 | 365.9 |
| `v7x_p11f_02_hf_sbdp_sa1024_hybridac20_tile512_mb80_1x_log1` | v7x-16 HF PR#123 + SBDP + ChunkedOpt + sa=1024 + HybridAC20 + tile512 (mb=80 = 3,840 tok/exp, 1x) | `PASS` | **11,951.1** | **18.01%** | 18.04% | 31.5 | 12.31053 -> 3.80508 | 402.4 |
| `v7x_p11f_03_hf_sbdp_sa1024_hybridac26_tile512_mb96_1x_log1` | v7x-16 HF PR#123 + SBDP + ChunkedOpt + sa=1024 + HybridAC26 + tile512 (mb=96 = 4,608 tok/exp, 1x) | `FAIL(rc=1)` | **660.0** | **0.99%** | 0.99% | 37.56 | 12.31077 -> 12.31077 | 321.6 |
| `v7x_p11f_04_hf_sbdp_sa1024_hybridac36_tile512_mb112_1x_log1` | v7x-16 HF PR#123 + SBDP + ChunkedOpt + sa=1024 + HybridAC36 + tile512 (mb=112 = 5,376 tok/exp, 1x) | `FAIL(rc=-9)` | **0.0** | **0.00%** | 0.00% | - | - | 359.7 |
| `v7x_p11f_05_hf_sbdp_sa1024_hybridac46_tile1024_mb128_1x_log1` | v7x-16 HF PR#123 + SBDP + ChunkedOpt + sa=1024 + HybridAC46 + tile1024 (mb=128 = 6,144 tok/exp, 1x) | `FAIL(rc=-9)` | **0.0** | **0.00%** | 0.00% | - | - | 423.6 |
| `v7x_p11f_06_hf_sbdp_sa1024_hybridac26_tile512_mb96_ga2_log1` | v7x-16 HF PR#123 + SBDP + ChunkedOpt + sa=1024 + HybridAC26 + tile512 (mb=96 = 4,608 tok/exp, 2xGA = 12.58M tok/step) | `FAIL(rc=1)` | **1,755.0** | **2.65%** | 2.65% | 37.56 | 12.31063 -> 12.31063 | 248.4 |
| `v7x_p11f_07_split7_fsdppipe_staticep32_zero3_paired512_mb48_ga2_log1` | v7x-16 Split-7 fsdp_pipeline + StaticEP=32 (0 device mismatch, 0 expert AG/RS) + ZeRO-3 + PairedGMM-512 (n36, mb=48 2xGA) | `FAIL(rc=-9)` | **0.0** | **0.00%** | 0.00% | - | - | 754.4 |
