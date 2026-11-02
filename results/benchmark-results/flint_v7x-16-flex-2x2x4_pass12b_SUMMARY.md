# Flint (`RecursiveSI/flint` 40B MoE) TPU v7x `PASS12B` (MFU Sweep + `EP/DP/TP` Matrix) — `v7x-16-flex-2x2x4`

- **Updated (UTC):** `2026-11-01T08:55:26.768915+00:00`
- **Chip / World Size:** `v7x` (`32` TensorCore ranks, `1153.5` BF16 TFLOP/s/rank)

| Experiment | Lane | Status | Steady tok/s/rank | Cluster tok/s | Steady MFU (%) | Peak MFU (%) | Peak HBM (GiB) | Loss (start -> end) | Wall Time (s) |
|---|---|---|---:|---:|---:|---:|---:|---|---:|
| `v7x_p12_01_native_tile1024_sbdp_bf16rope_fusedopt_libtpu_n43_mb48_ga2_log5` | v7x-16 Native PR#122 + tile1024 + SBDP + BF16-RoPE + PureFusedOpt + CalibratedLibtpuFlags + n43 (mb=48, 2xGA = 6.29M tok/step, log5) | `FAIL(rc=1)` | **804.0** | 25,728 | **1.21%** | 1.21% | 28.04 | 12.32011 -> 12.32011 | 270.7 |
| `v7x_p12_02_native_tile1024_sbdp_bf16rope_fusedopt_libtpu_n45_mb48_ga4_log5` | v7x-16 Native PR#122 + tile1024 + SBDP + BF16-RoPE + PureFusedOpt + CalibratedLibtpuFlags + n45 (mb=48, 4xGA = 12.58M tok/step, log5) | `FAIL(rc=-9)` | **4,747.0** | 151,904 | **7.15%** | 7.15% | 27.88 | 12.3202 -> 12.3202 | 111.5 |
| `v7x_p12_03_native_tile1024_sbdp_bf16rope_fusedopt_libtpu_n36_mb64_ga2_log5` | v7x-16 Native PR#122 + tile1024 + SBDP + BF16-RoPE + PureFusedOpt + CalibratedLibtpuFlags + n36 (mb=64 = 3,072 tok/exp, 2xGA = 8.39M tok/step, log5) | `FAIL(rc=1)` | **0.0** | 0 | **0.00%** | 0.00% | - | - | 276.2 |
| `v7x_p12b_04_hf_pr123_sbdp_purefused_sa1024_tile1024_hybridac12_mb64_ga4_log5` | v7x-16 HF PR#123 + SBDP + PureFusedOpt + sa=1024 + tile1024 + HybridAC12 (mb=64, 4xGA = 16.78M tok/step, log5) | `PASS` | **12,311.0** | 393,952 | **18.55%** | 18.56% | 25.5 | 12.31054 -> 4.18721 | 886.7 |
| `v7x_p12_08_native_intrachip_staticep2_zero3_paired512_n43_mb48_ga2_log5` | v7x-16 Intra-Chip StaticEP=2 x FSDP=16 (Zero Inter-Chip A2A, 4,608 tok/exp) + ZeRO-3 + PairedGMM-512 + n43 (mb=48, 2xGA, log5) | `PASS` | **7,860.0** | 251,520 | **11.85%** | 11.85% | 19.73 | 12.31885 -> 3.69214 | 611.5 |
| `v7x_p12_09_native_intratray_staticep4_zero3_paired512_n40_mb48_ga2_log5` | v7x-16 Intra-Tray StaticEP=4 x FSDP=8 (2-Chip Intra-Host A2A, 9,216 tok/exp) + ZeRO-3 + PairedGMM-512 + n40 (mb=48, 2xGA, log5) | `PASS` | **4,759.5** | 152,304 | **7.17%** | 7.19% | 21.05 | 12.31767 -> 3.6906 | 844.7 |
