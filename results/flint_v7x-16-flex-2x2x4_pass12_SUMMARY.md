# Flint (`RecursiveSI/flint` 40B MoE) TPU v7x `PASS12` (MFU Frontier + `EP/DP/TP` Matrix) — `v7x-16-flex-2x2x4`

- **Updated (UTC):** `2026-11-01T07:53:06.317586+00:00`
- **Chip / World Size:** `v7x` (`32` TensorCore ranks, `1153.5` BF16 TFLOP/s/rank)

| Experiment | Lane | Status | Steady tok/s/rank | Cluster tok/s | Steady MFU (%) | Peak MFU (%) | Peak HBM (GiB) | Loss (start -> end) | Wall Time (s) |
|---|---|---|---:|---:|---:|---:|---:|---|---:|
| `v7x_p12_01_native_tile1024_sbdp_bf16rope_fusedopt_libtpu_n43_mb48_ga2_log5` | v7x-16 Native PR#122 + tile1024 + SBDP + BF16-RoPE + PureFusedOpt + CalibratedLibtpuFlags + n43 (mb=48, 2xGA = 6.29M tok/step, log5) | `FAIL(rc=1)` | **759.0** | 24,288 | **1.14%** | 1.14% | 28.04 | 12.32011 -> 12.32011 | 313.8 |
| `v7x_p12_02_native_tile1024_sbdp_bf16rope_fusedopt_libtpu_n45_mb48_ga4_log5` | v7x-16 Native PR#122 + tile1024 + SBDP + BF16-RoPE + PureFusedOpt + CalibratedLibtpuFlags + n45 (mb=48, 4xGA = 12.58M tok/step, log5) | `FAIL(rc=-9)` | **4,641.0** | 148,512 | **6.99%** | 6.99% | 27.88 | 12.3202 -> 12.3202 | 117.5 |
| `v7x_p12_03_native_tile1024_sbdp_bf16rope_fusedopt_libtpu_n36_mb64_ga2_log5` | v7x-16 Native PR#122 + tile1024 + SBDP + BF16-RoPE + PureFusedOpt + CalibratedLibtpuFlags + n36 (mb=64 = 3,072 tok/exp, 2xGA = 8.39M tok/step, log5) | `FAIL(rc=-9)` | **0.0** | 0 | **0.00%** | 0.00% | - | - | 279.7 |
| `v7x_p12_04_hf_pr123_sbdp_purefused_sa1024_tile1024_hybridac12_mb64_ga2_log5` | v7x-16 HF PR#123 + SBDP + PureFusedOpt + sa=1024 + tile1024 + HybridAC12 (mb=64 = 3,072 tok/exp, 2xGA = 8.39M tok/step, log5) | `PASS` | **12,331.0** | 394,592 | **18.58%** | 18.59% | 25.5 | 12.31044 -> 3.76404 | 564.5 |
| `v7x_p12_05_hf_pr123_sbdp_purefused_sa1024_tile1024_hybridac16_mb72_ga2_log5` | v7x-16 HF PR#123 + SBDP + PureFusedOpt + sa=1024 + tile1024 + HybridAC16 (mb=72 = 3,456 tok/exp, 2xGA = 9.44M tok/step, log5) | `PASS` | **12,256.0** | 392,192 | **18.47%** | 18.48% | 28.49 | 12.3105 -> 3.79098 | 568.4 |
