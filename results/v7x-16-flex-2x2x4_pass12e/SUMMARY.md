# Flint (`RecursiveSI/flint` 40B MoE) TPU v7x `PASS12E` (MFU Frontier + `EP/DP/TP` Matrix) — `v7x-16-flex-2x2x4`

- **Updated (UTC):** `2026-11-01T10:54:24.789339+00:00`
- **Chip / World Size:** `v7x` (`32` TensorCore ranks, `1153.5` BF16 TFLOP/s/rank)

| Experiment | Lane | Status | Steady tok/s/rank | Cluster tok/s | Steady MFU (%) | Peak MFU (%) | Peak HBM (GiB) | Loss (start -> end) | Wall Time (s) |
|---|---|---|---:|---:|---:|---:|---:|---|---:|
| `v7x_p12e_01_native_p9base_sbdp_bf16rope_n49_mb48_ga4_log5` | v7x-16 Native PR#122 (n49 = 0% Attn Recompute! + 4xGA = 12.58M tok/step, SBDP + BF16-RoPE, mb=48, log5) | `PASS` | **13,760.7** | 440,342 | **20.74%** | 20.77% | 23.02 | 12.3202 -> 3.27832 | 1248.2 |
| `v7x_p12e_02_hf_pr123_tp2_dp16_sbdp_purefused_sa1024_tile1024_hybridac24_mb48_ga2_log5` | v7x-16 Intra-Chip TP=2 x FSDP=16 + SBDP + PureFusedOpt + sa=1024 + tile1024 + HybridAC24 + SinksTP + PadFix (mb=48/dp_rank, 2xGA = 3.15M tok/step, log5) | `FAIL(rc=1)` | **0.0** | 0 | **0.00%** | 0.00% | - | - | 66.5 |
| `v7x_p12e_03_native_p9base_sbdp_bf16rope_n49_mb64_ga2_log5` | v7x-16 Native PR#122 (n49 Single-Template + SBDP + BF16-RoPE, mb=64 = 3,072 tok/exp, 2xGA = 8.39M tok/step, log5) | `FAIL(rc=-9)` | **0.0** | 0 | **0.00%** | 0.00% | - | - | 203.8 |
| `v7x_p12e_04_native_fulloffload_sbdp_bf16rope_mb64_ga2_log5` | v7x-16 Native PR#122 (Full Host Offload prefetch=2 + SBDP + BF16-RoPE, mb=64 = 3,072 tok/exp, 2xGA = 8.39M tok/step, log5) | `PASS` | **12,342.7** | 394,966 | **18.60%** | 18.61% | 34.91 | 12.32061 -> 3.27944 | 632.7 |
| `v7x_p12e_05_native_intrachip_staticep2_zero2_paired512_n49_mb64_ga4_log5` | v7x-16 Intra-Chip StaticEP=2 x FSDP=16 (Zero Inter-Chip A2A, 6,144 tok/exp) + ZeRO-2 + PairedGMM-512 + n49 (mb=64, 4xGA = 16.78M tok/step) | `FAIL(rc=-9)` | **0.0** | 0 | **0.00%** | 0.00% | - | - | 194.0 |
| `v7x_p12e_06_hf_pr123_hsdp2x16_sbdp_purefused_sa1024_tile1024_hybridac36_mb64_ga2_log5` | v7x-16 HSDP (DP_rep=2 x FSDP=16) + SBDP + PureFusedOpt + sa=1024 + tile1024 + HybridAC36 (mb=64 = 3,072 tok/exp, 2xGA = 8.39M tok/step, log5) | `FAIL(rc=1)` | **1,168.0** | 37,376 | **1.76%** | 1.76% | 25.73 | 12.43463 -> 12.43463 | 257.1 |
