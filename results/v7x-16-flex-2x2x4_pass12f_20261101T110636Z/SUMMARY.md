# Flint (`RecursiveSI/flint` 40B MoE) TPU v7x `PASS12F` (MFU Frontier + `EP/DP/TP` Matrix) — `v7x-16-flex-2x2x4`

- **Updated (UTC):** `2026-11-01T11:36:38.953471+00:00`
- **Chip / World Size:** `v7x` (`32` TensorCore ranks, `1153.5` BF16 TFLOP/s/rank)

| Experiment | Lane | Status | Steady tok/s/rank | Cluster tok/s | Steady MFU (%) | Peak MFU (%) | Peak HBM (GiB) | Loss (start -> end) | Wall Time (s) |
|---|---|---|---:|---:|---:|---:|---:|---|---:|
| `v7x_p12f_01_native_p9base_sbdp_bf16rope_n49_mb48_ga8_log5` | v7x-16 Native PR#122 (n49 = 0% Attn Recompute! + 8xGA = 25.17M tok/step, SBDP + BF16-RoPE, mb=48, log5) | `PASS` | **13,766.5** | 440,528 | **20.75%** | 20.77% | 23.02 | 12.32021 -> 3.71241 | 1049.0 |
| `v7x_p12f_02_native_p9base_sbdp_bf16rope_n49_mb52_ga4_log5` | v7x-16 Native PR#122 (n49 + SBDP + BF16-RoPE, mb=52 = 2,496 tok/exp, 4xGA = 13.63M tok/step, log5) | `FAIL(rc=-9)` | **2,155.0** | 68,960 | **3.25%** | 3.25% | 21.61 | 12.32009 -> 12.32009 | 227.8 |
| `v7x_p12f_03_hf_pr123_tp2_dp16_sbdp_purefused_sa1024_tile1024_hybridac24_mb48_ga2_log5` | v7x-16 Intra-Chip TP=2 x FSDP=16 + SBDP + PureFusedOpt + sa=1024 + tile1024 + HybridAC24 + SinksTP + EmbPadFix (mb=48/dp_rank, 2xGA = 3.15M tok/step, log5) | `FAIL(rc=1)` | **0.0** | 0 | **0.00%** | 0.00% | - | - | 157.6 |
| `v7x_p12f_04_hf_pr123_hsdp2x16_sbdp_purefused_sa1024_tile1024_fullac49_mb64_ga2_log5` | v7x-16 HSDP (DP_rep=2 x FSDP=16) + SBDP + PureFusedOpt + sa=1024 + tile1024 + FullAC49 (mb=64 = 3,072 tok/exp, 2xGA = 8.39M tok/step, log5) | `FAIL(rc=1)` | **1,161.0** | 37,152 | **1.75%** | 1.75% | 25.73 | 12.43463 -> 12.43463 | 264.2 |
| `v7x_p12f_05_native_intrachip_staticep2_zero3_sbdp_paired512_n43_mb56_ga2_log5` | v7x-16 Intra-Chip StaticEP=2 x FSDP=16 + ZeRO-3 + SBDP + BF16-RoPE + PairedGMM-512 + n43 (mb=56 = 5,376 tok/exp, 2xGA = 7.34M tok/step) | `FAIL(rc=1)` | **0.0** | 0 | **0.00%** | 0.00% | - | - | 24.3 |
