# Flint (`RecursiveSI/flint` 40B MoE) TPU v7x `PASS12G` (MFU Sweep + `EP/DP/TP` Matrix) — `v7x-16-flex-2x2x4`

- **Updated (UTC):** `2026-11-01T12:42:01.687425+00:00`
- **Chip / World Size:** `v7x` (`32` TensorCore ranks, `1153.5` BF16 TFLOP/s/rank)

| Experiment | Lane | Status | Steady tok/s/rank | Cluster tok/s | Steady MFU (%) | Peak MFU (%) | Peak HBM (GiB) | Loss (start -> end) | Wall Time (s) |
|---|---|---|---:|---:|---:|---:|---:|---|---:|
| `v7x_p12g_01_native_p9base_sbdp_bf16rope_fusedopt_n49_mb48_ga16_log5` | v7x-16 Native PR#122 (n49 + PureFusedOpt + 16xGA = 50.33M tok/step, SBDP + BF16-RoPE, mb=48, log5) | `PASS` | **13,787.0** | 441,184 | **20.78%** | 20.79% | 23.02 | 12.32016 -> 3.76355 | 1909.7 |
| `v7x_p12g_02_native_p9base_sbdp_bf16rope_fusedopt_n49_mb48_ga8_log5` | v7x-16 Native PR#122 (n49 + PureFusedOpt + 8xGA = 25.17M tok/step, SBDP + BF16-RoPE, mb=48, log5) | `PASS` | **13,802.0** | 441,664 | **20.80%** | 20.82% | 23.02 | 12.32021 -> 3.71241 | 938.4 |
| `v7x_p12g_03_hf_pr123_tp2_dp16_sbdp_purefused_sa1024_tile1024_hybridac24_mb48_ga2_log5` | v7x-16 Intra-Chip TP=2 x FSDP=16 + SBDP + PureFusedOpt + sa=1024 + tile1024 + HybridAC24 + TP2PallasCE (mb=48/dp_rank, 2xGA = 3.15M tok/step, log5) | `FAIL(rc=1)` | **0.0** | 0 | **0.00%** | 0.00% | - | - | 156.1 |
| `v7x_p12g_04_native_intrachip_staticep2_zero3_sbdp_paired512_n43_mb56_ga2_log5` | v7x-16 Intra-Chip StaticEP=2 x FSDP=16 + ZeRO-3 + SBDP + BF16-RoPE + PairedGMM-512 + n43 (mb=56 = 5,376 tok/exp, 2xGA = 7.34M tok/step) | `FAIL(rc=1)` | **0.0** | 0 | **0.00%** | 0.00% | - | - | 42.6 |
| `v7x_p12g_05_hf_pr123_hsdp2x16_sbdp_purefused_sa1024_tile1024_fullac49_mb40_ga2_log5` | v7x-16 HSDP (DP_rep=2 x FSDP=16) + SBDP + PureFusedOpt + sa=1024 + tile1024 + FullAC49 (mb=40 = 1,920 tok/exp, 2xGA = 5.24M tok/step, log5) | `FAIL(rc=1)` | **0.0** | 0 | **0.00%** | 0.00% | - | - | 132.8 |
