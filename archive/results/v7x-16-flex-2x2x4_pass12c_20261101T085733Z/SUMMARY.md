# Flint (`RecursiveSI/flint` 40B MoE) TPU v7x `PASS12C` (MFU Sweep + `EP/DP/TP` Matrix) — `v7x-16-flex-2x2x4`

- **Updated (UTC):** `2026-11-01T09:26:34.738063+00:00`
- **Chip / World Size:** `v7x` (`32` TensorCore ranks, `1153.5` BF16 TFLOP/s/rank)

| Experiment | Lane | Status | Steady tok/s/rank | Cluster tok/s | Steady MFU (%) | Peak MFU (%) | Peak HBM (GiB) | Loss (start -> end) | Wall Time (s) |
|---|---|---|---:|---:|---:|---:|---:|---|---:|
| `v7x_p12c_01_native_p9base_sbdp_bf16rope_n41_mb48_ga2_log5` | v7x-16 Native PR#122 (v7x_p9_02 base + SBDP + BF16-RoPE + n41, mb=48, 2xGA = 6.29M tok/step, log5) | `PASS` | **13,675.7** | 437,622 | **20.61%** | 20.63% | 25.48 | 12.32011 -> 3.27156 | 559.7 |
| `v7x_p12c_02_native_p9base_sbdp_bf16rope_n30_mb64_ga2_log5` | v7x-16 Native PR#122 (v7x_p9_02 base + SBDP + BF16-RoPE + n30, mb=64 = 3,072 tok/exp, 2xGA = 8.39M tok/step, log5) | `FAIL(rc=-9)` | **0.0** | 0 | **0.00%** | 0.00% | - | - | 281.9 |
| `v7x_p12c_03_native_intrachip_staticep2_zero3_paired512_n36_mb80_ga2_log5` | v7x-16 Intra-Chip StaticEP=2 x FSDP=16 (Zero Inter-Chip A2A, 7,680 tok/exp) + ZeRO-3 + PairedGMM-512 + n36 (mb=80, 2xGA = 10.49M tok/step) | `FAIL(rc=1)` | **0.0** | 0 | **0.00%** | 0.00% | - | - | 283.3 |
| `v7x_p12c_04_native_intrachip_staticep2_zero2_paired512_n36_mb80_ga2_log5` | v7x-16 Intra-Chip StaticEP=2 x FSDP=16 (Zero Inter-Chip A2A, 7,680 tok/exp) + ZeRO-2 (No Bwd AG) + PairedGMM-512 + n36 (mb=80, 2xGA = 10.49M tok/step) | `FAIL(rc=1)` | **0.0** | 0 | **0.00%** | 0.00% | - | - | 89.4 |
| `v7x_p12c_05_hf_pr123_hsdp2x16_sbdp_purefused_sa1024_tile1024_hybridac28_mb48_ga2_log5` | v7x-16 HSDP (DP_rep=2 x FSDP=16) + SBDP + PureFusedOpt + sa=1024 + tile1024 + HybridAC28 (mb=48, 2xGA = 6.29M tok/step, log5) | `PASS` | **12,109.5** | 387,504 | **18.25%** | 18.26% | 20.1 | 12.43497 -> 3.7939 | 455.6 |
| `v7x_p12c_06_hf_pr123_tp2_dp16_sbdp_purefused_sa1024_tile1024_hybridac16_mb48_ga2_log5` | v7x-16 Intra-Chip TP=2 x FSDP=16 + SBDP + PureFusedOpt + sa=1024 + tile1024 + HybridAC16 + SinksTP (mb=48/dp_rank, 2xGA = 3.15M tok/step, log5) | `FAIL(rc=1)` | **0.0** | 0 | **0.00%** | 0.00% | - | - | 20.0 |
