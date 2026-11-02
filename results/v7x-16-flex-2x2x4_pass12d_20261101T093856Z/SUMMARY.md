# Flint (`RecursiveSI/flint` 40B MoE) TPU v7x `PASS12D` (MFU Sweep + `EP/DP/TP` Matrix) — `v7x-16-flex-2x2x4`

- **Updated (UTC):** `2026-11-01T10:05:26.507068+00:00`
- **Chip / World Size:** `v7x` (`32` TensorCore ranks, `1153.5` BF16 TFLOP/s/rank)

| Experiment | Lane | Status | Steady tok/s/rank | Cluster tok/s | Steady MFU (%) | Peak MFU (%) | Peak HBM (GiB) | Loss (start -> end) | Wall Time (s) |
|---|---|---|---:|---:|---:|---:|---:|---|---:|
| `v7x_p12d_01_native_p9base_sbdp_bf16rope_n47_mb48_ga4_log5` | v7x-16 Native PR#122 (v7x_p12c_01 + n47 + 4xGA = 12.58M tok/step, SBDP + BF16-RoPE, mb=48, log5) | `PASS` | **13,719.7** | 439,030 | **20.68%** | 20.69% | 23.02 | 12.3202 -> 3.27799 | 764.0 |
| `v7x_p12d_02_native_hostoffload_sbdp_bf16rope_mb64_ga2_log5` | v7x-16 Native PR#122 (Full Host Offload prefetch=2 + SBDP + BF16-RoPE, mb=64 = 3,072 tok/exp, 2xGA = 8.39M tok/step, log5) | `FAIL(rc=1)` | **0.0** | 0 | **0.00%** | 0.00% | - | - | 10.9 |
| `v7x_p12d_03_native_intrachip_staticep2_zero2_sbdp_paired512_n30_mb64_ga2_log5` | v7x-16 Intra-Chip StaticEP=2 x FSDP=16 (Zero Inter-Chip A2A, 6,144 tok/exp) + ZeRO-2 + SBDP + PairedGMM-512 + n30 (mb=64, 2xGA = 8.39M tok/step) | `FAIL(rc=1)` | **0.0** | 0 | **0.00%** | 0.00% | - | - | 17.7 |
| `v7x_p12d_04_hf_pr123_hsdp2x16_sbdp_purefused_sa1024_tile1024_hybridac16_mb64_ga2_log5` | v7x-16 HSDP (DP_rep=2 x FSDP=16) + SBDP + PureFusedOpt + sa=1024 + tile1024 + HybridAC16 (mb=64 = 3,072 tok/exp, 2xGA = 8.39M tok/step, log5) | `FAIL(rc=1)` | **1,168.0** | 37,376 | **1.76%** | 1.76% | 25.73 | 12.43463 -> 12.43463 | 257.5 |
| `v7x_p12d_05_hf_pr123_tp2_dp16_sbdp_purefused_sa1024_tile1024_hybridac16_mb48_ga2_log5` | v7x-16 Intra-Chip TP=2 x FSDP=16 + SBDP + PureFusedOpt + sa=1024 + tile1024 + HybridAC16 + SinksTP + PadFix (mb=48/dp_rank, 2xGA = 3.15M tok/step, log5) | `FAIL(rc=1)` | **0.0** | 0 | **0.00%** | 0.00% | - | - | 498.1 |
