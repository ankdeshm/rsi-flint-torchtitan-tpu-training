# Flint (`RecursiveSI/flint` 40B MoE) TPU v5p `PASS9B` (`mb=96..160` + `StaticEP=4/8/16 ZeRO-2`) — `v5p-32-2x4x4`

- **Updated (UTC):** `2026-10-31T08:18:19.008464+00:00`
- **Chip / World Size:** `v5p` (`32` chips/ranks, `459.0` BF16 TFLOP/s/rank)

| Experiment | Lane | Status | Steady tok/s/rank | Steady MFU (%) | Peak MFU (%) | Peak HBM (GiB) | Loss (start -> end) | Wall Time (s) |
|---|---|---|---:|---:|---:|---:|---|---:|
| `v5p_p9b_01_hf_sbdp_sa1024_hybridac16_mb96_1x_log5` | v5p-32 HF PR#123 + SBDP + FusedOpt + sa=1024 + HybridAC16 + log5 (mb=96 = 4,608 tok/exp, 1x) | `FAIL(rc=1)` | **410.0** | **1.55%** | 1.55% | 37.87 | 12.2945 -> 12.2945 | 511.2 |
| `v5p_p9b_02_hf_sbdp_sa1024_hybridac16_tile512_mb128_1x_log5` | v5p-32 HF PR#123 + SBDP + FusedOpt + sa=1024 + HybridAC16 + tile512 + log5 (mb=128 = 6,144 tok/exp, 1x) | `FAIL(rc=-9)` | **0.0** | **0.00%** | 0.00% | - | - | 575.0 |
| `v5p_p9b_03_hf_sbdp_sa1024_hybridac24_tile512_mb160_1x_log5` | v5p-32 HF PR#123 + SBDP + FusedOpt + sa=1024 + HybridAC24 + tile512 + log5 (mb=160 = 7,680 tok/exp, 1x) | `FAIL(rc=-9)` | **0.0** | **0.00%** | 0.00% | - | - | 831.1 |
| `v5p_p9b_04_hf_hsdp2x16_sbdp_sa1024_hybridac20_mb96_ga2_log5` | v5p-32 HF PR#123 + HSDP 2x16 + SBDP + sa=1024 + HybridAC20 (mb=96, 2xGA) | `FAIL(rc=-9)` | **0.0** | **0.00%** | 0.00% | - | - | 445.4 |
| `v5p_p9b_05_native_staticep8_zero2_paired256_mb48_ga2_log5` | v5p-32 Native TpuGraphTrainer + StaticEP=8 + ZeRO-2 + PairedGMM-256 (18,432 tok/exp, mb=48 2xGA) | `FAIL(rc=1)` | **0.0** | **0.00%** | 0.00% | - | - | 22.9 |
