# Flint (`RecursiveSI/flint` 40B MoE) TPU v5p `PASS9D` (`mb=96..160` + `StaticEP=4/8/16 ZeRO-2`) — `v5p-32-2x4x4`

- **Updated (UTC):** `2026-10-31T09:20:33.756681+00:00`
- **Chip / World Size:** `v5p` (`32` chips/ranks, `459.0` BF16 TFLOP/s/rank)

| Experiment | Lane | Status | Steady tok/s/rank | Steady MFU (%) | Peak MFU (%) | Peak HBM (GiB) | Loss (start -> end) | Wall Time (s) |
|---|---|---|---:|---:|---:|---:|---|---:|
| `v5p_p9d_01_hf_sbdp_sa1024_hybridac16_mb96_1x_log1` | v5p-32 HF PR#123 + SBDP + FusedOpt + sa=1024 + HybridAC16 + log1 (mb=96 = 4,608 tok/exp, 1x) | `FAIL(rc=-9)` | **410.0** | **1.55%** | 1.55% | 37.87 | 12.2945 -> 12.2945 | 519.4 |
| `v5p_p9d_02_hf_sbdp_sa1024_hybridac26_tile512_mb128_1x_log1` | v5p-32 HF PR#123 + SBDP + FusedOpt + sa=1024 + HybridAC26 + tile512 + log1 (mb=128 = 6,144 tok/exp, 1x) | `FAIL(rc=-9)` | **0.0** | **0.00%** | 0.00% | - | - | 575.4 |
| `v5p_p9d_03_native_staticep8_zero2_compgn_paired256_mb48_ga2_log1` | v5p-32 Native + StaticEP=8 + ZeRO-2 + EPCompiledGradNorm + SplitMeshOpt + PairedGMM-256 (18,432 tok/exp, mb=48 2xGA) | `FAIL(rc=-9)` | **616.0** | **2.33%** | 2.33% | 25.51 | 12.32572 -> 12.32572 | 359.6 |
| `v5p_p9d_04_native_staticep16_zero2_compgn_paired512_mb48_ga2_log1` | v5p-32 Native + StaticEP=16 + ZeRO-2 + EPCompiledGradNorm + SplitMeshOpt + PairedGMM-512 (36,864 tok/exp, mb=48 2xGA) | `FAIL(rc=-9)` | **725.0** | **2.75%** | 2.75% | 25.54 | 12.32657 -> 12.32657 | 313.7 |
| `v5p_p9d_05_native_wholestep_staticep16_zero2_compgn_mb32_ga2_log1` | v5p-32 Native WholeStep + StaticEP=16 + ZeRO-2 + EPCompiledGradNorm + SplitMeshOpt + PairedGMM-512 (mb=32 2xGA) | `PASS` | **3,180.4** | **12.05%** | 12.07% | 58.45 | 12.32692 -> 4.21299 | 1463.3 |
