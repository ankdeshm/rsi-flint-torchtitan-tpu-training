# Flint (`RecursiveSI/flint` 40B MoE) TPU v5p `PASS9B` (`mb=96..160` + `StaticEP=4/8/16 ZeRO-2`) — `v5p-16-2x2x4`

- **Updated (UTC):** `2026-10-31T10:09:28.504304+00:00`
- **Chip / World Size:** `v5p` (`16` chips/ranks, `459.0` BF16 TFLOP/s/rank)

| Experiment | Lane | Status | Steady tok/s/rank | Steady MFU (%) | Peak MFU (%) | Peak HBM (GiB) | Loss (start -> end) | Wall Time (s) |
|---|---|---|---:|---:|---:|---:|---|---:|
| `v5p_p9b_01_hf_sbdp_sa1024_hybridac16_mb96_1x_log1` | v5p-16 HF PR#123 + SBDP + FusedOpt + sa=1024 + HybridAC16 + log1 (mb=96 = 4,608 tok/exp, 1x) | `FAIL(rc=-9)` | **0.0** | **0.00%** | 0.00% | - | - | 447.5 |
| `v5p_p9b_02_hf_sbdp_sa1024_hybridac26_tile512_mb128_1x_log1` | v5p-16 HF PR#123 + SBDP + FusedOpt + sa=1024 + HybridAC26 + tile512 + log1 (mb=128 = 6,144 tok/exp, 1x) | `FAIL(rc=-9)` | **0.0** | **0.00%** | 0.00% | - | - | 568.9 |
| `v5p_p9b_03_hf_sbdp_sa1024_hybridac34_tile512_mb160_1x_log1` | v5p-16 HF PR#123 + SBDP + FusedOpt + sa=1024 + HybridAC34 + tile512 + log1 (mb=160 = 7,680 tok/exp, 1x) | `FAIL(rc=-9)` | **0.0** | **0.00%** | 0.00% | - | - | 826.4 |
| `v5p_p9b_04_hf_hsdp2x8_sbdp_sa1024_hybridac20_mb96_ga2_log1` | v5p-16 HF PR#123 + HSDP 2x8 + SBDP + sa=1024 + HybridAC20 (mb=96, 2xGA) | `FAIL(rc=-9)` | **0.0** | **0.00%** | 0.00% | - | - | 404.7 |
| `v5p_p9b_05_native_staticep8_zero2_paired256_mb48_ga2_log1` | v5p-16 Native TpuGraphTrainer + StaticEP=8 + ZeRO-2 + PairedGMM-256 (18,432 tok/exp, mb=48 2xGA) | `FAIL(rc=-9)` | **0.0** | **0.00%** | 0.00% | - | - | 303.6 |
