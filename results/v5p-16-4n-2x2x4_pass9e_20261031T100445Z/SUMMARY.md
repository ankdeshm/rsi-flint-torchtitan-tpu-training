# Flint (`RecursiveSI/flint` 40B MoE) TPU v5p `PASS9E` (`mb=96..160` + `StaticEP=4/8/16 ZeRO-2`) — `v5p-16-4n-2x2x4`

- **Updated (UTC):** `2026-10-31T10:33:18.893002+00:00`
- **Chip / World Size:** `v5p` (`16` chips/ranks, `459.0` BF16 TFLOP/s/rank)

| Experiment | Lane | Status | Steady tok/s/rank | Steady MFU (%) | Peak MFU (%) | Peak HBM (GiB) | Loss (start -> end) | Wall Time (s) |
|---|---|---|---:|---:|---:|---:|---|---:|
| `v5p_p9e_01_hf_sbdp_sa1024_hybridac30_tile512_mb80_1x_log1` | v5p-16 HF PR#123 + SBDP + ChunkedOpt + sa=1024 + HybridAC30 + tile512 (mb=80 = 3,840 tok/exp, 1x) | `FAIL(rc=-9)` | **409.0** | **1.55%** | 1.55% | 28.55 | 12.49536 -> 12.49536 | 437.5 |
| `v5p_p9e_02_hf_sbdp_sa1024_hybridac42_tile512_mb96_1x_log1` | v5p-16 HF PR#123 + SBDP + ChunkedOpt + sa=1024 + HybridAC42 + tile512 (mb=96 = 4,608 tok/exp, 1x) | `FAIL(rc=-9)` | **0.0** | **0.00%** | 0.00% | - | - | 403.8 |
| `v5p_p9e_03_hf_sbdp_sa1024_hybridac42_tile512_mb96_ga2_log1` | v5p-16 HF PR#123 + SBDP + ChunkedOpt + sa=1024 + HybridAC42 + tile512 (mb=96 = 4,608 tok/exp, 2xGA = 6.29M tok/step) | `FAIL(rc=-9)` | **0.0** | **0.00%** | 0.00% | - | - | 296.9 |
| `v5p_p9e_04_hf_sbdp_sa1024_fullac48_tile512_mb112_1x_log1` | v5p-16 HF PR#123 + SBDP + ChunkedOpt + sa=1024 + FullAC48 + tile512 (mb=112 = 5,376 tok/exp, 1x) | `FAIL(rc=-9)` | **0.0** | **0.00%** | 0.00% | - | - | 503.7 |
| `v5p_p9e_05_split7_fsdppipe_staticep16_zero3_paired512_mb48_ga2_log1` | v5p-16 Split-7 fsdp_pipeline + StaticEP=16 (0 device mismatch) + ZeRO-3 + PairedGMM-512 (n36, mb=48 2xGA) | `FAIL(rc=1)` | **0.0** | **0.00%** | 0.00% | - | - | 24.3 |
