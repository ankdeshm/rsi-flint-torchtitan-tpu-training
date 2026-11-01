# Flint (`RecursiveSI/flint` 40B MoE) TPU v5p `PASS9G` (`mb=96..160` + `StaticEP=4/8/16 ZeRO-2`) — `v5p-16-4n-2x2x4`

- **Updated (UTC):** `2026-10-31T17:44:39.361028+00:00`
- **Chip / World Size:** `v5p` (`16` chips/ranks, `459.0` BF16 TFLOP/s/rank)

| Experiment | Lane | Status | Steady tok/s/rank | Steady MFU (%) | Peak MFU (%) | Peak HBM (GiB) | Loss (start -> end) | Wall Time (s) |
|---|---|---|---:|---:|---:|---:|---|---:|
| `v5p_p9g_01_hf_sbdp_sa1024_fullac48_microflush_mb80_1x` | v5p-16 HF PR#123 + SBDP + MicroFlushOpt24 + sa=1024 + FullAC48(1 exec) + NativeTile1024 (mb=80 = 3,840 tok/exp) | `FAIL(rc=1)` | **414.0** | **1.57%** | 1.57% | 28.71 | 12.49534 -> 12.49534 | 426.8 |
| `v5p_p9g_02_hf_sbdp_sa1024_fullac48_microflush_mb96_1x` | v5p-16 HF PR#123 + SBDP + MicroFlushOpt24 + sa=1024 + FullAC48(1 exec) + NativeTile1024 (mb=96 = 4,608 tok/exp) | `FAIL(rc=1)` | **449.0** | **1.70%** | 1.70% | 34.43 | 12.49542 -> 12.49542 | 467.5 |
| `v5p_p9g_03_hf_sbdp_sa1024_fullac48_microflush_mb112_1x` | v5p-16 HF PR#123 + SBDP + MicroFlushOpt24 + sa=1024 + FullAC48(1 exec) + NativeTile1024 (mb=112 = 5,376 tok/exp) | `FAIL(rc=-9)` | **0.0** | **0.00%** | 0.00% | - | - | 503.6 |
| `v5p_p9g_04_hf_sbdp_sa1024_hybridac24_microflush_mb80_1x` | v5p-16 HF PR#123 + SBDP + MicroFlushOpt24 + sa=1024 + HybridAC24(50% zero recomp) + NativeTile1024 (mb=80 = 3,840 tok/exp) | `FAIL(rc=1)` | **733.0** | **2.78%** | 2.78% | 28.71 | 12.49534 -> 12.49534 | 255.3 |
| `v5p_p9g_05_native_fp8dense_sbdp_microflush_mb64_ga2` | v5p-16 Native TpuGraphTrainer + Rowwise FP8 Dense(use_fast_accum=False) + SBDP + MicroFlushOpt24 (mb=64, 2xGA) | `FAIL(rc=-9)` | **0.0** | **0.00%** | 0.00% | - | - | 1205.9 |
| `v5p_p9g_06_hf_fp8dense_sbdp_sa1024_fullac48_microflush_mb96_1x` | v5p-16 HF PR#123 + Rowwise FP8 Dense(use_fast_accum=False) + SBDP + MicroFlushOpt24 + sa=1024 + FullAC48 (mb=96, 1x) | `FAIL(rc=-9)` | **0.0** | **0.00%** | 0.00% | - | - | 567.6 |
