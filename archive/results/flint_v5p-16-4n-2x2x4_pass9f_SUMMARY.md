# Flint (`RecursiveSI/flint` 40B MoE) TPU v5p `PASS9F` (`mb=96..160` + `StaticEP=4/8/16 ZeRO-2`) — `v5p-16-4n-2x2x4`

- **Updated (UTC):** `2026-10-31T16:35:26.507282+00:00`
- **Chip / World Size:** `v5p` (`16` chips/ranks, `459.0` BF16 TFLOP/s/rank)

| Experiment | Lane | Status | Steady tok/s/rank | Steady MFU (%) | Peak MFU (%) | Peak HBM (GiB) | Loss (start -> end) | Wall Time (s) |
|---|---|---|---:|---:|---:|---:|---|---:|
| `v5p_p9f_01_hf_sbdp_sa1024_attnac0_mb64_1x` | v5p-16 HF PR#123 + SBDP + PureFusedOpt + sa=1024 + AttnAC(N=0, 0% MoE/Attn recomp, 1 exec) + NativeTile1024 (mb=64 = 3,072 tok/exp) | `FAIL(rc=-9)` | **0.0** | **0.00%** | 0.00% | - | - | 254.9 |
| `v5p_p9f_02_hf_sbdp_sa1024_attnac0_mb72_1x` | v5p-16 HF PR#123 + SBDP + PureFusedOpt + sa=1024 + AttnAC(N=0, 0% MoE/Attn recomp, 1 exec) + NativeTile1024 (mb=72 = 3,456 tok/exp) | `FAIL(rc=-9)` | **0.0** | **0.00%** | 0.00% | - | - | 248.7 |
| `v5p_p9f_03_hf_sbdp_sa1024_attnac0_mb80_1x` | v5p-16 HF PR#123 + SBDP + PureFusedOpt + sa=1024 + AttnAC(N=0, 0% MoE/Attn recomp, 1 exec) + NativeTile1024 (mb=80 = 3,840 tok/exp) | `FAIL(rc=-9)` | **0.0** | **0.00%** | 0.00% | - | - | 282.6 |
| `v5p_p9f_04_hf_sbdp_sa1024_fullac48_mb96_1x` | v5p-16 HF PR#123 + SBDP + PureFusedOpt + sa=1024 + FullAC48(1 exec, 33.9G headroom) + NativeTile1024 (mb=96 = 4,608 tok/exp) | `FAIL(rc=1)` | **452.0** | **1.71%** | 1.71% | 34.43 | 12.49542 -> 12.49542 | 464.1 |
| `v5p_p9f_05_native_fp8dense_sbdp_mb64_1x` | v5p-16 Native TpuGraphTrainer + Rowwise FP8 Dense + SBDP + PureFusedOpt + NativeTile1024 (mb=64, 1x) | `FAIL(rc=1)` | **0.0** | **0.00%** | 0.00% | - | - | 87.5 |
| `v5p_p9f_06_hf_fp8dense_sbdp_sa1024_attnac0_mb64_1x` | v5p-16 HF PR#123 + Rowwise FP8 Dense + SBDP + PureFusedOpt + sa=1024 + AttnAC(N=0) (mb=64, 1x) | `FAIL(rc=1)` | **0.0** | **0.00%** | 0.00% | - | - | 70.6 |
