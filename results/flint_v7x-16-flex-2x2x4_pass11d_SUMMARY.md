# Flint (`RecursiveSI/flint` 40B MoE) TPU v7x `PASS11D` (`StaticEP=8/16/32 ZeRO-2` + `WholeStep` + `Offload`) — `v7x-16-flex-2x2x4`

- **Updated (UTC):** `2026-10-31T09:27:11.507584+00:00`
- **Chip / World Size:** `v7x` (`32` TensorCore ranks, `1153.5` BF16 TFLOP/s/rank)

| Experiment | Lane | Status | Steady tok/s/rank | Steady MFU (%) | Peak MFU (%) | Peak HBM (GiB) | Loss (start -> end) | Wall Time (s) |
|---|---|---|---:|---:|---:|---:|---|---:|
| `v7x_p11d_01_split_staticep32_zero2_compgn_paired512_mb48_ga2_log2` | v7x-16 GraphSplit + StaticEP=32 + ZeRO-2 + EPCompiledGradNorm + SplitMeshOpt + PairedGMM-512 (0 expert AG/RS, 73,728 tok/exp, n45, mb=48 2xGA, log2) | `FAIL(rc=-9)` | **752.0** | **1.13%** | 1.13% | 25.88 | 12.32026 -> 12.32026 | 297.8 |
| `v7x_p11d_02_split_staticep32_zero2_compgn_paired512_mb64_1x_log1` | v7x-16 GraphSplit + StaticEP=32 + ZeRO-2 + EPCompiledGradNorm + SplitMeshOpt + PairedGMM-512 (98,304 tok/exp, n36, mb=64 1x, log1) | `FAIL(rc=1)` | **0.0** | **0.00%** | 0.00% | - | - | 287.7 |
| `v7x_p11d_03_split_staticep16_zero2_compgn_paired256_mb48_ga2_log1` | v7x-16 GraphSplit + StaticEP=16 + ZeRO-2 + EPCompiledGradNorm + SplitMeshOpt + PairedGMM-256 (36,864 tok/exp, n40, mb=48 2xGA, log1) | `PASS` | **3,760.7** | **5.67%** | 5.69% | 24.48 | 12.32101 -> 4.21118 | 852.1 |
| `v7x_p11d_04_split_staticep8_zero3_compgn_paired256_mb48_ga2_log1` | v7x-16 GraphSplit + Intra-Host StaticEP=8 + ZeRO-3 + EPCompiledGradNorm + SplitMeshOpt + PairedGMM-256 (100% intra-host A2A, n36, mb=48 2xGA, log1) | `PASS` | **4,463.0** | **6.73%** | 6.81% | 22.69 | 12.32147 -> 4.19128 | 771.5 |
| `v7x_p11d_05_wholestep_staticep32_zero2_compgn_paired512_mb32_ga2_log2` | v7x-16 WholeStep Single-Graph + StaticEP=32 + ZeRO-2 + EPCompiledGradNorm + SplitMeshOpt + PairedGMM-512 (n45, 49,152 tok/exp, mb=32 2xGA, log2) | `PASS` | **3,118.5** | **4.70%** | 4.72% | 55.84 | 12.32057 -> 4.18993 | 1343.4 |
| `v7x_p11d_06_wholestep_staticep32_zero2_ingraph_paired512_mb48_1x_log1` | v7x-16 WholeStep + in_graph Fused AdamW + StaticEP=32 + ZeRO-2 + PairedGMM-512 (n36, 73,728 tok/exp, mb=48 1x, log1) | `FAIL(rc=1)` | **0.0** | **0.00%** | 0.00% | - | - | 47.9 |
