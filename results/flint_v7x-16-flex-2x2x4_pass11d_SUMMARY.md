# Flint (`RecursiveSI/flint` 40B MoE) TPU v7x `PASS11D` (`StaticEP=8/16/32 ZeRO-2` + `WholeStep` + `Offload`) — `v7x-16-flex-2x2x4`

- **Updated (UTC):** `2026-10-31T08:31:23.604858+00:00`
- **Chip / World Size:** `v7x` (`32` TensorCore ranks, `1153.5` BF16 TFLOP/s/rank)

| Experiment | Lane | Status | Steady tok/s/rank | Steady MFU (%) | Peak MFU (%) | Peak HBM (GiB) | Loss (start -> end) | Wall Time (s) |
|---|---|---|---:|---:|---:|---:|---|---:|
| `v7x_p11d_01_split_staticep32_zero2_compgn_paired512_mb48_ga2_log2` | v7x-16 GraphSplit + StaticEP=32 + ZeRO-2 + EPCompiledGradNorm + SplitMeshOpt + PairedGMM-512 (0 expert AG/RS, 73,728 tok/exp, n45, mb=48 2xGA, log2) | `FAIL(rc=-9)` | **752.0** | **1.13%** | 1.13% | 25.88 | 12.32026 -> 12.32026 | 297.8 |
