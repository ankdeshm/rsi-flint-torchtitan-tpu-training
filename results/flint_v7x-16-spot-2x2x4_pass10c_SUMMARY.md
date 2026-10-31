# Flint (`RecursiveSI/flint` 40B MoE) TPU v7x `PASS10C` (`EP=8/16/32 ZeRO-2` + `WholeStep` + `Scan`) — `v7x-16-spot-2x2x4`

- **Updated (UTC):** `2026-10-31T07:15:18.302942+00:00`
- **Chip / World Size:** `v7x` (`32` TensorCore ranks, `1153.5` BF16 TFLOP/s/rank)

| Experiment | Lane | Status | Steady tok/s/rank | Steady MFU (%) | Peak MFU (%) | Peak HBM (GiB) | Loss (start -> end) | Wall Time (s) |
|---|---|---|---:|---:|---:|---:|---|---:|
| `v7x_p10c_01_wholestep_ep32_zero2_rematbar_mb32_1x` | v7x-16 WholeStep Graph + EP=32 + ZeRO-2 (0 expert AG/RS, 50.3MB/layer, 49,152 tok/exp, n49, mb=32 1x, log_freq=5) | `FAIL(rc=1)` | **0.0** | **0.00%** | 0.00% | - | - | 22.0 |
| `v7x_p10c_02_wholestep_ep16_zero2_rematbar_mb32_ga2` | v7x-16 WholeStep Graph + EP=16 + ZeRO-2 (100.6MB/layer, 24,576 tok/exp, n40, mb=32 2xGA, log_freq=5) | `FAIL(rc=1)` | **0.0** | **0.00%** | 0.00% | - | - | 21.9 |
| `v7x_p10c_03_split_ep32_zero2_n40_mb48_ga2` | v7x-16 GraphSplit + EP=32 + ZeRO-2 (0 expert AG/RS, 73,728 tok/exp, n40, mb=48 2xGA, log_freq=5) | `FAIL(rc=1)` | **0.0** | **0.00%** | 0.00% | - | - | 40.3 |
| `v7x_p10c_04_split_ep16_zero2_n36_mb48_ga2` | v7x-16 GraphSplit + EP=16 + ZeRO-2 (36,864 tok/exp, n36, mb=48 2xGA, log_freq=5) | `FAIL(rc=1)` | **0.0** | **0.00%** | 0.00% | - | - | 21.5 |
| `v7x_p10c_05_split_ep8_zero2_n36_mb48_ga2` | v7x-16 GraphSplit + Intra-Host EP=8 + ZeRO-2 (100% intra-host AllToAll, 18,432 tok/exp, n36, mb=48 2xGA) | `FAIL(rc=1)` | **0.0** | **0.00%** | 0.00% | - | - | 40.6 |
