# Flint (`RecursiveSI/flint` 40B MoE) TPU v7x `PASS10D` (`EP=8/16/32 ZeRO-2` + `WholeStep` + `Scan`) — `v7x-16-flex-2x2x4`

- **Updated (UTC):** `2026-10-31T07:14:41.034871+00:00`
- **Chip / World Size:** `v7x` (`32` TensorCore ranks, `1153.5` BF16 TFLOP/s/rank)

| Experiment | Lane | Status | Steady tok/s/rank | Steady MFU (%) | Peak MFU (%) | Peak HBM (GiB) | Loss (start -> end) | Wall Time (s) |
|---|---|---|---:|---:|---:|---:|---|---:|
| `v7x_p10d_01_wholestep_ep32_zero2_ingraph_opt_mb48_1x` | v7x-16 WholeStep Graph + EP=32 + ZeRO-2 + in_graph AdamW (73,728 tok/exp, n36, mb=48 1x, log_freq=5) | `FAIL(rc=1)` | **0.0** | **0.00%** | 0.00% | - | - | 22.6 |
| `v7x_p10d_02_wholestep_ep8_zero2_rematbar_mb32_ga2` | v7x-16 WholeStep Graph + Intra-Host EP=8 + ZeRO-2 (100% intra-host AllToAll, 12,288 tok/exp, n32, mb=32 2xGA) | `FAIL(rc=1)` | **0.0** | **0.00%** | 0.00% | - | - | 20.5 |
| `v7x_p10d_03_split_ep32_zero2_n28_mb64_1x` | v7x-16 GraphSplit + EP=32 + ZeRO-2 (98,304 tok/exp, n28, mb=64 = 4.19M tok/step 1x, log_freq=5) | `FAIL(rc=1)` | **0.0** | **0.00%** | 0.00% | - | - | 22.3 |
| `v7x_p10d_04_scanned_layers_q235_flags_mb64_1x` | v7x-16 ScannedTpuFlintModel (use_scan_layers + full_and_scan_offload + Q235 scan flags + SBDP, mb=64 1x) | `FAIL(rc=1)` | **0.0** | **0.00%** | 0.00% | - | - | 18.3 |
| `v7x_p10d_05_ep16_fulloffload_mb96_1x` | v7x-16 GraphSplit + EP=16 + Full Host Offload (save_last_n=0, prefetch=2, 73,728 tok/exp, mb=96 = 6.29M tok/step 1x) | `FAIL(rc=1)` | **0.0** | **0.00%** | 0.00% | - | - | 22.3 |
