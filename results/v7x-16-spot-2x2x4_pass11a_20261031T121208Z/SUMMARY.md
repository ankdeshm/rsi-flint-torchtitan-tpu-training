# Flint (`RecursiveSI/flint` 40B MoE) TPU v7x `PASS11A` (`StaticEP=8/16/32 ZeRO-2` + `WholeStep` + `Offload`) — `v7x-16-spot-2x2x4`

- **Updated (UTC):** `2026-10-31T12:40:03.364913+00:00`
- **Chip / World Size:** `v7x` (`32` TensorCore ranks, `1153.5` BF16 TFLOP/s/rank)

| Experiment | Lane | Status | Steady tok/s/rank | Steady MFU (%) | Peak MFU (%) | Peak HBM (GiB) | Loss (start -> end) | Wall Time (s) |
|---|---|---|---:|---:|---:|---:|---|---:|
| `v7x_p11a_01_split_staticep8_zero2_paired256_mb48_ga2` | v7x-16 GraphSplit + Intra-Host StaticEP=8 + ZeRO-2 + PairedGMM-256 (100% intra-host A2A, 18,432 tok/exp, mb=48 2xGA) | `FAIL(rc=-9)` | **719.0** | **1.08%** | 1.08% | 24.45 | 12.32147 -> 12.32147 | 311.4 |
| `v7x_p11a_02_split_staticep16_zero2_paired256_mb48_ga2` | v7x-16 GraphSplit + StaticEP=16 + ZeRO-2 + PairedGMM-256 (36,864 tok/exp, 100.6MB/layer, n40, mb=48 2xGA) | `PASS` | **3,759.5** | **5.67%** | 5.70% | 24.48 | 12.32101 -> 3.69192 | 1003.6 |
| `v7x_p11a_03_split_staticep32_zero2_paired512_mb48_ga2` | v7x-16 GraphSplit + StaticEP=32 + ZeRO-2 + PairedGMM-512 (0 expert AG/RS, 50.3MB/layer, 73,728 tok/exp, n45, mb=48 2xGA) | `FAIL(rc=1)` | **0.0** | **0.00%** | 0.00% | - | - | 20.0 |
| `v7x_p11a_04_split_staticep32_zero2_paired512_mb64_1x` | v7x-16 GraphSplit + StaticEP=32 + ZeRO-2 + PairedGMM-512 (98,304 tok/exp, n36, mb=64 = 4.19M tok/step 1x) | `FAIL(rc=1)` | **0.0** | **0.00%** | 0.00% | - | - | 284.1 |
| `v7x_p11a_05_ep1_sbdp_paired256_libtpuflags_log5_mb48_ga2` | v7x-16 EP=1 SBDP + PairedGMM-256 + Direct config.libtpu_init_flags + log_freq=5 (n36, mb=48 2xGA) | `FAIL(rc=1)` | **0.0** | **0.00%** | 0.00% | - | - | 18.5 |
