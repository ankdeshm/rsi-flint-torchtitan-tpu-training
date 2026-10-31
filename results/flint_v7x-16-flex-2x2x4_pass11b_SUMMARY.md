# Flint (`RecursiveSI/flint` 40B MoE) TPU v7x `PASS11B` (`StaticEP=8/16/32 ZeRO-2` + `WholeStep` + `Offload`) — `v7x-16-flex-2x2x4`

- **Updated (UTC):** `2026-10-31T07:51:41.219612+00:00`
- **Chip / World Size:** `v7x` (`32` TensorCore ranks, `1153.5` BF16 TFLOP/s/rank)

| Experiment | Lane | Status | Steady tok/s/rank | Steady MFU (%) | Peak MFU (%) | Peak HBM (GiB) | Loss (start -> end) | Wall Time (s) |
|---|---|---|---:|---:|---:|---:|---|---:|
| `v7x_p11b_01_wholestep_staticep32_zero2_paired512_mb32_1x` | v7x-16 WholeStep Single-Graph + StaticEP=32 + ZeRO-2 + PairedGMM-512 (n49, 49,152 tok/exp, mb=32 1x) | `FAIL(rc=1)` | **0.0** | **0.00%** | 0.00% | - | - | 18.2 |
| `v7x_p11b_02_wholestep_staticep8_zero2_paired256_mb32_ga2` | v7x-16 WholeStep Single-Graph + Intra-Host StaticEP=8 + ZeRO-2 + PairedGMM-256 (n36, mb=32 2xGA) | `FAIL(rc=1)` | **0.0** | **0.00%** | 0.00% | - | - | 18.2 |
| `v7x_p11b_03_wholestep_staticep32_zero2_ingraph_mb48_1x` | v7x-16 WholeStep + in_graph Fused AdamW + StaticEP=32 + ZeRO-2 (n36, 73,728 tok/exp, mb=48 1x) | `FAIL(rc=1)` | **0.0** | **0.00%** | 0.00% | - | - | 17.1 |
| `v7x_p11b_04_ep1_sbdp_fulloffload_mb96_ga2` | v7x-16 EP=1 SBDP + PairedGMM-256 + Full Host Offload (`save_last_n=0, prefetch=2`, mb=96 = 12.58M tok/step 2xGA) | `PASS` | **12,037.5** | **18.14%** | 18.16% | 53.43 | 12.32021 -> 3.67903 | 759.2 |
| `v7x_p11b_05_staticep16_fulloffload_mb96_1x` | v7x-16 GraphSplit + StaticEP=16 + Full Host Offload (`save_last_n=0, prefetch=2`, 73,728 tok/exp, mb=96 1x) | `FAIL(rc=1)` | **0.0** | **0.00%** | 0.00% | - | - | 17.7 |
