# RSI Flint (`RecursiveSI/flint` 40B MoE) Pass 3 Surgical MFU Hillclimb — `v7x-16-2x2x4`

- **Updated (UTC):** `2026-10-30T08:53:33.107591+00:00`
- **Chip / World Size:** `v7x` (`32` ranks, `1153.5` BF16 TFLOP/s/rank, `17.385` GFLOP/token)

| Experiment | Lane | Status | Steady tok/s/rank | Cluster tok/s | Steady MFU (%) | Peak MFU (%) | Loss (start -> end) | Wall Time (s) |
|---|---|---|---:|---:|---:|---:|---|---:|
| `p3_01_native_n49_tile1024_comp_opt_gradnorm` | Pass3 Native (n49 + tile_m=1024 + CompiledOpt + CompiledGradNorm, mb=32) | `PASS` | **11,448.6** | 366,355 | **17.25%** | 17.32% | 12.32117 -> 3.68717 | 299.1 |
| `p3_02_native_n49_tile1024_moe_routing_comp_opt` | Pass3 Native (n49 Attn+MoERouting + tile_m=1024 + CompiledOpt/GradNorm, mb=32) | `PASS` | **11,540.8** | 369,306 | **17.39%** | 17.46% | 12.32117 -> 3.68881 | 252.9 |
| `p3_03_native_n49_tile1024_moe_routing_comp_opt_grp2` | Pass3 Native (n49 Attn+MoERouting + CompiledOpt/GradNorm + 2-Layer Split, mb=32) | `PASS` | **11,405.3** | 364,970 | **17.19%** | 17.27% | 12.32115 -> 3.6884 | 260.2 |
| `p3_04_native_n49_tile1024_moe_routing_in_graph_opt` | Pass3 Native (n49 Attn+MoERouting + InGraph Optimizer, mb=32) | `FAIL(rc=1)` | **0.0** | 0 | **0.00%** | 0.00% | - | 19.0 |
| `p3_05_native_n28_tile1024_moe_routing_comp_opt_mb48` | Pass3 Native (n28 Attn + All49 MoERouting + CompiledOpt, mb=48 / 98K tok/rank) | `PASS` | **12,019.2** | 384,614 | **18.11%** | 18.18% | 12.32154 -> 3.68007 | 384.4 |
| `p3_06_native_fullremat_tile1024_comp_opt_mb64` | Pass3 Native (FullRemat + tile_m=1024 + CompiledOpt, mb=64 / 131K tok/rank) | `PASS` | **11,492.7** | 367,766 | **17.32%** | 17.38% | 12.32045 -> 3.67667 | 385.0 |
| `p3_07_hf_pr123_paired_gmm_b2048_mb32` | Pass3 HF Backend (AttnOnlyAC + Paired W1/W3 Pallas GMM + b=2048, mb=32) | `FAIL(rc=1)` | **0.0** | 0 | **0.00%** | 0.00% | - | 144.3 |
| `p3_08_hf_pr123_paired_gmm_b2048_mb48` | Pass3 HF Backend (AttnOnlyAC + Paired W1/W3 Pallas GMM + b=2048, mb=48 / 98K tok/rank) | `FAIL(rc=1)` | **0.0** | 0 | **0.00%** | 0.00% | - | 157.2 |
