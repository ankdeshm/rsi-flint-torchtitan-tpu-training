# Flint (`RecursiveSI/flint` 40B MoE) Pass 5 Final Hillclimb — `v7x-16-2x2x4`

- **Updated (UTC):** `2026-10-30T09:37:11.566984+00:00`
- **Chip / World Size:** `v7x` (`32` ranks, `1153.5` BF16 TFLOP/s/rank, `17.385` GFLOP/token)

| Experiment | Lane | Status | Steady tok/s/rank | Cluster tok/s | Steady MFU (%) | Peak MFU (%) | Loss (start -> end) | Wall Time (s) |
|---|---|---|---:|---:|---:|---:|---|---:|
| `p5_01_native_n49_moe_routing_comp_opt_mb48` | Pass5 Native (ALL49 Attn + ALL49 MoERouting + CompiledOpt, mb=48 / 98K tok/rank) | `FAIL(rc=1)` | **511.0** | 16,352 | **0.77%** | 0.77% | 12.32154 -> 12.32154 | 216.9 |
| `p5_02_native_n43_moe_routing_comp_opt_tgmm1024_mb48` | Pass5 Native (n43 Attn + ALL49 MoERouting + CompiledOpt + BwdTGMM1024, mb=48) | `FAIL(rc=1)` | **0.0** | 0 | **0.00%** | 0.00% | - | 182.7 |
| `p5_03_native_n49_moe_routing_comp_opt_tgmm1024_mb44` | Pass5 Native (ALL49 Attn + ALL49 MoERouting + CompiledOpt + BwdTGMM1024, mb=44) | `FAIL(rc=1)` | **0.0** | 0 | **0.00%** | 0.00% | - | 197.9 |
| `p5_04_hf_pr123_paired_gmm_b1024_mb32` | Pass5 HF Backend PR #123 (Real Flint + AttnOnlyAC + Paired W1/W3 Pallas GMM + b=1024, mb=32) | `PASS` | **11,728.6** | 375,315 | **17.68%** | 17.72% | 12.31149 -> 3.44342 | 237.1 |
