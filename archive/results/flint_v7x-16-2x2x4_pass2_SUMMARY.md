# Flint (`RecursiveSI/flint` 40B MoE) Pass 2 MFU Hillclimb — `v7x-16-2x2x4`

- **Updated (UTC):** `2026-10-30T08:19:57.298482+00:00`
- **Chip / World Size:** `v7x` (`32` ranks, `1153.5` BF16 TFLOP/s/rank, `17.385` GFLOP/token)

| Experiment | Lane | Status | Steady tok/s/rank | Steady MFU (%) | Peak MFU (%) | Peak HBM (GiB) | Loss (start -> end) | Wall Time (s) |
|---|---|---|---:|---:|---:|---:|---|---:|
| `p2_01_hf_no_ac_mb32_b2048` | Pass2 HF Backend (NoAC + b=2048 + mb=32) | `FAIL(rc=1)` | **0.0** | **0.00%** | 0.00% | - | - | 126.9 |
| `p2_02_hf_selective_ac_mb64_b2048` | Pass2 HF Backend (SelectiveAC + b=2048 + mb=64 / 131K tok/rank) | `FAIL(rc=1)` | **0.0** | **0.00%** | 0.00% | - | - | 194.9 |
| `p2_03_hf_no_ac_mb64_b2048` | Pass2 HF Backend (NoAC + b=2048 + mb=64 / 131K tok/rank) | `FAIL(rc=1)` | **0.0** | **0.00%** | 0.00% | - | - | 172.6 |
| `p2_04_hf_attnac_mb96_b2048` | Pass2 HF Backend (AttnOnlyAC + b=2048 + mb=96 / 196K tok/rank) | `FAIL(rc=1)` | **0.0** | **0.00%** | 0.00% | - | - | 155.7 |
| `p2_05_native_no_remat_mb32_grp2` | Pass2 Native GraphTrainer (Zero Remat + CompiledOpt + 2-Layer Split, mb=32) | `FAIL(rc=1)` | **0.0** | **0.00%** | 0.00% | - | - | 205.4 |
| `p2_06_native_opt_moe_route_mb64_grp2` | Pass2 Native GraphTrainer (Save Attn+MoERouting + CompiledOpt + 2-Layer Split, mb=64) | `FAIL(rc=1)` | **0.0** | **0.00%** | 0.00% | - | - | 214.9 |
| `p2_07_native_no_remat_mb64_grp2` | Pass2 Native GraphTrainer (Zero Remat + CompiledOpt + 2-Layer Split, mb=64) | `FAIL(rc=1)` | **0.0** | **0.00%** | 0.00% | - | - | 212.2 |
