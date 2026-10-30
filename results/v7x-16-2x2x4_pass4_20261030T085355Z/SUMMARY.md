# Flint (`RecursiveSI/flint` 40B MoE) Pass 4 Peak-MFU + Real Checkpoint — `v7x-16-2x2x4`

- **Updated (UTC):** `2026-10-30T09:22:16.513888+00:00`
- **Chip / World Size:** `v7x` (`32` ranks, `1153.5` BF16 TFLOP/s/rank, `17.385` GFLOP/token)

| Experiment | Lane | Status | Steady tok/s/rank | Cluster tok/s | Steady MFU (%) | Peak MFU (%) | Loss (start -> end) | Wall Time (s) |
|---|---|---|---:|---:|---:|---:|---|---:|
| `p4_01_native_n43_tile1024_moe_routing_comp_opt_mb48` | Pass4 Native (n43 Attn + All49 MoERouting + CompiledOpt, mb=48 / 98K tok/rank) | `PASS` | **12,040.9** | 385,309 | **18.15%** | 18.21% | 12.32154 -> 3.67991 | 398.9 |
| `p4_02_native_n45_tile1024_comp_opt_mb48` | Pass4 Native (n45 Attn + CompiledOpt, mb=48 / 98K tok/rank) | `PASS` | **11,997.4** | 383,917 | **18.08%** | 18.15% | 12.32154 -> 3.67985 | 367.8 |
| `p4_03_native_n22_tile1024_moe_routing_comp_opt_mb56` | Pass4 Native (n22 Attn + All49 MoERouting + CompiledOpt, mb=56 / 115K tok/rank) | `PASS` | **11,776.8** | 376,858 | **17.75%** | 17.88% | 12.32045 -> 3.68118 | 421.5 |
| `p4_04_native_n14_tile1024_moe_routing_comp_opt_mb64` | Pass4 Native (n14 Attn + All49 MoERouting + CompiledOpt, mb=64 / 131K tok/rank) | `PASS` | **11,743.5** | 375,792 | **17.70%** | 17.75% | 12.32045 -> 3.67648 | 450.1 |
| `p4_05_native_real_flint_hf_checkpoint_30steps_mb48` | Pass4 Native Real RecursiveSI/flint HF Weights + Tokenizer (30 steps, n38+MoERouting, mb=48) | `FAIL(rc=1)` | **0.0** | 0 | **0.00%** | 0.00% | - | 10.9 |
