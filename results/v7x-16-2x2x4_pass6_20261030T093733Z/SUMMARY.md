# RSI Flint (`RecursiveSI/flint` 40B MoE) Pass 6 Real HF Checkpoint & n49@mb44 — `v7x-16-2x2x4`

- **Updated (UTC):** `2026-10-30T09:49:15.420746+00:00`
- **Chip / World Size:** `v7x` (`32` ranks, `1153.5` BF16 TFLOP/s/rank, `17.385` GFLOP/token)

| Experiment | Lane | Status | Steady tok/s/rank | Cluster tok/s | Steady MFU (%) | Peak MFU (%) | Loss (start -> end) | Wall Time (s) |
|---|---|---|---:|---:|---:|---:|---|---:|
| `p6_01_native_n49_moe_routing_comp_opt_mb44` | Pass6 Native (ALL49 Attn + ALL49 MoERouting + CompiledOpt + VMEM-safe TGMM, mb=44 / 90K tok/rank) | `PASS` | **11,774.8** | 376,794 | **17.75%** | 17.82% | 12.32128 -> 3.6835 | 321.5 |
| `p6_02_native_real_flint_hf_checkpoint_30steps_mb48` | Pass6 Native Real RecursiveSI/flint 15-Shard HF Safetensors Fine-Tuning (30 steps, n43+MoERouting+CompiledOpt, mb=48) | `FAIL(rc=1)` | **286.0** | 9,152 | **0.43%** | 0.43% | 3.43601 -> 3.43601 | 367.3 |
