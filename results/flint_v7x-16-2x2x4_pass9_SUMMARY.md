# Flint (`RecursiveSI/flint` 40B MoE) TPU v7x Pass 9 (`BF16 Reduce` + `Scout/HierSC XLA`) — `v7x-16-2x2x4`

- **Updated (UTC):** `2026-10-31T03:05:01.716401+00:00`
- **Chip / World Size:** `v7x` (`32` TensorCore ranks, `1153.5` BF16 TFLOP/s/rank)

| Experiment | Lane | Status | Steady tok/s/rank | Steady MFU (%) | Peak MFU (%) | Peak HBM (GiB) | Loss (start -> end) | Wall Time (s) |
|---|---|---|---:|---:|---:|---:|---|---:|
| `v7x_p9_01_native_pr122_bf16red_scout_n45_mb48_1x` | v7x-16 Native TpuGraphTrainer PR#122 (n45 + All49 MoERouting + BF16 Reduce + Scout/HierSC XLA, mb=48 = 3.15M tok/step) | `PASS` | **12,297.1** | **18.53%** | 18.62% | 26.23 | 12.32154 -> 4.17208 | 378.9 |
| `v7x_p9_02_native_pr122_bf16red_scout_n43_mb48_ga2` | v7x-16 Native TpuGraphTrainer PR#122 (2x GradAccum @ mb=48 = 6.29M tok/step + n43 + BF16 Reduce + Scout/HierSC XLA) | `PASS` | **12,392.8** | **18.68%** | 18.76% | 27.12 | 12.3201 -> 4.16058 | 295.9 |
| `v7x_p9_03_hf_pr123_bf16red_scout_compopt_mb48_1x` | v7x-16 FSDP-32 HF PR#123 (Paired GMM tile_m=1024 + BF16 Reduce + Scout/HierSC XLA + CompiledOpt, mb=48 = 3.15M tok/step) | `FAIL(rc=1)` | **0.0** | **0.00%** | 0.00% | - | - | 10.4 |
| `v7x_p9_04_native_pr122_bf16red_scout_n22_mb56_ga2` | v7x-16 Native TpuGraphTrainer PR#122 (2x GradAccum @ mb=56 = 7.34M tok/step + n22 + BF16 Reduce + Scout/HierSC XLA) | `PASS` | **11,804.6** | **17.79%** | 17.88% | 29.38 | 12.32083 -> 4.16494 | 503.6 |
| `v7x_p9_05_hf_pr123_hsdp2x16_bf16red_hybridac28_mb48_1x` | v7x-16 HSDP 2x FSDP-16 (8-Chip / 16-TC Shards, HybridAC v3 N=28 + BF16 Reduce + Scout/HierSC XLA, mb=48 = 3.15M tok/step) | `FAIL(rc=1)` | **0.0** | **0.00%** | 0.00% | - | - | 10.2 |
