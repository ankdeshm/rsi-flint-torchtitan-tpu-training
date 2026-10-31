# Flint (`RecursiveSI/flint` 40B MoE) v5p-32 Pass 8 (`Llama4-Scout` XLA + BF16 Reduce + 2xGA) — `v5p-32-2x4x4_pass8`

- **Updated (UTC):** `2026-10-31T03:13:17.287153+00:00`
- **Chip / World Size:** `v5p` (`32` ranks, `459.0` BF16 TFLOP/s/rank)

| Experiment | Lane | Status | Steady tok/s/rank | Steady MFU (%) | Peak MFU (%) | Peak HBM (GiB) | Loss (start -> end) | Wall Time (s) |
|---|---|---|---:|---:|---:|---:|---|---:|
| `v5p_p8_01_hf_scout_xla_attnac_mb48` | v5p-32 FSDP-32 HF PR#123 + Llama4-Scout v5p XLA LHS/AG-Bwd-Pipe + VMEM=80MiB (AttnOnlyAC, mb=48 = 3.15M tok/step) | `PASS` | **6,147.5** | **23.28%** | 23.42% | 16.92 | 12.29472 -> 4.41749 | 441.0 |
| `v5p_p8_02_hf_scout_xla_bf16red_mb48` | v5p-32 FSDP-32 HF PR#123 + Llama4-Scout XLA + BF16 Grad Reduce (AttnOnlyAC, mb=48 = 3.15M tok/step) | `PASS` | **6,565.3** | **24.87%** | 24.95% | 16.92 | 12.29472 -> 4.40614 | 344.2 |
| `v5p_p8_03_hf_scout_xla_bf16red_2xga_mb48` | v5p-32 FSDP-32 HF PR#123 + Llama4-Scout XLA + BF16 Reduce + 2x GradAccum (6.29M tok/step) | `FAIL(rc=1)` | **1,734.5** | **6.57%** | 6.58% | 16.92 | 12.29451 -> 12.29451 | 153.4 |
| `v5p_p8_04_hf_scout_xla_bf16red_hybridac22_2xga_mb64` | v5p-32 FSDP-32 HF PR#123 + Llama4-Scout XLA + BF16 Reduce + HybridAC N=22 + 2x GradAccum (mb=64 = 8.39M tok/step) | `FAIL(rc=-9)` | **1,950.1** | **7.39%** | 7.46% | 27.09 | 12.29473 -> 4.38613 | 1848.0 |
| `v5p_p8_05_native_pr122_scout_xla_bf16red_mb48` | v5p-32 Native TpuGraphTrainer PR#122 (flint_40b_v5p_n40) + Llama4-Scout XLA + BF16 Reduce (mb=48 = 3.15M tok/step) | `PASS` | **5,840.4** | **22.12%** | 22.26% | 28.11 | 12.32875 -> 4.17352 | 570.5 |
| `v5p_p8_06_hf_hsdp2x16_scout_xla_bf16red_2xga_mb48` | v5p-32 HSDP 2x FSDP-16 + Llama4-Scout XLA + BF16 Reduce + HybridAC N=28 + 2x GradAccum (6.29M tok/step) | `PASS` | **6,285.7** | **23.81%** | 23.83% | 21.08 | 12.49579 -> 3.77894 | 632.1 |
