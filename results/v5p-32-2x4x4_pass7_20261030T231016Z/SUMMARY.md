# Flint (`RecursiveSI/flint` 40B MoE) v5p-32 Pass 7 Results — `v5p-32-2x4x4_pass7`

- **Updated (UTC):** `2026-10-30T23:49:28.102964+00:00`
- **Chip / World Size:** `v5p` (`32` ranks, `459.0` BF16 TFLOP/s/rank)

| Experiment | Lane | Status | Steady tok/s/rank | Steady MFU (%) | Peak MFU (%) | Peak HBM (GiB) | Loss (start -> end) | Wall Time (s) |
|---|---|---|---:|---:|---:|---:|---|---:|
| `v5p_p7_01_hf_pr123_fsdp32_attnac_compopt_mb48` | v5p-32 FSDP-32 HF PR#123 (AttnOnlyAC + DEFER_AND_FUSE + Compiled AdamW/GradNorm, mb=48 = 3.15M tok/step) | `PASS` | **5,921.3** | **22.43%** | 22.51% | 16.91 | 12.29472 -> 4.39088 | 445.4 |
| `v5p_p7_02_hf_pr123_fsdp32_hybridac22_defer_mb64` | v5p-32 FSDP-32 HF PR#123 (Compiler-Native HybridAC N=22 + DEFER_AND_FUSE, mb=64 = 4.19M tok/step) | `PASS` | **2,096.4** | **7.94%** | 8.00% | 27.07 | 12.29461 -> 4.41966 | 980.1 |
| `v5p_p7_03_hf_pr123_fsdp32_hybridac24_compopt_mb64` | v5p-32 FSDP-32 HF PR#123 (HybridAC N=24 + DEFER_AND_FUSE + Compiled AdamW/GradNorm, mb=64 = 4.19M tok/step) | `PASS` | **2,097.9** | **7.95%** | 8.00% | 27.07 | 12.29461 -> 4.4255 | 904.0 |
