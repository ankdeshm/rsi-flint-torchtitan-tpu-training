# RSI Flint (`RecursiveSI/flint` 40B MoE) TorchTPU 0.2.0 GA Training Results — `v7x-16-2x2x4`

- **Updated (UTC):** `2026-10-30T07:56:54.886402+00:00`
- **Chip / World Size:** `v7x` (`32` ranks, `1153.5` BF16 TFLOP/s/rank, `17.385` GFLOP/token)

| Experiment | Lane | Status | Steady tok/s/rank | Steady MFU (%) | Peak MFU (%) | Peak HBM (GiB) | Loss (start -> end) | Wall Time (s) |
|---|---|---|---:|---:|---:|---:|---|---:|
| `01_hf_backend_pr123_flint_40b` | PR#123 HF Backend | `PASS` | **10,532.0** | **15.87%** | 15.91% | 14.31 | 12.3115 -> 3.43581 | 248.7 |
| `02_native_pr122_v011_n40_baseline` | PR#122 Native Flint | `PASS` | **10,985.3** | **16.56%** | 16.69% | 15.74 | 12.32117 -> 3.68596 | 354.0 |
| `03_native_v012_hier_sparsecore_rs` | PR#122 Hillclimb 1 (Hier ND RS) | `PASS` | **11,011.9** | **16.60%** | 16.70% | 15.74 | 12.32117 -> 3.68596 | 353.4 |
| `04_native_v013_save_all_49_layers` | PR#122 Hillclimb 2 (Save All 49 Attn + Hier RS) | `PASS` | **11,135.8** | **16.78%** | 16.86% | 14.63 | 12.32117 -> 3.68665 | 190.8 |
| `05_native_v014_n49_tile_m_1024` | PR#122 Hillclimb 3 (n49 + tile_m=1024 + Hier RS) | `PASS` | **11,196.2** | **16.87%** | 16.94% | 14.63 | 12.32117 -> 3.68665 | 263.8 |
| `06_native_v014_n49_tile1024_mb48` | PR#122 Hillclimb 4 (FSDP32 mb=48 seqs / 98K tok/rank) | `FAIL(rc=1)` | **515.0** | **0.78%** | 0.78% | 21.95 | 12.32154 -> 12.32154 | 213.4 |
| `07_native_v015_compiled_optimizers` | PR#122 Hillclimb 5 (n49 + Compiled Optimizer) | `FAIL(rc=1)` | **0.0** | **0.00%** | 0.00% | - | - | 10.9 |
