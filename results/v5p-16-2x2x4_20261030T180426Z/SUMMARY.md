# RSI Flint (`RecursiveSI/flint` 40B MoE) TorchTPU 0.2.0 GA Training Results — `v5p-16-2x2x4`

- **Updated (UTC):** `2026-10-30T18:15:07.434335+00:00`
- **Chip / World Size:** `v5p` (`16` ranks, `459.0` BF16 TFLOP/s/rank, `17.385` GFLOP/token)

| Experiment | Lane | Status | Steady tok/s/rank | Steady MFU (%) | Peak MFU (%) | Peak HBM (GiB) | Loss (start -> end) | Wall Time (s) |
|---|---|---|---:|---:|---:|---:|---|---:|
| `01_hf_backend_pr123_flint_v5p` | PR#123 HF Backend (v5p) | `PASS` | **5,215.8** | **19.76%** | 19.79% | 11.78 | 12.49684 -> 3.43675 | 383.2 |
| `02_native_pr122_v5p_n40` | PR#122 Native Flint (v5p n40) | `FAIL(rc=1)` | **0.0** | **0.00%** | 0.00% | - | - | 116.8 |
| `03_native_pr122_v5p_n49_tile1024` | PR#122 Hillclimb (v5p n49 + tile_m=1024) | `FAIL(rc=1)` | **0.0** | **0.00%** | 0.00% | - | - | 113.6 |
