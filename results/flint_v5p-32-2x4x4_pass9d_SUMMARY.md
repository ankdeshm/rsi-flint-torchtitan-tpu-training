# Flint (`RecursiveSI/flint` 40B MoE) TPU v5p `PASS9D` (`mb=96..160` + `StaticEP=4/8/16 ZeRO-2`) — `v5p-32-2x4x4`

- **Updated (UTC):** `2026-10-31T08:34:44.364296+00:00`
- **Chip / World Size:** `v5p` (`32` chips/ranks, `459.0` BF16 TFLOP/s/rank)

| Experiment | Lane | Status | Steady tok/s/rank | Steady MFU (%) | Peak MFU (%) | Peak HBM (GiB) | Loss (start -> end) | Wall Time (s) |
|---|---|---|---:|---:|---:|---:|---|---:|
| `v5p_p9d_01_hf_sbdp_sa1024_hybridac16_mb96_1x_log1` | v5p-32 HF PR#123 + SBDP + FusedOpt + sa=1024 + HybridAC16 + log1 (mb=96 = 4,608 tok/exp, 1x) | `FAIL(rc=-9)` | **410.0** | **1.55%** | 1.55% | 37.87 | 12.2945 -> 12.2945 | 519.4 |
