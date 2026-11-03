# Historical Sweep Archive (`v5p32_pass10`) — `RecursiveSI/flint` (`40B-A2.5B` MoE) on `32` × TPU `v5p`

> [!IMPORTANT]
> **Historical `c4_test` EoD-Skipping & FLOPs Convention Note:**
> 1. **Dataset Packing (`c4_test` Short-Doc EoD Skipping):** These archived `10–15`-step tuning sweeps ran on `c4_test` (`~400–500` tok/doc) where `SbdConfig(document_aware_skipping=True)` skips cross-document attention blocks in Pallas `SplashAttention`. For **60-step (`50` steady post-warmup steps `11..60`) True Unbroken Full-Sequence Causal Attention (`TT_CONTIGUOUS_FULL_SEQ=1`)** across `2K..64K` at strictly fixed recipe and constant `65,536` tok/rank, see [`../customer_val60/`](../customer_val60/) and the root [`README.md`](../../README.md).
> 2. **Dual MFU Accounting:** Upstream TorchTitan (`torchtitan/models/utils.py:442`) logs the **PaLM Full-Square Convention** (`12 * H * D * S` on the `14` full layers), which `2×` overcounts causal full-attention FLOPs (`61.073` vs **`38.521` GFLOP/tok** at `64K`). Both **Steady Causal-Masked MFU** and **Steady PaLM Full-Square MFU** are reported side-by-side below, led by hardware-agnostic **`Steady tok/s/chip`**.

- **Hardware:** `32` × TPU `v5p` chips (`32` TensorCore ranks, `459.0` BF16 TFLOP/s/chip)

| Experiment | Seq Len | Status | **Steady tok/s/chip** | Steady tok/s/rank | Slice tok/s | Causal GF/tok | **Steady Causal MFU** | PaLM GF/tok | Steady PaLM MFU | Peak HBM (GiB) | Loss (`start -> end`) |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `v5p32_p10_01_hf_pr123_s4096_mb32_1x_sbdp_fusedopt_sa1024_ac16` | `4096` | `PASS` | **6,494.7** | 6,494.7 | 207,830 | `17.3302` | **24.52%** | `18.7943` | 26.59% | 22.84 | `12.29401 -> 5.27365` |
| `v5p32_p10_02_hf_pr123_s8192_mb16_1x_sbdp_fusedopt_sa1024_ac16` | `8192` | `PASS` | **6,291.7** | 6,291.7 | 201,334 | `18.7671` | **25.72%** | `21.6129` | 29.63% | 22.84 | `12.29378 -> 5.51056` |
| `v5p32_p10_03_hf_pr123_s16384_mb8_1x_sbdp_fusedopt_sa1024_ac18` | `16384` | `PASS` | **5,951.6** | 5,951.6 | 190,451 | `21.5994` | **28.01%** | `27.2500` | 35.33% | 22.84 | `12.29381 -> 5.30979` |
| `v5p32_p10_04_hf_pr123_s16384_mb12_1x_sbdp_fusedopt_sa1024_ac36` | `16384` | `FAIL(rc=1)` | **459.0** | 459.0 | 14,688 | `21.5994` | **2.16%** | `27.2500` | 2.73% | 37.87 | `12.29372 -> 12.29372` |
| `v5p32_p10_05_hf_pr123_hsdp2x16_s16384_mb8_ga2_sbdp_fusedopt_sa1024_ac28` | `16384` | `FAIL(rc=-9)` | **795.0** | 795.0 | 25,440 | `21.5994` | **3.74%** | `27.2500` | 4.72% | 25.58 | `12.49596 -> 12.49596` |
| `v5p32_p10_06_native_pr122_s8192_mb12_1x_sbdp_bf16rope_fusedopt` | `8192` | `PASS` | **6,036.3** | 6,036.3 | 193,162 | `18.7671` | **24.68%** | `21.6129` | 28.42% | 29.21 | `12.33347 -> 4.16189` |
| `v5p32_p10_07_native_pr122_s16384_mb6_1x_sbdp_bf16rope_fusedopt` | `16384` | `PASS` | **5,670.7** | 5,670.7 | 181,462 | `21.5994` | **26.68%** | `27.2500` | 33.67% | 29.21 | `12.33417 -> 4.15166` |
