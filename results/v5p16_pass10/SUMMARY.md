# Historical Sweep Archive (`v5p16_pass10`) — `RecursiveSI/flint` (`40B-A2.5B` MoE) on `16` × TPU `v5p`

> [!IMPORTANT]
> **Historical `c4_test` EoD-Skipping & FLOPs Convention Note:**
> 1. **Dataset Packing (`c4_test` Short-Doc EoD Skipping):** These archived `10–15`-step tuning sweeps ran on `c4_test` (`~400–500` tok/doc) where `SbdConfig(document_aware_skipping=True)` skips cross-document attention blocks in Pallas `SplashAttention`. For **60-step (`50` steady post-warmup steps `11..60`) True Unbroken Full-Sequence Causal Attention (`TT_CONTIGUOUS_FULL_SEQ=1`)** across `2K..64K` at strictly fixed recipe and constant `65,536` tok/rank, see [`../customer_val60/`](../customer_val60/) and the root [`README.md`](../../README.md).
> 2. **Dual MFU Accounting:** Upstream TorchTitan (`torchtitan/models/utils.py:442`) logs the **PaLM Full-Square Convention** (`12 * H * D * S` on the `14` full layers), which `2×` overcounts causal full-attention FLOPs (`61.073` vs **`38.521` GFLOP/tok** at `64K`). Both **Steady Causal-Masked MFU** and **Steady PaLM Full-Square MFU** are reported side-by-side below, led by hardware-agnostic **`Steady tok/s/chip`**.

- **Hardware:** `16` × TPU `v5p` chips (`16` TensorCore ranks, `459.0` BF16 TFLOP/s/chip)

| Experiment | Seq Len | Status | **Steady tok/s/chip** | Steady tok/s/rank | Slice tok/s | Causal GF/tok | **Steady Causal MFU** | PaLM GF/tok | Steady PaLM MFU | Peak HBM (GiB) | Loss (`start -> end`) |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `v5p16_p10_01_hf_pr123_s4096_mb32_1x_sbdp_fusedopt_sa1024_ac16` | `4096` | `PASS` | **6,339.7** | 6,339.7 | 101,435 | `17.3302` | **23.94%** | `18.7943` | 25.96% | 22.84 | `12.49604 -> 3.6804` |
| `v5p16_p10_02_hf_pr123_s8192_mb16_1x_sbdp_fusedopt_sa1024_ac16` | `8192` | `PASS` | **6,212.4** | 6,212.4 | 99,398 | `18.7671` | **25.40%** | `21.6129` | 29.25% | 22.84 | `12.49606 -> 3.61466` |
| `v5p16_p10_03_hf_pr123_s16384_mb8_1x_sbdp_fusedopt_sa1024_ac18` | `16384` | `PASS` | **5,931.0** | 5,931.0 | 94,896 | `21.5994` | **27.91%** | `27.2500` | 35.21% | 22.84 | `12.49622 -> 3.55225` |
| `v5p16_p10_04_hf_pr123_s16384_mb12_1x_sbdp_fusedopt_sa1024_ac36` | `16384` | `FAIL(rc=-9)` | — | — | — | `21.5994` | — | `27.25` | — | `OOM/FAIL` | — |
| `v5p16_p10_05_hf_pr123_s16384_mb8_ga2_sbdp_fusedopt_sa1024_ac24` | `16384` | `FAIL(rc=1)` | **1,536.0** | 1,536.0 | 24,576 | `21.5994` | **7.23%** | `27.2500` | 9.12% | 22.84 | `12.49581 -> 12.49581` |
| `v5p16_p10_06_native_pr122_s8192_mb12_1x_sbdp_bf16rope_fusedopt` | `8192` | `FAIL(rc=1)` | **262.0** | 262.0 | 4,192 | `18.7671` | **1.07%** | `21.6129` | 1.23% | 29.55 | `12.31932 -> 12.31932` |
| `v5p16_p10_07_native_pr122_s16384_mb6_1x_sbdp_bf16rope_fusedopt` | `16384` | `FAIL(rc=1)` | **279.0** | 279.0 | 4,464 | `21.5994` | **1.31%** | `27.2500` | 1.66% | 29.55 | `12.32027 -> 12.32027` |
