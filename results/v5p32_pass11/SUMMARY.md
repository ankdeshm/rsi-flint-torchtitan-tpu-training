# Historical Sweep Archive (`v5p32_pass11`) — `RecursiveSI/flint` (`40B-A2.5B` MoE) on `32` × TPU `v5p`

> [!IMPORTANT]
> **Historical `c4_test` EoD-Skipping & FLOPs Convention Note:**
> 1. **Dataset Packing (`c4_test` Short-Doc EoD Skipping):** These archived `10–15`-step tuning sweeps ran on `c4_test` (`~400–500` tok/doc) where `SbdConfig(document_aware_skipping=True)` skips cross-document attention blocks in Pallas `SplashAttention`. For **60-step (`50` steady post-warmup steps `11..60`) True Unbroken Full-Sequence Causal Attention (`TT_CONTIGUOUS_FULL_SEQ=1`)** across `2K..64K` at strictly fixed recipe and constant `65,536` tok/rank, see [`../customer_val60/`](../customer_val60/) and the root [`README.md`](../../README.md).
> 2. **Dual MFU Accounting:** Upstream TorchTitan (`torchtitan/models/utils.py:442`) logs the **PaLM Full-Square Convention** (`12 * H * D * S` on the `14` full layers), which `2×` overcounts causal full-attention FLOPs (`61.073` vs **`38.521` GFLOP/tok** at `64K`). Both **Steady Causal-Masked MFU** and **Steady PaLM Full-Square MFU** are reported side-by-side below, led by hardware-agnostic **`Steady tok/s/chip`**.

- **Hardware:** `32` × TPU `v5p` chips (`32` TensorCore ranks, `459.0` BF16 TFLOP/s/chip)

| Experiment | Seq Len | Status | **Steady tok/s/chip** | Steady tok/s/rank | Slice tok/s | Causal GF/tok | **Steady Causal MFU** | PaLM GF/tok | Steady PaLM MFU | Peak HBM (GiB) | Loss (`start -> end`) |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `v5p32_p11_01_native_s32768_mb3_ga4_sbdp_bf16rope_fusedopt_n44` | `32768` | `PASS` | **5,439.5** | 5,439.5 | 174,064 | `27.2435` | **32.29%** | `38.5243` | 45.65% | 27.52 | `12.33327 -> 4.16861` |
| `v5p32_p11_02_native_s65536_mb1_ga4_sbdp_bf16rope_fusedopt_n44` | `65536` | `PASS` | **4,674.5** | 4,674.5 | 149,584 | `38.5212` | **39.23%** | `61.0729` | 62.20% | 18.67 | `12.33374 -> 4.17085` |
| `v5p32_p11_03_native_s65536_mb1_ga8_overlap_tile512_n49` | `65536` | `PASS` | **4,734.5** | 4,734.5 | 151,504 | `38.5212` | **39.73%** | `61.0729` | 63.00% | 14.96 | `12.33341 -> 4.20836` |
| `v5p32_p11_05_native_s16384_mb6_ga4_overlap_tile512_n44` | `16384` | `PASS` | **5,868.8** | 5,868.8 | 187,802 | `21.5994` | **27.62%** | `27.2500` | 34.84% | 27.52 | `12.33313 -> 4.17916` |
