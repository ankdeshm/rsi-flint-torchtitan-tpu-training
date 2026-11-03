# Historical Sweep Archive (`v5p16_pass11`) — `RecursiveSI/flint` (`40B-A2.5B` MoE) on `16` × TPU `v5p`

> [!IMPORTANT]
> **Historical `c4_test` EoD-Skipping & FLOPs Convention Note:**
> 1. **Dataset Packing (`c4_test` Short-Doc EoD Skipping):** These archived `10–15`-step tuning sweeps ran on `c4_test` (`~400–500` tok/doc) where `SbdConfig(document_aware_skipping=True)` skips cross-document attention blocks in Pallas `SplashAttention`. For **60-step (`50` steady post-warmup steps `11..60`) True Unbroken Full-Sequence Causal Attention (`TT_CONTIGUOUS_FULL_SEQ=1`)** across `2K..64K` at strictly fixed recipe and constant `65,536` tok/rank, see [`../customer_val60/`](../customer_val60/) and the root [`README.md`](../../README.md).
> 2. **Dual MFU Accounting:** Upstream TorchTitan (`torchtitan/models/utils.py:442`) logs the **PaLM Full-Square Convention** (`12 * H * D * S` on the `14` full layers), which `2×` overcounts causal full-attention FLOPs (`61.073` vs **`38.521` GFLOP/tok** at `64K`). Both **Steady Causal-Masked MFU** and **Steady PaLM Full-Square MFU** are reported side-by-side below, led by hardware-agnostic **`Steady tok/s/chip`**.

- **Hardware:** `16` × TPU `v5p` chips (`16` TensorCore ranks, `459.0` BF16 TFLOP/s/chip)

| Experiment | Seq Len | Status | **Steady tok/s/chip** | Steady tok/s/rank | Slice tok/s | Causal GF/tok | **Steady Causal MFU** | PaLM GF/tok | Steady PaLM MFU | Peak HBM (GiB) | Loss (`start -> end`) |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `v5p16_p11_02_native_s65536_mb1_ga4_sbdp_bf16rope_fusedopt_n44` | `65536` | `PASS` | **4,913.2** | 4,913.2 | 78,611 | `38.5212` | **41.23%** | `61.0729` | 65.37% | 18.87 | `12.32201 -> 4.06841` |
| `v5p16_p11_03_native_s65536_mb1_ga8_overlap_tile512_n49` | `65536` | `PASS` | **4,919.8** | 4,919.8 | 78,717 | `38.5212` | **41.29%** | `61.0729` | 65.46% | 18.83 | `12.32187 -> 4.00942` |
