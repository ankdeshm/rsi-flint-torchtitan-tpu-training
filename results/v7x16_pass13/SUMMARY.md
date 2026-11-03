# Historical Sweep Archive (`v7x16_pass13`) — `RecursiveSI/flint` (`40B-A2.5B` MoE) on `16` × TPU `v7x`

> [!IMPORTANT]
> **Historical `c4_test` EoD-Skipping & FLOPs Convention Note:**
> 1. **Dataset Packing (`c4_test` Short-Doc EoD Skipping):** These archived `10–15`-step tuning sweeps ran on `c4_test` (`~400–500` tok/doc) where `SbdConfig(document_aware_skipping=True)` skips cross-document attention blocks in Pallas `SplashAttention`. For **60-step (`50` steady post-warmup steps `11..60`) True Unbroken Full-Sequence Causal Attention (`TT_CONTIGUOUS_FULL_SEQ=1`)** across `2K..64K` at strictly fixed recipe and constant `65,536` tok/rank, see [`../customer_val60/`](../customer_val60/) and the root [`README.md`](../../README.md).
> 2. **Dual MFU Accounting:** Upstream TorchTitan (`torchtitan/models/utils.py:442`) logs the **PaLM Full-Square Convention** (`12 * H * D * S` on the `14` full layers), which `2×` overcounts causal full-attention FLOPs (`61.073` vs **`38.521` GFLOP/tok** at `64K`). Both **Steady Causal-Masked MFU** and **Steady PaLM Full-Square MFU** are reported side-by-side below, led by hardware-agnostic **`Steady tok/s/chip`**.

- **Hardware:** `16` × TPU `v7x` chips (`32` TensorCore ranks, `2307.0` BF16 TFLOP/s/chip)

| Experiment | Seq Len | Status | **Steady tok/s/chip** | Steady tok/s/rank | Slice tok/s | Causal GF/tok | **Steady Causal MFU** | PaLM GF/tok | Steady PaLM MFU | Peak HBM (GiB) | Loss (`start -> end`) |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `v7x_p13_01_native_s4096_mb24_ga8_sbdp_bf16rope_fusedopt_n49` | `4096` | `PASS` | **26,911.0** | 13,455.5 | 430,576 | `17.3302` | **20.22%** | `18.7943` | 21.92% | 23.52 | `12.32219 -> 3.7291` |
| `v7x_p13_02_native_s8192_mb12_ga8_sbdp_bf16rope_fusedopt_n49` | `8192` | `PASS` | **24,687.0** | 12,343.5 | 394,992 | `18.7671` | **20.08%** | `21.6129` | 23.13% | 20.52 | `12.32358 -> 3.74033` |
| `v7x_p13_03_native_s16384_mb6_ga8_sbdp_bf16rope_fusedopt_n49` | `16384` | `PASS` | **22,988.0** | 11,494.0 | 367,808 | `21.5994` | **21.52%** | `27.2500` | 27.15% | 20.52 | `12.32431 -> 3.73677` |
| `v7x_p13_04_native_s16384_mb8_ga4_sbdp_bf16rope_fusedopt_n36` | `16384` | `FAIL(rc=-9)` | — | — | — | `21.5994` | — | `27.25` | — | `OOM/FAIL` | — |
| `v7x_p13_05_hf_pr123_s4096_mb32_ga2_sbdp_fusedopt_sa1024_tile1024_ac12` | `4096` | `PASS` | **24,037.0** | 12,018.5 | 384,592 | `17.3302` | **18.06%** | `18.7943` | 19.58% | 25.5 | `12.30972 -> 4.05178` |
| `v7x_p13_06_hf_pr123_s8192_mb16_ga2_sbdp_fusedopt_sa1024_tile1024_ac12` | `8192` | `PASS` | **23,034.0** | 11,517.0 | 368,544 | `18.7671` | **18.74%** | `21.6129` | 21.58% | 25.5 | `12.30955 -> 3.51351` |
| `v7x_p13_07_hf_pr123_s16384_mb8_ga2_sbdp_fusedopt_sa1024_tile1024_ac16` | `16384` | `PASS` | **21,427.0** | 10,713.5 | 342,832 | `21.5994` | **20.06%** | `27.2500` | 25.31% | 25.5 | `12.30953 -> 3.98046` |
