# Historical Sweep Archive (`v7x16_pass14`) — `RecursiveSI/flint` (`40B-A2.5B` MoE) on `16` × TPU `v7x`

> [!IMPORTANT]
> **Historical `c4_test` EoD-Skipping & FLOPs Convention Note:**
> 1. **Dataset Packing (`c4_test` Short-Doc EoD Skipping):** These archived `10–15`-step tuning sweeps ran on `c4_test` (`~400–500` tok/doc) where `SbdConfig(document_aware_skipping=True)` skips cross-document attention blocks in Pallas `SplashAttention`. For **60-step (`50` steady post-warmup steps `11..60`) True Unbroken Full-Sequence Causal Attention (`TT_CONTIGUOUS_FULL_SEQ=1`)** across `2K..64K` at strictly fixed recipe and constant `65,536` tok/rank, see [`../customer_val60/`](../customer_val60/) and the root [`README.md`](../../README.md).
> 2. **Dual MFU Accounting:** Upstream TorchTitan (`torchtitan/models/utils.py:442`) logs the **PaLM Full-Square Convention** (`12 * H * D * S` on the `14` full layers), which `2×` overcounts causal full-attention FLOPs (`61.073` vs **`38.521` GFLOP/tok** at `64K`). Both **Steady Causal-Masked MFU** and **Steady PaLM Full-Square MFU** are reported side-by-side below, led by hardware-agnostic **`Steady tok/s/chip`**.

- **Hardware:** `16` × TPU `v7x` chips (`32` TensorCore ranks, `2307.0` BF16 TFLOP/s/chip)

| Experiment | Seq Len | Status | **Steady tok/s/chip** | Steady tok/s/rank | Slice tok/s | Causal GF/tok | **Steady Causal MFU** | PaLM GF/tok | Steady PaLM MFU | Peak HBM (GiB) | Loss (`start -> end`) |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `v7x_p14_01_native_s65536_mb1_ga8_sbdp_bf16rope_fusedopt_n49` | `65536` | `PASS` | **18,103.4** | 9,051.7 | 289,654 | `38.5212` | **30.23%** | `61.0729` | 47.92% | 14.02 | `12.32456 -> 4.53533` |
| `v7x_p14_02_native_s65536_mb2_ga4_bylayer1_sa1024_n45` | `65536` | `FAIL(rc=-9)` | — | — | — | `38.5212` | — | `61.0729` | — | `OOM/FAIL` | — |
| `v7x_p14_03_native_s131072_mb1_ga4_bylayer1_sa1024_n45` | `131072` | `FAIL(rc=-9)` | — | — | — | `61.0715` | — | `106.17` | — | `OOM/FAIL` | — |
| `v7x_p14_04_native_s131072_mb1_ga4_offload_p2_sa1024` | `131072` | `FAIL(rc=-9)` | — | — | — | `61.0715` | — | `106.17` | — | `OOM/FAIL` | — |
| `v7x_p14_07_native_s16384_mb8_ga6_bylayer1_sbdp_bf16rope_fusedopt_n45` | `16384` | `FAIL(rc=1)` | — | — | — | `21.5994` | — | `27.25` | — | `OOM/FAIL` | — |
| `v7x_p14_08_native_s16384_mb6_ga8_sa1024_ag_rs_overlap_n49` | `16384` | `FAIL(rc=1)` | — | — | — | `21.5994` | — | `27.25` | — | `OOM/FAIL` | — |
| `v7x_p14_09_native_s32768_mb3_ga8_sbdp_bf16rope_fusedopt_n49` | `32768` | `PASS` | **20,524.4** | 10,262.2 | 328,390 | `27.2435` | **24.24%** | `38.5243` | 34.27% | 20.52 | `12.32438 -> 4.18028` |
| `v7x_p14_10_hf_pr123_s16384_mb10_ga4_sa2048_tile1024_ac20` | `16384` | `FAIL(rc=-9)` | — | — | — | `21.5994` | — | `27.25` | — | `OOM/FAIL` | — |
