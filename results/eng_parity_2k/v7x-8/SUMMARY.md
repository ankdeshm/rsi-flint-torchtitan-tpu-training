# RSI Flint (`RecursiveSI/flint` 40B-A2.5B MoE) 60-Step Customer Validation (`ENG_PARITY_2K`) — `v7x-8-2x2x2`

- **Updated (UTC):** `2026-11-03T03:24:14.162726+00:00`
- **Hardware:** `8` × `v7x` chips (`16` TensorCore ranks, `2307.0` BF16 TFLOP/s/chip)
- **Methodology:** `60` optimizer steps per run (`log_freq=1`), Steady-State Mean over **50 post-warmup steps (`11..60`)**; True Unskipped Contiguous `seq_len` Causal Attention (`TT_CONTIGUOUS_FULL_SEQ=1`) unless marked `c4eod`.
- **Apples-to-apples with google-pytorch/torchtitan PR #122 (native lane, 16.23% MFU) and PR #123 (HF backend, 16.29% MFU):** same slice shape (`tpu7x-2x2x2` = v7x-8, `16` TensorCore ranks, SimpleFSDP over 16), same `seq_len=2048`, same global batch `512 x 2048 = 1,048,576` tok/step (`mb=32`, `ga=1`) unless a row says otherwise, same `c4_test` EoD packing as upstream (`C4-EoD`) unless marked `Full-Seq`, and the same MFU definition (TorchTitan counter `17.385` GFLOP/token = **PaLM GF/tok** column, `1,153.5` BF16 TFLOP/s per TensorCore).
- Each run is `60` steps; the steady column averages steps `11..60`. The two PR-window columns re-score the same run with each PR's own window (PR #122: median of steps `3..8`; PR #123: mean of steps `2..10`).
- `eng_*` rows reproduce the PR recipes verbatim with this repo's image; `abl_*` rows add one group of changes at a time; `ours_*` rows are the full recipe of this repo (`ours_05` also changes the batch, so it is not iso-batch).

| Experiment | Seq | Seq Mode | Steady Steps | **tok/s/chip** | tok/s/rank (±std) | Slice tok/s | Causal GF/tok | **Steady Causal MFU** | PaLM GF/tok | Steady PaLM MFU | `6*N_act` MFU | PR #122-window PaLM MFU (median 3..8) | PR #123-window PaLM MFU (mean 2..10) | HBM (GiB) | Loss (`1 -> 60`) | Status |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| `v7x8_eng_01_pr122_v011_n40_c4eod_mb32_ga1` | `2048` | `C4-EoD` | `50` (`11..60`) | **22,134.5** | 11,067.3 (±414.4) | 177,076 | `16.5704` | **15.90%** | `17.385` | 16.68% | 14.48% | 16.77% | 16.70% | 15.93 | 12.23862 -> 2.8391 | `PASS` |
| `v7x8_ours_02_val60_recipe_c4eod_mb32_ga1` | `2048` | `C4-EoD` | `50` (`11..60`) | **24,582.8** | 12,291.4 (±528.6) | 196,662 | `16.5704` | **17.66%** | `17.385` | 18.52% | 16.08% | 18.61% | 18.50% | 15.52 | 12.23864 -> 2.84265 | `PASS` |
| `v7x8_ours_03_val60_recipe_fullseq_mb32_ga1` | `2048` | `Full-Seq` | `50` (`11..60`) | **24,396.8** | 12,198.4 (±546.7) | 195,175 | `16.5704` | **17.52%** | `17.385` | 18.38% | 15.96% | 18.50% | 18.43% | 15.52 | 12.23512 -> 2.80278 | `PASS` |
| `v7x8_eng_04_pr123_hf_attnac_r3_c4eod_mb32_ga1` | `2048` | `C4-EoD` | `50` (`11..60`) | **20,558.3** | 10,279.2 (±53.1) | 164,467 | `16.5704` | **14.77%** | `17.385` | 15.49% | 13.45% | 15.49% | 15.50% | 14.37 | 12.43618 -> 2.88305 | `PASS` |
| `v7x8_ours_05_val60_recipe_c4eod_mb48_ga2` | `2048` | `C4-EoD` | `0` (`-`) | **0.0** | 0.0 (±0.0) | 0 | `16.5704` | **0.00%** | `17.385` | 0.00% | 0.00% | 0.00% | 0.00% | - | - -> - | `FAIL(rc=1)` |
| `v7x8_abl_06_v011n40_bf16red_ourflags_c4eod_mb32_ga1` | `2048` | `C4-EoD` | `50` (`11..60`) | **22,131.5** | 11,065.8 (±422.0) | 177,052 | `16.5704` | **15.90%** | `17.385` | 16.68% | 14.48% | 16.76% | 16.69% | 16.35 | 12.23862 -> 2.83789 | `PASS` |
| `v7x8_abl_07_v014_bf16red_ourflags_c4eod_mb32_ga1` | `2048` | `C4-EoD` | `50` (`11..60`) | **22,640.0** | 11,320.0 (±443.0) | 181,120 | `16.5704` | **16.26%** | `17.385` | 17.06% | 14.81% | 17.14% | 17.01% | 14.76 | 12.23862 -> 2.83674 | `PASS` |
| `v7x8_abl_08_v014_sbdp_rope_bf16red_ourflags_c4eod_mb32_ga1` | `2048` | `C4-EoD` | `50` (`11..60`) | **23,774.2** | 11,887.1 (±472.6) | 190,193 | `16.5704` | **17.08%** | `17.385` | 17.92% | 15.55% | 17.97% | 17.92% | 15.54 | 12.23864 -> 2.83929 | `PASS` |
