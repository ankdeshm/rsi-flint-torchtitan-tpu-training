# RSI Flint (`RecursiveSI/flint` 40B-A2.5B MoE) 60-Step Customer Validation (`ENG_PARITY_2K`) — `v7x-8-2x2x2`

- **Updated (UTC):** `2026-11-03T03:32:50.071554+00:00`
- **Hardware:** `8` × `v7x` chips (`16` TensorCore ranks, `2307.0` BF16 TFLOP/s/chip)
- **Methodology:** `60` optimizer steps per run (`log_freq=1`), Steady-State Mean over **50 post-warmup steps (`11..60`)**; True Unskipped Contiguous `seq_len` Causal Attention (`TT_CONTIGUOUS_FULL_SEQ=1`) unless marked `c4eod`.
- **Apples-to-apples with google-pytorch/torchtitan PR #122 (native lane, 16.23% MFU) and PR #123 (HF backend, 16.29% MFU):** same slice shape (`tpu7x-2x2x2` = v7x-8, `16` TensorCore ranks, SimpleFSDP over 16), same `seq_len=2048`, same global batch `512 x 2048 = 1,048,576` tok/step (`mb=32`, `ga=1`) unless a row says otherwise, same `c4_test` EoD packing as upstream (`C4-EoD`) unless marked `Full-Seq`, and the same MFU definition (TorchTitan counter `17.385` GFLOP/token = **PaLM GF/tok** column, `1,153.5` BF16 TFLOP/s per TensorCore).
- Each run is `60` steps; the steady column averages steps `11..60`. The two PR-window columns re-score the same run with each PR's own window (PR #122: median of steps `3..8`; PR #123: mean of steps `2..10`).
- `eng_*` rows reproduce the PR recipes verbatim with this repo's image; `abl_*` rows add one group of changes at a time; `ours_*` rows are the full recipe of this repo (`ours_05` also changes the batch, so it is not iso-batch).

| Experiment | Seq | Seq Mode | Steady Steps | **tok/s/chip** | tok/s/rank (±std) | Slice tok/s | Causal GF/tok | **Steady Causal MFU** | PaLM GF/tok | Steady PaLM MFU | `6*N_act` MFU | PR #122-window PaLM MFU (median 3..8) | PR #123-window PaLM MFU (mean 2..10) | HBM (GiB) | Loss (`1 -> 60`) | Status |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| `v7x8_ours_05b_val60_recipe_c4eod_mb40_ga2` | `2048` | `C4-EoD` | `0` (`-`) | **0.0** | 0.0 (±0.0) | 0 | `16.5704` | **0.00%** | `17.385` | 0.00% | 0.00% | 0.00% | 0.00% | - | - -> - | `FAIL(rc=1)` |
