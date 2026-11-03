# Flint 40B on TPU `v7x`: this repo vs `google-pytorch/torchtitan` PR #122 / PR #123

Apples-to-apples comparison of `RecursiveSI/flint` (39.8B total / 2.52B active MoE) pre-training throughput against the engineering team's two PRs: [PR #122](https://github.com/google-pytorch/torchtitan/pull/122) (native TorchTPU lane, **16.23% MFU**, 10,779 tok/s/device) and [PR #123](https://github.com/google-pytorch/torchtitan/pull/123) (HF transformers backend, **16.29% MFU**, ~10,810 tok/s/device). Both report `v7x-8` (`tpu7x-2x2x2`, 16 TensorCores), `seq_len=2048`, global batch `512 x 2048`. A 'device' in the PRs is one TensorCore.

## TL;DR

- **Same slice (`v7x-8`), same `2K` context, same `512 x 2048` batch, same packing, same MFU counter:** this repo's recipe runs at **18.52% MFU** (12,291 tok/s per TensorCore) vs **16.23%** in PR #122 (**+14.0%** throughput) and **16.29%** in PR #123 (**+13.7%**).
- **Both PR recipes were re-run unchanged in this environment:** PR #122 = 16.68% (published 16.23%), PR #123 = 15.49% (published 16.29%). Scored with each PR's own short window the re-runs give 16.77% and 15.50%, so the window is not the cause of any gap. PR #123 re-runs ~5% below its published number in this image (TorchTPU `0.2.0` GA, `libtpu 0.0.50`); PR #123 was measured with its own `...-flint-tune2` image, which we did not have.
- **Against our PR #122 re-run the gain is +11.1%**, from four changes (table in section 3): BF16 gradient reduce-scatter + async-collective flags (no measurable gain at this size), Splash output saved in all 49 layers + grouped-mm `tile_m=1024` (+2.3%), `scale_before_down_proj` + BF16 RoPE (+5.0%), saved MoE routing + compiled fused AdamW / grad-norm (+3.4%).
- **Without end-of-document attention skipping** (every row attends over the full 2,048 tokens): 18.38%, only 0.14 MFU points lower, so the packing mode barely matters at `2K`.
- **Not apples-to-apples, labelled as such:** a larger batch, `mb=40 x ga=2` is running (results will be added); `mb=48 x ga=2` does not fit 16-way FSDP HBM on `v7x-8` (OOM by 0.6 GiB). On `v7x-16` (32 TensorCores) the same recipe gives 18.97% at the PR per-rank batch and 20.31% with `mb=48 x ga=2` (full-sequence attention).
- **MFU definition:** both PRs use the TorchTitan counter (`17.385` GFLOP/token at `2K`), which also counts the masked half of full-attention FLOPs. Counting attention exactly (`16.570` GFLOP/token) lowers every number by ~4.7% relative at `2K`; that column is shown in section 2.

## 1. Matching the setup

| Dimension | PR #122 / PR #123 | This comparison |
|---|---|---|
| Slice | `v7x-8` (`tpu7x-2x2x2`), 2 hosts x 8 processes = 16 TensorCore ranks | Same shape (`v7x-8` pool, 2 hosts x 8 processes) |
| Parallelism | SimpleFSDP over 16 ranks | Same |
| Sequence length | 2,048 | 2,048 |
| Global batch | `512 x 2048 = 1,048,576` tok/step (`mb=32` per rank, no gradient accumulation) | Same, except rows marked *not iso-batch* |
| Data / attention | `c4_test`, documents packed with EoD; Splash skips cross-document blocks | Same (`C4-EoD` rows) plus a `Full-Seq` row with no skipping |
| MFU definition | TorchTitan counter: `6 x N_active` + window-aware full-square attention = `17.385` GFLOP/token; peak `1,153.5` BF16 TFLOP/s per TensorCore | Same (headline column); exact causal FLOPs (`16.570` GFLOP/token) shown alongside |
| Measurement | PR #122: median of steps 3-8 (8 steps); PR #123: mean of steps 2-10 (10 steps) | 60 steps per run, mean of steps 11-60; each PR's own window is also re-scored on the same runs |
| Software | PR images (`torchtitan-container:flint-v012-small-levers`, `...hf-backend...-flint-tune2`) | This repo's image: TorchTPU `0.2.0` GA, `libtpu 0.0.50`, built from the PR #122 / #123 branches ([`docker/`](../docker/)) |

## 2. Headline vs the published numbers

| Run | What it is | tok/s per TensorCore | MFU (TorchTitan counter) | vs PR #122 published | vs PR #123 published |
|---|---|---:|---:|---:|---:|
| PR #122 (published) | Engineering run: native lane, 8 steps, median of steps 3-8 | 10,779 | 16.23% | - | - |
| PR #123 (published) | Engineering run: HF backend, 10 steps, mean of steps 2-10 | ~10,810 | 16.29% | - | - |
| `eng_01` | PR #122 recipe, unchanged, re-run here | 11,067.3 | 16.68% | +2.7% | +2.4% |
| `eng_04` | PR #123 recipe, unchanged, re-run here | 10,279.2 | 15.49% | -4.6% | -4.9% |
| **`ours_02`** | **This repo's recipe, same batch and packing** | **12,291.4** | **18.52%** | **+14.0%** | **+13.7%** |
| `ours_03` | This repo's recipe, true full-sequence attention (no EoD skipping) | 12,198.4 | 18.38% | +13.2% | +12.8% |
| `ours_05b` | This repo's recipe, `mb=40 x ga=2` (*not iso-batch*) | running | - | - | - |

## 3. What we changed, one step at a time (iso-batch, `C4-EoD`, same seed)

| Step | Change | tok/s per TensorCore | MFU | Step gain |
|---|---|---:|---:|---:|
| `eng_01` | PR #122 as published: FP32 gradient reduce-scatter, config XLA flags, Splash attention output saved in 40 of 49 layers, Pallas grouped-mm `tile_m=512`, eager AdamW and grad-norm | 11,067.3 | 16.68% | - |
| `abl_06` | + BF16 gradient reduce-scatter (`--training.mixed_precision_reduce=bfloat16`, half the FSDP reduce-scatter bytes) + async-collective / latency-hiding XLA flags, including PR #123's hierarchical SparseCore reduce-scatter set | 11,065.8 | 16.68% | -0.0% |
| `abl_07` | + config `flint_40b_v014_tile1024`: Splash attention output saved in all 49 layers (no attention recompute) and Pallas grouped-mm `tile_m` 512 -> 1024 for the expert matmuls | 11,320.0 | 17.06% | +2.3% |
| `abl_08` | + `scale_before_down_proj` (router weights applied before the expert down-projection, so the combine is a plain sum) + RoPE computed in BF16 instead of FP32 | 11,887.1 | 17.92% | +5.0% |
| `ours_02` | + MoE routing outputs saved for backward (no router recompute) + AdamW and grad-norm compiled into their own graph with the fused optimizer implementation = this repo's full recipe | 12,291.4 | 18.52% | +3.4% (total +11.1%) |

Numerics: BF16 gradient reduce-scatter and BF16 RoPE change rounding; `scale_before_down_proj` changes the order of the routing-weight multiply (mathematically identical). Every row is 60 steps from the same seed and the loss curves stay in the same band (section 4).

## 4. All runs on `v7x-8` (16 TensorCores)

| Run | What it is | Tok/step | Attention | **Steady tok/s per TensorCore** | **MFU (TorchTitan counter, steps 11-60)** | MFU in PR #122 / PR #123 windows | Causal MFU | vs PR #122 re-run | Loss (`1 -> 60`) | Peak HBM (GiB) | Status |
|---|---|---:|---|---:|---:|---|---:|---:|---|---:|---|
| `v7x8_eng_01_pr122_v011_n40_c4eod_mb32_ga1` | PR #122 recipe, unchanged, re-run here | `1.05M` | C4-EoD (as upstream) | **11,067.3** | **16.68%** | 16.77% / 16.70% | 15.90% | +0.0% | `12.23862 -> 2.8391` | 15.93 | `PASS` |
| `v7x8_eng_04_pr123_hf_attnac_r3_c4eod_mb32_ga1` | PR #123 recipe, unchanged, re-run here | `1.05M` | C4-EoD (as upstream) | **10,279.2** | **15.49%** | 15.49% / 15.50% | 14.77% | -7.1% | `12.43618 -> 2.88305` | 14.37 | `PASS` |
| `v7x8_ours_02_val60_recipe_c4eod_mb32_ga1` | This repo's recipe, same batch and packing | `1.05M` | C4-EoD (as upstream) | **12,291.4** | **18.52%** | 18.61% / 18.50% | 17.66% | +11.1% | `12.23864 -> 2.84265` | 15.52 | `PASS` |
| `v7x8_ours_03_val60_recipe_fullseq_mb32_ga1` | This repo's recipe, true full-sequence attention (no EoD skipping) | `1.05M` | Full-Seq | **12,198.4** | **18.38%** | 18.50% / 18.43% | 17.52% | +10.2% | `12.23512 -> 2.80278` | 15.52 | `PASS` |
| `v7x8_ours_05_val60_recipe_c4eod_mb48_ga2` | This repo's recipe, `mb=48 x ga=2` (*not iso-batch*) | `3.15M` | C4-EoD (as upstream) | - | - | - | - | - | - | HBM OOM at compile (needs 12.04 GiB more, 11.46 GiB free per TensorCore) |
| `v7x8_ours_05b` | This repo's recipe, `mb=40 x ga=2` (*not iso-batch*) | - | - | - | - | - | - | - | - | running |
| `v7x8_abl_06_v011n40_bf16red_ourflags_c4eod_mb32_ga1` | PR #122 + BF16 gradient reduce-scatter + async-collective XLA flags | `1.05M` | C4-EoD (as upstream) | **11,065.8** | **16.68%** | 16.76% / 16.69% | 15.90% | -0.0% | `12.23862 -> 2.83789` | 16.35 | `PASS` |
| `v7x8_abl_07_v014_bf16red_ourflags_c4eod_mb32_ga1` | + Splash output saved in all 49 layers + grouped-mm `tile_m=1024` | `1.05M` | C4-EoD (as upstream) | **11,320.0** | **17.06%** | 17.14% / 17.01% | 16.26% | +2.3% | `12.23862 -> 2.83674` | 14.76 | `PASS` |
| `v7x8_abl_08_v014_sbdp_rope_bf16red_ourflags_c4eod_mb32_ga1` | + `scale_before_down_proj` + BF16 RoPE | `1.05M` | C4-EoD (as upstream) | **11,887.1** | **17.92%** | 17.97% / 17.92% | 17.08% | +7.4% | `12.23864 -> 2.83929` | 15.54 | `PASS` |

## 5. Numbers that are *not* apples-to-apples (and why they differ)

| Setting | Run | Tok/step | tok/s per TensorCore | MFU (TorchTitan counter) | Causal MFU | Why it differs |
|---|---|---:|---:|---:|---:|---|
| `v7x-16` (32 TensorCores), PR per-rank batch | `v7x_val60_01_iso65k_s2048_mb32_ga1_fullseq` | `2.10M` | 12,584.2 | 18.97% | 18.08% | 2x the ranks and 2x the global batch; full-sequence attention |
| `v7x-16`, larger batch | `v7x_val60_09_opt_s2048_mb48_ga2_fullseq` | `6.29M` | 13,473.9 | 20.31% | 19.36% | `mb=48`, `ga=2`: 6.3M tok/step; full-sequence attention |
| `v7x-16`, `32K` context | `v7x_val60_03_iso65k_s32768_mb2_ga1_fullseq` | `2.10M` | 7,433.0 | 24.82% | 17.56% | Attention FLOPs grow with context; the TorchTitan counter also counts the masked half |
| `v7x-16`, `64K` context | `v7x_val60_04_iso65k_s65536_mb1_ga1_fullseq` | `2.10M` | 5,361.1 | 28.38% | 17.90% | Same; full-sequence attention |

- The repo's older long-context headline numbers (`47.92%` on `v7x-16` and `63–65%` on `v5p` at `64K`) were 10–15-step tuning runs with `c4_test` EoD block skipping **and** the TorchTitan counter. With true full-sequence attention the same recipe measures 28.38% (TorchTitan) / 17.90% (causal) on `v7x-16` and 42.83% / 27.01% on `v5p-32` at `64K` (see [`README.md`](../README.md#validated-60-step-results-customer_val60)).

## 6. TPU `v5p`

Neither PR reports `v5p`. PR #122's native recipe does not compile on `v5p` as published (`16 MiB` scoped-VMEM limit at grouped-mm `tile_m=512`); PR #123's default recipe measured `19.76%` on `v5p-16` in this repo (Pass 1, short run). Same counter, same 60-step method (`459` BF16 TFLOP/s per chip), true full-sequence attention, `2K`, `mb=32`:

| Slice | Run | Tok/step | tok/s per chip | MFU (TorchTitan counter) | Causal MFU |
|---|---|---:|---:|---:|---:|
| `v5p-16` | `v5p_val60_01_iso65k_s2048_mb32_ga1_fullseq` | `1.05M` | 5,874.4 | 22.25% | 21.21% |
| `v5p-32` | `v5p_val60_01_iso65k_s2048_mb32_ga1_fullseq` | `2.10M` | 5,925.7 | 22.44% | 21.39% |

## 7. Reproduce

- Runner: [`scripts/run_flint_customer_60step_validation.py`](../scripts/run_flint_customer_60step_validation.py) (`--suite=eng_parity_2k`; experiment list in `build_eng_parity_experiments`; `--only-filter=ours_05b` for the follow-up).
- Manifest: [`k8s/v7x-8-flint-engparity.yaml`](../k8s/v7x-8-flint-engparity.yaml) (2-host Indexed Job, lockstep runner).
- Raw per-step metrics: [`results/eng_parity_2k/v7x-8/summary.json`](../results/eng_parity_2k/v7x-8/summary.json).
