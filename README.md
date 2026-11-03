# `RecursiveSI/flint` (40B MoE) Pre-Training on Cloud TPU (`v7x` & `v5p`)

Distributed pre-training, context-length scaling (`2K` to `64K`), HuggingFace safetensors fine-tuning, numerical checkpoint parity verification, and static-shape Expert Parallelism (`StaticEP=2..32`) for **[`RecursiveSI/flint`](https://huggingface.co/RecursiveSI/flint)** (`39.8B` total / `~2.52B` active parameters per token) on **TPU `v7x-16`** and **TPU `v5p-16` / `v5p-32`** using **TorchTitan** and **TorchTPU `0.2.0` GA**.

- **Full Experiment Table (`195+` runs):** [`docs/TRAINING_RESULTS.md`](./docs/TRAINING_RESULTS.md) · **Engineering comparison:** [`docs/ENGINEERING_COMPARISON.md`](./docs/ENGINEERING_COMPARISON.md)
- **Directory Indexes:** [`scripts/README.md`](./scripts/README.md) · [`results/README.md`](./results/README.md)

<!-- HEADLINE:BEGIN -->
## Headline Results (validated: 60 steps per run, steady-state mean of steps `11..60`)

### A. Apples-to-apples vs the engineering team (`google-pytorch/torchtitan` PR #122 / PR #123): TPU `v7x-8`, `2K` context

Same slice (`tpu7x-2x2x2`, 16 TensorCores, SimpleFSDP over 16), same `seq_len=2048`, same global batch (`512 x 2048 = 1.05M` tokens/step), same `c4_test` end-of-document packing, same MFU counter (TorchTitan: `17.385` GFLOP/token, `1,153.5` BF16 TFLOP/s per TensorCore). Full write-up: [`docs/ENGINEERING_COMPARISON.md`](./docs/ENGINEERING_COMPARISON.md) · raw per-step data: [`results/eng_parity_2k/v7x-8/`](./results/eng_parity_2k/v7x-8/SUMMARY.md).

| Run | What it is | tok/s per TensorCore | MFU (TorchTitan counter) | vs PR #122 published | vs PR #123 published |
|---|---|---:|---:|---:|---:|
| PR #122 (published) | Engineering run: native lane, 8 steps, median of steps 3-8 | 10,779 | 16.23% | - | - |
| PR #123 (published) | Engineering run: HF backend, 10 steps, mean of steps 2-10 | ~10,810 | 16.29% | - | - |
| `eng_01` | PR #122 recipe, unchanged, re-run here | 11,067.3 | 16.68% | +2.7% | +2.4% |
| `eng_04` | PR #123 recipe, unchanged, re-run here | 10,279.2 | 15.49% | -4.6% | -4.9% |
| **`ours_02`** | **This repo's recipe, same batch and packing** | **12,291.4** | **18.52%** | **+14.0%** | **+13.7%** |
| `ours_03` | This repo's recipe, true full-sequence attention (no EoD skipping) | 12,198.4 | 18.38% | +13.2% | +12.8% |
| `ours_05b` | This repo's recipe, `mb=40 x ga=2` (*not iso-batch*) | HBM OOM at program load (needs 17.26 GiB, 16.58 GiB reservable per TensorCore) | - | - | - |

How the +11.1% over the PR #122 re-run was obtained (each row adds one group of changes; same batch, same packing, same seed):

| Step | Change | tok/s per TensorCore | MFU | Step gain |
|---|---|---:|---:|---:|
| `eng_01` | PR #122 as published: FP32 gradient reduce-scatter, config XLA flags, Splash attention output saved in 40 of 49 layers, Pallas grouped-mm `tile_m=512`, eager AdamW and grad-norm | 11,067.3 | 16.68% | - |
| `abl_06` | + BF16 gradient reduce-scatter (`--training.mixed_precision_reduce=bfloat16`, half the FSDP reduce-scatter bytes) + async-collective / latency-hiding XLA flags, including PR #123's hierarchical SparseCore reduce-scatter set | 11,065.8 | 16.68% | -0.0% |
| `abl_07` | + config `flint_40b_v014_tile1024`: Splash attention output saved in all 49 layers (no attention recompute) and Pallas grouped-mm `tile_m` 512 -> 1024 for the expert matmuls | 11,320.0 | 17.06% | +2.3% |
| `abl_08` | + `scale_before_down_proj` (router weights applied before the expert down-projection, so the combine is a plain sum) + RoPE computed in BF16 instead of FP32 | 11,887.1 | 17.92% | +5.0% |
| `ours_02` | + MoE routing outputs saved for backward (no router recompute) + AdamW and grad-norm compiled into their own graph with the fused optimizer implementation = this repo's full recipe | 12,291.4 | 18.52% | +3.4% (total +11.1%) |

### B. Validated results across slices and context lengths (true full-sequence attention)

Same recipe at every context length, constant `65,536` tokens per microbatch per rank (`mb=32 -> 1` from `2K -> 64K`), no end-of-document block skipping. MFU is shown with exact causal attention FLOPs and with the TorchTitan / PaLM counter used by the PRs. Per-run details: [Validated 60-Step Results](#validated-60-step-results-customer_val60).

| Slice | Metric | `2K` | `4K` | `8K` | `16K` | `32K` | `64K` |
|---|---|---:|---:|---:|---:|---:|---:|
| TPU `v7x-16` (16 chips / 32 TC) | tok/s per chip | 25,168.4 | 24,552.1 | 22,326.5 | 18,958.8 | 14,866.0 | 10,722.3 |
|  | MFU causal / TorchTitan | 18.08% / 18.97% | 18.44% / 20.00% | 18.16% / 20.92% | 17.75% / 22.39% | 17.56% / 24.82% | 17.90% / 28.38% |
| TPU `v5p-32` (32 chips) | tok/s per chip | 5,925.7 | 5,809.7 | 5,513.6 | 4,995.9 | 4,214.4 | 3,218.7 |
|  | MFU causal / TorchTitan | 21.39% / 22.44% | 21.94% / 23.79% | 22.54% / 25.96% | 23.51% / 29.66% | 25.01% / 35.37% | 27.01% / 42.83% |
| TPU `v5p-16` (16 chips) | tok/s per chip | 5,874.4 | 5,799.7 | 5,504.7 | 4,949.7 | 4,170.0 | 3,159.2 |
|  | MFU causal / TorchTitan | 21.21% / 22.25% | 21.90% / 23.75% | 22.51% / 25.92% | 23.29% / 29.39% | 24.75% / 35.00% | 26.51% / 42.04% |

- **Larger-batch variants (not iso-batch):** `v7x-16` `64K` `mb=1 x ga=4`: 10,829.4 tok/s/chip (18.08% / 28.67%); `v7x-16` `2K` `mb=48 x ga=2`: 26,947.8 tok/s/chip (19.36% / 20.31%); `v7x-16` `16K` `mb=6 x ga=2`: 19,323.3 tok/s/chip (18.09% / 22.82%); `v7x-16` `32K` `mb=3 x ga=2`: 15,202.8 tok/s/chip (17.95% / 25.39%); `v5p-32` `64K` `mb=1 x ga=4`: 3,240.9 tok/s/chip (27.20% / 43.12%); `v5p-16` `64K` `mb=1 x ga=4`: 3,199.3 tok/s/chip (26.85% / 42.57%).
- **`64K` with `c4_test` EoD block skipping (control row, same recipe):** `v5p-16` 4,870.9 tok/s/chip (40.88% / 64.81%); `v5p-32` 4,765.8 tok/s/chip (40.00% / 63.41%); `v7x-16` 18,155.0 tok/s/chip (30.31% / 48.06%). This is the mode behind the historical `63–66%` (`v5p`) and `48%` (`v7x-16`) `64K` headline numbers further down.
- **`128K` / `256K`:** not reachable in this stack. Flint in this TorchTitan TPU build rejects context and tensor parallelism, so each rank must hold one full `131,072`-token sequence: with the Splash output saved in `49` or `32` layers the step runs out of HBM, and with `16` saved layers the first step did not finish within 30 minutes on `v5p-16` (2 attempts) or `v5p-32` (1 attempt). `256K` was not attempted.

Older 10–15-step tuning results (with `c4_test` EoD block skipping and the TorchTitan counter) are kept below as [historical results](#historical-short-run-tuning-results--script-reference).
<!-- HEADLINE:END -->

---

## Validated 60-Step Results (`CUSTOMER_VAL60`)

- **What this is:** the same fixed recipe at every context length, `60` optimizer steps per run, steady-state **mean over steps `11..60`** (no single-step peaks), constant `65,536` tokens/microbatch/rank (`mb=32 -> 1` from `2K -> 64K`), and **true unskipped full-sequence causal attention** (`TT_CONTIGUOUS_FULL_SEQ=1`: no `c4_test` end-of-document block skipping). Each `64K` table also has one `c4_test` EoD control row.
- **Recipe:** Native `TpuGraphTrainer` lane — `flint_40b_v014_tile1024` (`v7x`) / `flint_40b_v5p_n40` with Pallas paired GMM (`v5p`), plus `scale_before_down_proj`, BF16 RoPE, Splash output saved in all `49` layers, MoE routing saved, compiled grad-norm, fused AdamW, BF16 gradient reduce-scatter — see [`scripts/run_flint_customer_60step_validation.py`](./scripts/run_flint_customer_60step_validation.py) and [`k8s/v5p-16-flint-val60.yaml`](./k8s/v5p-16-flint-val60.yaml) · [`k8s/v5p-32-flint-val60.yaml`](./k8s/v5p-32-flint-val60.yaml) · [`k8s/v7x-16-flint-val60.yaml`](./k8s/v7x-16-flint-val60.yaml).
- **Two MFU columns:** **Causal MFU** counts attention exactly (lower triangle on the `14` full layers, causal ramp on the `35` `W=513` layers). **PaLM / TorchTitan MFU** is the upstream counter (`12·H·D·S` full square), i.e. the convention used by `google-pytorch/torchtitan` PR #122 / #123; it equals causal MFU plus the masked half of full-layer attention, so it grows with context length (`17.385` vs `16.570` GF/tok at `2K`, `61.073` vs `38.521` GF/tok at `64K`).
- **Why the historical `64K` headline numbers are higher:** the `c4_test` EoD control (`~400–500`-token documents) lets Splash skip cross-document blocks, so it runs `+54%` (`v5p-16`) / `+48%` (`v5p-32`) more tok/s than true full-sequence `64K` with the same recipe. The historical `63–65%` `64K` MFUs further down were measured in that mode and with the PaLM counter.
- **Per-slice files:** [`v5p-16`](./results/customer_val60/v5p-16/SUMMARY.md) · [`v5p-32`](./results/customer_val60/v5p-32/SUMMARY.md) · [`v7x-16`](./results/customer_val60/v7x-16/SUMMARY.md) (each with `summary.json` holding every per-step record).

| Slice | Run | Seq | `mb` × `ga` (tok/step) | Attention | **Steady tok/s/chip** | **Causal MFU** | PaLM / TorchTitan MFU | HBM (GiB) | Loss (`1 -> 60`) | Status |
|---|---|---:|---|---|---:|---:|---:|---:|---|---|
| TPU `v5p-16` (`2x2x4`, 16 chips) | `01_iso65k_s2048_mb32_ga1_fullseq` | `2,048` | `32` × `1` (`1.05M`) | Full-Seq | **5,874.4** | **21.21%** | 22.25% | 17.55 | `12.32226 -> 2.75305` | `PASS` |
| TPU `v5p-16` (`2x2x4`, 16 chips) | `02_iso65k_s16384_mb4_ga1_fullseq` | `16,384` | `4` × `1` (`1.05M`) | Full-Seq | **4,949.7** | **23.29%** | 29.39% | 17.55 | `12.33612 -> 2.79439` | `PASS` |
| TPU `v5p-16` (`2x2x4`, 16 chips) | `03_iso65k_s32768_mb2_ga1_fullseq` | `32,768` | `2` × `1` (`1.05M`) | Full-Seq | **4,170.0** | **24.75%** | 35.00% | 19.43 | `12.33914 -> 2.86616` | `PASS` |
| TPU `v5p-16` (`2x2x4`, 16 chips) | `04_iso65k_s65536_mb1_ga1_fullseq` | `65,536` | `1` × `1` (`1.05M`) | Full-Seq | **3,159.2** | **26.51%** | 42.04% | 18.83 | `12.34034 -> 2.86275` | `PASS` |
| TPU `v5p-16` (`2x2x4`, 16 chips) | `05_iso65k_s4096_mb16_ga1_fullseq` | `4,096` | `16` × `1` (`1.05M`) | Full-Seq | **5,799.7** | **21.90%** | 23.75% | 17.55 | `12.3292 -> 3.02743` | `PASS` |
| TPU `v5p-16` (`2x2x4`, 16 chips) | `06_iso65k_s8192_mb8_ga1_fullseq` | `8,192` | `8` × `1` (`1.05M`) | Full-Seq | **5,504.7** | **22.51%** | 25.92% | 17.55 | `12.33347 -> 3.06289` | `PASS` |
| TPU `v5p-16` (`2x2x4`, 16 chips) | `07_ctrl_s65536_mb1_ga1_c4eod` | `65,536` | `1` × `1` (`1.05M`) | `c4_test` EoD (control) | **4,870.9** | **40.88%** | 64.81% | 18.83 | `12.32183 -> 2.78776` | `PASS` |
| TPU `v5p-16` (`2x2x4`, 16 chips) | `08_opt_s65536_mb1_ga4_fullseq` | `65,536` | `1` × `4` (`4.19M`) | Full-Seq | **3,199.3** | **26.85%** | 42.57% | 18.83 | `12.33868 -> 2.87057` | `PASS` |
| TPU `v5p-32` (`2x4x4`, 32 chips) | `01_iso65k_s2048_mb32_ga1_fullseq` | `2,048` | `32` × `1` (`2.10M`) | Full-Seq | **5,925.7** | **21.39%** | 22.44% | 17.53 | `12.32683 -> 2.7928` | `PASS` |
| TPU `v5p-32` (`2x4x4`, 32 chips) | `02_iso65k_s16384_mb4_ga1_fullseq` | `16,384` | `4` × `1` (`2.10M`) | Full-Seq | **4,995.9** | **23.51%** | 29.66% | 17.53 | `12.3406 -> 2.80076` | `PASS` |
| TPU `v5p-32` (`2x4x4`, 32 chips) | `03_iso65k_s32768_mb2_ga1_fullseq` | `32,768` | `2` × `1` (`2.10M`) | Full-Seq | **4,214.4** | **25.01%** | 35.37% | 17.53 | `12.34433 -> 2.77856` | `PASS` |
| TPU `v5p-32` (`2x4x4`, 32 chips) | `04_iso65k_s65536_mb1_ga1_fullseq` | `65,536` | `1` × `1` (`2.10M`) | Full-Seq | **3,218.7** | **27.01%** | 42.83% | 14.96 | `12.34367 -> 2.78643` | `PASS` |
| TPU `v5p-32` (`2x4x4`, 32 chips) | `05_iso65k_s4096_mb16_ga1_fullseq` | `4,096` | `16` × `1` (`2.10M`) | Full-Seq | **5,809.7** | **21.94%** | 23.79% | 17.53 | `12.33364 -> 2.79027` | `PASS` |
| TPU `v5p-32` (`2x4x4`, 32 chips) | `06_iso65k_s8192_mb8_ga1_fullseq` | `8,192` | `8` × `1` (`2.10M`) | Full-Seq | **5,513.6** | **22.54%** | 25.96% | 17.53 | `12.33846 -> 2.78548` | `PASS` |
| TPU `v5p-32` (`2x4x4`, 32 chips) | `07_ctrl_s65536_mb1_ga1_c4eod` | `65,536` | `1` × `1` (`2.10M`) | `c4_test` EoD (control) | **4,765.8** | **40.00%** | 63.41% | 14.96 | `12.33451 -> 2.82996` | `PASS` |
| TPU `v5p-32` (`2x4x4`, 32 chips) | `08_opt_s65536_mb1_ga4_fullseq` | `65,536` | `1` × `4` (`8.39M`) | Full-Seq | **3,240.9** | **27.20%** | 43.12% | 14.96 | `12.34191 -> 2.84033` | `PASS` |
| TPU `v7x-16` (`2x2x4`, 16 chips / 32 TensorCores) | `01_iso65k_s2048_mb32_ga1_fullseq` | `2,048` | `32` × `1` (`2.10M`) | Full-Seq | **25,168.4** | **18.08%** | 18.97% | 15.52 | `12.31906 -> 2.82746` | `PASS` |
| TPU `v7x-16` (`2x2x4`, 16 chips / 32 TensorCores) | `02_iso65k_s16384_mb4_ga1_fullseq` | `16,384` | `4` × `1` (`2.10M`) | Full-Seq | **18,958.8** | **17.75%** | 22.39% | 14.02 | `12.33117 -> 2.78966` | `PASS` |
| TPU `v7x-16` (`2x2x4`, 16 chips / 32 TensorCores) | `03_iso65k_s32768_mb2_ga1_fullseq` | `32,768` | `2` × `1` (`2.10M`) | Full-Seq | **14,866.0** | **17.56%** | 24.82% | 14.02 | `12.33477 -> 2.78622` | `PASS` |
| TPU `v7x-16` (`2x2x4`, 16 chips / 32 TensorCores) | `04_iso65k_s65536_mb1_ga1_fullseq` | `65,536` | `1` × `1` (`2.10M`) | Full-Seq | **10,722.3** | **17.90%** | 28.38% | 14.02 | `12.33406 -> 2.78358` | `PASS` |
| TPU `v7x-16` (`2x2x4`, 16 chips / 32 TensorCores) | `05_iso65k_s4096_mb16_ga1_fullseq` | `4,096` | `16` × `1` (`2.10M`) | Full-Seq | **24,552.1** | **18.44%** | 20.00% | 15.52 | `12.32523 -> 2.80467` | `PASS` |
| TPU `v7x-16` (`2x2x4`, 16 chips / 32 TensorCores) | `06_iso65k_s8192_mb8_ga1_fullseq` | `8,192` | `8` × `1` (`2.10M`) | Full-Seq | **22,326.5** | **18.16%** | 20.92% | 16.02 | `12.32958 -> 2.78572` | `PASS` |
| TPU `v7x-16` (`2x2x4`, 16 chips / 32 TensorCores) | `07_ctrl_s65536_mb1_ga1_c4eod` | `65,536` | `1` × `1` (`2.10M`) | `c4_test` EoD (control) | **18,155.0** | **30.31%** | 48.06% | 14.02 | `12.32568 -> 2.822` | `PASS` |
| TPU `v7x-16` (`2x2x4`, 16 chips / 32 TensorCores) | `08_opt_s65536_mb1_ga4_fullseq` | `65,536` | `1` × `4` (`8.39M`) | Full-Seq | **10,829.4** | **18.08%** | 28.67% | 14.02 | `12.33261 -> 2.85837` | `PASS` |
| TPU `v7x-16` (`2x2x4`, 16 chips / 32 TensorCores) | `09_opt_s2048_mb48_ga2_fullseq` | `2,048` | `48` × `2` (`6.29M`) | Full-Seq | **26,947.8** | **19.36%** | 20.31% | 23.02 | `12.31829 -> 2.86263` | `PASS` |
| TPU `v7x-16` (`2x2x4`, 16 chips / 32 TensorCores) | `10_opt_s16384_mb6_ga2_fullseq` | `16,384` | `6` × `2` (`6.29M`) | Full-Seq | **19,323.3** | **18.09%** | 22.82% | 20.52 | `12.32923 -> 2.81084` | `PASS` |
| TPU `v7x-16` (`2x2x4`, 16 chips / 32 TensorCores) | `11_opt_s32768_mb3_ga2_fullseq` | `32,768` | `3` × `2` (`6.29M`) | Full-Seq | **15,202.8** | **17.95%** | 25.39% | 20.52 | `12.33113 -> 2.80935` | `PASS` |

- **Apples-to-apples comparison with `google-pytorch/torchtitan` PR #122 (`16.23%`) and PR #123 (`16.29%`):** see [Headline Results § A](#headline-results-validated-60-steps-per-run-steady-state-mean-of-steps-1160) and [`docs/ENGINEERING_COMPARISON.md`](./docs/ENGINEERING_COMPARISON.md).


---

## Historical Short-Run Tuning Results & Script Reference

> **Historical numbers.** These are 10–15-step tuning runs with `c4_test` end-of-document block skipping and the TorchTitan / PaLM MFU counter. At `32K`/`64K` they are not comparable to the validated full-sequence results above (see the `c4_test` EoD control rows).

Use the table below to jump directly to the script, Kubernetes manifest, result files, and configuration changes for each historical operating point:

| Configuration Summary | Baseline MFU → Optimized MFU (`Throughput`) | Run ID & Pass | Script & Line Anchor | K8s Manifest | Result Files | Key Changes Over Baseline |
|---|---|---|---|---|---|---|
| **TPU `v5p-16` (`64K` Native)** | `19.76%` → **`65.46%` / `66.20%`** (`78,717` tok/s = **`4.81` PF/s**) | `v5p32_p11_03` (`Pass 11`) | [`run_flint_v5p_pass11_mfu_frontier.py#L263`](./scripts/run_flint_v5p_pass11_mfu_frontier.py#L263-L270) | [`v5p-16-flint-pass11.yaml`](./k8s/v5p-16-flint-pass11.yaml) | [`SUMMARY.md`](./results/v5p16_pass11/SUMMARY.md) · [`JSON`](./results/v5p16_pass11/summary.json) | Recursive 49-layer `64K` (`max_context_length=65536`) + `AG/RS Overlap` + `PairedGMM tile_m=512` (bwd `256`) + `SBDP` + `BF16` RoPE + `PureFusedOpt` + `n49 = 0%` Attn Recompute (`mb=1`, `8×` GA = `8.39M` tok/step, `18.83 GiB` HBM) |
| **TPU `v5p-32` (`64K` Native)** | `19.76%` → **`63.00%` / `64.07%`** (`151,504` tok/s = **`9.25` PF/s**) | `v5p32_p11_03` (`Pass 11`) | [`run_flint_v5p_pass11_mfu_frontier.py#L263`](./scripts/run_flint_v5p_pass11_mfu_frontier.py#L263-L270) | [`v5p-32-flint-pass11.yaml`](./k8s/v5p-32-flint-pass11.yaml) | [`SUMMARY.md`](./results/v5p32_pass11/SUMMARY.md) · [`JSON`](./results/v5p32_pass11/summary.json) | Same `Pass 11` `64K` overlap recipe across `32` `v5p` chips (`8×` GA = `16.78M` tok/step, only **`14.96 GiB` HBM**, `96.2%` weak-scaling efficiency from `v5p-16`) |
| **TPU `v5p-32` (`32K` Native)** | `19.76%` → **`45.65%` / `45.89%`** (`174,064` tok/s = **`6.70` PF/s**) | `v5p32_p11_01` (`Pass 11`) | [`run_flint_v5p_pass11_mfu_frontier.py#L247`](./scripts/run_flint_v5p_pass11_mfu_frontier.py#L247-L254) | [`v5p-32-flint-pass11.yaml`](./k8s/v5p-32-flint-pass11.yaml) | [`SUMMARY.md`](./results/v5p32_pass11/SUMMARY.md) · [`JSON`](./results/v5p32_pass11/summary.json) | Recursive 49-layer `32K` (`max_context_length=32768`) + `tile_m=256` + `SBDP` + `BF16` RoPE + `PureFusedOpt` + `n44` (`mb=3 = 98.3K` tok/chip, `4×` GA = `12.58M` tok/step, `27.52 GiB` HBM) |
| **TPU `v7x-16` (`64K` Native)** | `16.23%` → **`47.92%` / `48.57%`** (`289,654` tok/s = **`17.69` PF/s**) | `v7x_p14_01` (`Pass 14`) | [`run_flint_v7x_pass14_mfu_frontier.py#L512`](./scripts/run_flint_v7x_pass14_mfu_frontier.py#L512-L519) | [`v7x-16-flint-pass14.yaml`](./k8s/v7x-16-flint-pass14.yaml) | [`SUMMARY.md`](./results/v7x16_pass14/SUMMARY.md) · [`JSON`](./results/v7x16_pass14/summary.json) | Recursive 49-layer `64K` (`max_context_length=65536`) + `Scout/HierSC XLA` + `BF16` Reduce + `SBDP` + `BF16` RoPE + `PureFusedOpt` + `n49 = 0%` Attn Recompute + `tile_m=1024` (`mb=1`, `8×` GA = `16.78M` tok/step, only **`14.02 GiB` HBM**) |
| **TPU `v7x-16` (`32K` Native)** | `16.23%` → **`34.27%` / `34.60%`** (`328,390` tok/s = **`12.65` PF/s**) | `v7x_p14_09` (`Pass 14`) | [`run_flint_v7x_pass14_mfu_frontier.py#L585`](./scripts/run_flint_v7x_pass14_mfu_frontier.py#L585-L595) | [`v7x-16-flint-pass14.yaml`](./k8s/v7x-16-flint-pass14.yaml) | [`SUMMARY.md`](./results/v7x16_pass14/SUMMARY.md) · [`JSON`](./results/v7x16_pass14/summary.json) | Recursive 49-layer `32K` (`max_context_length=32768`) + `SBDP` + `BF16` RoPE + `PureFusedOpt` + `n49` + `tile_m=1024` (`mb=3 = 98.3K` tok/rank, `8×` GA = `25.17M` tok/step, `20.52 GiB` HBM) |
| **TPU `v5p-32` (`16K` HF)** | `19.76%` → **`35.33%` / `36.20%`** (`190,451` tok/s) | `v5p32_p10_03` (`Pass 10`) | [`run_flint_v5p_pass10_longctx.py#L413`](./scripts/run_flint_v5p_pass10_longctx.py#L413-L420) | [`v5p-32-flint-pass10.yaml`](./k8s/v5p-32-flint-pass10.yaml) | [`SUMMARY.md`](./results/v5p32_pass10/SUMMARY.md) · [`JSON`](./results/v5p32_pass10/summary.json) | `PairedGMM tile_m=256` (`16 MiB` VMEM-safe) + [`SBDP` (`#L175`)](./scripts/run_flint_v5p_pass10_longctx.py#L175-L199) + `PureFusedOpt` + `Splash sa=1024` + [`HybridAC18` (`#L151`)](./scripts/run_flint_v5p_pass10_longctx.py#L151-L168) + `BF16` Reduce + [`XLA LHS` (`#L295`)](./scripts/run_flint_v5p_pass10_longctx.py#L295-L310) + `131K` tok/chip at `16K` (`mb=8`, `22.84 GiB` HBM) |
| **TPU `v5p-16` (`16K` HF)** | `19.76%` → **`35.21%` / `35.92%`** (`94,896` tok/s) | `v5p16_p10_03` (`Pass 10`) | [`run_flint_v5p_pass10_longctx.py#L413`](./scripts/run_flint_v5p_pass10_longctx.py#L413-L420) | [`v5p-16-flint-pass10.yaml`](./k8s/v5p-16-flint-pass10.yaml) | [`SUMMARY.md`](./results/v5p16_pass10/SUMMARY.md) · [`JSON`](./results/v5p16_pass10/summary.json) | Same `Pass 10` HF `16K` configuration on 16-chip `2x2x4` slice (`mb=8`, `2.10M` tok/step, `22.84 GiB` HBM, `+15.45 pp` over baseline) |
| **TPU `v5p-32` (`16K` Native)** | `VMEM OOM` → **`33.67%` / `34.63%`** (`181,462` tok/s) | `v5p32_p10_07` (`Pass 10`) | [`run_flint_v5p_pass10_longctx.py#L470`](./scripts/run_flint_v5p_pass10_longctx.py#L470-L477) | [`v5p-32-flint-pass10.yaml`](./k8s/v5p-32-flint-pass10.yaml) | [`SUMMARY.md`](./results/v5p32_pass10/SUMMARY.md) · [`JSON`](./results/v5p32_pass10/summary.json) | Resolved `v5p` Scoped VMEM OOM (`tile_m=512 -> 256` in [`#L87-L109`](./scripts/run_flint_v5p_pass10_longctx.py#L87-L109)) + 49-layer `max_context_length=16384` + `SBDP` + `BF16` RoPE + `PureFusedOpt` (`mb=6`) |
| **TPU `v7x-16` (`16K` Native)** | `16.23%` → **`27.15%` / `27.20%`** (`367,808` tok/s) | `v7x_p13_03` (`Pass 13`) | [`run_flint_v7x_pass13_longctx.py#L441`](./scripts/run_flint_v7x_pass13_longctx.py#L441-L447) | [`v7x-16-flint-pass13.yaml`](./k8s/v7x-16-flint-pass13.yaml) | [`SUMMARY.md`](./results/v7x16_pass13/SUMMARY.md) · [`JSON`](./results/v7x16_pass13/summary.json) | [`Scout/HierSC XLA` (`#L298`)](./scripts/run_flint_v7x_pass13_longctx.py#L298-L314) + `BF16` Reduce + [`SBDP` + `BF16` RoPE + `PureFusedOpt` + `n49 = 0%` Attn Recompute (`#L89`)](./scripts/run_flint_v7x_pass13_longctx.py#L89-L106) + `tile_m=1024` + `8×` GA at `16K` (`mb=6`, `20.52 GiB` HBM) |
| **TPU `v7x-16` (`16K` HF)** | `15.87%` → **`25.31%` / `25.35%`** (`342,832` tok/s) | `v7x_p13_07` (`Pass 13`) | [`run_flint_v7x_pass13_longctx.py#L477`](./scripts/run_flint_v7x_pass13_longctx.py#L477-L483) | [`v7x-16-flint-pass13.yaml`](./k8s/v7x-16-flint-pass13.yaml) | [`SUMMARY.md`](./results/v7x16_pass13/SUMMARY.md) · [`JSON`](./results/v7x16_pass13/summary.json) | [`PairedGMM tile_m=1024` (`#L348`)](./scripts/run_flint_v7x_pass13_longctx.py#L348-L354) + [`SBDP` (`#L176`)](./scripts/run_flint_v7x_pass13_longctx.py#L176-L200) + `PureFusedOpt` + `Splash sa=1024` + `HybridAC16` + `BF16` Reduce (`mb=8, ga=2`, `25.50 GiB` HBM) |
| **TPU `v7x-16` (`2K` Native)** | `16.23%` → **`20.80%` / `20.81%`** (`441,664` tok/s) | `v7x_p12g_02` (`Pass 12G`) | [`run_flint_v7x_pass12.py#L2005`](./scripts/run_flint_v7x_pass12.py#L2005-L2020) | [`v7x-16-flint-pass12g.yaml`](./k8s/v7x-16-flint-pass12g.yaml) | [`SUMMARY.md`](./results/v7x-16-flex-2x2x4_pass12g_20261101T114752Z/SUMMARY.md) · [`JSON`](./results/v7x-16-flex-2x2x4_pass12g_20261101T114752Z/summary.json) | `2K` throughput config: `mb=48` (`98.3K` tok/rank) + `8×` GA (`25.17M` tok/step) + `SBDP` + `BF16` RoPE + `PureFusedOpt` + `n49 = 0%` Attn Recompute (`23.02 GiB` HBM) |
| **TPU `v5p` (`2K` HF)** | `19.76%` → **`25.76%` (`v5p-16`) / `24.88%` (`v5p-32`)** | `v5p_p9_02` (`Pass 9`) | [`run_flint_v5p_pass9.py#L349`](./scripts/run_flint_v5p_pass9.py#L349-L365) | [`v5p-32-and-16-flint-pass9.yaml`](./k8s/v5p-32-and-16-flint-pass9.yaml) | [`v5p-16 MD`](./results/flint_v5p-16-2x2x4_pass9_SUMMARY.md) · [`v5p-32 MD`](./results/flint_v5p-32-2x4x4_pass9_SUMMARY.md) | `2K` `mb=64` (`131K` tok/chip) + `SBDP` + `FusedCompiledOpt` + `Splash sa=1024` + `HybridAC16` + `BF16` Reduce (`+6.00 pp` over baseline) |
| **TPU `v7x-16` (`StaticEP=2`)** | `Recompile/OOM` → **`11.85%` / `12.24%`** (`251,520` tok/s) | `v7x_p12_08` (`Pass 12B`) | [`run_flint_v7x_pass12.py#L53`](./scripts/run_flint_v7x_pass12.py#L53-L140) · [`#L1239`](./scripts/run_flint_v7x_pass12.py#L1239-L1248) | [`v7x-16-flint-pass12b.yaml`](./k8s/v7x-16-flint-pass12b.yaml) | [`SUMMARY.md`](./results/flint-results/v7x-16-flex-2x2x4_pass12b_20261101T080434Z/SUMMARY.md) · [`JSON`](./results/flint-results/v7x-16-flex-2x2x4_pass12b_20261101T080434Z/summary.json) | Replaced dynamic-shape `TokenDispatcher` (`torch.bincount` guard error) with fixed-capacity `[EP, E_local, C, D]` `StaticTpuAllToAllTokenDispatcher` within dual-TensorCore `v7x` package (`0` inter-chip hops) |

---


## 1. Model Overview

| Property | Value |
|---|---|
| **Hugging Face Checkpoint** | [`https://huggingface.co/RecursiveSI/flint`](https://huggingface.co/RecursiveSI/flint) (`15` safetensors shards, `79.59 GB` BF16) |
| **Architecture** | Sparse Mixture-of-Experts, `49` Transformer layers (`2` dense SwiGLU `inter=6144` + `47` MoE), `hidden_size=2048`, `vocab_size=150,144`, `32` Q / `4` KV heads × `head_dim=128`, `14` full-causal NoPE layers + `35` sliding-window (`W=513`) RoPE layers, per-head attention sinks |
| **MoE Routing** | `256` routed experts per layer, **top-6** active routing (`~2.52B` active parameters / `39.8B` total) |
| **FLOPs Accounting** | PaLM / TorchTitan counter: `17.385` GFLOP/tok (`2K`) · `18.794` (`4K`) · `21.613` (`8K`) · `27.250` (`16K`) · `38.524` (`32K`) · `61.073` (`64K`); exact causal: `16.570` · `17.330` · `18.767` · `21.599` · `27.244` · `38.521`; `6·N_active` = `15.093` — see [`flint_flops_per_token()`](./scripts/run_flint_v5p_pass10_longctx.py#L41-L45) |
| **Hardware Peak BF16** | TPU `v7x`: `1,153.5` BF16 TFLOP/s per TensorCore (`2` TensorCores/chip) · TPU `v5p`: `459.0` BF16 TFLOP/s per chip |

---

## 2. Software & Hardware Stack

- **Runtime Environment** (built via [`docker/Dockerfile`](./docker/Dockerfile) + config patcher [`docker/patch_flint_configs.py`](./docker/patch_flint_configs.py)):
  - `python:3.14`, `torch==2.15.0+cpu`, `torch_tpu==0.2.0` (GA)
  - `jax==0.11.1`, `libtpu==0.0.50`, `tokamax==0.0.14`
  - `transformers>=5.22.0`, `safetensors`, `accelerate`, `sentencepiece`
  - `libtcmalloc.so.4` (`LD_PRELOAD` enabled at runtime)
- **Two TorchTitan Execution Lanes Included in One Image:**
  1. **Native `TpuGraphTrainer` Lane (`/torchtitan-native`):** Built on [`google-pytorch/torchtitan` PR #122 (`alekseyv_flint`)](https://github.com/google-pytorch/torchtitan/pull/122) — native `TpuGroupedExperts`, `GraphSplitter` per-layer XLA executables, `SimpleFSDP`, `StaticTpuAllToAllTokenDispatcher`, and `FlintStateDictAdapter`.
  2. **HuggingFace Modeling Backend Lane (`/torchtitan-hf`):** Built on [`google-pytorch/torchtitan` PR #123 (`hf-backend-tpu-20261028-v020-flint-tune`)](https://github.com/google-pytorch/torchtitan/pull/123) — `transformers_modeling_backend` with Pallas GroupedMM expert replacement and SplashAttention.

---

## 3. Benchmark Results by Platform (historical short runs)

> 10–15-step tuning runs with `c4_test` EoD block skipping and the TorchTitan counter; validated 60-step numbers are in [Headline Results](#headline-results-validated-60-steps-per-run-steady-state-mean-of-steps-1160).

| Platform & Topology | Execution Lane, Run ID & Context Length | Baseline MFU | Optimized Steady MFU | Optimized Peak MFU | Per-Rank Tok/s | Slice Tok/s | C4 Loss (`Start -> End`) & HBM | Script & Manifest | Result Files |
|---|---|---:|---:|---:|---:|---:|---|---|---|
| **TPU `v5p-32` (`2x4x4`, 32c)** | HF `16K` (`v5p32_p10_03`: `SBDP` + `FusedOpt` + `sa=1024` + `HybridAC18`, `mb=8`, `4.19M` tok/step) | `19.76%` | **`35.33%`** *(+15.57 pp)* | **`36.20%`** | **`5,951.6`** | **`190,451`** | `12.29 -> 5.31` (`22.84 GiB`) | [`pass10.py#L413`](./scripts/run_flint_v5p_pass10_longctx.py#L413-L420) · [`yaml`](./k8s/v5p-32-flint-pass10.yaml) | [`MD`](./results/v5p32_pass10/SUMMARY.md) · [`JSON`](./results/v5p32_pass10/summary.json) |
| **TPU `v5p-16` (`2x2x4`, 16c)** | HF `16K` (`v5p16_p10_03`: `SBDP` + `FusedOpt` + `sa=1024` + `HybridAC18`, `mb=8`, `2.10M` tok/step) | `19.76%` | **`35.21%`** *(+15.45 pp)* | **`35.92%`** | **`5,931.0`** | **`94,896`** | `12.50 -> 3.55` (`22.84 GiB`) | [`pass10.py#L413`](./scripts/run_flint_v5p_pass10_longctx.py#L413-L420) · [`yaml`](./k8s/v5p-16-flint-pass10.yaml) | [`MD`](./results/v5p16_pass10/SUMMARY.md) · [`JSON`](./results/v5p16_pass10/summary.json) |
| **TPU `v5p-32` (`2x4x4`, 32c)** | Native `16K` (`v5p32_p10_07`: `SBDP` + `BF16-RoPE` + `FusedOpt` + `tile_m=256`, `mb=6`, `3.15M` tok/step) | `VMEM OOM` | **`33.67%`** *(+9.78 pp vs `2K`)* | **`34.63%`** | **`5,670.7`** | **`181,462`** | `12.33 -> 4.15` (`29.21 GiB`) | [`pass10.py#L470`](./scripts/run_flint_v5p_pass10_longctx.py#L470-L477) · [`yaml`](./k8s/v5p-32-flint-pass10.yaml) | [`MD`](./results/v5p32_pass10/SUMMARY.md) · [`JSON`](./results/v5p32_pass10/summary.json) |
| **TPU `v5p-32` (`2x4x4`, 32c)** | HF `8K` (`v5p32_p10_02`: `SBDP` + `FusedOpt` + `sa=1024` + `HybridAC16`, `mb=16`) | `19.76%` | **`29.63%`** *(+9.87 pp)* | **`29.84%`** | **`6,291.7`** | **`201,334`** | `12.29 -> 5.51` (`22.84 GiB`) | [`pass10.py#L404`](./scripts/run_flint_v5p_pass10_longctx.py#L404-L411) · [`yaml`](./k8s/v5p-32-flint-pass10.yaml) | [`MD`](./results/v5p32_pass10/SUMMARY.md) · [`JSON`](./results/v5p32_pass10/summary.json) |
| **TPU `v5p-16` (`2x2x4`, 16c)** | HF `8K` (`v5p16_p10_02`: `SBDP` + `FusedOpt` + `sa=1024` + `HybridAC16`, `mb=16`) | `19.76%` | **`29.25%`** *(+9.49 pp)* | **`29.42%`** | **`6,212.4`** | **`99,398`** | `12.50 -> 3.61` (`22.84 GiB`) | [`pass10.py#L404`](./scripts/run_flint_v5p_pass10_longctx.py#L404-L411) · [`yaml`](./k8s/v5p-16-flint-pass10.yaml) | [`MD`](./results/v5p16_pass10/SUMMARY.md) · [`JSON`](./results/v5p16_pass10/summary.json) |
| **TPU `v5p-32` (`2x4x4`, 32c)** | Native `8K` (`v5p32_p10_06`: `SBDP` + `BF16-RoPE` + `FusedOpt` + `tile_m=256`, `mb=12`) | `VMEM OOM` | **`28.42%`** | **`28.61%`** | **`6,036.3`** | **`193,162`** | `12.33 -> 4.16` (`29.21 GiB`) | [`pass10.py#L462`](./scripts/run_flint_v5p_pass10_longctx.py#L462-L469) · [`yaml`](./k8s/v5p-32-flint-pass10.yaml) | [`MD`](./results/v5p32_pass10/SUMMARY.md) · [`JSON`](./results/v5p32_pass10/summary.json) |
| **TPU `v5p-32` (`2x4x4`, 32c)** | HF `4K` (`v5p32_p10_01`: `SBDP` + `FusedOpt` + `sa=1024` + `HybridAC16`, `mb=32`) | `19.76%` | **`26.59%`** *(+6.83 pp)* | **`26.64%`** | **`6,494.7`** | **`207,830`** | `12.29 -> 5.27` (`22.84 GiB`) | [`pass10.py#L395`](./scripts/run_flint_v5p_pass10_longctx.py#L395-L402) · [`yaml`](./k8s/v5p-32-flint-pass10.yaml) | [`MD`](./results/v5p32_pass10/SUMMARY.md) · [`JSON`](./results/v5p32_pass10/summary.json) |
| **TPU `v5p-16` (`2x2x4`, 16c)** | HF `4K` (`v5p16_p10_01`: `SBDP` + `FusedOpt` + `sa=1024` + `HybridAC16`, `mb=32`) | `19.76%` | **`25.96%`** *(+6.20 pp)* | **`26.00%`** | **`6,339.7`** | **`101,435`** | `12.50 -> 3.68` (`22.84 GiB`) | [`pass10.py#L395`](./scripts/run_flint_v5p_pass10_longctx.py#L395-L402) · [`yaml`](./k8s/v5p-16-flint-pass10.yaml) | [`MD`](./results/v5p16_pass10/SUMMARY.md) · [`JSON`](./results/v5p16_pass10/summary.json) |
| **TPU `v5p-16` (`2x2x4`, 16c)** | HF `2K` (`v5p_p9_02`: `SBDP` + `FusedOpt` + `sa=1024` + `HybridAC16`, `mb=64`) | `19.76%` | **`25.76%`** *(+6.00 pp)* | **`25.81%`** | **`6,802.0`** | **`108,832`** | `12.50 -> 3.97` (`22.84 GiB`) | [`pass9.py#L349`](./scripts/run_flint_v5p_pass9.py#L349-L365) · [`yaml`](./k8s/v5p-32-and-16-flint-pass9.yaml) | [`MD`](./results/flint_v5p-16-2x2x4_pass9_SUMMARY.md) · [`JSON`](./results/flint_v5p-16-2x2x4_pass9_summary.json) |
| **TPU `v5p-32` (`2x4x4`, 32c)** | HF `2K` (`v5p_p9_02`: `SBDP` + `FusedOpt` + `sa=1024` + `HybridAC16`, `mb=64`) | `19.76%` | **`24.88%`** *(+5.12 pp)* | **`24.91%`** | **`6,568.4`** | **`210,189`** | `12.30 -> 4.31` (`22.84 GiB`) | [`pass9.py#L349`](./scripts/run_flint_v5p_pass9.py#L349-L365) · [`yaml`](./k8s/v5p-32-and-16-flint-pass9.yaml) | [`MD`](./results/flint_v5p-32-2x4x4_pass9_SUMMARY.md) · [`JSON`](./results/flint_v5p-32-2x4x4_pass9_summary.json) |
| **TPU `v5p-32` (`2x4x4`, 32c)** | Native `2K` (`v5p_p9_06`: `SBDP` + `BF16-RoPE` + `FusedOpt` + `tile_m=256`, `mb=48`) | `VMEM OOM` | **`23.89%`** | **`24.00%`** | **`6,308.1`** | **`201,859`** | `12.33 -> 4.17` (`28.11 GiB`) | [`pass9.py#L421`](./scripts/run_flint_v5p_pass9.py#L421-L435) · [`yaml`](./k8s/v5p-32-and-16-flint-pass9.yaml) | [`MD`](./results/flint_v5p-32-2x4x4_pass9_SUMMARY.md) · [`JSON`](./results/flint_v5p-32-2x4x4_pass9_summary.json) |
| **TPU `v5p-32` (`2x4x4`, 32c)** | HF `2K` + `HSDP 2×16` (`v5p_p8_06`, `2x` GA, `6.29M` tok/step) | `19.76%` | **`23.81%`** *(+4.05 pp)* | **`23.83%`** | **`6,285.7`** | **`201,142`** | `12.50 -> 3.78` (`21.08 GiB`) | [`pass8.py#L303`](./scripts/run_flint_v5p_pass8.py#L303-L320) · [`yaml`](./k8s/v5p-32-flint-pass8.yaml) | [`MD`](./results/flint_v5p-32-2x4x4_pass8_SUMMARY.md) · [`JSON`](./results/flint_v5p-32-2x4x4_pass8_summary.json) |
| **TPU `v7x-16` (`2x2x4`, 32 TC)** | Native `16K` (`v7x_p13_03`: Scout XLA + `SBDP` + `BF16-RoPE` + `PureFusedOpt` + `tile1024` + `n49`, `8x` GA = `25.17M`, `mb=6`) | `16.23%` / `16.56%` | **`27.15%`** *(+10.92 pp)* | **`27.20%`** *(`10.04` PF/s)* | **`11,494.0`** | **`367,808`** | `12.32 -> 3.74` (`20.52 GiB`) | [`pass13.py#L441`](./scripts/run_flint_v7x_pass13_longctx.py#L441-L447) · [`yaml`](./k8s/v7x-16-flint-pass13.yaml) | [`MD`](./results/v7x16_pass13/SUMMARY.md) · [`JSON`](./results/v7x16_pass13/summary.json) |
| **TPU `v7x-16` (`2x2x4`, 32 TC)** | HF `16K` (`v7x_p13_07`: `SBDP` + `PureFusedOpt` + `sa=1024` + `tile1024` + `HybridAC16`, `mb=8`, `2x` GA = `8.39M`) | `16.29%` / `15.87%` | **`25.31%`** *(+9.44 pp)* | **`25.35%`** *(`9.36` PF/s)* | **`10,713.5`** | **`342,832`** | `12.31 -> 3.98` (`25.50 GiB`) | [`pass13.py#L477`](./scripts/run_flint_v7x_pass13_longctx.py#L477-L483) · [`yaml`](./k8s/v7x-16-flint-pass13.yaml) | [`MD`](./results/v7x16_pass13/SUMMARY.md) · [`JSON`](./results/v7x16_pass13/summary.json) |
| **TPU `v7x-16` (`2x2x4`, 32 TC)** | Native `8K` (`v7x_p13_02`: Scout XLA + `SBDP` + `BF16-RoPE` + `PureFusedOpt` + `tile1024` + `n49`, `8x` GA, `mb=12`) | `16.23%` / `16.56%` | **`23.13%`** *(+6.90 pp)* | **`23.13%`** *(`8.54` PF/s)* | **`12,343.5`** | **`394,992`** | `12.32 -> 3.74` (`20.52 GiB`) | [`pass13.py#L432`](./scripts/run_flint_v7x_pass13_longctx.py#L432-L438) · [`yaml`](./k8s/v7x-16-flint-pass13.yaml) | [`MD`](./results/v7x16_pass13/SUMMARY.md) · [`JSON`](./results/v7x16_pass13/summary.json) |
| **TPU `v7x-16` (`2x2x4`, 32 TC)** | Native `4K` (`v7x_p13_01`: Scout XLA + `SBDP` + `BF16-RoPE` + `PureFusedOpt` + `tile1024` + `n49`, `8x` GA, `mb=24`) | `16.23%` / `16.56%` | **`21.92%`** *(+5.69 pp)* | **`21.93%`** | **`13,455.5`** | **`430,576`** | `12.32 -> 3.73` (`23.52 GiB`) | [`pass13.py#L423`](./scripts/run_flint_v7x_pass13_longctx.py#L423-L429) · [`yaml`](./k8s/v7x-16-flint-pass13.yaml) | [`MD`](./results/v7x16_pass13/SUMMARY.md) · [`JSON`](./results/v7x16_pass13/summary.json) |
| **TPU `v7x-16` (`2x2x4`, 32 TC)** | HF `8K` (`v7x_p13_06`: `SBDP` + `PureFusedOpt` + `sa=1024` + `tile1024` + `HybridAC12`, `mb=16`, `2x` GA) | `16.29%` / `15.87%` | **`21.58%`** *(+5.71 pp)* | **`21.61%`** | **`11,517.0`** | **`368,544`** | `12.31 -> 3.51` (`25.50 GiB`) | [`pass13.py#L468`](./scripts/run_flint_v7x_pass13_longctx.py#L468-L474) · [`yaml`](./k8s/v7x-16-flint-pass13.yaml) | [`MD`](./results/v7x16_pass13/SUMMARY.md) · [`JSON`](./results/v7x16_pass13/summary.json) |
| **TPU `v7x-16` (`2x2x4`, 32 TC)** | Native `2K` (`v7x_p12g_02`: Scout XLA + `SBDP` + `BF16-RoPE` + `PureFusedOpt` + `n49`, `8x` GA = `25.17M`, `mb=48`) | `16.23%` / `16.56%` | **`20.80%`** *(+4.57 pp)* | **`20.81%`** | **`13,802.0`** | **`441,664`** | `12.32 -> 3.71` (`23.02 GiB`) | [`pass12.py#L2005`](./scripts/run_flint_v7x_pass12.py#L2005-L2020) · [`yaml`](./k8s/v7x-16-flint-pass12g.yaml) | [`MD`](./results/v7x-16-flex-2x2x4_pass12g_20261101T114752Z/SUMMARY.md) · [`JSON`](./results/v7x-16-flex-2x2x4_pass12g_20261101T114752Z/summary.json) |
| **TPU `v7x-16` (`2x2x4`, 32 TC)** | Native `2K` (`v7x_p12g_01`: `SBDP` + `BF16-RoPE` + `PureFusedOpt` + `n49`, `16x` GA = `50.33M`, `mb=48`) | `16.23%` / `16.56%` | **`20.78%`** *(+4.55 pp)* | **`20.79%`** | **`13,787.0`** | **`441,184`** | `12.32 -> 3.76` (`23.02 GiB`) | [`pass12.py#L1987`](./scripts/run_flint_v7x_pass12.py#L1987-L2004) · [`yaml`](./k8s/v7x-16-flint-pass12g.yaml) | [`MD`](./results/v7x-16-flex-2x2x4_pass12g_20261101T114752Z/SUMMARY.md) · [`JSON`](./results/v7x-16-flex-2x2x4_pass12g_20261101T114752Z/summary.json) |
| **TPU `v7x-16` (`2x2x4`, 32 TC)** | HF `4K` (`v7x_p13_05`: `SBDP` + `PureFusedOpt` + `sa=1024` + `tile1024` + `HybridAC12`, `mb=32`, `2x` GA) | `16.29%` / `15.87%` | **`19.58%`** *(+3.71 pp)* | **`19.59%`** | **`12,018.5`** | **`384,592`** | `12.31 -> 4.05` (`25.50 GiB`) | [`pass13.py#L459`](./scripts/run_flint_v7x_pass13_longctx.py#L459-L465) · [`yaml`](./k8s/v7x-16-flint-pass13.yaml) | [`MD`](./results/v7x16_pass13/SUMMARY.md) · [`JSON`](./results/v7x16_pass13/summary.json) |
| **TPU `v7x-16` (`2x2x4`, 32 TC)** | Native Full Host Offload (`v7x_p12e_04`: `save_last_n=0, prefetch=2` + `SBDP` + `BF16-RoPE`, `mb=64`, `2x` GA) | `OOM` | **`18.60%`** | **`18.61%`** | **`12,342.7`** | **`394,966`** | `12.32 -> 3.28` (`34.91 GiB`) | [`pass12.py#L1785`](./scripts/run_flint_v7x_pass12.py#L1785-L1805) · [`yaml`](./k8s/v7x-16-flint-pass12f.yaml) | [`MD`](./results/v7x-16-flex-2x2x4_pass12e/SUMMARY.md) · [`JSON`](./results/v7x-16-flex-2x2x4_pass12e/summary.json) |
| **TPU `v7x-16` (`2x2x4`, 32 TC)** | HF `2K` (`v7x_p12_04`: `SBDP` + `PureFusedOpt` + `sa=1024` + `tile1024` + `HybridAC12`, `mb=64`, `2x` GA) | `16.29%` / `15.87%` | **`18.58%`** *(+2.71 pp)* | **`18.59%`** | **`12,331.0`** | **`394,592`** | `12.31 -> 3.76` (`25.50 GiB`) | [`pass12.py#L1193`](./scripts/run_flint_v7x_pass12.py#L1193-L1210) · [`yaml`](./k8s/v7x-16-flint-pass12.yaml) | [`MD`](./results/flint_v7x-16-flex-2x2x4_pass12_SUMMARY.md) · [`JSON`](./results/flint_v7x-16-flex-2x2x4_pass12_summary.json) |
| **TPU `v7x-16` (`2x2x4`, 32 TC)** | HF `2K` + `HSDP 2×16` (`v7x_p12c_05`: `SBDP` + `PureFusedOpt` + `tile1024`, `mb=48`, `2x` GA) | `15.87%` | **`18.25%`** *(+2.38 pp)* | **`18.27%`** | **`12,114.8`** | **`387,674`** | `12.43 -> 3.58` (`21.22 GiB`) | [`pass12.py#L1490`](./scripts/run_flint_v7x_pass12.py#L1490-L1515) · [`yaml`](./k8s/v7x-16-flint-pass12c.yaml) | [`MD`](./results/v7x-16-flex-2x2x4_pass12c_20261101T085733Z/SUMMARY.md) · [`JSON`](./results/v7x-16-flex-2x2x4_pass12c_20261101T085733Z/summary.json) |
| **TPU `v7x-16` (`2x2x4`, 32 TC)** | Intra-Chip `StaticEP=2 × FSDP=16 + ZeRO-3 + PairedGMM-512` (`v7x_p12_08`, `0` Inter-Chip A2A) | `Recompile/OOM` | **`11.85%`** *(+76% vs EP8)* | **`12.24%`** | **`7,860.0`** | **`251,520`** | `12.32 -> 3.69` (`19.73 GiB`) | [`pass12.py#L1239`](./scripts/run_flint_v7x_pass12.py#L1239-L1248) · [`yaml`](./k8s/v7x-16-flint-pass12b.yaml) | [`MD`](./results/flint-results/v7x-16-flex-2x2x4_pass12b_20261101T080434Z/SUMMARY.md) · [`JSON`](./results/flint-results/v7x-16-flex-2x2x4_pass12b_20261101T080434Z/summary.json) |
| **TPU `v7x-16` (`2x2x4`, 32 TC)** | Intra-Tray `StaticEP=4 × FSDP=8 + ZeRO-3 + PairedGMM-512` (`v7x_p12_09`, `2`-Chip A2A) | `Recompile/OOM` | **`7.17%`** | **`7.37%`** | **`4,759.5`** | **`152,304`** | `12.32 -> 3.69` (`21.05 GiB`) | [`pass12.py#L1249`](./scripts/run_flint_v7x_pass12.py#L1249-L1260) · [`yaml`](./k8s/v7x-16-flint-pass12b.yaml) | [`MD`](./results/flint-results/v7x-16-flex-2x2x4_pass12b_20261101T080434Z/SUMMARY.md) · [`JSON`](./results/flint-results/v7x-16-flex-2x2x4_pass12b_20261101T080434Z/summary.json) |
| **TPU `v5p-32` (`2x4x4`, 32c)** | Native `StaticEP=8 × FSDP=4 + ZeRO-2 + PairedGMM-256` (`v5p_p9f_02`) | `Recompile/OOM` | **`7.30%`** | **`7.34%`** | **`1,926.1`** | **`61,635`** | `12.33 -> 4.21` (`37.98 GiB`) | [`pass9b.py#L1138`](./scripts/run_flint_v5p_pass9b.py#L1138-L1155) · [`yaml`](./k8s/v5p-32-flint-pass9f.yaml) | [`MD`](./results/flint_v5p-16-4n-2x2x4_pass9f_SUMMARY.md) · [`JSON`](./results/flint_v5p-16-4n-2x2x4_pass9f_summary.json) |
| **TPU `v7x-16` (`2x2x4`, 32 TC)** | Intra-Host `StaticEP=8 × FSDP=4 + ZeRO-3 + PairedGMM-256` (`v7x_p11d_04`, `4`-Chip A2A) | `Recompile/OOM` | **`6.73%`** | **`6.81%`** | **`4,463.0`** | **`142,816`** | `12.32 -> 4.19` (`22.69 GiB`) | [`pass11.py#L1158`](./scripts/run_flint_v7x_pass11.py#L1158-L1175) · [`yaml`](./k8s/v7x-16-flint-pass11d.yaml) | [`MD`](./results/flint_v7x-16-flex-2x2x4_pass11d_SUMMARY.md) · [`JSON`](./results/flint_v7x-16-flex-2x2x4_pass11d_summary.json) |

---

## 4. MFU Optimization Progression (historical short runs)

### 4.1 Native Trainer on TPU `v7x-16` (`16.23%` to `47.92%` MFU)

- **Baseline (`flint_40b_v011_n40`):** `TpuGraphTrainer` with `GraphSplitter` (`1` fwd + `1` bwd XLA executable/layer), `SimpleFSDP`, `(E, K, N)` weight layout, Paired W1/W3 Pallas GroupedMM at `tile_m=512`, `mb=32` (`65,536` tok/rank), saving SplashAttention in `40/49` layers (`n40`, recomputing 9 layers), FP32 ReduceScatter, FP32 RoPE casts, and eager DTensor AdamW + GradNorm (`16.23%` MFU on `v7x-8`, `16.56%` on `v7x-16`).

| Step | Experiment ID | Code / Config Change Over Previous Step | Steady / Cluster `tok/s` | Steady / Peak MFU | Delta vs `v7x-8` | Script / Config Anchor | Result Files |
|---|---|---|---|---|---:|---|---|
| **0 (Ref)** | *PR #122 `v7x-8`* | Upstream `flint_40b_v011_n40` (`16` ranks, `mb=32`, `tile_m=512`, `n40`, FP32 Reduce) | `10,779.0` / `172,464` | `16.23%` / — | — | [PR #122](https://github.com/google-pytorch/torchtitan/pull/122) | — |
| **1 (`Pass 1`)** | `02_native_pr122_v011_n40` | Direct `v7x-16` baseline (`32` ranks, `SimpleFSDP32`, `2.10M` tok/step) | `10,985.3` / `351,530` | `16.56%` / `16.69%` | `+0.33 pp` | [`run_flint_sweep.py#L102`](./scripts/run_flint_sweep.py#L102-L117) | [`Pass 1 MD`](./results/flint_v7x-16-2x2x4_SUMMARY.md) |
| **2 (`Pass 1`)** | `03_native_flint_40b_v012_hier_rs` | Enable multi-axis ND Hierarchical SparseCore Reduce-Scatter across `2x2x4` | `11,011.9` / `352,381` | `16.60%` / `16.70%` | `+0.37 pp` | [`patch_flint_configs.py#L21`](./docker/patch_flint_configs.py#L21-L29) | [`Pass 1 MD`](./results/flint_v7x-16-2x2x4_SUMMARY.md) |
| **3 (`Pass 1`)** | `04_native_flint_40b_v013_n49` | Save SplashAttention across all 49 layers (`n49`) (`98.0%` XLA cache hit rate) | `11,135.8` / `356,346` | `16.78%` / `16.86%` | `+0.55 pp` | [`patch_flint_configs.py#L30`](./docker/patch_flint_configs.py#L30-L36) | [`Pass 1 MD`](./results/flint_v7x-16-2x2x4_SUMMARY.md) |
| **4 (`Pass 1`)** | `05_native_flint_40b_v014_tile1024` | Widen forward Paired W1/W3 GroupedMM MXU tile `512 -> 1024` (`(1024, 2048, 2048)`) + VMEM-safe bwd `(512, 2048, 0)` | `11,196.2` / `358,278` | `16.87%` / `16.94%` | `+0.64 pp` | [`patch_flint_configs.py#L37`](./docker/patch_flint_configs.py#L37-L52) | [`Pass 1 MD`](./results/flint_v7x-16-2x2x4_SUMMARY.md) |
| **5 (`Pass 3`)** | `p3_01_native_comp_opt_and_gradnorm_n49` | Compile 779-param DTensor AdamW (`optimizer_compile="separate_graph"`) + global L2 GradNorm | `11,448.6` / `366,355` | `17.25%` / `17.32%` | `+1.02 pp` | [`run_flint_pass3.py#L54`](./scripts/run_flint_pass3.py#L54-L61) | [`Pass 3 MD`](./results/flint_v7x-16-2x2x4_pass3_SUMMARY.md) |
| **6 (`Pass 3`)** | `p3_02_native_save_moe_routing_n49` | Save `moe.routing` across all 49 layers (skip backward router logits + top-6 sort recompute) | `11,540.8` / `369,306` | `17.39%` / `17.46%` | `+1.16 pp` | [`run_flint_pass3.py#L62`](./scripts/run_flint_pass3.py#L62-L72) | [`Pass 3 MD`](./results/flint_v7x-16-2x2x4_pass3_SUMMARY.md) |
| **7 (`Pass 7`)** | `p7_01_native_grad_accum_2x_mb48_n43` | Increase microbatch `1.5×` (`mb=48`) + `2×` gradient accumulation (`6.29M` tok/step) + `n43` | `12,182.1` / `389,827` | `18.36%` / `18.42%` | `+2.13 pp` | [`run_flint_pass7_v7x_next.py#L5`](./scripts/run_flint_pass7_v7x_next.py#L5-L45) | [`Pass 7 Dir`](./results/v7x-16-2x2x4_pass7_20261030T190252Z) |
| **8 (`Pass 9`)** | `v7x_p9_02_native_pr122_bf16red_scout_n43_mb48_ga2` | Switch gradient `ReduceScatter` to `BF16` (`159.2 -> 79.6 GB` ICI) + Scout async AG/RS overlap | `12,392.8` / `396,570` | `18.68%` / `18.76%` | `+2.45 pp` | [`run_flint_v7x_pass9.py#L270`](./scripts/run_flint_v7x_pass9.py#L270-L290) | [`Pass 9 MD`](./results/flint_v7x-16-2x2x4_pass9_SUMMARY.md) |
| **9 (`Pass 12C`)** | `v7x_p12c_01_native_p9base_sbdp_bf16rope_n41_mb48_ga2_log5` | Add `scale_before_down_proj=True` (`SBDP`) + `BF16` RoPE (`25.48 GiB` HBM) | `13,675.7` / `437,622` | `20.61%` / `20.63%` | `+4.38 pp` | [`run_flint_v7x_pass12.py#L1520`](./scripts/run_flint_v7x_pass12.py#L1520-L1540) | [`Pass 12C MD`](./results/flint_v7x-16-flex-2x2x4_pass12c_SUMMARY.md) |
| **10 (`Pass 12G` `2K`)** | `v7x_p12g_02_native_p9base_sbdp_bf16rope_fusedopt_n49_mb48_ga8_log5` | `n49 = 0%` Attn Recompute + `optimizer.implementation="fused"` at `8×` GA (`25.17M` tok/step, `23.02 GiB` HBM) | **`13,802.0` / `441,664`** | **`20.80%` / `20.81%`** | **`+4.57 pp`** | [`run_flint_v7x_pass12.py#L2005`](./scripts/run_flint_v7x_pass12.py#L2005-L2020) | [`Pass 12G MD`](./results/v7x-16-flex-2x2x4_pass12g_20261101T114752Z/SUMMARY.md) |
| **11 (`Pass 13` `4K`)** | `v7x_p13_01_native_s4096_mb24_ga8_sbdp_bf16rope_fusedopt_n49` | Scale sequence length `2K -> 4K` (`mb=24 = 98,304` tok/rank, 49-layer `max_context_length` in [`#L65`](./scripts/run_flint_v7x_pass13_longctx.py#L65-L87)) | **`13,455.5` / `430,576`** | **`21.92%` / `21.93%`** | **`+5.69 pp`** | [`run_flint_v7x_pass13_longctx.py#L423`](./scripts/run_flint_v7x_pass13_longctx.py#L423-L429) | [`Pass 13 MD`](./results/v7x16_pass13/SUMMARY.md) |
| **12 (`Pass 13` `8K`)** | `v7x_p13_02_native_s8192_mb12_ga8_sbdp_bf16rope_fusedopt_n49` | Scale sequence length `4K -> 8K` (`mb=12 = 98,304` tok/rank, `8.539` PFLOP/s sustained, `20.52 GiB` HBM) | **`12,343.5` / `394,992`** | **`23.13%` / `23.13%`** | **`+6.90 pp`** | [`run_flint_v7x_pass13_longctx.py#L432`](./scripts/run_flint_v7x_pass13_longctx.py#L432-L438) | [`Pass 13 MD`](./results/v7x16_pass13/SUMMARY.md) |
| **13 (`Pass 13` `16K`)** | `v7x_p13_03_native_s16384_mb6_ga8_sbdp_bf16rope_fusedopt_n49` | Scale sequence length `8K -> 16K` (`mb=6 = 98,304` tok/rank, `10.023` PFLOP/s sustained / `10.039` PFLOP/s peak, `20.52 GiB` HBM) | **`11,494.0` / `367,808`** | **`27.15%` / `27.20%`** | **`+10.92 pp`** | [`run_flint_v7x_pass13_longctx.py#L441`](./scripts/run_flint_v7x_pass13_longctx.py#L441-L447) | [`Pass 13 MD`](./results/v7x16_pass13/SUMMARY.md) |
| **14 (`Pass 14` `32K`)** | `v7x_p14_09_native_s32768_mb3_ga8_sbdp_bf16rope_fusedopt_n49` | Scale sequence length `16K -> 32K` (`mb=3 = 98,304` tok/rank, `12.65` PFLOP/s sustained, `20.52 GiB` HBM) | **`10,262.2` / `328,390`** | **`34.27%` / `34.60%`** | **`+18.04 pp`** | [`run_flint_v7x_pass14_mfu_frontier.py#L585`](./scripts/run_flint_v7x_pass14_mfu_frontier.py#L585-L595) | [`Pass 14 MD`](./results/v7x16_pass14/SUMMARY.md) |
| **15 (`Pass 14` `64K`)** | `v7x_p14_01_native_s65536_mb1_ga8_sbdp_bf16rope_fusedopt_n49` | Scale sequence length `32K -> 64K` (`mb=1 = 65,536` tok/rank, **`17.69` PFLOP/s sustained / `17.93` PFLOP/s peak**, only **`14.02 GiB` HBM**) | **`9,051.7` / `289,654`** | **`47.92%` / `48.57%`** | **`+31.69 pp`** | [`run_flint_v7x_pass14_mfu_frontier.py#L512`](./scripts/run_flint_v7x_pass14_mfu_frontier.py#L512-L519) | [`Pass 14 MD`](./results/v7x16_pass14/SUMMARY.md) |

---

### 4.2 HuggingFace Backend on TPU `v7x-16` (`15.87%` to `25.31%` MFU)

| Step | Experiment ID | Code / Config Change Over Previous Step | Steady / Cluster `tok/s` | Steady / Peak MFU | Peak HBM | Script / Patch Anchor | Result Files |
|---|---|---|---|---|---:|---|---|
| **1 (`Pass 1` Base)** | `01_hf_backend_pr123_flint_40b` | Default PR #123 (`use_pallas_paired_grouped_mm=False`, FP32 Reduce, eager AdamW, `mb=32`, `sa=1024`) | `10,532.0` / `337,024` | `15.87%` / `15.91%` | `11.78 GiB` | [`run_flint_sweep.py#L80`](./scripts/run_flint_sweep.py#L80-L100) | [`Pass 1 MD`](./results/flint_v7x-16-2x2x4_SUMMARY.md) |
| **2 (`Pass 9`)** | `v7x_p9_03_hf_pr123_bf16red_scout_compopt_mb48_1x` | Enable Paired W1/W3 GMM (`tile_m=1024`, bwd `512:2048:0`) + `BF16` Reduce + Scout XLA + CompiledOpt (`mb=48`) | `12,132.8` / `388,250` | `18.29%` / `18.34%` | `16.92 GiB` | [`run_flint_v7x_pass9.py#L290`](./scripts/run_flint_v7x_pass9.py#L290-L320) | [`Pass 9 MD`](./results/flint_v7x-16-2x2x4_pass9_SUMMARY.md) |
| **3 (`Pass 12` `2K`)** | `v7x_p12_04_hf_pr123_sbdp_purefused_sa1024_tile1024_hybridac12_mb64_ga2` | Add `TT_HF_SBDP=1` + `PureFusedOpt` + `HybridAC N=12` + scale `mb=64` (`131K` tok/rank, `2×` GA = `8.39M` tok/step) | `12,331.0` / `394,592` | **`18.58%` / `18.59%`** | `25.50 GiB` | [`run_flint_v7x_pass12.py#L1193`](./scripts/run_flint_v7x_pass12.py#L1193-L1210) | [`Pass 12 MD`](./results/flint_v7x-16-flex-2x2x4_pass12_SUMMARY.md) |
| **4 (`Pass 13` `4K`)** | `v7x_p13_05_hf_pr123_s4096_mb32_ga2_sbdp_fusedopt_sa1024_tile1024_ac12` | Scale sequence length `2K -> 4K` at constant `131,072` tok/rank (`mb=32`, `2×` GA = `8.39M` tok/step, `25.50 GiB` HBM) | `12,018.5` / `384,592` | **`19.58%` / `19.59%`** | `25.50 GiB` | [`run_flint_v7x_pass13_longctx.py#L459`](./scripts/run_flint_v7x_pass13_longctx.py#L459-L465) | [`Pass 13 MD`](./results/v7x16_pass13/SUMMARY.md) |
| **5 (`Pass 13` `8K`)** | `v7x_p13_06_hf_pr123_s8192_mb16_ga2_sbdp_fusedopt_sa1024_tile1024_ac12` | Scale sequence length `4K -> 8K` at constant `131,072` tok/rank (`mb=16`, `2×` GA = `8.39M` tok/step, `25.50 GiB` HBM) | `11,517.0` / `368,544` | **`21.58%` / `21.61%`** | `25.50 GiB` | [`run_flint_v7x_pass13_longctx.py#L468`](./scripts/run_flint_v7x_pass13_longctx.py#L468-L474) | [`Pass 13 MD`](./results/v7x16_pass13/SUMMARY.md) |
| **6 (`Pass 13` `16K`)** | `v7x_p13_07_hf_pr123_s16384_mb8_ga2_sbdp_fusedopt_sa1024_tile1024_ac16` | Scale sequence length `8K -> 16K` (`mb=8`, `HybridAC N=16`, `9.342` / `9.357` PFLOP/s, `25.50 GiB` HBM) | **`10,713.5` / `342,832`** | **`25.31%` / `25.35%`** | `25.50 GiB` | [`run_flint_v7x_pass13_longctx.py#L477`](./scripts/run_flint_v7x_pass13_longctx.py#L477-L483) | [`Pass 13 MD`](./results/v7x16_pass13/SUMMARY.md) |

---

### 4.3 TPU `v5p-16` & `v5p-32` Progression (`19.76%` to `65.46%` MFU)

| Step | Slice | Experiment ID | Code / Config Change Over Previous Step | Steady / Cluster `tok/s` | Steady / Peak MFU | Peak HBM | Script / Patch Anchor | Result Files |
|---|---|---|---|---|---:|---:|---|---|
| **1 (`Pass 1` Base)** | `v5p-16` | `01_hf_backend_pr123_flint_v5p` | Default HF PR #123 (`2K`, `mb=32`, unpaired GMM, FP32 Reduce); Native PR #122 hit `v5p` `16 MiB` VMEM limit at `tile_m=512` | `5,215.8` / `83,453` | `19.76%` / `19.79%` | `11.78 GiB` | [`run_flint_sweep.py#L201`](./scripts/run_flint_sweep.py#L201-L225) | [`v5p Pass 1 MD`](./results/flint_v5p-16-2x2x4_SUMMARY.md) |
| **2 (`Pass 8`)** | `v5p-32` | `v5p_p8_02_hf_scout_xla_bf16red_mb48` | Add `BF16` ReduceScatter + XLA LHS / AG-Bwd Pipelining + `VMEM=80MiB` + `PairedGMM tile_m=256` (`mb=48`) | `6,565.3` / `210,090` | `24.87%` / `24.95%` | `16.92 GiB` | [`run_flint_v5p_pass8.py#L230`](./scripts/run_flint_v5p_pass8.py#L230-L265) | [`v5p-32 Pass 8 MD`](./results/flint_v5p-32-2x4x4_pass8_SUMMARY.md) |
| **3A (`Pass 9` `2K`)** | `v5p-16` | `v5p_p9_02_hf_pr123_sbdp_fusedopt_sa1024_hybridac16_mb64_1x` | Add `TT_HF_SBDP=1` + `FusedCompiledOpt` + `Splash sa=1024` + `HybridAC N=16` + scale `mb=64` (`131K` tok/chip) | `6,802.0` / `108,832` | **`25.76%` / `25.81%`** | `22.84 GiB` | [`run_flint_v5p_pass9.py#L349`](./scripts/run_flint_v5p_pass9.py#L349-L365) | [`v5p-16 Pass 9 MD`](./results/flint_v5p-16-2x2x4_pass9_SUMMARY.md) |
| **3B (`Pass 9` `2K`)** | `v5p-32` | `v5p_p9_02_hf_pr123_sbdp_fusedopt_sa1024_hybridac16_mb64_1x` | Same `Pass 9` `mb=64` recipe across `32` `v5p` chips (`4.19M` tok/step); Native `v5p_p9_06` (`tile_m=256`) reached `23.89%` / `24.00%` | `6,568.4` / `210,189` | **`24.88%` / `24.91%`** | `22.84 GiB` | [`run_flint_v5p_pass9.py#L349`](./scripts/run_flint_v5p_pass9.py#L349-L435) | [`v5p-32 Pass 9 MD`](./results/flint_v5p-32-2x4x4_pass9_SUMMARY.md) |
| **4 (`Pass 10` `4K`)** | `v5p-32` / `v5p-16` | `v5p32_p10_01` / `v5p16_p10_01` | Scale sequence length `2K -> 4K` at constant `131,072` tok/chip (`mb=32`, `22.84 GiB` HBM) | `6,494.7` / `207,830` *(32c)* | **`26.59%`** *(32c)* / **`25.96%`** *(16c)* | `22.84 GiB` | [`run_flint_v5p_pass10_longctx.py#L395`](./scripts/run_flint_v5p_pass10_longctx.py#L395-L402) | [`v5p-32 MD`](./results/v5p32_pass10/SUMMARY.md) · [`v5p-16 MD`](./results/v5p16_pass10/SUMMARY.md) |
| **5 (`Pass 10` `8K`)** | `v5p-32` / `v5p-16` | `v5p32_p10_02` / `v5p16_p10_02` | Scale sequence length `4K -> 8K` at constant `131,072` tok/chip (`mb=16`, `22.84 GiB` HBM; Native `v5p32_p10_06` reached `28.42%`) | `6,291.7` / `201,334` *(32c)* | **`29.63%`** *(32c)* / **`29.25%`** *(16c)* | `22.84 GiB` | [`run_flint_v5p_pass10_longctx.py#L404`](./scripts/run_flint_v5p_pass10_longctx.py#L404-L411) | [`v5p-32 MD`](./results/v5p32_pass10/SUMMARY.md) · [`v5p-16 MD`](./results/v5p16_pass10/SUMMARY.md) |
| **6A (`Pass 10` `16K` HF)** | `v5p-32` | `v5p32_p10_03_hf_pr123_s16384_mb8_1x_sbdp_fusedopt_sa1024_ac18` | Scale sequence length `8K -> 16K` (`mb=8 = 131K` tok/chip, `HybridAC N=18` in [`#L151`](./scripts/run_flint_v5p_pass10_longctx.py#L151-L168), `22.84 GiB` HBM, `+15.57 pp` vs baseline) | **`5,951.6` / `190,451`** | **`35.33%` / `36.20%`** | **`22.84 GiB`** | [`run_flint_v5p_pass10_longctx.py#L413`](./scripts/run_flint_v5p_pass10_longctx.py#L413-L420) | [`SUMMARY.md`](./results/v5p32_pass10/SUMMARY.md) · [`JSON`](./results/v5p32_pass10/summary.json) |
| **6B (`Pass 10` `16K` HF)** | `v5p-16` | `v5p16_p10_03_hf_pr123_s16384_mb8_1x_sbdp_fusedopt_sa1024_ac18` | Same `16K` `mb=8` configuration on 16-chip `v5p-16` (`2.10M` tok/step, `22.84 GiB` HBM, `+15.45 pp` vs baseline) | **`5,931.0` / `94,896`** | **`35.21%` / `35.92%`** | **`22.84 GiB`** | [`run_flint_v5p_pass10_longctx.py#L413`](./scripts/run_flint_v5p_pass10_longctx.py#L413-L420) | [`SUMMARY.md`](./results/v5p16_pass10/SUMMARY.md) · [`JSON`](./results/v5p16_pass10/summary.json) |
| **6C (`Pass 10` `16K` Native)** | `v5p-32` | `v5p32_p10_07_native_pr122_s16384_mb6_1x_sbdp_bf16rope_fusedopt` | Native `TpuGraphTrainer` `16K` (`mb=6 = 98.3K` tok/chip, `tile_m=256`, `SBDP`, `BF16` RoPE, `PureFusedOpt` in [`#L87`](./scripts/run_flint_v5p_pass10_longctx.py#L87-L128)) | **`5,670.7` / `181,462`** | **`33.67%` / `34.63%`** | **`29.21 GiB`** | [`run_flint_v5p_pass10_longctx.py#L470`](./scripts/run_flint_v5p_pass10_longctx.py#L470-L477) | [`SUMMARY.md`](./results/v5p32_pass10/SUMMARY.md) · [`JSON`](./results/v5p32_pass10/summary.json) |
| **6D (`Pass 11` `16K` Native Overlap)** | `v5p-32` | `v5p32_p11_05_native_s16384_mb6_ga4_overlap_tile512_n44` | Add `AG/RS Overlap` + `tile_m=512` (bwd `256`) + `n44` at `16K` (`mb=6 = 98.3K` tok/chip, `4×` GA = `12.58M` tok/step; `+6,340` tok/s & `-1.69 GiB` HBM vs `Pass 10`) | **`5,868.8` / `187,802`** | **`34.84%` / `35.16%`** | **`27.52 GiB`** | [`run_flint_v5p_pass11_mfu_frontier.py#L279`](./scripts/run_flint_v5p_pass11_mfu_frontier.py#L279-L286) | [`SUMMARY.md`](./results/v5p32_pass11/SUMMARY.md) · [`JSON`](./results/v5p32_pass11/summary.json) |
| **7 (`Pass 11` `32K` Native)** | `v5p-32` | `v5p32_p11_01_native_s32768_mb3_ga4_sbdp_bf16rope_fusedopt_n44` | Scale sequence length `16K -> 32K` (`mb=3 = 98.3K` tok/chip, `4×` GA = `12.58M` tok/step, **`6.70` PFLOP/s** sustained, `+25.89 pp` vs baseline) | **`5,439.5` / `174,064`** | **`45.65%` / `45.89%`** | **`27.52 GiB`** | [`run_flint_v5p_pass11_mfu_frontier.py#L247`](./scripts/run_flint_v5p_pass11_mfu_frontier.py#L247-L254) | [`SUMMARY.md`](./results/v5p32_pass11/SUMMARY.md) · [`JSON`](./results/v5p32_pass11/summary.json) |
| **8A (`Pass 11` `64K` Native)** | `v5p-32` | `v5p32_p11_03_native_s65536_mb1_ga8_overlap_tile512_n49` | Scale sequence length `32K -> 64K` + `AG/RS Overlap` + `tile_m=512` + `n49 = 0%` Attn Recompute (`mb=1`, `8×` GA = `16.78M` tok/step, **`9.25` PFLOP/s** sustained, `+43.24 pp` vs baseline) | **`4,734.5` / `151,504`** | **`63.00%` / `64.07%`** | **`14.96 GiB`** | [`run_flint_v5p_pass11_mfu_frontier.py#L263`](./scripts/run_flint_v5p_pass11_mfu_frontier.py#L263-L270) | [`SUMMARY.md`](./results/v5p32_pass11/SUMMARY.md) · [`JSON`](./results/v5p32_pass11/summary.json) |
| **8B (`Pass 11` `64K` Native)** | `v5p-16` | `v5p32_p11_03_native_s65536_mb1_ga8_overlap_tile512_n49` | Same `Pass 11` `64K` overlap recipe on 16-chip `2x2x4` slice (`8×` GA = `8.39M` tok/step, **`4.81` PFLOP/s** sustained, **`+45.70 pp`** vs baseline; `p11_02` hit `66.20%` peak MFU) | **`4,919.8` / `78,717`** | **`65.46%` / `66.20%`** | **`18.83 GiB`** | [`run_flint_v5p_pass11_mfu_frontier.py#L263`](./scripts/run_flint_v5p_pass11_mfu_frontier.py#L263-L270) | [`SUMMARY.md`](./results/v5p16_pass11/SUMMARY.md) · [`JSON`](./results/v5p16_pass11/summary.json) |

---

### 4.4 Static-Shape Expert Parallelism (`EP=2` to `EP=32`)

- **Implementation ([`StaticTpuAllToAllTokenDispatcher` in `scripts/run_flint_v7x_pass12.py#L53-L140`](./scripts/run_flint_v7x_pass12.py#L53-L140)):** Upstream `TokenDispatcher` (`EP > 1`) triggers dynamic-shape XLA recompilation and `SymNode` / `Eq(u46, 128)` Dynamo guard errors (`torch.bincount`). Replacing token dispatch with a fixed-capacity `[EP, E_local, C, D]` `all_to_all_single` counting-bucket dispatcher (`(recv_lexp.unsqueeze(1) == arange(E)).sum(0)`), `EPCompiledGradNorm`, and `SplitMeshOpt` enabled static-shape training across `EP = 2, 4, 8, 16, 32`:
  - **`StaticEP=2 × FSDP=16 + ZeRO-3` (`v7x_p12_08` — Intra-Package Die-to-Die, `0` Inter-Chip Hops):** `11.85%` steady / `12.24%` peak MFU (`251,520` tok/s, `19.73 GiB` HBM — [`script#L1239`](./scripts/run_flint_v7x_pass12.py#L1239-L1248) · [`Pass 12B MD`](./results/flint-results/v7x-16-flex-2x2x4_pass12b_20261101T080434Z/SUMMARY.md)).
  - **`StaticEP=4 × FSDP=8 + ZeRO-3` (`v7x_p12_09` — 2-Chip Intra-Tray):** `7.17%` steady / `7.37%` peak MFU (`152,304` tok/s, `21.05 GiB` HBM — [`script#L1249`](./scripts/run_flint_v7x_pass12.py#L1249-L1260) · [`Pass 12B MD`](./results/flint-results/v7x-16-flex-2x2x4_pass12b_20261101T080434Z/SUMMARY.md)).
  - **`StaticEP=8 × FSDP=4` (`4`-Chip Intra-Host):** `7.30%` / `7.34%` MFU on `v5p-32` (`v5p_p9f_02` — [`script#L1138`](./scripts/run_flint_v5p_pass9b.py#L1138-L1155) · [`Pass 9F MD`](./results/flint_v5p-16-4n-2x2x4_pass9f_SUMMARY.md)) and `6.73%` / `6.81%` MFU on `v7x-16` (`v7x_p11d_04` — [`script#L1158`](./scripts/run_flint_v7x_pass11.py#L1158-L1175) · [`Pass 11D MD`](./results/flint_v7x-16-flex-2x2x4_pass11d_SUMMARY.md)).
  - **`StaticEP=16 × FSDP=2 + ZeRO-2` (`2`-Host Inter-Host):** `6.78%` MFU on `v5p-32` (`v5p_p9e_03` — [`script#L1359`](./scripts/run_flint_v5p_pass9b.py#L1359-L1375) · [`Pass 9E MD`](./results/flint_v5p-16-4n-2x2x4_pass9e_SUMMARY.md)) and `5.85%` / `6.11%` MFU on `v7x-16` (`v7x_p11d_03` — [`Pass 11D MD`](./results/flint_v7x-16-flex-2x2x4_pass11d_SUMMARY.md)).
  - **`WholeStep StaticEP=32 + ZeRO-2` (`4`-Host Full Slice, Single Graph):** `4.70%` / `4.86%` MFU on `v7x-16` (`v7x_p11d_05` — [`Pass 11D MD`](./results/flint_v7x-16-flex-2x2x4_pass11d_SUMMARY.md)).

---

## 5. Repository Structure

```text
├── README.md                              # Summary + script, manifest & result traceability
├── docs/
│   └── TRAINING_RESULTS.md            # Complete 195+ experiment TorchTitan table across v7x-16 & v5p
├── docker/
│   ├── Dockerfile                         # Builds PyTorch 2.15 + TorchTPU 0.2.0 GA + PR #122 & PR #123
│   └── patch_flint_configs.py             # Base Native configs (v012..v014, v5p_base..n40)
├── scripts/
│   ├── README.md                          # Guide mapping each pass script to its experiments
│   ├── run_flint_v5p_pass10_longctx.py    # v5p-32 (35.33% MFU) & v5p-16 (35.21% MFU) 4K/8K/16K sweep
│   ├── run_flint_v7x_pass13_longctx.py    # v7x-16 Native (27.15% MFU) & HF (25.31% MFU) 4K/8K/16K sweep
│   ├── run_flint_v7x_pass12.py            # v7x-16 2K (441,664 tok/s, 20.80% MFU), HF 2K (18.58%), Offload & StaticEP=2/4
│   ├── run_flint_v5p_pass9.py             # v5p-16 (25.76% MFU) & v5p-32 (24.88% MFU) 2K sweep
│   └── run_flint_pass*.py                 # Earlier ablation passes (Pass 1..11 on v7x, Pass 1..9B on v5p)
├── k8s/                                   # Multi-host GKE TPU Indexed Job & GCS Fuse manifests
└── results/
    ├── README.md                          # Guide mapping each result folder/file to its pass
    ├── v5p32_pass10/                      # Results for 35.33% / 36.20% MFU (HF) & 33.67% (Native) on v5p-32
    ├── v5p16_pass10/                      # Results for 35.21% / 35.92% MFU on v5p-16
    ├── v7x16_pass13/                      # Results for 27.15% / 27.20% Native & 25.31% / 25.35% HF MFU on v7x-16
    └── ...                                # JSON & SUMMARY.md archive for earlier passes
```

---

## 6. Reproduction Commands

```bash
# 1. Build & Push Container Image
docker build -f docker/Dockerfile -t us-central1-docker.pkg.dev/YOUR_GCP_PROJECT_ID/torchtpu/torchtitan-flint:v0.2.0 ./docker
docker push us-central1-docker.pkg.dev/YOUR_GCP_PROJECT_ID/torchtpu/torchtitan-flint:v0.2.0

# 2. Stage RecursiveSI/flint (15 shards, 79.59 GB BF16) to GCS
kubectl apply -f k8s/storage-gcsfuse.yaml
kubectl apply -f k8s/stage-flint-job.yaml

# 3A. Run TPU v5p-32 4K/8K/16K sweep (up to 35.33% steady / 36.20% peak MFU -> scripts/run_flint_v5p_pass10_longctx.py)
kubectl apply -f k8s/v5p-32-flint-pass10.yaml

# 3B. Run TPU v5p-16 4K/8K/16K sweep (up to 35.21% steady / 35.92% peak MFU -> scripts/run_flint_v5p_pass10_longctx.py)
kubectl apply -f k8s/v5p-16-flint-pass10.yaml

# 3C. Run TPU v7x-16 4K/8K/16K sweep (27.15% Native & 25.31% HF MFU -> scripts/run_flint_v7x_pass13_longctx.py)
kubectl apply -f k8s/v7x-16-flint-pass13.yaml

# 3D. Run TPU v7x-16 2K sweep (441,664 tok/s, 20.80% MFU -> scripts/run_flint_v7x_pass12.py)
kubectl apply -f k8s/v7x-16-flint-pass12g.yaml
```
