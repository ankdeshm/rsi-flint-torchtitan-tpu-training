# `RecursiveSI/flint` Pre-Training on Google Cloud TPU (`v7x` & `v5p`) with TorchTitan

Pre-training benchmarks and step-by-step optimization recipes for **`RecursiveSI/flint`** (`39.8B` total parameters, `~2.52B` active per token, `256` routed experts + `1` shared expert, `49` layers mixing full and sliding-window causal attention) on **Google Cloud TPU `v7x`** and **TPU `v5p`**, using **PyTorch / TorchTPU `0.2.0` GA** and **[`google-pytorch/torchtitan`](https://github.com/google-pytorch/torchtitan)**.

**Starting point.** This work is built on the TorchTitan TPU training recipes for Flint in [`google-pytorch/torchtitan` PR #122](https://github.com/google-pytorch/torchtitan/pull/122) (native `TpuGraphTrainer` lane) and [PR #123](https://github.com/google-pytorch/torchtitan/pull/123) (Hugging Face modeling-backend lane). Those recipes report `16.23%` and `16.29%` MFU on TPU `v7x-8` at `2K` context. This repository then:

1. **Reproduces that baseline** on TPU `v7x-8` with 60-step runs (Section 1, rows 1–2).
2. **Changes one thing at a time on top of it** and measures each change (Section 1, rows 3–7). After four changes, training throughput is **`+11.1%`** tokens/s higher than the reproduced baseline (exact causal MFU `15.90% → 17.66%`).
3. **Validates the resulting recipe from `2K` to `64K` context** on TPU `v7x-16`, `v5p-32` and `v5p-16`, using 60-step runs with full-sequence attention (Sections 2–3).

| Result | Slice | Headline (60-step runs, steps `11..60`) |
|---|---|---|
| Baseline reproduction (`2K`) | `v7x-8` | `11,067` tok/s per TensorCore · `15.90%` exact causal MFU (`16.68%` TorchTitan counter) |
| Step-by-step recipe (`2K`) | `v7x-8` | `12,291` tok/s per TensorCore (`+11.1%`) · `17.66%` exact causal MFU (`18.52%` TorchTitan counter) |
| Validated context scaling (`2K`–`64K`) | `v7x-16` | `10,722`–`26,948` tok/s per chip · `17.6%`–`19.4%` exact causal MFU |
| Validated context scaling (`2K`–`64K`) | `v5p-32` / `v5p-16` | `3,159`–`5,926` tok/s per chip · `21.2%`–`27.2%` exact causal MFU |

---

## 1. `2K` Baseline Reproduction & Step-by-Step Runs (`TPU v7x-8`, 60 Steps)

All runs below use the same **TPU `v7x-8`** slice (`tpu7x-2x2x2`, `8` chips = `16` TensorCore ranks, `SimpleFSDP` across `16` ranks, `2,307.0` BF16 TFLOP/s per chip = `1,153.5` per TensorCore), `seq_len = 2,048`, global batch `512 × 2,048 = 1,048,576` tokens/step (`mb=32`, `ga=1`) and `seed=42`. Every number is the average of **steps `11..60`** of a 60-step run. Per-step details: [`docs/BASELINE_AND_ABLATIONS.md`](docs/BASELINE_AND_ABLATIONS.md) and [`results/baseline_2k/v7x-8/SUMMARY.md`](results/baseline_2k/v7x-8/SUMMARY.md).

**How to read this table**
- Rows 1–2 reproduce the two published recipes unchanged. Rows 3–7 each add **one change** on top of the row above, starting from row 1.
- **Gain vs. row above**: the change in training throughput (tokens/s per chip) compared with the previous row. **Total gain**: the same comparison against the baseline reproduction in row 1. For example, `+11.1%` means 11.1% more tokens per second on identical hardware, batch, data and seed.
- **pp**: the change in MFU in percentage points (for example `15.90% → 17.66%` is `+1.76 pp`).
- **Attention**: `C4-EoD` is the upstream default. It uses the `c4_test` dataset, and attention is skipped across document boundaries. `Full-Seq` means every token attends to the whole sequence.
- **MFU**: *Exact causal* counts only the attention work that causal attention needs. *TorchTitan counter* is the value printed by upstream TorchTitan logs; it counts the full attention square, so it reads slightly higher. See Section 2.

| # | Run: what changed | Attention | tok/s per chip | tok/s per TensorCore | Exact causal MFU | TorchTitan-counter MFU | Gain vs. row above | Total gain vs. row 1 | Peak HBM (GiB) |
|---:|---|---|---:|---:|---:|---:|---:|---|---:|
| 1 | **Baseline reproduction: native lane** ([PR #122](https://github.com/google-pytorch/torchtitan/pull/122) recipe, unchanged)<br>FP32 gradient reduce-scatter; attention (Splash) output kept for backward in `40` of `49` layers; expert grouped-matmul tile `tile_m=512`; eager AdamW + grad-norm.<br>*Reported in PR #122: `10,779` tok/s per TensorCore, `16.23%` (median of steps 3–8).* | `C4-EoD` | **22,134.6** | 11,067.3 | **15.90%** | 16.68% | — | baseline | 15.93 |
| 2 | **Baseline reproduction: Hugging Face backend lane** ([PR #123](https://github.com/google-pytorch/torchtitan/pull/123) recipe, unchanged)<br>`hf_moe_auto_max_attnac`, Splash `512` blocks, `tile_m=1024`, cross-entropy block `1024`, hierarchical SparseCore reduce-scatter flags.<br>*Reported in PR #123: `~10,810` tok/s per TensorCore, `16.29%` (mean of steps 2–10, pre-GA image).* | `C4-EoD` | **20,558.4** | 10,279.2 | **14.77%** | 15.49% | — | second reference lane (rows 3–7 build on row 1) | 14.37 |
| 3 | **+ BF16 gradient reduce-scatter & async-collective XLA flags** (on top of row 1)<br>`--training.mixed_precision_reduce=bfloat16` halves FSDP reduce-scatter traffic (`159.2 → 79.6 GB` per step); adds async-collective / latency-hiding scheduler flags. | `C4-EoD` | **22,131.6** | 11,065.8 | **15.90%** | 16.68% | `0.0%` | `0.0%` (`+0.00 pp`) | 16.35 |
| 4 | **+ Keep attention output for all 49 layers & larger expert matmul tile**<br>No attention recompute in backward (`40 → 49` kept layers); grouped-matmul `tile_m` `512 → 1,024`. | `C4-EoD` | **22,640.0** | 11,320.0 | **16.26%** | 17.06% | `+2.3%` | `+2.3%` (`+0.36 pp` causal / `+0.38 pp` counter) | 14.76 |
| 5 | **+ Apply router weights before the expert down-projection & BF16 RoPE**<br>`scale_before_down_proj` turns the expert combine into a plain sum; RoPE runs in BF16 instead of FP32. | `C4-EoD` | **23,774.2** | 11,887.1 | **17.08%** | 17.92% | `+5.0%` | `+7.4%` (`+1.18 pp` / `+1.24 pp`) | 15.54 |
| 6 | **+ Save MoE routing for backward & compile fused AdamW + grad-norm** (the tuned recipe)<br>No router recompute in backward; optimizer step and gradient clipping compiled into one XLA graph. | `C4-EoD` | **24,582.8** | 12,291.4 | **17.66%** | 18.52% | `+3.4%` | **`+11.1%`** (`+1.76 pp` / `+1.84 pp`) | 15.52 |
| 7 | **Tuned recipe with full-sequence attention** (`TT_CONTIGUOUS_FULL_SEQ=1`)<br>Same as row 6, but attention is no longer skipped at document boundaries: every token attends across all `2,048` positions. **Every run in Section 3 uses this recipe.** | `Full-Seq` | **24,396.8** | 12,198.4 | **17.52%** | 18.38% | `-0.8%` | `+10.2%` (`+1.62 pp` / `+1.70 pp`) | 15.52 |

> **Reproduction vs. reported numbers.** The reproductions land close to the reported baselines. The native lane measured `11,067` vs. `10,779` tok/s per TensorCore. The HF lane measured `10,279` vs. `~10,810`, but its reported number came from a pre-GA image. Scoring row 1 over the reported step window (median of steps 3–8) gives `16.77%` instead of `16.68%`, so the choice of step window does not explain the small differences.

---

## 2. How the Results Are Measured (60-Step Validated Runs)

> [!NOTE]
> **In plain words, a "validated 60-step run" means:**
> 1. **Long enough to be steady.** Every run trains for 60 steps. Steps 1–10 (compilation and warm-up) are dropped, and every number is the average of steps 11–60.
> 2. **One fixed recipe.** Every context length uses the same recipe (row 7 above). Only the sequence length and the number of sequences per step change.
> 3. **Full attention, no shortcuts.** Every token attends across the whole sequence. The upstream test data packs many short web documents into each sequence and lets attention skip across document boundaries. At `64K` that skips almost all of the attention work and makes MFU look far higher than it is. Each slice also has one deliberately skipped *control* run so the size of that effect is visible.
> 4. **Two MFU numbers, both shown.** *Exact causal MFU* counts only the attention work that causal attention needs; this is the recommended number. *TorchTitan-counter MFU* is the number upstream TorchTitan logs print. It counts the full attention square, so it reads higher, and the gap grows with context: about `5%` higher at `2K` and about `59%` higher at `64K`.

Short benchmark windows and packed short-document datasets can overstate long-context (`16K`–`64K`) MFU considerably. The rules below are implemented in [`scripts/run_flint_60step_validation.py`](scripts/run_flint_60step_validation.py):

1. **60 Optimizer Steps per Run (Averaging Steps `11..60`):**
   Steps `1..10` absorb XLA compilation, HBM allocator stabilization, and data pipeline startup. Averaging `50` consecutive post-warmup steps (`11..60`) eliminates compilation skew and single-step timing noise.
2. **True Contiguous Full-Sequence Causal Attention (`TT_CONTIGUOUS_FULL_SEQ=1`):**
   Upstream `c4_test` packs short web articles (`~450` tokens on average) separated by End-of-Document (`EoD`) segment IDs, and SplashAttention skips `100%`-masked cross-document tiles. At `2K` context that changes throughput by `<1%` (`17.66%` vs. `17.52%` Causal MFU above). At `64K` context (`65,536` tokens), however, packing `~145` short articles per row causes SplashAttention to **skip `>98%` of the attention matrix** in all `14` global causal layers while the FLOPs formula still credits full `65,536`-token attention. Setting `TT_CONTIGUOUS_FULL_SEQ=1` feeds contiguous `seq_len` sequences so every token computes full causal attention across the entire context window.
3. **Exact Causal FLOPs Alongside the Upstream TorchTitan FLOPs Counter:**
   Upstream TorchTitan (`palm_flops`) counts unmasked `S × S` square attention (`12 × H × D × S` per global layer = `61.07` GFLOP/tok at `64K`), which includes the masked upper triangle of causal attention. The **Exact Causal MFU** column (`flint_causal_flops_per_token`) integrates the exact lower-triangular causal FLOPs across Flint's `14` full causal layers and `35` one-sided `W=513` sliding-window layers (`38.52` GFLOP/tok at `64K`), and both columns are reported side by side:

   | Context Length (`seq_len`) | Non-Embedding Parameter FLOPs (`6 × N_active`) | **Exact Causal GFLOP/tok** (`14` global causal + `35` `W=513` SWA) | **TorchTitan Counter GFLOP/tok** (upstream unmasked square convention) |
   |---:|---:|---:|---:|
   | `2,048` (`2K`) | `15.0932` | **`16.5704`** | `17.3850` |
   | `4,096` (`4K`) | `15.0932` | **`17.3302`** | `18.7943` |
   | `8,192` (`8K`) | `15.0932` | **`18.7671`** | `21.6129` |
   | `16,384` (`16K`) | `15.0932` | **`21.5994`** | `27.2500` |
   | `32,768` (`32K`) | `15.0932` | **`27.2435`** | `38.5243` |
   | `65,536` (`64K`) | `15.0932` | **`38.5212`** | `61.0729` |

---

## 3. Validated Context-Length Scaling (`2K` to `64K`, 60 Steps, Full-Sequence Attention)

All rows below run the **same fixed Native `TpuGraphTrainer` tuned recipe** (`SimpleFSDP` across all TensorCore ranks, `49/49` Splash outputs saved, `scale_before_down_proj`, `BF16` RoPE, saved MoE routing, compiled fused AdamW + grad-norm, and `BF16` gradient ReduceScatter; `tile_m=1024` on `v7x` and `tile_m=256` on `v5p` to fit `v5p`'s `16 MiB` Scoped VMEM limit). Raw step-by-step telemetry is in [`results/validated_60step/`](results/validated_60step/).

### 3.1 Cross-Slice Summary (`2K` to `64K`, Contiguous Full-Sequence Attention)

| TPU Slice | `2K` (`seq=2,048`) | `4K` (`seq=4,096`) | `8K` (`seq=8,192`) | `16K` (`seq=16,384`) | `32K` (`seq=32,768`) | `64K` (`seq=65,536`) |
|---|---|---|---|---|---|---|
| **TPU `v7x-16`**<br>(`16` chips / `32` TCs, `2,307` TF/s/chip) | **`26,947.8` tok/s/chip**<br>**`19.36%`** Causal MFU<br>(`20.31%` TorchTitan MFU)<br>*(`mb=48, ga=2`)* | **`24,552.1` tok/s/chip**<br>**`18.44%`** Causal MFU<br>(`20.00%` TorchTitan MFU)<br>*(`mb=16, ga=1`)* | **`22,326.5` tok/s/chip**<br>**`18.16%`** Causal MFU<br>(`20.92%` TorchTitan MFU)<br>*(`mb=8, ga=1`)* | **`19,323.3` tok/s/chip**<br>**`18.09%`** Causal MFU<br>(`22.82%` TorchTitan MFU)<br>*(`mb=6, ga=2`)* | **`15,202.8` tok/s/chip**<br>**`17.95%`** Causal MFU<br>(`25.39%` TorchTitan MFU)<br>*(`mb=3, ga=2`)* | **`10,829.4` tok/s/chip**<br>**`18.08%`** Causal MFU<br>(`28.67%` TorchTitan MFU)<br>*(`mb=1, ga=4`)* |
| **TPU `v5p-32`**<br>(`32` chips, `459` TF/s/chip) | **`5,925.7` tok/s/chip**<br>**`21.39%`** Causal MFU<br>(`22.44%` TorchTitan MFU)<br>*(`mb=32, ga=1`)* | **`5,809.7` tok/s/chip**<br>**`21.94%`** Causal MFU<br>(`23.79%` TorchTitan MFU)<br>*(`mb=16, ga=1`)* | **`5,513.6` tok/s/chip**<br>**`22.54%`** Causal MFU<br>(`25.96%` TorchTitan MFU)<br>*(`mb=8, ga=1`)* | **`4,995.9` tok/s/chip**<br>**`23.51%`** Causal MFU<br>(`29.66%` TorchTitan MFU)<br>*(`mb=4, ga=1`)* | **`4,214.4` tok/s/chip**<br>**`25.01%`** Causal MFU<br>(`35.37%` TorchTitan MFU)<br>*(`mb=2, ga=1`)* | **`3,240.9` tok/s/chip**<br>**`27.20%`** Causal MFU<br>(`43.12%` TorchTitan MFU)<br>*(`mb=1, ga=4`)* |
| **TPU `v5p-16`**<br>(`16` chips, `459` TF/s/chip) | **`5,874.4` tok/s/chip**<br>**`21.21%`** Causal MFU<br>(`22.25%` TorchTitan MFU)<br>*(`mb=32, ga=1`)* | **`5,799.7` tok/s/chip**<br>**`21.90%`** Causal MFU<br>(`23.75%` TorchTitan MFU)<br>*(`mb=16, ga=1`)* | **`5,504.7` tok/s/chip**<br>**`22.51%`** Causal MFU<br>(`25.92%` TorchTitan MFU)<br>*(`mb=8, ga=1`)* | **`4,949.7` tok/s/chip**<br>**`23.29%`** Causal MFU<br>(`29.39%` TorchTitan MFU)<br>*(`mb=4, ga=1`)* | **`4,170.0` tok/s/chip**<br>**`24.75%`** Causal MFU<br>(`35.00%` TorchTitan MFU)<br>*(`mb=2, ga=1`)* | **`3,199.3` tok/s/chip**<br>**`26.85%`** Causal MFU<br>(`42.57%` TorchTitan MFU)<br>*(`mb=1, ga=4`)* |

> **Scaling Efficiency (`v5p-16` → `v5p-32`):** Doubling from `16` to `32` `v5p` chips (`4` hosts → `8` hosts) achieves **`100.2%`–`101.9%` weak-scaling per-chip efficiency** across every context length from `2K` (`93,990 -> 189,622` slice tok/s) to `64K` (`51,188 -> 103,709` slice tok/s). On `v7x`, scaling from `v7x-8` (`8` chips) to `v7x-16` (`16` chips) at `2K` Full-Seq (`mb=32, ga=1`) increases per-chip throughput by **`+3.2% tok/s/chip`** (`24,396.8 -> 25,168.4` tok/s/chip) and unlocks larger `mb=48, ga=2` batches (**`+10.5% tok/s/chip`**, `26,947.8` tok/s/chip = `431,165` slice tok/s) thanks to `32`-way FSDP parameter sharding.

---

### 3.2 TPU `v7x-16` (`2x2x4`, 16 Chips / 32 TensorCores) — Full 60-Step Table

Full summary and step logs: [`results/validated_60step/v7x-16/SUMMARY.md`](results/validated_60step/v7x-16/SUMMARY.md).

| Configuration (`seq_len`, Microbatch `mb` × GradAccum `ga`) | Global Tokens/Step | Attention Mode | Steady tok/s per Chip | Steady tok/s per TensorCore | Slice tok/s | **Exact Causal MFU** | TorchTitan Counter MFU | Peak HBM (GiB) | Loss (`Step 1 -> 60`) |
|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| **`2K` Iso-Token (`seq=2,048`, `mb=32 × ga=1`)** | `2,097,152` | `Full-Seq` | **25,168.4** | 12,584.2 | 402,694 | **18.08%** | 18.97% | 15.52 | `12.31906 -> 2.82746` |
| **`2K` Higher Batch (`seq=2,048`, `mb=48 × ga=2`)** | `6,291,456` | `Full-Seq` | **26,947.8** | 13,473.9 | 431,165 | **19.36%** | 20.31% | 23.02 | `12.31829 -> 2.86263` |
| **`4K` Iso-Token (`seq=4,096`, `mb=16 × ga=1`)** | `2,097,152` | `Full-Seq` | **24,552.1** | 12,276.0 | 392,833 | **18.44%** | 20.00% | 15.52 | `12.32523 -> 2.80467` |
| **`8K` Iso-Token (`seq=8,192`, `mb=8 × ga=1`)** | `2,097,152` | `Full-Seq` | **22,326.5** | 11,163.2 | 357,224 | **18.16%** | 20.92% | 16.02 | `12.32958 -> 2.78572` |
| **`16K` Iso-Token (`seq=16,384`, `mb=4 × ga=1`)** | `2,097,152` | `Full-Seq` | **18,958.8** | 9,479.4 | 303,340 | **17.75%** | 22.39% | 14.02 | `12.33117 -> 2.78966` |
| **`16K` Higher Batch (`seq=16,384`, `mb=6 × ga=2`)** | `6,291,456` | `Full-Seq` | **19,323.3** | 9,661.6 | 309,172 | **18.09%** | 22.82% | 20.52 | `12.32923 -> 2.81084` |
| **`32K` Iso-Token (`seq=32,768`, `mb=2 × ga=1`)** | `2,097,152` | `Full-Seq` | **14,866.0** | 7,433.0 | 237,855 | **17.56%** | 24.82% | 14.02 | `12.33477 -> 2.78622` |
| **`32K` Higher Batch (`seq=32,768`, `mb=3 × ga=2`)** | `6,291,456` | `Full-Seq` | **15,202.8** | 7,601.4 | 243,245 | **17.95%** | 25.39% | 20.52 | `12.33113 -> 2.80935` |
| **`64K` Iso-Token (`seq=65,536`, `mb=1 × ga=1`)** | `2,097,152` | `Full-Seq` | **10,722.3** | 5,361.1 | 171,556 | **17.90%** | 28.38% | 14.02 | `12.33406 -> 2.78358` |
| **`64K` Grad-Accum (`seq=65,536`, `mb=1 × ga=4`)** | `8,388,608` | `Full-Seq` | **10,829.4** | 5,414.7 | 173,271 | **18.08%** | 28.67% | 14.02 | `12.33261 -> 2.85837` |
| *`64K` Short-Doc EoD Control (`seq=65,536`, `mb=1 × ga=1`, `c4_test` EoD skipping)* | `2,097,152` | *`C4-EoD` (skips >98% global attn)* | *18,155.0* | *9,077.5* | *290,481* | *30.31% (inflated)* | *48.06% (inflated)* | 14.02 | `12.32568 -> 2.82200` |

> The italic control row repeats the `64K` run with document-boundary skipping switched back on. It reproduces the `~48%` TorchTitan-counter MFU seen in earlier short tuning runs, which shows where that number came from. The validated `64K` result is the full-sequence row above it (`17.90%`–`18.08%` exact causal MFU). See the [appendix](docs/APPENDIX_HISTORICAL_RUNS.md).

---

### 3.3 TPU `v5p-32` (`2x4x4`, 32 Chips) & `v5p-16` (`2x2x4`, 16 Chips) — Full 60-Step Table

Full summaries and step logs: [`results/validated_60step/v5p-32/SUMMARY.md`](results/validated_60step/v5p-32/SUMMARY.md) and [`results/validated_60step/v5p-16/SUMMARY.md`](results/validated_60step/v5p-16/SUMMARY.md).

| TPU Slice | Configuration (`seq_len`, `mb × ga`) | Global Tokens/Step | Attention Mode | Steady tok/s per Chip | Slice tok/s | **Exact Causal MFU** | TorchTitan Counter MFU | Peak HBM (GiB) | Loss (`Step 1 -> 60`) |
|---|---|---:|---|---:|---:|---:|---:|---:|---|
| **`v5p-32`** (`32` chips) | **`2K` (`seq=2,048`, `mb=32 × ga=1`)** | `2,097,152` | `Full-Seq` | **5,925.7** | 189,622 | **21.39%** | 22.44% | 17.53 | `12.32683 -> 2.79280` |
| **`v5p-32`** (`32` chips) | **`4K` (`seq=4,096`, `mb=16 × ga=1`)** | `2,097,152` | `Full-Seq` | **5,809.7** | 185,911 | **21.94%** | 23.79% | 17.53 | `12.33364 -> 2.79027` |
| **`v5p-32`** (`32` chips) | **`8K` (`seq=8,192`, `mb=8 × ga=1`)** | `2,097,152` | `Full-Seq` | **5,513.6** | 176,435 | **22.54%** | 25.96% | 17.53 | `12.33846 -> 2.78548` |
| **`v5p-32`** (`32` chips) | **`16K` (`seq=16,384`, `mb=4 × ga=1`)** | `2,097,152` | `Full-Seq` | **4,995.9** | 159,868 | **23.51%** | 29.66% | 17.53 | `12.34060 -> 2.80076` |
| **`v5p-32`** (`32` chips) | **`32K` (`seq=32,768`, `mb=2 × ga=1`)** | `2,097,152` | `Full-Seq` | **4,214.4** | 134,861 | **25.01%** | 35.37% | 17.53 | `12.34433 -> 2.77856` |
| **`v5p-32`** (`32` chips) | **`64K` (`seq=65,536`, `mb=1 × ga=1`)** | `2,097,152` | `Full-Seq` | **3,218.7** | 102,997 | **27.01%** | 42.83% | 14.96 | `12.34367 -> 2.78643` |
| **`v5p-32`** (`32` chips) | **`64K` Grad-Accum (`seq=65,536`, `mb=1 × ga=4`)** | `8,388,608` | `Full-Seq` | **3,240.9** | 103,709 | **27.20%** | 43.12% | 14.96 | `12.34191 -> 2.84033` |
| **`v5p-32`** (`32` chips) | *`64K` Short-Doc EoD Control (`mb=1 × ga=1`, `c4_test`)* | `2,097,152` | *`C4-EoD`* | *4,765.8* | *152,504* | *40.00% (inflated)* | *63.41% (inflated)* | 14.96 | `12.33451 -> 2.82996` |
| **`v5p-16`** (`16` chips) | **`2K` (`seq=2,048`, `mb=32 × ga=1`)** | `1,048,576` | `Full-Seq` | **5,874.4** | 93,990 | **21.21%** | 22.25% | 17.55 | `12.32226 -> 2.75305` |
| **`v5p-16`** (`16` chips) | **`4K` (`seq=4,096`, `mb=16 × ga=1`)** | `1,048,576` | `Full-Seq` | **5,799.7** | 92,796 | **21.90%** | 23.75% | 17.55 | `12.32920 -> 3.02743` |
| **`v5p-16`** (`16` chips) | **`8K` (`seq=8,192`, `mb=8 × ga=1`)** | `1,048,576` | `Full-Seq` | **5,504.7** | 88,075 | **22.51%** | 25.92% | 17.55 | `12.33347 -> 3.06289` |
| **`v5p-16`** (`16` chips) | **`16K` (`seq=16,384`, `mb=4 × ga=1`)** | `1,048,576` | `Full-Seq` | **4,949.7** | 79,195 | **23.29%** | 29.39% | 17.55 | `12.33612 -> 2.79439` |
| **`v5p-16`** (`16` chips) | **`32K` (`seq=32,768`, `mb=2 × ga=1`)** | `1,048,576` | `Full-Seq` | **4,170.0** | 66,721 | **24.75%** | 35.00% | 19.43 | `12.33914 -> 2.86616` |
| **`v5p-16`** (`16` chips) | **`64K` (`seq=65,536`, `mb=1 × ga=1`)** | `1,048,576` | `Full-Seq` | **3,159.2** | 50,547 | **26.51%** | 42.04% | 18.83 | `12.34034 -> 2.86275` |
| **`v5p-16`** (`16` chips) | **`64K` Grad-Accum (`seq=65,536`, `mb=1 × ga=4`)** | `4,194,304` | `Full-Seq` | **3,199.3** | 51,188 | **26.85%** | 42.57% | 18.83 | `12.33868 -> 2.87057` |
| **`v5p-16`** (`16` chips) | *`64K` Short-Doc EoD Control (`mb=1 × ga=1`, `c4_test`)* | `1,048,576` | *`C4-EoD`* | *4,870.9* | *77,934* | *40.88% (inflated)* | *64.81% (inflated)* | 18.83 | `12.32183 -> 2.78776` |

> **Single-Rank Context Limit (`128K` Probe):** In this TorchTitan TPU branch, `RecursiveSI/flint` enforces pure data/FSDP sharding (`context_parallel` and `tensor_parallel` are disabled for the hybrid MoE graph). At `128K` (`seq_len = 131,072`, `mb=1`), each rank must hold a full `131,072`-token sequence of MoE activations (`2×` the `65,536`-token budget), exceeding per-rank memory even when attention-output checkpointing is lowered (`attn49 -> attn32 -> attn16`; see [`results/longctx_128k_attempts/`](results/longctx_128k_attempts/)). Scaling pre-training beyond `64K` per sequence requires Sequence/Context Parallelism (Ring Attention) across ranks.

---

## 4. Model & Software Stack Overview

- **Model (`RecursiveSI/flint`):**
  - `39.8B` total parameters (`~2.52B` active non-embedding parameters per token; `15.8 : 1` total-to-active sparsity ratio)
  - `49` layers, `d_model = 2,048`: `2` dense SwiGLU layers (`d_ff = 6,144`) + `47` MoE layers with `256` routed experts (`top-6`, sigmoid router, expert `d_ff = 512`) and `1` shared expert (`d_ff = 1,024`)
  - Attention: `14` full causal layers (no RoPE) + `35` sliding-window causal layers (`W = 513`, RoPE), learnable attention sinks; GQA with `32` query heads / `4` KV heads, `head_dim = 128`; `vocab_size = 150,144`; `max_position_embeddings = 262,144`
- **Container & Runtime ([`docker/Dockerfile`](docker/Dockerfile)):**
  - `torch==2.15.0+cpu`, `torch_tpu==0.2.0` GA, `jax==0.11.1`, `libtpu==0.0.50`, `tokamax==0.0.14`, `transformers>=5.22.0`
  - Upstream TorchTitan TPU lanes fetched directly by PR ref (`refs/pull/122/head` -> `/torchtitan-native`, `refs/pull/123/head` -> `/torchtitan-hf`) plus [`docker/patch_flint_configs.py`](docker/patch_flint_configs.py) (`v5p` `tile_m=256` Scoped VMEM fix and `v7x` `tile_m=1024` / `n49` configs)

---

## 5. Repository Structure

```text
rsi-flint-torchtitan-tpu-training/
├── README.md                                        # Baseline reproduction, step-by-step runs & validated 2K..64K results
├── docker/
│   ├── Dockerfile                                   # TorchTPU 0.2.0 GA + TorchTitan PR #122 & PR #123 image
│   ├── patch_flint_configs.py                       # Portable v5p (tile_m=256) and v7x (tile_m=1024, n49) configs
│   └── cloudbuild.yaml                              # Google Cloud Build specification
├── k8s/
│   ├── storage-gcsfuse.yaml                         # GCSFuse CSI PV / PVC for shared GCS weights & logs
│   ├── stage-flint-job.yaml                         # One-shot HuggingFace -> GCS weight staging Job
│   ├── v7x-8-flint-baseline-2k.yaml                 # 60-step 2K baseline & ablation Indexed Job (v7x-8)
│   ├── v7x-16-flint-val60.yaml                      # 60-step 2K..64K full-sequence Indexed Job (v7x-16)
│   ├── v7x-16-flint-val60-rerun.yaml                # Lockstep coordinator re-run manifest (v7x-16)
│   ├── v5p-16-flint-val60.yaml                      # 60-step 2K..64K full-sequence Indexed Job (v5p-16)
│   ├── v5p-16-flint-val60-longctx.yaml              # 128K single-rank HBM boundary Job (v5p-16)
│   ├── v5p-32-flint-val60.yaml                      # 60-step 2K..64K full-sequence Indexed Job (v5p-32)
│   └── v5p-32-flint-val60-longctx.yaml              # 128K single-rank HBM boundary Job (v5p-32)
├── scripts/
│   ├── README.md                                    # Guide to active staging and validation scripts
│   ├── stage_hf_to_gcs.py                           # Stages 15 RecursiveSI/flint safetensors shards to GCS
│   └── run_flint_60step_validation.py               # 60-step runner (suites: baseline_2k, val60, longctx_128k)
├── results/
│   ├── README.md                                    # Index of validated 60-step summaries & per-step JSONs
│   ├── baseline_2k/v7x-8/                           # 60-step 2K baseline & ablation results (v7x-8)
│   ├── validated_60step/{v7x-16,v5p-32,v5p-16}/     # 60-step 2K..64K full-sequence results
│   └── longctx_128k_attempts/{v5p-32,v5p-16}/       # 128K attempts (out of HBM without context parallelism)
├── docs/
│   ├── BASELINE_AND_ABLATIONS.md                    # Detailed walkthrough of 2K baselines & incremental gains
│   └── APPENDIX_HISTORICAL_RUNS.md                  # Appendix explaining earlier 10-15 step tuning sweeps
└── archive/
    ├── scripts/                                     # Archived exploratory tuning scripts (Pass 1..14)
    ├── k8s/                                         # Archived exploratory Kubernetes Job manifests
    └── results/                                     # Archived 10-15 step tuning summaries & JSON logs
```

---

## 6. Quick-Start Reproduction on GKE TPU Slices

### Step 1: Build the Container Image

```bash
export PROJECT_ID="YOUR_GCP_PROJECT_ID"
export REGION="us-central1"
export GCS_BUCKET="YOUR_GCS_BUCKET"

gcloud builds submit ./docker \
  --project="${PROJECT_ID}" \
  --config=docker/cloudbuild.yaml \
  --substitutions="_IMAGE=${REGION}-docker.pkg.dev/${PROJECT_ID}/torchtpu/torchtitan-flint:v0.2.0"
```

### Step 2: Mount GCS & Stage `RecursiveSI/flint` Weights

```bash
kubectl create secret generic hf-token --from-literal=HF_TOKEN="${HF_TOKEN}"

sed "s/YOUR_GCS_BUCKET/${GCS_BUCKET}/g" k8s/storage-gcsfuse.yaml | kubectl apply -f -
kubectl create configmap flint-stage-script --from-file=stage_hf_to_gcs.py=scripts/stage_hf_to_gcs.py
sed "s/YOUR_GCP_PROJECT_ID/${PROJECT_ID}/g; s/YOUR_GCS_BUCKET/${GCS_BUCKET}/g" k8s/stage-flint-job.yaml | kubectl apply -f -
kubectl wait --for=condition=complete --timeout=3600s job/stage-flint-hf-to-gcs
```

### Step 3: Run the 60-Step Baseline Reproduction or `2K`–`64K` Validation Suite

```bash
kubectl create configmap flint-val60-script \
  --from-file=run_flint_60step_validation.py=scripts/run_flint_60step_validation.py \
  --dry-run=client -o yaml | kubectl apply -f -

# 2K baseline reproduction + step-by-step ablations on TPU v7x-8 (2 hosts, 16 TensorCore ranks):
sed "s/YOUR_GCP_PROJECT_ID/${PROJECT_ID}/g" k8s/v7x-8-flint-baseline-2k.yaml | kubectl apply -f -

# 2K..64K contiguous full-sequence 60-step validation on TPU v7x-16, v5p-32, or v5p-16:
sed "s/YOUR_GCP_PROJECT_ID/${PROJECT_ID}/g" k8s/v7x-16-flint-val60-rerun.yaml | kubectl apply -f -
sed "s/YOUR_GCP_PROJECT_ID/${PROJECT_ID}/g" k8s/v5p-32-flint-val60.yaml | kubectl apply -f -
sed "s/YOUR_GCP_PROJECT_ID/${PROJECT_ID}/g" k8s/v5p-16-flint-val60.yaml | kubectl apply -f -
```

---

## 7. Appendix & Additional Documentation

- **[`docs/BASELINE_AND_ABLATIONS.md`](docs/BASELINE_AND_ABLATIONS.md):** Step-by-step analysis of the `2K` TorchTitan baseline reproductions, incremental optimizations, numerical loss checks, and `16`-rank vs. `32`-rank FSDP batch ceilings.
- **[`docs/APPENDIX_HISTORICAL_RUNS.md`](docs/APPENDIX_HISTORICAL_RUNS.md):** Archive guide for the earlier `10–15`-step tuning passes (`Pass 1..14`, `StaticEP=2..32`, host activation offload) and why short-document `c4_test` EoD block skipping inflated early long-context MFU numbers.
