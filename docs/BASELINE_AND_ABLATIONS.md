# TorchTitan TPU `2K` Baseline Reproduction & Incremental Ablations (`v7x-8`)

This document details the `2,048`-token (`2K`) baseline reproduction and step-by-step optimization runs for `RecursiveSI/flint` (`39.8B` total / `~2.52B` active parameters per token) on a single 8-chip **TPU `v7x-8`** slice (`tpu7x-2x2x2`, 2 hosts × 8 processes = `16` TensorCore ranks).

All experiments build directly on the [`google-pytorch/torchtitan`](https://github.com/google-pytorch/torchtitan) TPU training recipes:
- **Native `TpuGraphTrainer` lane ([PR #122](https://github.com/google-pytorch/torchtitan/pull/122)):** Native `TpuGroupedExperts` / `TpuRoutedExperts`, per-layer `GraphSplitter` executables, `SimpleFSDP` over 16 TensorCore ranks, and `FlintStateDictAdapter`.
- **HuggingFace Modeling Backend lane ([PR #123](https://github.com/google-pytorch/torchtitan/pull/123)):** `transformers` modeling backend (`hf_moe_auto_max_attnac`) with Pallas grouped matmul (`tile_m=1024`), SplashAttention (`512` blocks), cross-entropy `b_block_size=1024`, and hierarchical SparseCore ReduceScatter flags.

---

## 1. Fixed Evaluation Setup (`v7x-8`, `2K` Context)

Every baseline and incremental ablation row holds the hardware topology, batch size, data seed, and 60-step measurement window constant unless explicitly labeled as a batch-scaling HBM probe:

| Parameter | Setting |
|---|---|
| **TPU Slice** | TPU `v7x-8` (`tpu7x-2x2x2`, `8` chips = `16` TensorCore ranks, `2,307.0` BF16 TFLOP/s/chip = `1,153.5` BF16 TFLOP/s/TensorCore) |
| **Parallelism** | `SimpleFSDP` across `16` TensorCore ranks |
| **Context Length (`seq_len`)** | `2,048` tokens |
| **Global Batch Size** | `512 × 2,048 = 1,048,576` tokens/step (`mb=32` sequences = `65,536` tokens/TensorCore, `ga=1`) |
| **Measurement Window** | `60` optimizer steps (`--debug.seed=42`); steady-state mean over **50 post-warmup steps (`11..60`)** |
| **MFU Metrics Reported** | **Exact Causal MFU** (`16.5704` GFLOP/tok at `2K`) and **TorchTitan Counter MFU** (`17.3850` GFLOP/tok at `2K`, matching upstream TorchTitan logs) |
| **Software Stack** | `torch==2.15.0+cpu`, `torch_tpu==0.2.0` GA, `jax==0.11.1`, `libtpu==0.0.50`, `tokamax==0.0.14` ([`docker/Dockerfile`](../docker/Dockerfile)) |

> **Note on measurement windows:** PR #122 reported the median of steps `3..8` (`10,779` tok/s/TensorCore, `16.23%` TorchTitan Counter MFU) and PR #123 reported the mean of steps `2..10` (`~10,810` tok/s/TensorCore, `16.29%` TorchTitan Counter MFU on a pre-GA `flint-tune2` image). Re-scoring the 60-step runs with those shorter step windows changes MFU by `<0.1` percentage points (`16.77%` / `16.70%` vs. `16.68%` steady on the Native baseline), confirming that the 50-step steady window (`11..60`) does not introduce window bias.

---

## 2. Baseline Reproduction & Step-by-Step Optimization Ladder

Starting from the unmodified Native `TpuGraphTrainer` baseline recipe (PR #122), each row below adds one group of changes while keeping `v7x-8`, `seq_len=2,048`, `1,048,576` tokens/step (`mb=32, ga=1`), and `seed=42` fixed:

| Configuration (Incremental Changes vs. Baseline) | Attention Mode | tok/s per Chip | tok/s per TensorCore | Exact Causal MFU | TorchTitan Counter MFU | Delta vs. Native Baseline Reproduction | Peak HBM (GiB) | Loss (`Step 1 -> 60`) |
|---|---|---:|---:|---:|---:|---|---:|---|
| **Baseline: Native `TpuGraphTrainer` (PR #122 recipe unchanged)**<br>`FP32` gradient ReduceScatter, Splash output saved in `40/49` layers (`n40`), Pallas grouped-MM `tile_m=512`, eager AdamW & grad-norm | `C4-EoD` | **22,134.6** | 11,067.3 | **15.90%** | 16.68% | Baseline (`0.0% tok/s`, `0.00 pp MFU`) | 15.93 | `12.23862 -> 2.83910` |
| **Baseline: HuggingFace Modeling Backend (PR #123 recipe unchanged)**<br>`hf_moe_auto_max_attnac`, Splash `512` blocks, Pallas grouped-MM `tile_m=1024`, `ce_b=1024`, hierarchical SparseCore RS flags | `C4-EoD` | **20,558.4** | 10,279.2 | **14.77%** | 15.49% | — (second reference lane; Steps 1–4 build on the Native baseline) | 14.37 | `12.43618 -> 2.88305` |
| **Step 1: + `BF16` Gradient ReduceScatter & Async Collective XLA Flags**<br>Adds `--training.mixed_precision_reduce=bfloat16` (halves FSDP ReduceScatter volume from `159.2 GB` to `79.6 GB`) + async collective / latency-hiding scheduler flags | `C4-EoD` | **22,131.6** | 11,065.8 | **15.90%** | 16.68% | `-0.0% tok/s` (`+0.00 pp` MFU at 2-host ICI scale) | 16.35 | `12.23862 -> 2.83789` |
| **Step 2: + Save Splash Output in All 49 Layers & Widen Expert `tile_m` (`512 -> 1024`)**<br>Eliminates attention backward recompute on the remaining 9 layers (`n40 -> n49`) and increases Pallas grouped-MM `tile_m` to `1,024` | `C4-EoD` | **22,640.0** | 11,320.0 | **16.26%** | 17.06% | **`+2.3% tok/s`** (`+0.36 pp` Causal / `+0.38 pp` TorchTitan MFU) | 14.76 | `12.23862 -> 2.83674` |
| **Step 3: + Pre-Down-Projection Router Scaling (`scale_before_down_proj`) & `BF16` RoPE**<br>Multiplies router weights before expert `down_proj` (so expert combine becomes an unweighted sum) and executes RoPE in `BF16` instead of `FP32` | `C4-EoD` | **23,774.2** | 11,887.1 | **17.08%** | 17.92% | **`+7.4% tok/s`** (`+1.18 pp` Causal / `+1.24 pp` TorchTitan MFU) | 15.54 | `12.23864 -> 2.83929` |
| **Step 4 (Full Tuned Recipe, `C4-EoD`): + Save MoE Routing & Compile Fused AdamW + Grad-Norm**<br>Saves router indices/weights for backward (no router recompute) and compiles fused AdamW + clip-grad-norm into a dedicated XLA graph | `C4-EoD` | **24,582.8** | 12,291.4 | **17.66%** | 18.52% | **`+11.1% tok/s`** (`+1.76 pp` Causal / `+1.84 pp` TorchTitan MFU) | 15.52 | `12.23864 -> 2.84265` |
| **Step 4 (Full Tuned Recipe, Contiguous Full-Sequence Attention)**<br>Same recipe with `TT_CONTIGUOUS_FULL_SEQ=1` so every row attends across the full `2,048` causal window without `c4_test` EoD block skipping | `Full-Seq` | **24,396.8** | 12,198.4 | **17.52%** | 18.38% | **`+10.2% tok/s`** (`+1.62 pp` Causal / `+1.70 pp` TorchTitan MFU) | 15.52 | `12.23512 -> 2.80278` |

### Key Takeaways from the `2K` Ablation Ladder

1. **Incremental Contribution Breakdown (`+11.1% tok/s` total over the Native baseline reproduction):**
   - **Step 1 (`BF16` ReduceScatter + async collective flags):** Neutral (`0.0% tok/s`) on a compact 2-host `v7x-8` (`2x2x2`) torus where ICI bandwidth already hides `ReduceScatter`, while freeing communication headroom on larger `4`-host (`v7x-16`, `v5p-16`) and `8`-host (`v5p-32`) slices.
   - **Step 2 (`49/49` Splash outputs saved + `tile_m=1024`):** **`+2.3% tok/s`** (`+0.36 pp` Exact Causal MFU) while *reducing* peak HBM from `16.35 GiB` to `14.76 GiB` due to cleaner XLA buffer liveness across larger expert tiles.
   - **Step 3 (`scale_before_down_proj` + `BF16` RoPE):** **`+5.0% tok/s` incremental** (`+7.4% tok/s` cumulative) by eliminating `FP32` RoPE casts and simplifying the post-expert reduction.
   - **Step 4 (Saved MoE routing + compiled fused AdamW & grad-norm):** **`+3.4% tok/s` incremental** (`+11.1% tok/s` cumulative over the reproduced Native baseline).
2. **Numerical Fidelity:**
   - All `C4-EoD` Native rows start from identical step-1 loss (`12.23862`–`12.23864`) and converge to the same step-60 loss band (`2.8367`–`2.8427`).
3. **Packing Sensitivity at `2K` vs. Long Context:**
   - At `2,048` tokens, disabling `c4_test` End-of-Document (`EoD`) block skipping reduces throughput by only `0.8% tok/s` (`0.14 pp` MFU: `17.66% -> 17.52%` Exact Causal MFU), because attention FLOPs are a small fraction of total compute at `2K`. At `32K`–`64K`, by contrast, `c4_test` EoD skipping omits `>98%` of global causal attention blocks and severely distorts MFU (see [`../README.md`](../README.md#why-60-step-contiguous-full-sequence-validation-is-required)).

---

## 3. Per-Rank Batch-Size Ceiling on 16-Way `SimpleFSDP` (`v7x-8`) vs. 32-Way `SimpleFSDP` (`v7x-16`)

On `v7x-8` (`16` TensorCore ranks), `39.8B` parameters + AdamW states sharded `16`-way leave less unreserved HBM per TensorCore than `32`-way sharding on `v7x-16`:

| Slice & Sharding | Batch Configuration | Global Tokens/Step | Outcome |
|---|---|---:|---|
| `v7x-8` (`16` TensorCore ranks, `SimpleFSDP-16`) | `mb=32, ga=1` (`65,536` tok/rank) | `1,048,576` | **`PASS`** (`24,582.8` tok/s/chip, `15.52 GiB` active HBM) |
| `v7x-8` (`16` TensorCore ranks, `SimpleFSDP-16`) | `mb=40, ga=2` (`81,920` tok/rank) | `2,621,440` | Program-load HBM OOM (`17.26 GiB` required vs. `16.58 GiB` reservable) |
| `v7x-8` (`16` TensorCore ranks, `SimpleFSDP-16`) | `mb=48, ga=2` (`98,304` tok/rank) | `3,145,728` | Compile-time HBM OOM (`0.58 GiB` over per-TensorCore budget) |
| `v7x-16` (`32` TensorCore ranks, `SimpleFSDP-32`) | `mb=32, ga=1` (`65,536` tok/rank, `Full-Seq`) | `2,097,152` | **`PASS`** (`25,168.4` tok/s/chip, `18.08%` Exact Causal MFU, `15.52 GiB` HBM) |
| `v7x-16` (`32` TensorCore ranks, `SimpleFSDP-32`) | `mb=48, ga=2` (`98,304` tok/rank, `Full-Seq`) | `6,291,456` | **`PASS`** (`26,947.8` tok/s/chip, `19.36%` Exact Causal MFU, `23.02 GiB` HBM) |

---

## 4. Reproducing This Suite

- **Runner script:** [`scripts/run_flint_60step_validation.py`](../scripts/run_flint_60step_validation.py) (`--suite=baseline_2k`, function `build_baseline_2k_experiments`)
- **Kubernetes Job manifest:** [`k8s/v7x-8-flint-baseline-2k.yaml`](../k8s/v7x-8-flint-baseline-2k.yaml)
- **Raw 60-step metrics & logs summary:** [`results/baseline_2k/v7x-8/SUMMARY.md`](../results/baseline_2k/v7x-8/SUMMARY.md) and [`results/baseline_2k/v7x-8/summary.json`](../results/baseline_2k/v7x-8/summary.json)
