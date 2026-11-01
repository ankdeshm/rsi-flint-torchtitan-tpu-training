# `RecursiveSI/flint` (40B MoE) Pre-Training & Long-Context MFU Optimization on Cloud TPU (`v7x` & `v5p`) with TorchTitan + TorchTPU 0.2.0 GA

End-to-end distributed pre-training, long-context (`2K` / `4K` / `8K` / `16K`) MFU scaling, HuggingFace safetensors fine-tuning, numerical checkpoint parity verification, and static-shape Expert Parallelism (`StaticEP=2..32`) for **[`RecursiveSI/flint`](https://huggingface.co/RecursiveSI/flint)** (`39.8B` total / `~2.52B` active parameters per token) on **TPU v7x (Ironwood)** and **TPU v5p** using **TorchTitan** and **TorchTPU `0.2.0` GA**.

See [`docs/TRAINING_LEADERBOARD.md`](./docs/TRAINING_LEADERBOARD.md) for the complete experiment-by-experiment TorchTitan leaderboard across all `v7x-16` (`Pass 1..13`) and `v5p` (`Pass 1..10`) sweeps.

---

## 1. Model Overview — [`RecursiveSI/flint`](https://huggingface.co/RecursiveSI/flint)

| Property | Value |
|---|---|
| **Hugging Face Checkpoint** | [`https://huggingface.co/RecursiveSI/flint`](https://huggingface.co/RecursiveSI/flint) (`15` safetensors shards, `79.59 GB` BF16) |
| **Architecture** | Sparse Mixture-of-Experts (`49` Transformer layers, `hidden_size=2048..2560`, `vocab_size=151,936`, `12` Global + `37` SWA layers) |
| **MoE Routing** | `256` routed experts per layer, **top-6** active routing (`~2.52B` active parameters / `39.8B` total) |
| **FLOPs Accounting** | `17.385` GFLOP/tok (`2K`) · `18.794` GFLOP/tok (`4K`) · `21.613` GFLOP/tok (`8K`) · `27.250` GFLOP/tok (`16K`) |
| **Hardware Peak BF16** | TPU `v7x`: `1,153.5` BF16 TFLOP/s per TensorCore (`2` TensorCores/chip) · TPU `v5p`: `459.0` BF16 TFLOP/s per chip |

---

## 2. Software & Hardware Requirements

- **Runtime Stack** (built via [`docker/Dockerfile`](./docker/Dockerfile)):
  - `python:3.14`
  - `torch==2.15.0+cpu`
  - `torch_tpu==0.2.0` (GA)
  - `jax==0.11.1`, `libtpu==0.0.50`, `tokamax==0.0.14`
  - `transformers>=5.22.0`, `safetensors`, `accelerate`, `sentencepiece`
  - `libtcmalloc.so.4` (`LD_PRELOAD` enabled at runtime)
- **Two TorchTitan Execution Lanes Included in One Image:**
  1. **Native `TpuGraphTrainer` Lane (`/torchtitan-native`):** Built on [`google-pytorch/torchtitan` PR #122 (`alekseyv_flint`)](https://github.com/google-pytorch/torchtitan/pull/122) — native `TpuGroupedExperts`, `GraphSplitter` per-layer XLA executables, `SimpleFSDP`, `StaticTpuAllToAllTokenDispatcher`, and `FlintStateDictAdapter`.
  2. **HuggingFace Modeling Backend Lane (`/torchtitan-hf`):** Built on [`google-pytorch/torchtitan` PR #123 (`hf-backend-tpu-20261028-v020-flint-tune`)](https://github.com/google-pytorch/torchtitan/pull/123) — `transformers_modeling_backend` with Pallas GroupedMM expert replacement and SplashAttention.

---

## 3. Summary of TorchTitan Results (`v7x-16` 32 TensorCores, `v5p-32` 32 Chips & `v5p-16` 16 Chips)

| Platform & Topology | Execution Lane & Context Length | Baseline Reference | Optimized Steady MFU | Optimized Peak MFU | Per-Rank Throughput | Slice Throughput | Real C4 Loss Convergence |
|---|---|---:|---:|---:|---:|---:|---|
| **TPU `v5p-32` (`2x4x4`, 32 Chips)** | **HF Backend `16K` + `SBDP` + `FusedOpt` + `sa=1024` + `HybridAC18` (`v5p32_p10_03`, `mb=8`, `4.19M` tok/step)** | `19.76%` *(v5p-16 2K base)* | **`35.33%`** *(+15.57 pp)* | **`36.20%`** *(step-10)* | **`5,951.6` tok/s** *(6,098 pk)* | **`190,451` tok/s** *(195.1K pk)* | `12.2938 -> 5.3098` *(`22.84 GiB` HBM — **#1 Overall MFU Record!**)* |
| **TPU `v5p-16` (`2x2x4`, 16 Chips)** | **HF Backend `16K` + `SBDP` + `FusedOpt` + `sa=1024` + `HybridAC18` (`v5p16_p10_03`, `mb=8`, `2.10M` tok/step)** | `19.76%` *(v5p-16 2K base)* | **`35.21%`** *(+15.45 pp)* | **`35.92%`** *(step-5)* | **`5,931.0` tok/s** *(6,051 pk)* | **`94,896` tok/s** *(96.8K pk)* | `12.4962 -> 3.5523` *(`22.84 GiB` HBM — **#1 16-Chip MFU Record!**)* |
| **TPU `v5p-32` (`2x4x4`, 32 Chips)** | **Native `TpuGraphTrainer` `16K` + `SBDP` + `BF16-RoPE` + `FusedOpt` (`v5p32_p10_07`, `mb=6`, `tile_m=256`, `3.15M` tok/step)** | `VMEM OOM` *(at tile_m=512)* | **`33.67%`** *(+9.78 pp vs 2K)* | **`34.63%`** *(step-6)* | **`5,670.7` tok/s** *(5,833 pk)* | **`181,462` tok/s** *(186.7K pk)* | `12.3335 -> 4.1517` *(`29.21 GiB` HBM — **#1 Native `v5p` Record!**)* |
| **TPU `v5p-32` (`2x4x4`, 32 Chips)** | **HF Backend `8K` + `SBDP` + `FusedOpt` + `sa=1024` + `HybridAC16` (`v5p32_p10_02`, `mb=16`)** | `19.76%` *(2K base)* | **`29.63%`** *(+9.87 pp)* | **`29.84%`** | **`6,291.7` tok/s** | **`201,334` tok/s** | `12.2938 -> 5.5106` *(`22.84 GiB` HBM)* |
| **TPU `v5p-16` (`2x2x4`, 16 Chips)** | **HF Backend `8K` + `SBDP` + `FusedOpt` + `sa=1024` + `HybridAC16` (`v5p16_p10_02`, `mb=16`)** | `19.76%` *(2K base)* | **`29.25%`** *(+9.49 pp)* | **`29.42%`** | **`6,212.4` tok/s** | **`99,398` tok/s** | `12.4961 -> 3.6147` *(`22.84 GiB` HBM)* |
| **TPU `v5p-32` (`2x4x4`, 32 Chips)** | **Native `TpuGraphTrainer` `8K` + `SBDP` + `BF16-RoPE` + `FusedOpt` (`v5p32_p10_06`, `mb=12`, `tile_m=256`)** | `VMEM OOM` *(at tile_m=512)* | **`28.42%`** | **`28.61%`** | **`6,036.3` tok/s** *(6,076 pk)* | **`193,162` tok/s** *(194.4K pk)* | `12.3335 -> 4.1619` *(`29.21 GiB` HBM)* |
| **TPU `v5p-32` (`2x4x4`, 32 Chips)** | **HF Backend `4K` + `SBDP` + `FusedOpt` + `sa=1024` + `HybridAC16` (`v5p32_p10_01`, `mb=32`)** | `19.76%` *(2K base)* | **`26.59%`** *(+6.83 pp)* | **`26.64%`** | **`6,494.7` tok/s** | **`207,830` tok/s** | `12.2940 -> 5.2737` *(`22.84 GiB` HBM)* |
| **TPU `v5p-16` (`2x2x4`, 16 Chips)** | **HF Backend `4K` + `SBDP` + `FusedOpt` + `sa=1024` + `HybridAC16` (`v5p16_p10_01`, `mb=32`)** | `19.76%` *(2K base)* | **`25.96%`** *(+6.20 pp)* | **`26.00%`** | **`6,339.7` tok/s** | **`101,435` tok/s** | `12.4960 -> 3.6804` *(`22.84 GiB` HBM)* |
| **TPU `v5p-16` (`2x2x4`, 16 Chips)** | **HF Backend `2K` + `SBDP` + `FusedOpt` + `sa=1024` + `HybridAC16` (`v5p_p9_02`, `mb=64`)** | `19.76%` *(v5p-16 base)* | **`25.76%`** *(+6.00 pp)* | **`25.81%`** | **`6,802.0` tok/s** | **`108,832` tok/s** | `12.4957 -> 3.9676` *(`22.84 GiB` HBM — **#1 `v5p-16` `2K` Record!**)* |
| **TPU `v5p-32` (`2x4x4`, 32 Chips)** | **HF Backend `2K` + `SBDP` + `FusedOpt` + `sa=1024` + `HybridAC16` (`v5p_p9_02`, `mb=64`)** | `19.76%` *(v5p-16)* / `19.95%` *(mb=32)* | **`24.88%`** *(+5.12 pp)* | **`24.91%`** *(24.95% `mb=48`)* | **`6,568.4` tok/s** | **`210,189` tok/s** | `12.2951 -> 4.3073` *(`22.84 GiB` HBM — **#1 `v5p-32` `2K` Record!**)* |
| **TPU `v5p-32` (`2x4x4`, 32 Chips)** | **Native `TpuGraphTrainer` `2K` + `SBDP` + `BF16-RoPE` + `FusedOpt` (`v5p_p9_06`, `mb=48`)** | `VMEM OOM` *(at tile_m=512)* | **`23.89%`** | **`24.00%`** | **`6,308.1` tok/s** | **`201,859` tok/s** | `12.3288 -> 4.1735` *(`28.11 GiB` HBM)* |
| **TPU `v5p-32` (`2x4x4`, 32 Chips)** | **HF Backend `2K` + `HSDP 2×16` (`v5p_p8_06`, `2x` GA, `6.29M` tok/step)** | `19.76%` *(v5p-16)* | **`23.81%`** *(+4.05 pp)* | **`23.83%`** | **`6,285.7` tok/s** | **`201,142` tok/s** | `12.4958 -> 3.7789` *(`21.08 GiB` HBM)* |
| **TPU `v7x-16` (`2x2x4`, 32 TensorCores)** | **Native `TpuGraphTrainer` `16K` (`v7x_p13_03`, Scout XLA + `SBDP` + `BF16-RoPE` + `PureFusedOpt` + `tile1024` + `n49 = 0%` Attn Recompute + `8x` GA = `25.17M` tok/step, `mb=6`)** | `16.23%` *(v7x-8)* / `16.56%` *(v7x-16 base)* | **`27.15%`** *(+10.92 pp!)* | **`27.20%`** *(`10.04` PFLOP/s!)* | **`11,494.0` tok/s** *(11,513 pk)* | **`367,808` tok/s** *(368.4K pk)* | `12.3243 -> 3.7368` *(`20.52 GiB` HBM — **#1 All-Time Ironwood MFU Record!**)* |
| **TPU `v7x-16` (`2x2x4`, 32 TensorCores)** | **Native `TpuGraphTrainer` `8K` (`v7x_p13_02`, Scout XLA + `SBDP` + `BF16-RoPE` + `PureFusedOpt` + `tile1024` + `n49 = 0%` Attn Recompute + `8x` GA = `25.17M` tok/step, `mb=12`)** | `16.23%` *(v7x-8)* / `16.56%` *(v7x-16 base)* | **`23.13%`** *(+6.90 pp)* | **`23.13%`** *(`8.54` PFLOP/s!)* | **`12,343.5` tok/s** *(12,346 pk)* | **`394,992` tok/s** *(395.1K pk)* | `12.3236 -> 3.7403` *(`20.52 GiB` HBM)* |
| **TPU `v7x-16` (`2x2x4`, 32 TensorCores)** | **Native `TpuGraphTrainer` `4K` (`v7x_p13_01`, Scout XLA + `SBDP` + `BF16-RoPE` + `PureFusedOpt` + `tile1024` + `n49` + `8x` GA = `25.17M` tok/step, `mb=24`)** | `16.23%` *(v7x-8)* / `16.56%` *(v7x-16 base)* | **`21.92%`** *(+5.69 pp)* | **`21.93%`** | **`13,455.5` tok/s** *(13,462 pk)* | **`430,576` tok/s** *(430.8K pk)* | `12.3222 -> 3.7291` *(`23.52 GiB` HBM)* |
| **TPU `v7x-16` (`2x2x4`, 32 TensorCores)** | **Native `TpuGraphTrainer` `2K` (`v7x_p12g_02`, Scout XLA + `SBDP` + `BF16-RoPE` + `PureFusedOpt` + `tile1024` + `n49 = 0%` Attn Recompute + `BF16` Reduce + `8x` GA = `25.17M` tok/step, `mb=48`)** | `16.23%` *(v7x-8)* / `16.56%` *(v7x-16 base)* | **`20.80%`** *(+4.57 pp)* | **`20.81%`** | **`13,802.0` tok/s** *(13,811 pk)* | **`441,664` tok/s** *(442.0K pk)* | `12.3202 -> 3.7124` *(`23.02 GiB` HBM — **All-Time Cluster Tok/s Record!**)* |
| **TPU `v7x-16` (`2x2x4`, 32 TensorCores)** | **Native `TpuGraphTrainer` `2K` (`v7x_p12g_01`, `SBDP` + `BF16-RoPE` + `PureFusedOpt` + `n49` + `16x` GA = `50.33M` tok/step, `mb=48`)** | `16.23%` *(v7x-8)* / `16.56%` *(v7x-16 base)* | **`20.78%`** *(+4.55 pp)* | **`20.79%`** | **`13,787.0` tok/s** *(13,793 pk)* | **`441,184` tok/s** *(441.4K pk)* | `12.3202 -> 3.7636` *(`23.02 GiB` HBM)* |
| **TPU `v7x-16` (`2x2x4`, 32 TensorCores)** | **Native Full Host Offload `save_last_n=0, prefetch=2` + `SBDP` + `BF16-RoPE` (`v7x_p12e_04`, `mb=64`, `2x` GA = `8.39M` tok/step)** | `OOM` *(without offload)* | **`18.60%`** | **`18.61%`** | **`12,342.7` tok/s** | **`394,966` tok/s** | `12.3206 -> 3.2794` *(`34.91 GiB` HBM — **#1 Full Offload Record!**)* |
| **TPU `v7x-16` (`2x2x4`, 32 TensorCores)** | **HF Backend `2K` + `SBDP` + `PureFusedOpt` + `sa=1024` + `tile1024` + `HybridAC12` (`v7x_p12_04`, `mb=64`, `2x` GA)** | `16.29%` *(v7x-8)* / `15.87%` *(v7x-16 base)* | **`18.58%`** *(+2.71 pp)* | **`18.59%`** | **`12,331.0` tok/s** | **`394,592` tok/s** | `12.3104 -> 3.7640` *(`25.50 GiB` HBM — **HF `v7x` Record!**)* |
| **TPU `v7x-16` (`2x2x4`, 32 TensorCores)** | **HF Backend `2K` + `HSDP 2×16` (`v7x_p12c_05`, `SBDP` + `PureFusedOpt` + `tile1024`, `mb=48`, `2x` GA)** | `15.87%` *(v7x-16 base)* | **`18.25%`** *(+2.38 pp)* | **`18.27%`** | **`12,114.8` tok/s** | **`387,674` tok/s** | `12.4342 -> 3.5752` *(`21.22 GiB` HBM — **HSDP Record!**)* |
| **TPU `v7x-16` (`2x2x4`, 32 TensorCores)** | **Intra-Chip `StaticEP=2 × FSDP=16 + ZeRO-3 + PairedGMM-512` (`v7x_p12_08`, `0` Inter-Chip A2A)** | `Recompile/OOM` *(dynamic EP)* | **`11.85%`** *(+76% vs EP=8)* | **`12.24%`** *(step-5)* | **`7,860.0` tok/s** *(8,120 pk)* | **`251,520` tok/s** *(259.8K pk)* | `12.3189 -> 3.6921` *(`19.73 GiB` HBM — **#1 StaticEP Record!**)* |
| **TPU `v7x-16` (`2x2x4`, 32 TensorCores)** | **Intra-Tray `StaticEP=4 × FSDP=8 + ZeRO-3 + PairedGMM-512` (`v7x_p12_09`, `2`-Chip A2A)** | `Recompile/OOM` *(dynamic EP)* | **`7.17%`** | **`7.37%`** *(step-5)* | **`4,759.5` tok/s** | **`152,304` tok/s** | `12.3177 -> 3.6906` *(`21.05 GiB` HBM — **#2 StaticEP**)* |
| **TPU `v5p-32` (`2x4x4`, 32 Chips)** | **Native `StaticEP=8 × FSDP=4 + ZeRO-2 + PairedGMM-256` (`v5p_p9f_02`)** | `Recompile/OOM` *(dynamic EP)* | **`7.30%`** | **`7.34%`** | **`1,926.1` tok/s** | **`61,635` tok/s** | `12.3288 -> 4.2121` *(`37.98 GiB` HBM)* |
| **TPU `v7x-16` (`2x2x4`, 32 TensorCores)** | **Intra-Host `StaticEP=8 × FSDP=4 + ZeRO-3 + PairedGMM-256` (`v7x_p11d_04`, `4`-Chip A2A)** | `Recompile/OOM` *(dynamic EP)* | **`6.73%`** | **`6.81%`** | **`4,463.0` tok/s** | **`142,816` tok/s** | `12.3215 -> 4.1913` *(`22.69 GiB` HBM)* |

---

## 4. From Baseline to `35.33%` / `36.20%` (`v5p-32`), `35.21%` / `35.92%` (`v5p-16`), & `27.15%` / `27.20%` (`v7x-16` — `10.04` PFLOP/s!) — Engineering Breakdown

### Track A: Native `TpuGraphTrainer` on Ironwood `v7x-16` (PR #122 `flint_40b_v011_n40` -> `20.80%` @ `2K`, `21.92%` @ `4K`, `23.13%` @ `8K`, `27.15%` / `27.20%` @ `16K`)

#### What the Baseline Did (`flint_40b_v011_n40`)
- Used `TpuGraphTrainer` with `GraphSplitter` (`1` forward + `1` backward XLA executable per layer), `SimpleFSDP`, kernel-native `(E, K, N)` weight layout, Paired W1/W3 Pallas GroupedMM at `tile_m=512`, `local_batch_size=32` (`65,536` tok/rank), saving SplashAttention outputs in only the last `40/49` layers (`n40`, recomputing 9 layers), FP32 gradient reduce-scatter, FP32 RoPE casts, and eager DTensor AdamW + gradient norm.
- Measured **`16.23%` MFU (`10,779` tok/s/rank)** on `v7x-8` (`16` TensorCores) and **`16.56%` Steady MFU (`10,985.3` tok/s/rank)** out-of-the-box on `v7x-16` (`32` TensorCores).

#### Exact Optimizations Applied Over the Baseline (`Pass 1 -> Pass 13`)

| Step | Experiment ID | Exact Change Over Previous Step | Steady tok/s/rank | Cluster tok/s | Steady MFU | Peak MFU | Delta vs `v7x-8` (`16.23%`) |
|---|---|---|---:|---:|---:|---:|---:|
| **0 (Ref)** | *PR #122 `v7x-8`* | Baseline `flint_40b_v011_n40` (`16` ranks, `mb=32`, `tile_m=512`, `n40`) | `10,779.0` | `172,464` | `16.23%` | — | — |
| **1 (Repro)** | `02_native_pr122_v011_n40` | Direct reproduction on `v7x-16` (`32` ranks, `SimpleFSDP32`, `2.10M` tok/step) | `10,985.3` | `351,530` | `16.56%` | `16.69%` | `+0.33 pp` |
| **2** | `03_native_v012_hier_rs` | Enable **multi-axis ND Hierarchical SparseCore Reduce-Scatter** across `2x2x4` torus | `11,011.9` | `352,381` | `16.60%` | `16.70%` | `+0.37 pp` |
| **3** | `04_native_v013_n49` | Save SplashAttention across **all 49 layers (`n49`)** (`98.0%` XLA cache hit rate) | `11,135.8` | `356,346` | `16.78%` | `16.86%` | `+0.55 pp` |
| **4** | `05_native_v014_tile1024` | Widen forward Paired W1/W3 Pallas GroupedMM MXU tile `512 -> 1024` (`(1024, 2048, 2048)`) with VMEM-safe backward `(512, 2048, 0)` | `11,196.2` | `358,278` | `16.87%` | `16.94%` | `+0.64 pp` |
| **5** | `p3_01_comp_opt_gradnorm` | Compile 779-parameter DTensor AdamW (`optimizer_compile="separate_graph"`) + global L2 GradNorm | `11,448.6` | `366,355` | `17.25%` | `17.32%` | `+1.02 pp` |
| **6** | `p3_02_save_moe_routing` | Save `moe.routing` across all 49 layers to skip backward router logits + top-6 sort recompute | `11,540.8` | `369,306` | `17.39%` | `17.46%` | `+1.16 pp` |
| **7** | `p7_01_grad_accum_2x_mb48` | Scale microbatch **1.5×** (`mb=48`) + **`2×` gradient accumulation (`6.29M` tok/step)** + `n43` | `12,182.1` | `389,827` | `18.36%` | `18.42%` | `+2.13 pp` |
| **8 (`Pass 9`)** | `v7x_p9_02_bf16red_scout` | Switch gradient `ReduceScatter` to **`BF16`** (`159.2 GB -> 79.6 GB` ICI) + async AG/RS collective overlap flags (`27.12 GiB` HBM) | `12,392.8` | `396,570` | `18.68%` | `18.76%` | `+2.45 pp` |
| **9 (`Pass 12C`)** | `v7x_p12c_01_sbdp_bf16rope` | Add **Sequence-Block Document Causal SplashAttention (`SBDP`)** + **pure `BF16` RoPE** (`n41`, `mb=48`, `2×` GA = `6.29M` tok/step, `25.48 GiB` HBM) | `13,675.7` | `437,622` | `20.61%` | `20.63%` | `+4.38 pp` |
| **10 (`Pass 12G` `2K` #1!)** | **`v7x_p12g_02_sbdp_bf16rope_fusedopt_n49_ga8`** | **`n49 = 0%` Attn Recompute** + **Pure `"fused"` Compiled AdamW (`PureFusedOpt`)** at `8×` GA (`25.17M` tok/step, `23.02 GiB` HBM — **`441,664` Steady / `441,952` Peak tok/s**!) | **`13,802.0`** | **`441,664`** *(442.0K pk)* | **`20.80%`** | **`20.81%`** | **`+4.57 pp`** |
| **11 (`Pass 13` `4K` #1!)** | **`v7x_p13_01_native_s4096_mb24_ga8`** | Scale context **`2K -> 4K`** (`mb=24`, constant `98,304` tok/rank, recursive 49-layer RoPE `max_context_length`, `23.52 GiB` HBM) | **`13,455.5`** | **`430,576`** *(430.8K pk)* | **`21.92%`** | **`21.93%`** | **`+5.69 pp`** |
| **12 (`Pass 13` `8K` #1!)** | **`v7x_p13_02_native_s8192_mb12_ga8`** | Scale context **`4K -> 8K`** (`mb=12`, constant `98,304` tok/rank, `25.17M` tok/step, **`20.52 GiB` (`21.38%`) HBM**, `8.539` PFLOP/s sustained!) | **`12,343.5`** | **`394,992`** *(395.1K pk)* | **`23.13%`** | **`23.13%`** | **`+6.90 pp`** |
| **13 (`Pass 13` `16K` #1!)** | **`v7x_p13_03_native_s16384_mb6_ga8`** | Scale context **`8K -> 16K`** (`mb=6`, constant `98,304` tok/rank, **`20.52 GiB` HBM**, **`10.023` PFLOP/s sustained / `10.039` PFLOP/s peak**, `loss: 12.3243 -> 3.7368`!) | **`11,494.0`** | **`367,808`** *(368.4K pk)* | **`27.15%`** | **`27.20%`** | **`+10.92 pp`** |

---

### Track B: HuggingFace Modeling Backend on `v7x-16` (PR #123 `hf_moe_auto_max_attnac` -> `v7x_p12_04` `18.58%` MFU)

- **Baseline (`01_hf_backend_pr123_flint_40b`):** Left `experts.use_pallas_paired_grouped_mm = False`, used FP32 ReduceScatter, eager optimizer, and `mb=32`, measuring **`15.87%` Steady MFU (`10,532.0` tok/s/rank)** on `v7x-16` (`16.29%` on `v7x-8`).
- **Progressive Hillclimb (`Pass 5 -> Pass 12C`):**
  1. **Fused Paired W1/W3 Pallas GroupedMM (`tile_m=1024`) + `32 MB` Scoped VMEM Backward TGMM (`512:2048:0`) + Chunked CE (`b=1024`) (`p7_03`):** Lifted MFU to **`18.02%` (`382,621` tok/s)** at `mb=48`.
  2. **`BF16` ReduceScatter + Async ICI Overlap (`v7x_p9_03`):** Lifted `mb=48` (`1×` step) to **`18.29%` Steady / `18.34%` Peak MFU (`388,250` tok/s, `16.92 GiB` HBM)**.
  3. **`mb=64` + `2×` / `4×` GA (`8.39M` & `16.78M` tok/step) + `PureFusedOpt` + `tile_m=1024` + `HybridAC N=12` (`v7x_p12_04` & `v7x_p12b_04`):** Set the **#1 HF Backend `v7x-16` `2K` Record at `18.58%` Steady / `18.59%` Peak MFU (`12,331.0` tok/s/rank = `394,592` tok/s, `25.50 GiB` HBM, `loss: 12.3104 -> 3.7640`)**, sustained **`18.55%` MFU** at `16.78M` tok/step (`4×` GA), and reached **`18.25%` / `18.27%` MFU (`387,674` tok/s, `21.22 GiB` HBM)** with **`HSDP 2×16` (`v7x_p12c_05`)**!

---

### Track C: Cross-Generation TorchTitan on TPU `v5p-32` (`35.33%`/`36.20%` HF & `33.67%`/`34.63%` Native MFU) & `v5p-16` (`35.21%`/`35.92%` MFU)

- **Baseline (`01_hf_backend_pr123_flint_v5p`):** Measured **`19.76%` Steady MFU (`5,215.8` tok/s/chip)** on `v5p-16` (`2x2x4`, `459` BF16 TFLOP/s/chip) at `2K`, while Native PR #122 failed XLA compilation out-of-the-box due to `v5p`'s **`16.00 MB` Scoped VMEM** limit (`tile_m=512` paired GMM requires `16.58 MB`).
- **HF Backend `v5p` `2K` Record (`v5p_p9_02`, Pass 9):** Scaling to `mb=64` (`131,072` tok/chip) with `SBDP` + `FusedCompiledOpt` + `Splash sa=1024` + `HybridAC N=16` + `BF16` Reduce + `tile_m=256` vaulted **`v5p-16` to `25.76%` Steady / `25.81%` Peak MFU (`108,832` tok/s, `22.84 GiB` HBM)** and **`v5p-32` to `24.88%` Steady / `24.95%` Peak MFU (`210,189` tok/s)**, alongside Native `TpuGraphTrainer` (`v5p_p9_06`, **`23.89%` / `24.00%` MFU** at VMEM-safe `tile_m=256`).
- **Long-Context (`4K` / `8K` / `16K`) TorchTitan MFU Records (`Pass 10` — `scripts/run_flint_v5p_pass10_longctx.py`):** Holding `131,072` tok/chip/microbatch constant kept HBM **100% flat at `22.84 GiB` (`24.04%`)** across `2K..16K` while lifting **`v5p-32` (`32` chips)** to **`26.59%` / `26.64%` (`4K`) -> `29.63%` / `29.84%` (`8K`) -> `35.33%` Steady / `36.20%` Peak MFU (`190,451` / `195,136` tok/s, `loss: 12.2938 -> 5.3098`) at `16K` (`v5p32_p10_03`)**, Native `TpuGraphTrainer` on `v5p-32` to **`28.42%` / `28.61%` (`8K`, `v5p32_p10_06`) -> `33.67%` Steady / `34.63%` Peak MFU (`181,462` / `186,656` tok/s, `29.21 GiB` HBM) at `16K` (`v5p32_p10_07`)**, and **`v5p-16` (`16` chips)** to **`25.96%` / `26.00%` (`4K`) -> `29.25%` / `29.42%` (`8K`) -> `35.21%` Steady / `35.92%` Peak MFU (`94,896` / `96,816` tok/s, `loss: 12.4962 -> 3.5523`) at `16K` (`v5p16_p10_03`)**!

---

### Track D: Zero-Recompile Static-Shape Expert Parallelism (`StaticEP=2..32`) Scaling Law

- **Engineering Fix (`StaticTpuAllToAllTokenDispatcher`):** Upstream `TokenDispatcher` (`EP > 1`) triggers dynamic-shape XLA recompilation and `SymNode` / `Eq(u46, 128)` Dynamo guard crashes (`torch.bincount`). Replacing token dispatch with a fixed-capacity `[EP, E_local, C, D]` `all_to_all_single` counting-bucket dispatcher (`(recv_lexp.unsqueeze(1) == arange(E)).sum(0)`), `EPCompiledGradNorm`, and `SplitMeshOpt` unlocked zero-recompile training across **`EP = 2, 4, 8, 16, 32`**.
- **Monotonic Ironwood Dual-Die Interconnect Law (`v7x-16`, `32` TensorCores):**
  - **`StaticEP=2 × FSDP=16 + ZeRO-3` (`v7x_p12_08` — Intra-Package Die-to-Die, `0` Inter-Chip ICI Hops):** **`11.85%` Steady / `12.24%` Step-5 Peak MFU** (`7,860.0` steady / `8,120.0` peak tok/s/rank = **`251,520` steady tok/s**, **`19.73 GiB` (`20.55%`) HBM**, `loss: 12.3189 -> 3.6921`) — **`+65%` faster** than `StaticEP=4` and **`+76%` faster** than `StaticEP=8`!
  - **`StaticEP=4 × FSDP=8 + ZeRO-3` (`v7x_p12_09` — 2-Chip Intra-Tray):** **`7.17%` Steady / `7.37%` Peak MFU** (`152,304` tok/s, `21.05 GiB` HBM).
  - **`StaticEP=8 × FSDP=4` (`4`-Chip Intra-Host):** **`6.73%` Steady / `6.81%` Peak MFU** on `v7x-16` (`v7x_p11d_04`, `22.69 GiB` HBM) and **`7.30%` Steady / `7.34%` Peak MFU** on `v5p-32` (`v5p_p9f_02`, `37.98 GiB` HBM).
  - **`StaticEP=16 × FSDP=2 + ZeRO-2` (`2`-Host Inter-Host):** **`5.85%` Steady / `6.11%` Peak MFU** on `v7x-16` (`v7x_p11d_03`, `24.48 GiB` HBM) and **`6.78%` MFU** on `v5p-32` (`v5p_p9e_03`, `23.62 GiB` HBM).
  - **`WholeStep StaticEP=32 + ZeRO-2` (`4`-Host Full Slice, Single Graph):** **`4.70%` Steady / `4.86%` Peak MFU** on `v7x-16` (`v7x_p11d_05`, `55.84 GiB` HBM).

---

## 5. Repository Structure

```text
├── README.md
├── docs/
│   └── TRAINING_LEADERBOARD.md        # Complete TorchTitan experiment leaderboard & production configs
├── docker/
│   ├── Dockerfile                     # Builds PyTorch 2.15 + TorchTPU 0.2.0 GA + PR #122 & PR #123
│   └── patch_flint_configs.py         # Appends hillclimb & v5p/v6e configs to Native config_registry.py
├── scripts/                           # All Pass 1..13 (v7x-16) & Pass 1..10 (v5p) training & long-context sweep scripts
├── k8s/                               # Multi-host GKE TPU Indexed Job & GCS Fuse manifests
└── results/                           # Output summary.json & SUMMARY.md artifacts across v7x-16 & v5p
```

---

## 6. Quickstart Reproduction Guide

```bash
# 1. Build & Push Container Image
docker build -f docker/Dockerfile -t us-central1-docker.pkg.dev/YOUR_GCP_PROJECT_ID/torchtpu/torchtitan-flint:v0.2.0 ./docker
docker push us-central1-docker.pkg.dev/YOUR_GCP_PROJECT_ID/torchtpu/torchtitan-flint:v0.2.0

# 2. Stage RecursiveSI/flint (15 shards) to GCS
kubectl apply -f k8s/storage-gcsfuse.yaml
kubectl apply -f k8s/stage-flint-job.yaml

# 3. Run Pass 13 (4K/8K/16K Long-Context MFU Sweep) & Pass 10 (v5p 35.33% / 36.20% MFU @ 16K)
kubectl apply -f k8s/v7x-16-flint-pass13.yaml
kubectl apply -f k8s/v5p-32-flint-pass10.yaml
kubectl logs -f job/flint-v7x-16-pass13
```
