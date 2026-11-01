# `RecursiveSI/flint` (40B MoE) Pre-Training & Fine-Tuning on Cloud TPU (`v7x` & `v5p`) with TorchTitan + TorchTPU 0.2.0 GA

End-to-end distributed pre-training, HuggingFace safetensors fine-tuning, numerical checkpoint parity verification, and Model FLOPs Utilization (MFU) optimization for **[`RecursiveSI/flint`](https://huggingface.co/RecursiveSI/flint)** (`39.8B` total / `~2.52B` active parameters per token) on **TPU v7x (Ironwood)** and **TPU v5p** using **TorchTitan** and **TorchTPU `0.2.0` GA**.

See [`docs/TRAINING_LEADERBOARD.md`](./docs/TRAINING_LEADERBOARD.md) for the complete experiment-by-experiment leaderboard across all `v7x-16` (`Pass 1..12G`) and `v5p` (`Pass 1..9G`) sweeps.

---

## 1. Model Overview — [`RecursiveSI/flint`](https://huggingface.co/RecursiveSI/flint)

| Property | Value |
|---|---|
| **Hugging Face Checkpoint** | [`https://huggingface.co/RecursiveSI/flint`](https://huggingface.co/RecursiveSI/flint) (`15` safetensors shards, `79.59 GB` BF16) |
| **Architecture** | Sparse Mixture-of-Experts (`49` Transformer layers, `hidden_size=2048..2560`, `vocab_size=151,936`) |
| **MoE Routing** | `256` routed experts per layer, **top-6** active routing (`~2.52B` active parameters / `39.8B` total) |
| **FLOPs Accounting** | **`17.385 GFLOP/token`** (`6 * active_params + attention FLOPs` at `seq_len=2,048`) |
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

## 3. Summary of Results (`v7x-16` 32 TensorCores, `v5p-32` 32 Chips & `v5p-16` 16 Chips)

| Platform & Topology | Execution Lane | Baseline Reference | Optimized Steady MFU | Optimized Peak MFU | Per-Rank Throughput | Slice Throughput | Real C4 Loss Convergence |
|---|---|---:|---:|---:|---:|---:|---|
| **TPU `v5p-16` (`2x2x4`, 16 Chips)** | **HF Backend + `SBDP` + `FusedOpt` + `sa=1024` + `HybridAC16` (`v5p_p9_02`, `mb=64`)** | `19.76%` *(v5p-16 base)* | **`25.76%`** *(+6.00 pp)* | **`25.81%`** | **`6,802.0` tok/s** | **`108,832` tok/s** | `12.4957 -> 3.9676` *(`22.84 GiB` HBM — **#1 `v5p-16` Record!**)* |
| **TPU `v5p-32` (`2x4x4`, 32 Chips)** | **HF Backend + `SBDP` + `FusedOpt` + `sa=1024` + `HybridAC16` (`v5p_p9_02`, `mb=64`)** | `19.76%` *(v5p-16)* / `19.95%` *(mb=32)* | **`24.88%`** *(+5.12 pp)* | **`24.91%`** *(24.95% `mb=48`)* | **`6,568.4` tok/s** | **`210,189` tok/s** | `12.2951 -> 4.3073` *(`22.84 GiB` HBM — **#1 `v5p-32` Record!**)* |
| **TPU `v5p-32` (`2x4x4`, 32 Chips)** | **Native `TpuGraphTrainer` + `SBDP` + `BF16-RoPE` + `FusedOpt` (`v5p_p9_06`, `mb=48`)** | `VMEM OOM` *(at tile_m=512)* | **`23.89%`** | **`24.00%`** | **`6,308.1` tok/s** | **`201,859` tok/s** | `12.3288 -> 4.1735` *(`28.11 GiB` HBM)* |
| **TPU `v5p-32` (`2x4x4`, 32 Chips)** | **HF Backend + `HSDP 2×16` (`v5p_p8_06`, `2x` GA, `6.29M` tok/step)** | `19.76%` *(v5p-16)* | **`23.81%`** *(+4.05 pp)* | **`23.83%`** | **`6,285.7` tok/s** | **`201,142` tok/s** | `12.4958 -> 3.7789` *(`21.08 GiB` HBM)* |
| **TPU `v7x-16` (`2x2x4`, 32 TensorCores)** | **Native `TpuGraphTrainer` (`v7x_p12g_02`, Scout XLA + `SBDP` + `BF16-RoPE` + `PureFusedOpt` + `tile1024` + `n49 = 0%` Attn Recompute + `BF16` Reduce + `8x` GA = `25.17M` tok/step, `mb=48`)** | `16.23%` *(v7x-8)* / `16.56%` *(v7x-16 base)* | **`20.80%`** *(+4.57 pp)* | **`20.81%`** | **`13,802.0` tok/s** *(13,811 pk)* | **`441,664` tok/s** *(442.0K pk)* | `12.3202 -> 3.7124` *(`23.02 GiB` HBM — **All-Time Throughput & Ironwood Record!**)* |
| **TPU `v7x-16` (`2x2x4`, 32 TensorCores)** | **Native `TpuGraphTrainer` (`v7x_p12g_01`, `SBDP` + `BF16-RoPE` + `PureFusedOpt` + `n49` + `16x` GA = `50.33M` tok/step, `mb=48`)** | `16.23%` *(v7x-8)* / `16.56%` *(v7x-16 base)* | **`20.78%`** *(+4.55 pp)* | **`20.79%`** | **`13,787.0` tok/s** *(13,793 pk)* | **`441,184` tok/s** *(441.4K pk)* | `12.3202 -> 3.7636` *(`23.02 GiB` HBM)* |
| **TPU `v7x-16` (`2x2x4`, 32 TensorCores)** | **Native `TpuGraphTrainer` (`v7x_p12f_01`, Scout XLA + `SBDP` + `BF16-RoPE` + `tile1024` + `n49` + `8x` GA = `25.17M` tok/step, `mb=48`)** | `16.23%` *(v7x-8)* / `16.56%` *(v7x-16 base)* | **`20.75%`** *(+4.52 pp)* | **`20.77%`** | **`13,766.5` tok/s** *(13,778 pk)* | **`440,528` tok/s** *(440.9K pk)* | `12.3202 -> 3.7124` *(`23.02 GiB` HBM)* |
| **TPU `v7x-16` (`2x2x4`, 32 TensorCores)** | **Native `TpuGraphTrainer` (`v7x_p12e_01`, Scout XLA + `SBDP` + `BF16-RoPE` + `tile1024` + `n49` + `4x` GA = `12.58M` tok/step, `mb=48`)** | `16.23%` *(v7x-8)* / `16.56%` *(v7x-16 base)* | **`20.74%`** *(+4.51 pp)* | **`20.77%`** | **`13,760.7` tok/s** *(13,780 pk)* | **`440,342` tok/s** *(441.0K pk)* | `12.3202 -> 3.2783` *(`23.02 GiB` HBM)* |
| **TPU `v7x-16` (`2x2x4`, 32 TensorCores)** | **Native `TpuGraphTrainer` (`v7x_p12c_01`, `SBDP` + `BF16-RoPE` + `tile1024` + `2x` GA = `6.29M` tok/step, `mb=48`)** | `16.23%` *(v7x-8)* / `16.56%` *(v7x-16 base)* | **`20.61%`** *(+4.38 pp)* | **`20.63%`** | **`13,675.7` tok/s** | **`437,622` tok/s** | `12.3201 -> 3.2716` *(`25.48 GiB` HBM)* |
| **TPU `v7x-16` (`2x2x4`, 32 TensorCores)** | **Native `TpuGraphTrainer` (`v7x_p9_02`, BF16 Reduce + Scout XLA + `2x` GA)** | `16.23%` *(v7x-8)* / `16.56%` *(v7x-16 base)* | **`18.68%`** *(+2.45 pp)* | **`18.76%`** | **`12,392.8` tok/s** *(12,447 peak)* | **`396,570` tok/s** *(398.3K peak)* | `12.3201 -> 4.1606` *(`27.12 GiB` HBM)* |
| **TPU `v7x-16` (`2x2x4`, 32 TensorCores)** | **Native Full Host Offload `save_last_n=0, prefetch=2` + `SBDP` + `BF16-RoPE` (`v7x_p12e_04`, `mb=64`, `2x` GA = `8.39M` tok/step)** | `OOM` *(without offload)* | **`18.60%`** | **`18.61%`** | **`12,342.7` tok/s** | **`394,966` tok/s** | `12.3206 -> 3.2794` *(`34.91 GiB` HBM — **#1 Full Offload Record!**)* |
| **TPU `v7x-16` (`2x2x4`, 32 TensorCores)** | **HF Backend + `SBDP` + `PureFusedOpt` + `sa=1024` + `tile1024` + `HybridAC12` (`v7x_p12_04`, `mb=64`, `2x` GA)** | `16.29%` *(v7x-8)* / `15.87%` *(v7x-16 base)* | **`18.58%`** *(+2.71 pp)* | **`18.59%`** | **`12,331.0` tok/s** | **`394,592` tok/s** | `12.3104 -> 3.7640` *(`25.50 GiB` HBM — **HF #1 Record!**)* |
| **TPU `v7x-16` (`2x2x4`, 32 TensorCores)** | **HF Backend + `SBDP` + `PureFusedOpt` + `sa=1024` + `tile1024` + `HybridAC12` (`v7x_p12b_04`, `mb=64`, `4x` GA = `16.78M` tok/step)** | `15.87%` *(v7x-16 base)* | **`18.55%`** *(+2.68 pp)* | **`18.56%`** | **`12,311.0` tok/s** | **`393,952` tok/s** | `12.3105 -> 4.1872` *(`25.50 GiB` HBM)* |
| **TPU `v7x-16` (`2x2x4`, 32 TensorCores)** | **HF Backend + `HSDP 2×16` (`v7x_p12c_05`, `SBDP` + `PureFusedOpt` + `tile1024`, `mb=48`, `2x` GA)** | `15.87%` *(v7x-16 base)* | **`18.25%`** *(+2.38 pp)* | **`18.27%`** | **`12,114.8` tok/s** | **`387,674` tok/s** | `12.4342 -> 3.5752` *(`21.22 GiB` HBM — **HSDP Record!**)* |
| **TPU `v7x-16` (`2x2x4`, 32 TensorCores)** | **Native Full Host Offload `save_last_n=0, prefetch=2` (`v7x_p11b_04`, `mb=96`, `2x` GA = `12.58M` tok/step)** | `OOM` *(without offload)* | **`18.14%`** | **`18.16%`** | **`12,037.5` tok/s** | **`385,200` tok/s** | `12.3202 -> 3.6790` *(`53.43 GiB` HBM)* |
| **TPU `v7x-16` (`2x2x4`, 32 TensorCores)** | **Intra-Chip `StaticEP=2 × FSDP=16 + ZeRO-3 + PairedGMM-512` (`v7x_p12_08`, `0` Inter-Chip A2A)** | `Recompile/OOM` *(dynamic EP)* | **`11.85%`** *(+76% vs EP=8)* | **`12.24%`** *(step-5)* | **`7,860.0` tok/s** *(8,120 pk)* | **`251,520` tok/s** *(259.8K pk)* | `12.3189 -> 3.6921` *(`19.73 GiB` HBM — **#1 StaticEP Record!**)* |
| **TPU `v7x-16` (`2x2x4`, 32 TensorCores)** | **Intra-Tray `StaticEP=4 × FSDP=8 + ZeRO-3 + PairedGMM-512` (`v7x_p12_09`, `2`-Chip A2A)** | `Recompile/OOM` *(dynamic EP)* | **`7.17%`** | **`7.37%`** *(step-5)* | **`4,759.5` tok/s** | **`152,304` tok/s** | `12.3177 -> 3.6906` *(`21.05 GiB` HBM — **#2 StaticEP**)* |
| **TPU `v5p-32` (`2x4x4`, 32 Chips)** | **Native `StaticEP=8 × FSDP=4 + ZeRO-2 + PairedGMM-256` (`v5p_p9f_02`)** | `Recompile/OOM` *(dynamic EP)* | **`7.30%`** | **`7.34%`** | **`1,926.1` tok/s** | **`61,635` tok/s** | `12.3288 -> 4.2121` *(`37.98 GiB` HBM)* |
| **TPU `v7x-16` (`2x2x4`, 32 TensorCores)** | **Intra-Host `StaticEP=8 × FSDP=4 + ZeRO-3 + PairedGMM-256` (`v7x_p11d_04`, `4`-Chip A2A)** | `Recompile/OOM` *(dynamic EP)* | **`6.73%`** | **`6.81%`** | **`4,463.0` tok/s** | **`142,816` tok/s** | `12.3215 -> 4.1913` *(`22.69 GiB` HBM)* |
| **TPU `v7x-16` (`2x2x4`, 32 TensorCores)** | **Inter-Host `StaticEP=16 × FSDP=2 + ZeRO-2 + PairedGMM-256` (`v7x_p11d_03`)** | `Blocked upstream` *(SymNode / EP>1 asserts)* | **`5.85%`** | **`6.11%`** | **`3,880.0` tok/s** | **`124,160` tok/s** | `12.3210 -> 3.6920` *(`24.48 GiB` HBM)* |

---

## 4. From Baseline to `25.81%` Peak (`v5p-16`), `24.95%` (`v5p-32`), `20.81%` (`v7x-16`) & `StaticEP=2..32` — Engineering Breakdown

### Track A: Native `TpuGraphTrainer` on Ironwood `v7x-16` (PR #122 `flint_40b_v011_n40` -> `v7x_p12g_02` `20.80%` MFU)

#### What the Baseline Did (`flint_40b_v011_n40`)
- Used `TpuGraphTrainer` with `GraphSplitter` (`1` forward + `1` backward XLA executable per layer), `SimpleFSDP`, kernel-native `(E, K, N)` weight layout, Paired W1/W3 Pallas GroupedMM at `tile_m=512`, `local_batch_size=32` (`65,536` tok/rank), saving SplashAttention outputs in only the last `40/49` layers (`n40`, recomputing 9 layers), FP32 gradient reduce-scatter, FP32 RoPE casts, and eager DTensor AdamW + gradient norm.
- Measured **`16.23%` MFU (`10,779` tok/s/rank)** on `v7x-8` (`16` TensorCores) and **`16.56%` Steady MFU (`10,985.3` tok/s/rank)** out-of-the-box on `v7x-16` (`32` TensorCores).

#### Exact Optimizations Applied Over the Baseline (`Pass 1 -> Pass 12G`)

| Step | Experiment ID | Exact Change Over Previous Step | Steady tok/s/rank | Cluster tok/s | Steady MFU | Peak MFU | Delta vs `v7x-8` (`16.23%`) |
|---|---|---|---:|---:|---:|---:|---:|
| **0 (Ref)** | *PR #122 `v7x-8`* | Baseline `flint_40b_v011_n40` (`16` ranks, `mb=32`, `tile_m=512`, `n40`) | `10,779.0` | `172,464` | `16.23%` | — | — |
| **1 (Repro)** | `02_native_pr122_v011_n40` | Direct reproduction on `v7x-16` (`32` ranks, `SimpleFSDP32`, `2.10M` tok/step) | `10,985.3` | `351,530` | `16.56%` | `16.69%` | `+0.33 pp` |
| **2** | `03_native_v012_hier_rs` | Enable **multi-axis ND Hierarchical SparseCore Reduce-Scatter** across `2x2x4` torus | `11,011.9` | `352,381` | `16.60%` | `16.70%` | `+0.37 pp` |
| **3** | `04_native_v013_n49` | Save SplashAttention across **all 49 layers (`n49`)** (`98.0%` XLA cache hit rate) | `11,135.8` | `356,346` | `16.78%` | `16.86%` | `+0.55 pp` |
| **4** | `05_native_v014_tile1024` | Widen forward Paired W1/W3 Pallas GroupedMM MXU tile `512 -> 1024` (`(1024, 2048, 2048)`) with VMEM-safe backward `(512, 2048, 0)` | `11,196.2` | `358,278` | `16.87%` | `16.94%` | `+0.64 pp` |
| **5** | `p3_01_comp_opt_gradnorm` | Compile 779-parameter DTensor AdamW (`optimizer_compile="separate_graph"`) + global L2 GradNorm | `11,448.6` | `366,355` | `17.25%` | `17.32%` | `+1.02 pp` |
| **6** | `p3_02_save_moe_routing` | Save `moe.routing` across all 49 layers to skip backward router logits + top-6 sort recompute | `11,540.8` | `369,306` | `17.39%` | `17.46%` | `+1.16 pp` |
| **7** | `p7_01_grad_accum_2x_mb48` | Scale microbatch **1.5×** (`mb=48`) + **`2×` gradient accumulation (`6.29M` tok/step)** + `n43` | `12,182.1` | `389,827` | `18.36%` | `18.42%` | `+1.13 pp` |
| **8 (`Pass 9`)** | `v7x_p9_02_bf16red_scout` | Switch gradient `ReduceScatter` to **`BF16`** (`159.2 GB -> 79.6 GB` ICI) + async AG/RS collective overlap flags (`27.12 GiB` HBM) | `12,392.8` | `396,570` | `18.68%` | `18.76%` | `+2.45 pp` |
| **9 (`Pass 12C`)** | `v7x_p12c_01_sbdp_bf16rope` | Add **Sequence-Block Document Causal SplashAttention (`SBDP`)** + **pure `BF16` RoPE** (`n41`, `mb=48`, `2×` GA = `6.29M` tok/step, `25.48 GiB` HBM) | `13,675.7` | `437,622` | `20.61%` | `20.63%` | `+4.38 pp` |
| **10 (`Pass 12E`)** | `v7x_p12e_01_sbdp_bf16rope_n49_ga4` | Save SplashAttention across **all 49 layers (`n49 = 0%` Attn Recompute)** at `mb=48`, `4×` GA (`12.58M` tok/step, `23.02 GiB` HBM) | `13,760.7` | `440,342` *(441.0K pk)* | `20.74%` | `20.77%` | `+4.51 pp` |
| **11 (`Pass 12F`)** | `v7x_p12f_01_sbdp_bf16rope_n49_ga8` | Scale to **`8×` Gradient Accumulation (`25.17M` tok/step)** with `n49 = 0%` Attn Recompute (`mb=48`, `23.02 GiB` HBM) | `13,766.5` | `440,528` *(440.9K pk)* | `20.75%` | `20.77%` | `+4.52 pp` |
| **12 (`#1 Record!`)** | **`v7x_p12g_02_sbdp_bf16rope_fusedopt_n49_ga8`** | Switch to **Pure `"fused"` Compiled AdamW (`PureFusedOpt`)** at `8×` GA (`25.17M` tok/step, `23.02 GiB` HBM — **`441,664` Steady / `441,952` Peak tok/s**!) | **`13,802.0`** | **`441,664`** *(442.0K pk)* | **`20.80%`** | **`20.81%`** | **`+4.57 pp`** |

---

### Track B: HuggingFace Modeling Backend on `v7x-16` (PR #123 `hf_moe_auto_max_attnac` -> `v7x_p12_04` `18.58%` MFU)

- **Baseline (`01_hf_backend_pr123_flint_40b`):** Left `experts.use_pallas_paired_grouped_mm = False`, used FP32 ReduceScatter, eager optimizer, and `mb=32`, measuring **`15.87%` Steady MFU (`10,532.0` tok/s/rank)** on `v7x-16` (`16.29%` on `v7x-8`).
- **Progressive Hillclimb (`Pass 5 -> Pass 12C`):**
  1. **Fused Paired W1/W3 Pallas GroupedMM (`tile_m=1024`) + `32 MB` Scoped VMEM Backward TGMM (`512:2048:0`) + Chunked CE (`b=1024`) (`p7_03`):** Lifted MFU to **`18.02%` (`382,621` tok/s)** at `mb=48`.
  2. **`BF16` ReduceScatter + Async ICI Overlap (`v7x_p9_03`):** Lifted `mb=48` (`1×` step) to **`18.29%` Steady / `18.34%` Peak MFU (`388,250` tok/s, `16.92 GiB` HBM)**.
  3. **`mb=64` (`4.19M` tok/step `1×`) + `SBDP` + `Splash sa=1024` + `HybridAC N=16` (`v7x_p11f_01`):** Reached **`18.39%` Steady / `18.41%` Peak MFU (`390,397` tok/s, `25.55 GiB` HBM)**, plus `mb=80` (`18.01%` MFU, `31.50 GiB` HBM).
  4. **`mb=64` + `2×` / `4×` GA (`8.39M` & `16.78M` tok/step) + `PureFusedOpt` + `tile_m=1024` + `HybridAC N=12` (`v7x_p12_04` & `v7x_p12b_04`):** Set the **#1 HF Backend `v7x-16` Record at `18.58%` Steady / `18.59%` Peak MFU (`12,331.0` tok/s/rank = `394,592` tok/s, `25.50 GiB` HBM, `loss: 12.3104 -> 3.7640`)**, sustained **`18.55%` MFU (`393,952` tok/s)** at `16.78M` tok/step (`4×` GA), and reached **`18.25%` Steady / `18.27%` Peak MFU (`387,674` tok/s, `21.22 GiB` HBM)** with **`HSDP 2×16` (`v7x_p12c_05`)**!

---

### Track C: Cross-Generation TorchTitan on TPU `v5p-32` (`24.88%`/`24.95%` MFU) & `v5p-16` (`25.76%`/`25.81%` MFU) + Long-Context (`4K`/`8K`/`16K`) Sweeps

- **Baseline (`01_hf_backend_pr123_flint_v5p`):** Measured **`19.76%` Steady MFU (`5,215.8` tok/s/chip)** on `v5p-16` (`2x2x4`, `459` BF16 TFLOP/s/chip), while Native PR #122 failed XLA compilation out-of-the-box due to `v5p`'s **`16.00 MB` Scoped VMEM** limit (`tile_m=512` paired GMM requires `16.58 MB`).
- **HF Backend `v5p` Record (`v5p_p9_02`, Pass 9):** Scaling to `mb=64` (`131,072` tok/chip) with `SBDP` + `FusedCompiledOpt` + `Splash sa=1024` + `HybridAC N=16` + `BF16` Reduce + `tile_m=256` vaulted **`v5p-16` (`16` chips) to `25.76%` Steady / `25.81%` Peak MFU (`6,802.0` tok/s/chip = `108,832` tok/s, `22.84 GiB` HBM)** (`+6.00 pp` over baseline!) and **`v5p-32` (`32` chips, `2x4x4`) to `24.88%` Steady / `24.95%` Peak MFU (`6,568.4` tok/s/chip = `210,189` tok/s)**, alongside Native `TpuGraphTrainer` (`v5p_p9_06`, **`23.89%` / `24.00%` MFU** at VMEM-safe `tile_m=256`).
- **Long-Context (`4K` / `8K` / `16K`) TorchTitan MFU Sweeps (`scripts/run_flint_v7x_pass13_longctx.py` & `scripts/run_flint_v5p_pass10_longctx.py`):** Evaluates constant-tokens-per-rank (`98,304` and `131,072` tok/rank) long-context scaling across `4,096`, `8,192`, and `16,384` sequence lengths for both Native `TpuGraphTrainer` (with recursive per-layer `attention.rope.max_context_length` override) and HF Modeling Backend across `v7x-16`, `v5p-32`, and `v5p-16`.

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
│   └── TRAINING_LEADERBOARD.md        # Complete experiment leaderboard & production configs
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

# 3. Run Pass 12G (#1 2K Native 20.80%/20.81% MFU = 441.95K tok/s) & Pass 13 (4K/8K/16K Long-Context Sweep) on TPU v7x-16
kubectl apply -f k8s/v7x-16-flint-pass12g.yaml
kubectl apply -f k8s/v7x-16-flint-pass13.yaml
kubectl logs -f job/flint-v7x-16-pass13
```
