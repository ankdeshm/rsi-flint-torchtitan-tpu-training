# `RecursiveSI/flint` (40B MoE) Pre-Training & Fine-Tuning on Cloud TPU (`v7x` & `v5p`) with TorchTitan + TorchTPU 0.2.0 GA

End-to-end distributed pre-training, HuggingFace safetensors fine-tuning, numerical checkpoint parity verification, and Model FLOPs Utilization (MFU) optimization for **[`RecursiveSI/flint`](https://huggingface.co/RecursiveSI/flint)** (`39.8B` total / `~2.52B` active parameters per token) on **TPU v7x (Ironwood)** and **TPU v5p** using **TorchTitan** and **TorchTPU `0.2.0` GA**.

See [`docs/TRAINING_LEADERBOARD.md`](./docs/TRAINING_LEADERBOARD.md) for the complete experiment-by-experiment leaderboard across all `v7x-16` (`Pass 1..12`) and `v5p` (`Pass 1..9F`) sweeps.

---

## 1. Model Overview — [`RecursiveSI/flint`](https://huggingface.co/RecursiveSI/flint)

| Property | Value |
|---|---|
| **Hugging Face Checkpoint** | [`https://huggingface.co/RecursiveSI/flint`](https://huggingface.co/RecursiveSI/flint) (`15` safetensors shards, `79.59 GB` BF16) |
| **Architecture** | Sparse Mixture-of-Experts (`49` Transformer layers, `hidden_size=2560`, `vocab_size=151,936`) |
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
  1. **Native `TpuGraphTrainer` Lane (`/torchtitan-native`):** Built on [`google-pytorch/torchtitan` PR #122 (`alekseyv_flint`)](https://github.com/google-pytorch/torchtitan/pull/122) — native `TpuGroupedExperts`, `GraphSplitter` per-layer XLA executables, `SimpleFSDP`, and `FlintStateDictAdapter`.
  2. **HuggingFace Modeling Backend Lane (`/torchtitan-hf`):** Built on [`google-pytorch/torchtitan` PR #123 (`hf-backend-tpu-20261028-v020-flint-tune`)](https://github.com/google-pytorch/torchtitan/pull/123) — `transformers_modeling_backend` with Pallas GroupedMM expert replacement and SplashAttention.

---

## 3. Summary of Results (`v7x-16` 32 TensorCores, `v5p-32` 32 Chips & `v5p-16` 16 Chips)

| Platform & Topology | Execution Lane | Baseline Reference | Optimized Steady MFU | Optimized Peak MFU | Per-Rank Throughput | Slice Throughput | Real C4 Loss Convergence |
|---|---|---:|---:|---:|---:|---:|---|
| **TPU `v5p-16` (`2x2x4`, 16 Chips)** | **HF Backend + `SBDP` + `FusedOpt` + `sa=1024` + `HybridAC16` (`v5p_p9_02`, `mb=64`)** | `19.76%` *(v5p-16 base)* | **`25.76%`** *(+6.00 pp)* | **`25.81%`** | **`6,802.0` tok/s** | **`108,832` tok/s** | `12.4957 -> 3.9676` *(`22.84 GiB` HBM — **All-Time MFU Record!**)* |
| **TPU `v5p-32` (`2x4x4`, 32 Chips)** | **HF Backend + `SBDP` + `FusedOpt` + `sa=1024` + `HybridAC16` (`v5p_p9_02`, `mb=64`)** | `19.76%` *(v5p-16)* / `19.95%` *(mb=32)* | **`24.88%`** *(+5.12 pp)* | **`24.91%`** *(24.95% `mb=48`)* | **`6,568.4` tok/s** | **`210,189` tok/s** | `12.2946 -> 6.2985` *(`22.84 GiB` HBM — **32-Chip Record!**)* |
| **TPU `v5p-32` (`2x4x4`, 32 Chips)** | **Native `TpuGraphTrainer` + `SBDP` + `BF16-RoPE` + `FusedOpt` (`v5p_p9_06`, `mb=48`)** | `VMEM OOM` *(at tile_m=512)* | **`23.89%`** | **`24.00%`** | **`6,308.1` tok/s** | **`201,859` tok/s** | `12.3288 -> 4.5064` *(`28.49 GiB` HBM)* |
| **TPU `v5p-32` (`2x4x4`, 32 Chips)** | **HF Backend + `HSDP 2×16` (`v5p_p8_06`, `2x` GA, `6.29M` tok/step)** | `19.76%` *(v5p-16)* | **`23.81%`** *(+4.05 pp)* | **`23.83%`** | **`6,285.7` tok/s** | **`201,142` tok/s** | `12.4958 -> 3.7789` *(`21.08 GiB` HBM)* |
| **TPU `v7x-16` (`2x2x4`, 32 TensorCores)** | **Native `TpuGraphTrainer` (`v7x_p9_02`, BF16 Reduce + Scout XLA + `2x` GA)** | `16.23%` *(v7x-8)* / `16.56%` *(v7x-16 base)* | **`18.68%`** *(+2.45 pp)* | **`18.76%`** | **`12,392.8` tok/s** *(12,447 peak)* | **`396,570` tok/s** *(398.3K peak)* | `12.3201 -> 4.1606` *(`27.12 GiB` HBM — **All-Time Throughput Record!**)* |
| **TPU `v7x-16` (`2x2x4`, 32 TensorCores)** | **Native `TpuGraphTrainer` (`v7x_p9_01`, `1x` step `n45` + BF16 Reduce)** | `16.23%` *(v7x-8)* / `16.56%` *(v7x-16 base)* | **`18.53%`** *(+2.30 pp)* | **`18.62%`** | **`12,297.1` tok/s** | **`393,507` tok/s** | `12.3215 -> 4.5422` *(`27.64 GiB` HBM)* |
| **TPU `v7x-16` (`2x2x4`, 32 TensorCores)** | **HF Backend + `SBDP` + `PureFusedOpt` + `sa=1024` + `tile1024` + `HybridAC12` (`v7x_p12_04`, `mb=64`, `2x` GA)** | `16.29%` *(v7x-8)* / `15.87%` *(v7x-16 base)* | **`18.58%`** *(+2.71 pp)* | **`18.59%`** | **`12,331.0` tok/s** | **`394,592` tok/s** | `12.3104 -> 3.7640` *(`25.50 GiB` HBM — **HF #1 Record!**)* |
| **TPU `v7x-16` (`2x2x4`, 32 TensorCores)** | **HF Backend + `SBDP` + `ChunkedOpt` + `sa=1024` + `HybridAC16` (`v7x_p11f_01`, `mb=64`, `1x` step)** | `16.29%` *(v7x-8)* / `15.87%` *(v7x-16 base)* | **`18.39%`** *(+2.52 pp)* | **`18.41%`** | **`12,199.9` tok/s** | **`390,397` tok/s** | `12.3104 -> 5.1286` *(`25.55 GiB` HBM)* |
| **TPU `v7x-16` (`2x2x4`, 32 TensorCores)** | **Native Full Host Offload `save_last_n=0, prefetch=2` (`v7x_p11b_04`, `mb=96`, `2x` GA = `12.58M` tok/step)** | `OOM` *(without offload)* | **`18.14%`** | **`18.16%`** | **`12,037.5` tok/s** | **`385,200` tok/s** | `12.3202 -> 3.6790` *(`53.43 GiB` HBM)* |
| **TPU `v7x-16` (`2x2x4`, 32 TensorCores)** | **Intra-Host `StaticEP=8 + ZeRO-3 + PairedGMM-256 + EPCompiledGradNorm` (`v7x_p11d_04`)** | `Recompile/OOM` *(dynamic EP)* | **`6.73%`** | **`6.81%`** | **`4,463.0` tok/s** | **`142,816` tok/s** | `12.3215 -> 4.1913` *(`22.69 GiB` HBM — **#1 StaticEP!**)* |
| **TPU `v7x-16` (`2x2x4`, 32 TensorCores)** | **Native `StaticEP=16` + `ZeRO-2` + `PairedGMM-256` (`v7x_p11c_02` / `p11d`)** | `Blocked upstream` *(SymNode / EP>1 asserts)* | **Verified `StaticEP=16`** | **`36,864` tok/exp** | *(`Pass 11D` live)* | *(`Pass 11D` live)* | **`12.3210 -> 3.6920`** *(`24.71 GiB` HBM — **First Static-Shape EP>1 Convergence!**)* |

---

## 4. From Baseline to `25.76%` (`v5p-16`), `24.88%` (`v5p-32`) & `StaticEP=16` (`v7x-16`) — Engineering Breakdown

### Track A: Native `TpuGraphTrainer` (PR #122 `flint_40b_v011_n40` -> `p7_01` & `p7_02`)

#### What the Baseline Did (`flint_40b_v011_n40`)
- Used `TpuGraphTrainer` with `GraphSplitter` (`1` forward + `1` backward XLA executable per layer), `SimpleFSDP`, kernel-native `(E, K, N)` weight layout, Paired W1/W3 Pallas GroupedMM at `tile_m=512`, `local_batch_size=32` (`65,536` tok/rank), saving SplashAttention outputs in only the last `40/49` layers (`n40`, recomputing 9 layers), eager/uncompiled DTensor AdamW + gradient norm, and single-axis SparseCore reduce-scatter.
- Measured **`16.23%` MFU (`10,779` tok/s/rank)** on `v7x-8` (`16` TensorCores) and **`16.56%` Steady MFU (`10,985.3` tok/s/rank)** out-of-the-box on `v7x-16` (`32` TensorCores).

#### Exact Optimizations Applied Over the Baseline

| Step | Experiment ID | Exact Change Over Previous Step | Steady tok/s/rank | Steady MFU | Peak MFU | Delta vs `v7x-8` (`16.23%`) |
|---|---|---|---:|---:|---:|---:|
| **0 (Ref)** | *PR #122 `v7x-8`* | Baseline `flint_40b_v011_n40` (`16` ranks, `mb=32`, `tile_m=512`, `n40`) | `10,779.0` | `16.23%` | — | — |
| **1 (Repro)** | `02_native_pr122_v011_n40` | Direct reproduction on `v7x-16` (`32` ranks, `SimpleFSDP32`, `2,097,152` tok/step) | `10,985.3` | `16.56%` | `16.69%` | `+0.33 pp` |
| **2** | `03_native_v012_hier_rs` | Enable **multi-axis ND Hierarchical SparseCore Reduce-Scatter** (`xla_tpu_enable_sparse_core_collective_offload_nd_reduce_scatter=True`, `xla_tpu_enable_sparse_core_hierarchical_reduce_scatter=True`, `use_single_sparse_core=False`) across `2x2x4` torus | `11,011.9` | `16.60%` | `16.70%` | `+0.37 pp` |
| **3** | `04_native_v013_n49` | Exploit `SimpleFSDP32` HBM savings to save SplashAttention across **all 49 layers (`n49`)** — eliminates 9 attention backward recomputations and achieves a **`98.0%` XLA compile cache hit rate** (`190.8s` wall time vs `354.0s`) | `11,135.8` | `16.78%` | `16.86%` | `+0.55 pp` |
| **4** | `05_native_v014_tile1024` | Widen forward Paired W1/W3 Pallas GroupedMM MXU tile `pallas_grouped_mm_tile_m: 512 -> 1024` (`gmm_tiling=(1024, 2048, 2048)`) while keeping VMEM-safe backward `tgmm_tiling=(512, 2048, 0)` | `11,196.2` | `16.87%` | `16.94%` | `+0.64 pp` |
| **5** | `p3_01_comp_opt_gradnorm` | Compile the 779-parameter DTensor AdamW step (`optimizer_compile="separate_graph"`) + global gradient L2 norm (`compile_grad_norm=True`) | `11,448.6` | `17.25%` | `17.32%` | `+1.02 pp` |
| **6** | `p3_02_save_moe_routing` | Add `save_regions=("moe.routing",)` across all 49 layers so backward skips recomputing the 256-expert top-6 router logits + top-k sort (`~0.11 GiB/layer`) | `11,540.8` | `17.39%` | `17.46%` | `+1.16 pp` |
| **7** | `p4_01_n43_mb48` | Scale microbatch **1.5×** to **`mb=48` (`98,304` tok/rank = `3,145,728` tok/step)** and calibrate `save_last_n_layers={"attention.inner_attention": 43}` + all-49 `moe.routing` to fit cleanly inside `96 GiB` XLA HBM | `12,040.9` | `18.15%` | `18.21%` | `+1.92 pp` |
| **8 (`1x` Best)** | **`p7_02_n45_all49_moe_mb48`** | Push saved SplashAttention layers from `n43 -> n45` alongside all-49 `moe.routing` at `mb=48` (`209.7s` total wall time) | **`12,136.2`** | **`18.29%`** | **`18.36%`** | **`+2.06 pp`** |
| **9 (`2x` Best)** | **`p7_01_grad_accum_2x_mb48`** | Add **`2×` gradient accumulation (`6,291,456` tok/step at `mb=48`)**, halving compiled AdamW + GradNorm invocation frequency per token | **`12,182.1`** | **`18.36%`** | **`18.42%`** | **`+2.13 pp`** |

---

### Track B: HuggingFace Modeling Backend (PR #123 `hf_moe_auto_max_attnac` -> `p7_03`)

#### What the Baseline Did (`hf_moe_auto_max_attnac`)
- Loaded `RecursiveSI/flint` via `transformers_modeling_backend`, replacing HF MoE experts with `TpuGroupedExperts` (`use_pallas_grouped_mm=True`), **but left `experts.use_pallas_paired_grouped_mm = False`**, launching separate `gate_proj` (`W1`) and `up_proj` (`W3`) Pallas GroupedMM kernels at every layer.
- Measured **`16.29%` MFU (`10,810` tok/s/rank)** on `v7x-8` and **`15.87%` Steady MFU (`10,532.0` tok/s/rank)** out-of-the-box on `v7x-16` (`01_hf_backend_pr123_flint_40b`).

#### Exact Optimizations Applied Over the Baseline (`p5_04` & `p7_03`)
1. **Fused Paired W1/W3 Pallas GroupedMM (`TT_HF_GMM`):** Enabled `use_pallas_paired_grouped_mm=true` with forward `pallas_grouped_mm_tile_m=1024` (`(1024, 2048, 2048)`), fusing `gate_proj` and `up_proj` into a single paired GroupedMM kernel per MoE layer.
2. **Hardware-Exact `32 MB` Scoped VMEM Backward TGMM Tiling:** Configured `pallas_grouped_mm_tgmm_tiling=512:2048:0`, `pallas_grouped_mm_w2_tgmm_tiling=512:512:0`, and `pallas_grouped_mm_w2_transposed_tgmm_tiling=512:0:2048` (whereas `tile_m=1024` in backward paired TGMM requires `35.68 MB > 32.00 MB` `v7x` Scoped VMEM).
3. **VMEM-Safe Chunked Cross-Entropy (`--loss.b_block_size=1024`) + `mb=48` Scaling (`p7_03`):** Kept the 151,936-vocab cross-entropy backward chunk inside `32 MB` Scoped VMEM (`b=2048` overflows Scoped VMEM) and scaled microbatch `32 -> 48` (`3,145,728` tok/step).
4. **Result:** Boosted PR #123 HF Backend from `15.87%` (`v7x-16` out-of-the-box) / `16.29%` (`v7x-8` ref) -> `17.68%` at `mb=32` (`p5_04`) and **`18.02%` Steady / `18.05%` Peak MFU (`11,956.9` steady / `11,973.0` peak tok/s/rank = `383,136` peak cluster tok/s)** at `mb=48` (`p7_03`), using only **`16.92 GiB` (`17.6%`)** persistent HBM while converging real C4 loss `12.3104 -> 3.3361`!

---

### Track C: Cross-Generation TPU `v5p` Scoped VMEM Calibration (`16.00 MB` Limit)
- On TPU `v5p-16` (`2x2x4`, `459` BF16 TFLOP/s/chip), PR #123 HF Backend (`01_hf_backend_pr123_flint_v5p`) achieves **`19.76%` Steady / `19.79%` Peak MFU** (`5,215.8` tok/s/chip).
- Native PR #122 Paired W1/W3 GMM fails XLA compilation out-of-the-box on `v5p` because `v5p` has **`16.00 MB` Scoped VMEM** (`half` of `v7x`'s `32.00 MB`), and `moe_gmm_w13_fwd` requires `20.00 MB` at `tile_m=1024` and `16.58 MB` at `tile_m=512`.
- Setting `pallas_grouped_mm_tile_m=256` and `tgmm_tiling=(256, 1024, 0)` ([`scripts/run_flint_v5p_pass2.py`](./scripts/run_flint_v5p_pass2.py)) compiles cleanly inside `16.00 MB` Scoped VMEM and achieves **`17.01%` Steady / `17.09%` Peak MFU** on `v5p-16`.

---

## 5. Repository Structure

```text
├── README.md
├── docs/
│   └── TRAINING_LEADERBOARD.md        # Complete experiment leaderboard & production configs
├── docker/
│   ├── Dockerfile                     # Builds PyTorch 2.15 + TorchTPU 0.2.0 GA + PR #122 & PR #123
│   └── patch_flint_configs.py         # Appends hillclimb & v5p/v6e configs to Native config_registry.py
├── scripts/                           # All Pass 1..7 & v5p training sweep & staging scripts
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

# 3. Run Pass 7 (#1 Native 18.36%/18.42% MFU & #1 HF Backend 18.02%/18.05% MFU) on TPU v7x-16 (2x2x4)
kubectl apply -f k8s/v7x-16-flint-pass7.yaml
kubectl logs -f job/flint-v7x-16-pass7
```
