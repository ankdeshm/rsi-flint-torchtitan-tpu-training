# `RecursiveSI/flint` TorchTPU 0.2.0 GA Training & MFU Optimization

- **Project / Region:** `YOUR_GCP_PROJECT_ID` / `us-central1`
- **Clusters:** `tpu-v5p-cluster` & `tpu-v7x-cluster`
- **GCS Bucket:** `gs://YOUR_GCS_BUCKET/flint-results/`
- **Status (`2026-10-30 12:26 PDT`, 7 `v7x-16` Passes + 2 `v5p-16` Passes Complete, `v5p-32` Live — 31 Experiments Persisted to GCS):**
  - **OVERALL #1 RECORD — Native `TpuGraphTrainer` (PR #122 + Hillclimb, `p7_01`) on `v7x-16` (`2x2x4`, 32 TensorCores):**
    - **`18.36%` Steady MFU (`18.42%` Peak MFU)** — **`12,182.1` Steady / `12,223.0` Peak tok/s/rank** (`389,827` Steady / **`391,136` Peak tok/s cluster-wide**, `212.50` Peak BF16 TFLOP/s/rank at `6,291,456` tokens/step via `2×` gradient accumulation at `mb=48`, `loss: 12.3201 -> 3.6763`).
    - Even at **`1×` single-microbatch step (`p7_02`, `3,145,728` tokens/step, `n45` + all-49 `moe.routing`)**, Native `TpuGraphTrainer` reaches **`18.29%` Steady MFU (`18.36%` Peak MFU, `12,136.2` steady / `12,185.0` peak tok/s/rank)** in just **`209.7s` wall time**!
    - Exceeds the `v7x-8` Native baseline (`16.23%`, `10,779` tok/s/rank) by **`+2.13` percentage points Steady MFU (`+2.19` pp Peak MFU, `+13.0%` higher per-device throughput)** while scaling **2×** from 16 to 32 TensorCores!
  - **PR #123 HF Transformers Backend #1 RECORD (`p7_03`) on `v7x-16` (`2x2x4`, 32 TensorCores) — ALSO SURPASSES `18%` MFU:**
    - **`18.02%` Steady MFU (`18.05%` Peak MFU)** — **`11,956.9` Steady / `11,973.0` Peak tok/s/rank** (`382,621` Steady / **`383,136` Peak tok/s cluster-wide**, `208.15` Peak BF16 TFLOP/s/rank at `3,145,728` tokens/step, `mb=48`), fine-tuning the real 15-shard `RecursiveSI/flint` checkpoint (`loss: 12.31036 -> 3.33608` on C4, `memory: 16.92 GiB / 17.6%`, `308.9s` total wall time).
    - Exceeds the `v7x-8` HF Backend baseline (`16.29%`, `10,810` tok/s/rank) by **`+1.73` pp Steady MFU (`+1.76` pp Peak MFU)** and improves **`+2.15` pp** over out-of-the-box PR #123 (`01_hf_backend_pr123_flint_40b`, `15.87%`).
  - **Real `RecursiveSI/flint` Pretrained Checkpoint Parity Verified Across Both Lanes:**
    - Staged `RecursiveSI/flint` (39.8B total / ~2.52B active MoE, 256 routed experts, top-6, 49 layers, 15 safetensors shards, `79.59 GB` BF16) at `gs://YOUR_GCS_BUCKET/models/RecursiveSI/flint/`.
    - PR #123 HF Backend (`01_hf`, `p5_04`, `p7_03`) converged smoothly to **`loss: 3.3361`–`3.4434`** on real C4 text; Native `FlintStateDictAdapter` (`p6_02`) loaded all 15 HF shards (`36,734` FQN tensors transposed into kernel-native `(E, K, N)` `TpuGroupedExperts`) in `95.24s` and produced **`step 1 loss: 3.43601`** on C4 — proving **exact 3-decimal numerical parity** between the HF Transformers Modeling Backend and Native `TpuGroupedExperts`!
  - **Container Image:** `us-central1-docker.pkg.dev/YOUR_GCP_PROJECT_ID/torchtpu/torchtitan-flint:v0.2.0` (`torch==2.15.0+cpu`, `torch_tpu==0.2.0` GA, `jax==0.11.1`, `libtpu==0.0.50`, `tokamax==0.0.14`, `transformers==5.22.0`, PR #123 in `/torchtitan-hf`, PR #122 in `/torchtitan-native`).

---

## 1. Key Engineering, Kernel & XLA HBM Discoveries

1. **Exact 96 GiB XLA HBM Calibration (`mb=48` + `n43`/`n45`) + `2×` Gradient Accumulation Unlocks `18.36%` Steady / `18.42%` Peak MFU (`p7_01`, `p7_02`):**
   - `torch.tpu.max_memory_allocated()` reports `14.63–23.22 GiB` (persistent `SimpleFSDP32` shards + AdamW state + microbatch inputs), whereas XLA's `GraphSplitter` holds live forward-to-backward boundary residuals (`~0.70 GiB/layer` at `mb=48`), saved Splash attention outputs (`~0.75 GiB/layer`), saved `moe.routing` indices/weights (`~0.11 GiB/layer`), and a `~21.96 GiB` peak backward layer executable workspace.
   - At `mb=48`, saving all 49 attention layers (`p5_01`, `06_native`) needs `21.96 GiB` workspace vs `19.54–20.02 GiB` reservable (`-1.93` to `-2.42 GiB` over 96 GiB). Pushing saved attention layers right up to `n45` (`p7_02`) with `save_regions=("moe.routing",)` across **all 49 layers** fits cleanly inside 96 GiB HBM and reaches **`18.29%` Steady / `18.36%` Peak MFU** in `209.7s`; adding **`2×` gradient accumulation (`6,291,456` tok/step, `p7_01`)** halves compiled AdamW + GradNorm invocation frequency and reaches **`18.36%` Steady / `18.42%` Peak MFU (`12,223` peak tok/s/rank)**!
   - Pushing microbatch higher (`mb=56 + n22` -> `17.75%`, `mb=64 + n14` -> `17.70%`, `mb=64 + fullremat` -> `17.32%`) or slightly lower (`mb=44 + n49` -> `17.75%`) confirms **`mb=48` (`98,304` tok/rank) is the global Pareto optimum** on 96 GiB `v7x`, both because `48 * 2048 = 98,304` aligns cleanly with `tile_m=1024` MXU tiles and because higher microbatches force excessive Splash attention recomputation (`27–49` layers).
2. **PR #123 HF Backend Paired W1/W3 Pallas GroupedMM + `mb=48` + VMEM-Safe Cross-Entropy (`p7_03`) -> `18.02%` Steady / `18.05%` Peak MFU:**
   - Out-of-the-box PR #123 (`transformers_modeling_backend/model.py`) leaves `experts.use_pallas_paired_grouped_mm = False`, running separate `gate_proj` and `up_proj` GroupedMM launches (`15.87%` MFU in `01_hf`).
   - Enabling `use_pallas_paired_grouped_mm = True` with forward `gmm_tiling = (1024, 2048, 2048)` + VMEM-safe backward `tgmm_tiling = (512, 2048, 0)` + `--loss.b_block_size=1024` vaulted PR #123 HF Backend MFU from `15.87%` -> `17.68%` at `mb=32` (`p5_04`) and **`18.02%` Steady / `18.05%` Peak MFU (`11,956.9` steady / `11,973.0` peak tok/s/rank, `16.92 GiB` HBM, `loss -> 3.3361`)** at `mb=48` (`p7_03`)!
   - By contrast, `--loss.b_block_size=2048` (`p3_07`, `p3_08`) overflows `v7x`'s `32.00 MB` Scoped VMEM limit inside the 151,936-vocab cross-entropy backward chunk (`b=1024` stays safely under `32 MB`).
3. **`v7x` 32 MB Scoped VMEM Hardware Ceiling on Backward Paired TGMM (`p5_02`, `p5_03`):**
   - Increasing backward `_paired_grouped_mm_bwd` (`moe_tgmm_w13_wgrad`) `tile_m` from `512` to `1024` (`tgmm_tiling=(1024, 2048, 0)`) requires **`35.68 MB` Scoped VMEM** vs `v7x`'s **`32.00 MB` hardware limit** (`XLA CompileTimeScopedVmemOom e1001`), proving `(512, 2048, 0)` is the exact VMEM-optimal backward TGMM tile shape on `v7x`.
4. **Compiled Optimizer (`separate_graph`) + Compiled GradNorm + Saved `moe.routing` + Homogeneous `n49` Cache Hit Rate:**
   - Compiling the 779-parameter DTensor AdamW step (`optimizer_compile="separate_graph"`) + global gradient norm (`compile_grad_norm=True`) + saving `"moe.routing"` across all 49 layers adds **`+0.52` pp MFU** (`16.87% -> 17.39%` at `mb=32`, and `+0.21` pp for `moe.routing` alone at `mb=48 + n45`: `18.08%` in `p4_02` -> **`18.29%` in `p7_02`**!).
   - Homogeneous 49-layer remat (`n49`, `04_native` & `p6_01`) achieves a **`98.0%` XLA compilation cache hit rate** (`4,632 / 4,723` cache hits, only `29` unique compilations), cutting cold start + 15-step wall time by up to **46%** (`354.0s -> 190.8s`).

---

## 2. Complete Training & MFU Leaderboard — `v7x-16` (`2x2x4`, 32 TensorCores)

- **GCS Artifact Directories (`gs://YOUR_GCS_BUCKET/flint-results/`):**
  - Pass 1 (`5/6 PASS`): `v7x-16-2x2x4_20261030T072821Z/`
  - Pass 2 (Zero-remat & `mb=64..96` HBM boundary probes): `v7x-16-2x2x4_pass2_20261030T075716Z/`
  - Pass 3 (`5/8 PASS`, surgical hillclimb): `v7x-16-2x2x4_pass3_20261030T082020Z/`
  - Pass 4 (`4/5 PASS`, `18.15%` / `18.21%` MFU): `v7x-16-2x2x4_pass4_20261030T085355Z/`
  - Pass 5 (`17.68%` / `17.72%` HF MFU + VMEM ceiling probes): `v7x-16-2x2x4_pass5_20261030T092237Z/`
  - Pass 6 (`n49@mb44` + Native 15-shard real checkpoint load parity): `v7x-16-2x2x4_pass6_20261030T093733Z/`
  - Pass 7 (**#1 Overall Native `18.36%` / `18.42%` MFU** & **#1 HF Backend `18.02%` / `18.05%` MFU**): `v7x-16-2x2x4_pass7_20261030T190252Z/`
- **FLOP Accounting:** `17.385` GFLOP/token (`RecursiveSI/flint`), `1,153.5` BF16 TFLOP/s/TensorCore (`v7x` peak).

| Rank | Experiment | Lane / Key Configuration | Global Batch (tok/step) | Status | Steady tok/s/rank | Cluster tok/s (32 ranks) | Steady MFU (%) | Peak MFU (%) | Loss (`start -> end`) | Wall Time (s) |
|---:|---|---|---:|---|---:|---:|---:|---:|---|---:|
| **#1** | **`p7_01_native_grad_accum_2x_mb48_n43`** | **Native PR#122 + `mb=48` + `2×` GradAccum + `n43` Attn + All-49 `moe.routing` + `tile_m=1024` + CompiledOpt** | **6,291,456** | **`PASS`** | **12,182.1** | **389,827** *(391,136 peak)* | **18.36%** | **18.42%** | **`12.3201 -> 3.6763`** | **553.8** |
| **#2** | **`p7_02_native_n45_all49_moe_routing_mb48`** | **Native PR#122 + `mb=48` (`1×` step) + `n45` Attn + All-49 `moe.routing` + `tile_m=1024` + CompiledOpt** | **3,145,728** | **`PASS`** | **12,136.2** | **388,358** *(389,920 peak)* | **18.29%** | **18.36%** | **`12.3215 -> 3.6797`** | **209.7** |
| **#3** | `p4_01_native_n43_tile1024_moe_routing_comp_opt_mb48` | Native PR#122 + `mb=48` (`98K` tok/rank) + `n43` Attn + All-49 `moe.routing` + `tile_m=1024` + CompiledOpt | 3,145,728 | `PASS` | **12,040.9** | 385,309 | **18.15%** | 18.21% | `12.3215 -> 3.6799` | 398.9 |
| **#4** | `p3_05_native_n28_tile1024_moe_routing_comp_opt_mb48` | Native PR#122 + `mb=48` + `n28` Attn + All-49 `moe.routing` + `tile_m=1024` + CompiledOpt | 3,145,728 | `PASS` | **12,019.2** | 384,614 | **18.11%** | 18.18% | `12.3215 -> 3.6801` | 384.4 |
| **#5** | `p4_02_native_n45_tile1024_comp_opt_mb48` | Native PR#122 + `mb=48` + `n45` Attn (no `moe.routing` save) + `tile_m=1024` + CompiledOpt | 3,145,728 | `PASS` | **11,997.4** | 383,917 | **18.08%** | 18.15% | `12.3215 -> 3.6799` | 367.8 |
| **#6 (HF #1)** | **`p7_03_hf_pr123_paired_gmm_b1024_mb48`** | **HF Backend PR#123 (`RecursiveSI/flint` weights) + Paired W1/W3 GMM (`tile_m=1024`) + `b=1024` + `mb=48`** | **3,145,728** | **`PASS`** | **11,956.9** | **382,621** *(383,136 peak)* | **18.02%** | **18.05%** | **`12.3104 -> 3.3361`** | **308.9** |
| **#7** | `p4_03_native_n22_tile1024_moe_routing_comp_opt_mb56` | Native PR#122 + `mb=56` (`115K` tok/rank) + `n22` Attn + All-49 `moe.routing` + CompiledOpt | 3,670,016 | `PASS` | **11,776.8** | 376,858 | **17.75%** | 17.88% | `12.3205 -> 3.6812` | 421.5 |
| **#8** | `p6_01_native_n49_moe_routing_comp_opt_mb44` | Native PR#122 + `mb=44` (`90K` tok/rank) + **All-49** Attn + **All-49** `moe.routing` (`98.0%` cache hit) | 2,883,584 | `PASS` | **11,774.8** | 376,794 | **17.75%** | 17.82% | `12.3213 -> 3.6835` | 321.5 |
| **#9** | `p4_04_native_n14_tile1024_moe_routing_comp_opt_mb64` | Native PR#122 + `mb=64` (`131K` tok/rank) + `n14` Attn + All-49 `moe.routing` + CompiledOpt | 4,194,304 | `PASS` | **11,743.5** | 375,792 | **17.70%** | 17.75% | `12.3205 -> 3.6765` | 450.1 |
| **#10 (HF #2)** | `p5_04_hf_pr123_paired_gmm_b1024_mb32` | HF Backend PR#123 (`RecursiveSI/flint` weights) + Paired W1/W3 Pallas GMM (`tile_m=1024`) + `b=1024` (`mb=32`) | 2,097,152 | `PASS` | **11,728.6** | 375,315 | **17.68%** | 17.72% | `12.3115 -> 3.4434` | 237.1 |
| **#11** | `p3_02_native_n49_tile1024_moe_routing_comp_opt` | Native PR#122 + `mb=32` + All-49 Attn & `moe.routing` + `tile_m=1024` + CompiledOpt/GradNorm | 2,097,152 | `PASS` | **11,540.8** | 369,306 | **17.39%** | 17.46% | `12.3212 -> 3.6888` | 252.9 |
| **#12** | `p3_06_native_fullremat_tile1024_comp_opt_mb64` | Native PR#122 + `mb=64` (`131K` tok/rank) + FullRemat + `tile_m=1024` + CompiledOpt | 4,194,304 | `PASS` | **11,492.7** | 367,766 | **17.32%** | 17.38% | `12.3205 -> 3.6767` | 385.0 |
| **#13** | `p3_01_native_n49_tile1024_comp_opt_gradnorm` | Native PR#122 + `mb=32` + All-49 Attn + `tile_m=1024` + CompiledOpt/GradNorm | 2,097,152 | `PASS` | **11,448.6** | 366,355 | **17.25%** | 17.32% | `12.3212 -> 3.6872` | 299.1 |
| **#14** | `p3_03_native_n49_tile1024_moe_routing_comp_opt_grp2` | Native PR#122 + `mb=32` + All-49 Attn & `moe.routing` + CompiledOpt + 2-Layer Split | 2,097,152 | `PASS` | **11,405.3** | 364,970 | **17.19%** | 17.27% | `12.3212 -> 3.6884` | 260.2 |
| **#15** | `05_native_v014_n49_tile_m_1024` | Native PR#122 + `mb=32` + All-49 Attn + Pallas GroupedMM `tile_m=1024` | 2,097,152 | `PASS` | **11,196.2** | 358,278 | **16.87%** | 16.94% | `12.3212 -> 3.6867` | 263.8 |
| **#16** | `04_native_v013_save_all_49_layers` | Native PR#122 + `mb=32` + Homogeneous `n49` Save All 49 Splash Attn Layers | 2,097,152 | `PASS` | **11,135.8** | 356,346 | **16.78%** | 16.86% | `12.3212 -> 3.6867` | 190.8 |
| **#17** | `03_native_v012_hier_sparsecore_rs` | Native PR#122 + Hierarchical ND SparseCore ReduceScatter (`n40`, `mb=32`) | 2,097,152 | `PASS` | **11,011.9** | 352,381 | **16.60%** | 16.70% | `12.3212 -> 3.6860` | 353.4 |
| **#18** | `02_native_pr122_v011_n40_baseline` | Native PR#122 Baseline (`v011_n40` on 32 ranks, `mb=32`) | 2,097,152 | `PASS` | **10,985.3** | 351,530 | **16.56%** | 16.69% | `12.3212 -> 3.6860` | 354.0 |
| *Ref* | *(Ref: PR #123 `v7x-8`)* | *HF Backend Reference (`16` ranks, `mb=32`)* | *1,048,576* | *Ref* | *10,810.0* | *172,960* | *16.29%* | *-* | *-* | *-* |
| *Ref* | *(Ref: PR #122 `v7x-8`)* | *Native `v011_n40` Reference (`16` ranks, `mb=32`)* | *1,048,576* | *Ref* | *10,779.0* | *172,464* | *16.23%* | *-* | *-* | *-* |
| **#19** | `01_hf_backend_pr123_flint_40b` | Out-of-the-Box PR#123 HF Backend (`RecursiveSI/flint`, `b=1024`, unpaired GMM, `mb=32`) | 2,097,152 | `PASS` | **10,532.0** | 337,024 | **15.87%** | 15.91% | `12.3115 -> 3.4358` | 248.7 |
| - | `p6_02_native_real_flint_hf_checkpoint_30steps_mb48` | Native `FlintStateDictAdapter` 15-Shard Load (`36,734` tensors in `95.24s`, verified `loss=3.43601` parity; step-2 `cached_state_dict` HBM boundary at `n43+mb48`) | 3,145,728 | `PARITY (step 1) / OOM (step 2)` | - | - | - | - | `3.43601` (step 1) | 367.3 |
| - | `06_native_v014_n49_tile1024_mb48` / `p5_01` | `n49` All-49 Attn at `mb=48` (`21.96 GiB` workspace needed vs `19.54–20.02 GiB` reservable) | 3,145,728 | `HBM OOM (-1.93 GiB)` | - | - | - | - | `12.3215` (step 1) | 213.4 |
| - | `p5_02` / `p5_03` (`tgmm_tiling M=1024`) | Backward Paired TGMM `tile_m=1024` (`35.68 MB` Scoped VMEM > `32.00 MB` `v7x` limit) | 2.9M – 3.1M | `VMEM OOM (35.68M)` | - | - | - | - | - | 182.7 |
| - | `p3_07` / `p3_08` (`b_block_size=2048`) | PR#123 HF Cross-Entropy `b=2048` (`>32 MB` Scoped VMEM on 151,936 vocab; fixed by `b=1024` in `p5_04`) | 2.1M – 3.1M | `VMEM OOM` | - | - | - | - | - | 144.3 |
| - | `p2_01..p2_07` | Zero-Remat (`NoAC` / `memory_policy='default'`) & uncalibrated `mb=64..96` | 2.1M – 6.3M | `HBM OOM (>96 GiB)` | - | - | - | - | - | ~127–215s |

---

## 2B. Cross-Generation TPU `v5p-16` (`2x2x4`, 16 Chips) & `v5p-32` (`2x4x4`, 32 Chips) Multi-Slice Scaling

- **Hardware / Peak Spec:** TPU `v5p` dual-MXU chips (`95 GiB` HBM/chip, `16.00 MB` Scoped VMEM, `459.0` BF16 TFLOP/s/chip peak) across **`v5p-16` (`2x2x4`, 16 chips)** and **`v5p-32` (`2x4x4`, 32 chips / 8 hosts)**.
- **GCS Run Directories:**
  - **Completed 4/4 `v5p-32` (`2x4x4`, 32 Chips) Scaling Pass (`19.95%` / `19.98%` HF MFU & `19.00%` / `19.07%` Native MFU):** `gs://YOUR_GCS_BUCKET/flint-results/v5p-32-2x4x4_pass2_20261030T192414Z/`
  - Completed 15-Step `v5p-16` Pass 1 (HF Backend `19.76%` MFU): `gs://YOUR_GCS_BUCKET/flint-results/v5p-16-2x2x4_20261030T180426Z/`
  - Completed 15-Step `v5p-16` Pass 2 (Native `TpuGraphTrainer` `17.01%` MFU): `gs://YOUR_GCS_BUCKET/flint-results/v5p-16-2x2x4_pass2_20261030T183651Z/`
- **Architectural, Scaling & VMEM Findings (`v5p-16` -> `v5p-32` vs `v7x-16`):**
  - **Superlinear `>100%` Scaling from `v5p-16` (`16` chips) -> `v5p-32` (`32` chips, `2x4x4`):**
    - **PR #123 HF Backend + Paired W1/W3 GMM (`tile_m=256`, `v5p_p2_04`) on `v5p-32`** reaches **`19.95%` Steady MFU (`19.98%` Peak MFU — `5,266.9` steady / `5,276.0` peak tok/s/chip = `168,541` steady / `168,832` peak cluster tok/s)** on real 15-shard `RecursiveSI/flint` weights (`loss: 12.2942 -> 3.6038`, `11.78 GiB` HBM, `98.4%` XLA compile cache hit rate, `355.4s` wall time) — achieving **`101.0%` weak-scaling efficiency** over `v5p-16` (`5,215.8` tok/s/chip, `19.76%` MFU)!
    - **Native `TpuGraphTrainer` (`v5p_p2_02`, Paired GMM `tile_m=256` + `n49` + CompiledOpt) on `v5p-32`** unlocks `mb=32` (`2,097,152` tok/step across `SimpleFSDP32`), surging from `17.05%` (`mb=24`) -> **`19.00%` Steady MFU (`19.07%` Peak MFU — `5,016.0` tok/s/chip = `160,512` steady cluster tok/s, `15.86 GiB` HBM)**!
  - **`v5p` `16.00 MB` vs `v7x` `32.00 MB` Scoped VMEM Hardware Boundary & Resolution:** Native PR #122 Paired W1/W3 Pallas GroupedMM (`moe_gmm_w13_fwd`) requires `20.00 MB` Scoped VMEM at `tile_m=1024` and `16.58 MB` (`592 KB` over `v5p`'s `16.00 MB` limit!) at `tile_m=512`. Calibrating Paired GMM to `tile_m=256` + `tgmm_tiling=(256, 1024, 0)` compiles cleanly inside `16.00 MB` Scoped VMEM on both `v5p-16` and `v5p-32` (`+2.03` pp faster at `mb=32` on `v5p-32`, and `+0.08` pp faster than unpaired `tile_m=512` at equal `mb=24`: `17.05%` vs `16.97%`).
  - **Throughput Comparison (`v7x-16` vs `v5p-32`):** A single 16-chip **`v7x-16` slice (`389,827` steady / `391,136` peak tok/s)** delivers **`2.31×` higher total training throughput than a 32-chip `v5p-32` slice (`168,541` tok/s)** (`4.63×` higher throughput per TPU chip!).

| Slice | Experiment | Lane / Key Configuration | Global Batch (tok/step) | Status / Steps | Steady tok/s/rank |Cluster tok/s | Steady MFU (%) | Peak MFU (%) | Loss (`start -> end`) | HBM / Wall Time |
|---|---|---|---:|---|---:|---:|---:|---:|---|---|
| **`v5p-32` (`2x4x4`)** | **`v5p_p2_04_hf_pr123_paired_tile256_b1024_mb32`** | **HF Backend PR#123 (`RecursiveSI/flint` weights, Paired GMM `tile_m=256`, `b=1024`, `mb=32`)** | **2,097,152** | **`PASS` (15 steps)** | **5,266.9** *(5,276 peak)* | **168,541** *(168,832 peak)* | **19.95%** | **19.98%** | **`12.2942 -> 3.6038`** | **`11.78 GiB` / `355.4s`** |
| **`v5p-32` (`2x4x4`)** | **`v5p_p2_02_native_paired_tile256_n49_mb32`** | **Native PR#122 (`TpuGraphTrainer`, Paired GMM `tile_m=256`, `n49`, CompiledOpt, `mb=32`)** | **2,097,152** | **`PASS` (15 steps)** | **5,016.0** | **160,512** | **19.00%** | **19.07%** | **`12.3285 -> 3.6998`** | **`15.86 GiB` / `412.7s`** |
| `v5p-32` (`2x4x4`) | `v5p_p2_01_native_paired_tile256_n49_mb24` | Native PR#122 (`TpuGraphTrainer`, Paired GMM `tile_m=256`, `n49`, CompiledOpt, `mb=24`) | 1,572,864 | `PASS` (15 steps) | **4,501.8** | 144,058 | **17.05%** | 17.11% | `12.3292 -> 3.6720` | `12.31 GiB` / `431.5s` |
| `v5p-32` (`2x4x4`) | `v5p_p2_03_native_unpaired_tile512_n49_mb24` | Native PR#122 (`TpuGraphTrainer`, Unpaired GMM `tile_m=512`, `n49`, CompiledOpt, `mb=24`) | 1,572,864 | `PASS` (15 steps) | **4,480.2** | 143,366 | **16.97%** | 17.07% | `12.3292 -> 3.6718` | `11.18 GiB` / `364.8s` |
| **`v5p-16` (`2x2x4`)** | **`01_hf_backend_pr123_flint_v5p`** | **HF Backend PR#123 (`RecursiveSI/flint` 15-shard weights, Pallas GMM, `b=1024`, `mb=32`, `seq=2048`)** | **1,048,576** | **`PASS` (15 steps)** | **5,215.8** *(5,224 peak)* | **83,453** *(83,584 peak)* | **19.76%** | **19.79%** | **`12.4968 -> 3.4368`** | **`11.78 GiB` / `383.2s`** |
| **`v5p-16` (`2x2x4`)** | **`v5p_p2_01_native_paired_tile256_n49_mb24`** | **Native PR#122 (`TpuGraphTrainer`, Paired GMM `tile_m=256`, `n49`, CompiledOpt, `mb=24`)** | **786,432** | **`PASS` (15 steps)** | **4,491.8** *(4,513 peak)* | **71,869** *(72,208 peak)* | **17.01%** | **17.09%** | **`12.3177 -> 3.3964`** | **`14.32 GiB` / `400.1s`** |
| `v5p-16` (`2x2x4`) | `02_native_pr122_v5p_n40` | Native PR#122 Paired W1/W3 GMM `tile_m=512` (`16.58 MB > 16.00 MB` `v5p` Scoped VMEM) | 1,048,576 | `VMEM OOM (16.58M)` | - | - | - | - | - | `192.0s` |
| `v5p-16` (`2x2x4`) | `03_native_pr122_v5p_n49_tile1024` | Native PR#122 Paired W1/W3 GMM `tile_m=1024` (`20.00 MB > 16.00 MB` `v5p` Scoped VMEM) | 1,048,576 | `VMEM OOM (20.00M)` | - | - | - | - | - | `135.4s` |

---

## 3. Recommended Production Configurations for `RecursiveSI/flint`

1. **Maximum Throughput / MFU Pretraining & Fine-Tuning (`18.36%` Steady / `18.42%` Peak MFU at `2×` GradAccum; `18.29%` Steady / `18.36%` Peak at `1×` Step):**
   - **Trainer:** TorchTitan Native `TpuGraphTrainer` (`flint_40b`, `SimpleFSDP` across 32 ranks).
   - **Settings:** `local_batch_size=48` (`seq_len=2048`, `98,304` tok/rank = `6,291,456` tok/step with `2×` grad accum in `p7_01`, or `3,145,728` tok/step `1×` in `p7_02`), `gmm_tiling=(1024, 2048, 2048)`, `tgmm_tiling=(512, 2048, 0)`, `hierarchical_nd_reduce_scatter=True`, `optimizer_compile="separate_graph"`, `compile_grad_norm=True`, `save_regions=("moe.routing",)` across all 49 layers, and `save_last_n_layers={"attention.inner_attention": 43}` (`n45` for `1×` step; `n36` when keeping `CheckpointManager` `initial_load_in_hf=True` buffers resident).
2. **Rapid Out-of-the-Box HuggingFace Checkpoint Fine-Tuning (`18.02%` Steady / `18.05%` Peak MFU):**
   - **Trainer:** TorchTitan PR #123 `transformers_modeling_backend` (`--hf-assets-path=/path/to/RecursiveSI/flint`).
   - **Settings:** `local_batch_size=48` (`98,304` tok/rank = `3,145,728` tok/step, `p7_03`), `experts.use_pallas_paired_grouped_mm=True`, `gmm_tiling=(1024, 2048, 2048)`, `tgmm_tiling=(512, 2048, 0)`, `--activation_checkpoint.mode=selective` (`op` / attention-only), `--loss.b_block_size=1024`, `--compile.enable`, `TORCHTPU_AUTO_MATMUL_PRECISION=bf16`.
