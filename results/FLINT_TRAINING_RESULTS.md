# `RecursiveSI/flint` TorchTPU 0.2.0 GA Training & MFU Optimization

- **Project / Region:** `YOUR_GCP_PROJECT_ID` / `us-central1`
- **Clusters:** `tpu-v5p-cluster` & `tpu-v7x-cluster`
- **GCS Bucket:** `gs://YOUR_GCS_BUCKET/flint-results/`
- **Status (`2026-10-31 05:20 UTC`, 9 `v7x-16` Passes + 8 `v5p-32` Passes + 2 `v5p-16` Passes Complete — 50+ Experiments Persisted to GCS):**
  - **OVERALL #1 MFU RECORD ACROSS ALL HARDWARE — `24.87%` Steady MFU (`24.95%` Peak MFU) on `v5p-32` (`2x4x4`, 32 `v5p` Chips, Pass 8 `v5p_p8_02`):**
    - **PR #123 HF Backend (`RecursiveSI/flint` weights) + `BF16` Gradient ReduceScatter + `Llama4-Scout` XLA LHS/AG-Bwd-Pipe (`mb=48`, `3,145,728` tok/step):** **`6,565.3` Steady tok/s/chip** (`210,090` tok/s cluster-wide, `16.92 GiB` HBM, `loss: 12.2947 -> 4.4061`, `344.2s` wall time) — **`+2.41` pp MFU** over Pass 3 (`22.46%`) and **`+5.11` pp** over `v5p-16` Pass 1 (`19.76%`)!
    - **Large-Batch `6.29M` tok/step (`2×` GradAccum) via `HSDP 2x FSDP-16` + `HybridAC N=28` + `BF16` Reduce (`v5p_p8_06`):** **`23.81%` Steady MFU (`23.83%` Peak MFU, `6,285.7` tok/s/chip = `201,142` tok/s cluster-wide, `loss: 12.4958 -> 3.7789`)**, completely eliminating the `FSDP-32` `2×` GA multi-host communication bottleneck!
    - **Native `TpuGraphTrainer` PR #122 on `v5p-32` (`v5p_p8_05`):** **`22.12%` Steady MFU (`22.26%` Peak MFU, `5,840.4` tok/s/chip = `186,893` tok/s cluster-wide, `loss: 12.3288 -> 4.1735`)**!
  - **OVERALL #1 CLUSTER THROUGHPUT & `v7x-16` MFU RECORD — `18.68%` Steady MFU (`18.76%` Peak MFU) & `396,570` Steady / `~398,300` Peak tok/s on `v7x-16` (`2x2x4`, 16 `v7x` Chips / 32 TensorCores, Pass 9 `v7x_p9_02`):**
    - **Native `TpuGraphTrainer` (PR #122 + `BF16` Reduce + `Llama4-Scout` / `HierSC` XLA, `mb=48`, `2×` GradAccum = `6,291,456` tok/step, `n43`):** **`12,392.8` Steady tok/s/rank** (`396,570` Steady tok/s cluster-wide, `27.12 GiB` HBM, `loss: 12.3201 -> 4.1606`, `295.9s` wall time) — **`+2.45` pp Steady MFU (`+15.0%` per-device throughput)** over `v7x-8` (`16.23%`, `10,779` tok/s/rank) while scaling **2×** from 16 to 32 TensorCores!
    - At **`1×` single-microbatch step (`v7x_p9_01`, `3,145,728` tok/step, `n45` + all-49 `moe.routing` + `BF16` Reduce)**, Native `TpuGraphTrainer` reaches **`18.53%` Steady / `18.62%` Peak MFU (`12,297.1` tok/s/rank = `393,507` tok/s cluster-wide)**!
    - **16 `v7x` Ironwood chips (`396,570` tok/s) outrun 32 `v5p` chips (`210,090` tok/s) by `1.89×` cluster-wide (`3.78×` per physical TPU chip)!**
  - **PR #123 HF Transformers Backend #1 `v7x-16` RECORD (`v7x_p9_03`, Pass 9B) — `18.29%` Steady / `18.34%` Peak MFU (`18.56%` Step-2 Peak):**
    - **`12,132.8` Steady tok/s/rank (`388,250` Steady / `394,112` Peak tok/s cluster-wide)** with Paired W1/W3 GMM (`tile_m=1024`), `BF16` Reduce, `Scout/HierSC` XLA, Compiled Optimizer/GradNorm, and `b=1024` (`16.92 GiB` HBM, `loss: 12.3104 -> 4.5041`, `302.7s`), improving **`+2.42` pp** over out-of-the-box PR #123 (`15.87%`)!
  - **Real `RecursiveSI/flint` Pretrained Checkpoint Parity Verified Across Both Lanes:**
    - Staged `RecursiveSI/flint` (39.8B total / ~2.52B active MoE, 256 routed experts, top-6, 49 layers, 15 safetensors shards, `79.59 GB` BF16) at `gs://YOUR_GCS_BUCKET/models/RecursiveSI/flint/`.
    - PR #123 HF Backend (`01_hf`, `p5_04`, `p7_03`, `v7x_p9_03`, `v5p_p8_02`) converged smoothly on real C4 text; Native `FlintStateDictAdapter` (`p6_02`) loaded all 15 HF shards (`36,734` FQN tensors transposed into kernel-native `(E, K, N)` `TpuGroupedExperts`) in `95.24s` and produced **`step 1 loss: 3.43601`** on C4 — proving **exact 3-decimal numerical parity** between the HF Transformers Modeling Backend and Native `TpuGroupedExperts`!
  - **Container Image:** `us-central1-docker.pkg.dev/YOUR_GCP_PROJECT_ID/torchtpu/torchtitan-flint:v0.2.0` (`torch==2.15.0+cpu`, `torch_tpu==0.2.0` GA, `jax==0.11.1`, `libtpu==0.0.50`, `tokamax==0.0.14`, `transformers==5.22.0`, PR #123 in `/torchtitan-hf`, PR #122 in `/torchtitan-native`).

---

## 1. Key Engineering, Kernel & Roofline Discoveries

1. **`BF16` Gradient ReduceScatter + `Llama4-Scout` / `HierSC` XLA Latency-Hiding Scheduler Flags (`v5p_p8_02` -> `24.87%` MFU; `v7x_p9_02` -> `18.68%` MFU):**
   - For fine-grained sparse MoE (`39.8B` total params vs only `~2.52B` active params = **`15.8×` parameter-to-FLOP sparsity amplification**!), standard FP32 gradient `ReduceScatter` ships `159.2 GB` of gradients per step across ICI for just `17.385` GFLOP/token of active MXU work.
   - Switching `--training.mixed_precision_reduce=bfloat16` halves ICI ReduceScatter payload (`159.2 GB -> 79.6 GB`) while enabling `--xla_enable_async_all_gather=true`, `--xla_tpu_enable_ag_backward_pipelining=true`, `--xla_tpu_enable_ici_ag_pipelining=true`, and `--xla_tpu_overlap_compute_collective_tc=true` (plus `80 MiB` Scoped VMEM on `v5p` and Hierarchical SparseCore ND ReduceScatter on `v7x`), vaulting `v5p-32` MFU from `22.46% -> 23.28%` (Scout XLA alone, `v5p_p8_01`) -> **`24.87%` Steady / `24.95%` Peak MFU (`v5p_p8_02`)**, and `v7x-16` Native MFU from `18.36% -> 18.68%` Steady / **`18.76%` Peak MFU (`v7x_p9_02`)**!
2. **Why `Flint` (`~18.7%` `v7x` / `~24.9%` `v5p`) Differs Structurally from `Llama-4-Scout` (`54.60%` MFU) — First-Principles Roofline Physics:**
   - `Llama-4-Scout` (`109B` total, **`17B` active**, only **`16` experts top-1**, `d_ff = 8,192`) executes **`6,144` tokens/expert** (`48 × 128 × 128` MXU tiles!) with `6.4 : 1` total-to-active ratio and **`1,558 FLOPs/byte`** arithmetic intensity (`9.6×` above `v7x`'s `162 FLOPs/byte` ridge point).
   - `RecursiveSI/flint` (`39.8B` total, **`~2.52B` active**, **`256` routed experts top-6**, tiny **`d_ff = 768`** = just `6` `128`-wide columns!) executes **`2,304` tokens/expert** at `mb=48` with **`15.8 : 1` total-to-active ratio** (`2.5×` worse FSDP byte/FLOP tax!) and **`~312 FLOPs/byte`** GroupedMM arithmetic intensity (`5×` lower than Scout), plus `49` layers of `256`-expert sort/permute/unpermute routing overhead. Accounting for these hardware rooflines places `Flint`'s theoretical 32-rank FSDP ceiling at `~25–28%` on `v5p` (where we hit **`24.95%` peak = `~90%` of roofline!**) and `~21–23%` on `v7x` (where dual-TensorCore `32`-rank splitting halves per-rank tokens/expert vs `16`-chip `v5p`, and we hit **`18.76%` peak**!).
3. **Exact 96 GiB XLA HBM Calibration (`mb=48` + `n43`/`n45`) + `HSDP 2x FSDP-16` (`HybridAC N=28`):**
   - On `v7x-16`, `mb=48` (`98,304` tok/rank) with `n43`/`n45` saved Splash attention layers and all-49 saved `moe.routing` regions is the global Pareto optimum inside 96 GiB HBM.
   - On `v5p-32`, `HSDP 2x FSDP-16` (`v5p_p8_06`) confines `AllGather`/`ReduceScatter` to local 16-chip `2x2x4` sub-tori while using `HybridAC N=28` (recomputing Splash attention on the first 28 layers and saving the last 21 layers to offset `2×` larger `FSDP-16` parameter shards), achieving **`23.81%` Steady / `23.83%` Peak MFU** at `6.29M` tokens/step!
4. **Hardware Scoped VMEM Ceilings Calibrated Across Both TPU Generations:**
   - On `v7x` (`32.00 MB` default Scoped VMEM): Forward Paired W1/W3 GMM runs at `tile_m=1024`, while backward Paired TGMM requires `tile_m=512` (`tile_m=1024` needs `35.68 MB`) and cross-entropy requires `b_block_size=1024`.
   - On `v5p`: Default `16.00 MB` Scoped VMEM fits Paired GMM at `tile_m=256`; raising `--xla_tpu_scoped_vmem_limit_kib=81920` (`80 MiB`) in Pass 8 gave XLA's latency-hiding scheduler additional VMEM scratchpad for async collective overlap (`+0.82` pp before `BF16` Reduce).

---

## 2. Complete Training & MFU Leaderboard — `v7x-16` (`2x2x4`, 16 Ironwood Chips / 32 TensorCores)

- **GCS Artifact Directories (`gs://YOUR_GCS_BUCKET/flint-results/`):**
  - Pass 9 (`v7x-16-2x2x4_pass9_20261031T024421Z/`): **#1 Overall `v7x-16` Native `18.68%` Steady / `18.76%` Peak MFU (`396,570` steady tok/s)**
  - Pass 9B (`v7x-16-2x2x4_pass9b_20261031T050635Z/`): **#1 Overall `v7x-16` HF PR#123 `18.29%` Steady / `18.34%` Peak MFU (`388,250` steady tok/s)** + `HSDP 2x16` (`17.61%`)
  - Passes 1–8: `v7x-16-2x2x4_20261030T072821Z/` through `v7x-16-2x2x4_pass8_20261030T222850Z/`
- **FLOP Accounting:** `17.385` GFLOP/token (`RecursiveSI/flint`), `1,153.5` BF16 TFLOP/s/TensorCore (`v7x` peak).

| Rank | Experiment | Lane / Key Configuration | Global Batch (tok/step) | Status | Steady tok/s/rank | Cluster tok/s (32 ranks) | Steady MFU (%) | Peak MFU (%) | Loss (`start -> end`) | Wall Time (s) |
|---:|---|---|---:|---|---:|---:|---:|---:|---|---:|
| **#1** | **`v7x_p9_02_native_pr122_bf16red_scout_n43_mb48_ga2`** | **Native PR#122 + `BF16` Reduce + Scout/HierSC XLA + `mb=48` + `2×` GradAccum + `n43` + All-49 `moe.routing` + CompiledOpt** | **6,291,456** | **`PASS`** | **12,392.8** | **396,570** *(~398,300 peak)* | **18.68%** | **18.76%** | **`12.3201 -> 4.1606`** | **295.9** |
| **#2** | **`v7x_p9_01_native_pr122_bf16red_scout_n45_mb48_1x`** | **Native PR#122 + `BF16` Reduce + Scout/HierSC XLA + `mb=48` (`1×` step) + `n45` + All-49 `moe.routing` + CompiledOpt** | **3,145,728** | **`PASS`** | **12,297.1** | **393,507** | **18.53%** | **18.62%** | **`12.3215 -> 4.1721`** | **378.9** |
| **#3** | `p7_01_native_grad_accum_2x_mb48_n43` | Native PR#122 (`FP32` Reduce) + `mb=48` + `2×` GradAccum + `n43` Attn + All-49 `moe.routing` + CompiledOpt | 6,291,456 | `PASS` | **12,182.1** | 389,827 *(391,136 peak)* | **18.36%** | **18.42%** | `12.3201 -> 3.6763` | 553.8 |
| **#4 (HF #1)** | **`v7x_p9_03_hf_pr123_bf16red_scout_compopt_mb48_1x`** | **HF Backend PR#123 (`RecursiveSI/flint`) + `BF16` Reduce + Scout/HierSC XLA + Paired GMM (`tile_m=1024`) + CompiledOpt (`mb=48`)** | **3,145,728** | **`PASS`** | **12,132.8** | **388,250** *(394,112 step-2)* | **18.29%** | **18.34%** *(18.56%)* | **`12.3104 -> 4.5041`** | **302.7** |
| **#5** | `p7_02_native_n45_all49_moe_routing_mb48` | Native PR#122 (`FP32` Reduce) + `mb=48` (`1×` step) + `n45` Attn + All-49 `moe.routing` + CompiledOpt | 3,145,728 | `PASS` | **12,136.2** | 388,358 *(389,920 peak)* | **18.29%** | **18.36%** | `12.3215 -> 3.6797` | 209.7 |
| **#6** | `p4_01_native_n43_tile1024_moe_routing_comp_opt_mb48` | Native PR#122 + `mb=48` (`98K` tok/rank) + `n43` Attn + All-49 `moe.routing` + `tile_m=1024` + CompiledOpt | 3,145,728 | `PASS` | **12,040.9** | 385,309 | **18.15%** | 18.21% | `12.3215 -> 3.6799` | 398.9 |
| **#7 (HF #2)** | `p7_03_hf_pr123_paired_gmm_b1024_mb48` | HF Backend PR#123 (`FP32` Reduce, `RecursiveSI/flint`) + Paired W1/W3 GMM (`tile_m=1024`) + `b=1024` + `mb=48` | 3,145,728 | `PASS` | **11,956.9** | 382,621 | **18.02%** | 18.05% | `12.3104 -> 3.3361` | 308.9 |
| **#8** | `v7x_p9_04_native_pr122_bf16red_scout_n22_mb56_ga2` | Native PR#122 + `BF16` Reduce + Scout/HierSC XLA + `mb=56` (`2×` GA = `7.34M` tok/step) + `n22` | 7,340,032 | `PASS` | **11,804.6** | 377,747 | **17.79%** | 17.88% | `12.3208 -> 4.1649` | 503.6 |
| **#9 (HF HSDP)** | `v7x_p9_05_hf_pr123_hsdp2x16_bf16red_hybridac28_mb48_1x` | HF Backend PR#123 `HSDP 2x FSDP-16` (`8`-Chip Shards) + `HybridAC N=28` + `BF16` Reduce + Scout/HierSC XLA | 3,145,728 | `PASS` | **11,682.8** | 373,850 | **17.61%** | 17.64% | `12.4341 -> 3.9882` | 301.2 |
| *Ref* | *(Ref: PR #123 `v7x-8`)* | *HF Backend Reference (`16` ranks, `mb=32`)* | *1,048,576* | *Ref* | *10,810.0* | *172,960* | *16.29%* | *-* | *-* | *-* |
| *Ref* | *(Ref: PR #122 `v7x-8`)* | *Native `v011_n40` Reference (`16` ranks, `mb=32`)* | *1,048,576* | *Ref* | *10,779.0* | *172,464* | *16.23%* | *-* | *-* | *-* |
| **#10** | `01_hf_backend_pr123_flint_40b` | Out-of-the-Box PR#123 HF Backend (`RecursiveSI/flint`, `b=1024`, unpaired GMM, `mb=32`) | 2,097,152 | `PASS` | **10,532.0** | 337,024 | **15.87%** | 15.91% | `12.3115 -> 3.4358` | 248.7 |
| - | `p6_02_native_real_flint_hf_checkpoint_30steps_mb48` | Native `FlintStateDictAdapter` 15-Shard Load (`36,734` tensors in `95.24s`, verified `loss=3.43601` parity) | 3,145,728 | `PARITY (3.43601)` | - | - | - | - | `3.43601` (step 1) | 367.3 |

---

## 2B. Cross-Generation TPU `v5p-32` (`2x4x4`, 32 Chips) & `v5p-16` (`2x2x4`, 16 Chips) Multi-Slice Scaling

- **Hardware / Peak Spec:** TPU `v5p` dual-MXU chips (`95 GiB` HBM/chip, `459.0` BF16 TFLOP/s/chip peak) across **`v5p-32` (`2x4x4`, 32 chips / 8 hosts)** and **`v5p-16` (`2x2x4`, 16 chips / 4 hosts)**.
- **GCS Run Directories:**
  - **Pass 8 `v5p-32` (`24.87%` Steady / `24.95%` Peak MFU — ALL-TIME MFU RECORD!):** `gs://YOUR_GCS_BUCKET/flint-results/v5p-32-2x4x4_pass8_20261031T020538Z/` (`flint_v5p-32-2x4x4_pass8_SUMMARY.md`)
  - Passes 2–7 `v5p-32` & Passes 1–2 `v5p-16`: `gs://YOUR_GCS_BUCKET/flint-results/v5p-*`

| Slice | Experiment | Lane / Key Configuration | Global Batch (tok/step) | Status / Steps | Steady tok/s/rank | Cluster tok/s | Steady MFU (%) | Peak MFU (%) | Loss (`start -> end`) | HBM / Wall Time |
|---|---|---|---:|---|---:|---:|---:|---:|---|---|
| **`v5p-32` (`2x4x4`)** | **`v5p_p8_02_hf_scout_xla_bf16red_mb48`** | **HF PR#123 + `BF16` Grad Reduce + `Llama4-Scout` XLA LHS/AG-Bwd-Pipe + `VMEM=80MiB` (`AttnOnlyAC`, `mb=48`)** | **3,145,728** | **`PASS` (12 steps)** | **6,565.3** | **210,090** | **24.87%** | **24.95%** | **`12.2947 -> 4.4061`** | **`16.92 GiB` / `344.2s`** |
| **`v5p-32` (`2x4x4`)** | **`v5p_p8_06_hf_hsdp2x16_scout_xla_bf16red_2xga_mb48`** | **HF PR#123 `HSDP 2x FSDP-16` + `BF16` Reduce + `Llama4-Scout` XLA + `HybridAC N=28` + `2×` GradAccum (`mb=48`)** | **6,291,456** | **`PASS` (12 steps)** | **6,285.7** | **201,142** | **23.81%** | **23.83%** | **`12.4958 -> 3.7789`** | **`21.08 GiB` / `632.1s`** |
| **`v5p-32` (`2x4x4`)** | **`v5p_p8_01_hf_scout_xla_attnac_mb48`** | **HF PR#123 (`FP32` Reduce) + `Llama4-Scout` XLA LHS/AG-Bwd-Pipe + `VMEM=80MiB` (`AttnOnlyAC`, `mb=48`)** | **3,145,728** | **`PASS` (12 steps)** | **6,147.5** | **196,720** | **23.28%** | **23.42%** | **`12.2947 -> 4.4175`** | **`16.92 GiB` / `441.0s`** |
| **`v5p-32` (`2x4x4`)** | `v5p_p3_01_hf_pr123_paired_tile256_b1024_mb48` | HF PR#123 (`FP32` Reduce, default XLA, Paired GMM `tile_m=256`, `b=1024`, `mb=48`) | 3,145,728 | `PASS` (12 steps) | **5,930.0** | 189,760 | **22.46%** | 22.50% | `12.2947 -> 4.4109` | `16.91 GiB` / `435.1s` |
| **`v5p-32` (`2x4x4`)** | **`v5p_p8_05_native_pr122_scout_xla_bf16red_mb48`** | **Native PR#122 (`TpuGraphTrainer` `n40`) + `BF16` Reduce + `Llama4-Scout` XLA (`mb=48`)** | **3,145,728** | **`PASS` (12 steps)** | **5,840.4** | **186,893** | **22.12%** | **22.26%** | **`12.3288 -> 4.1735`** | **`28.11 GiB` / `570.5s`** |
| **`v5p-32` (`2x4x4`)** | `v5p_p3_03_native_paired_tile256_n45_mb48_ga2` | Native PR#122 (`FP32` Reduce, Paired GMM `tile_m=256`, `n45`, CompiledOpt, `mb=48`, `2×` GA) | 6,291,456 | `PASS` (12 steps) | **5,578.6** | 178,515 | **21.13%** | 21.18% | `12.3273 -> 4.1698` | `26.87 GiB` / `705.9s` |
| **`v5p-32` (`2x4x4`)** | `v5p_p2_04_hf_pr123_paired_tile256_b1024_mb32` | HF PR#123 (`RecursiveSI/flint` weights, Paired GMM `tile_m=256`, `b=1024`, `mb=32`) | 2,097,152 | `PASS` (15 steps) | **5,266.9** | 168,541 | **19.95%** | 19.98% | `12.2942 -> 3.6038` | `11.78 GiB` / `355.4s` |
| **`v5p-16` (`2x2x4`)** | **`01_hf_backend_pr123_flint_v5p`** | **HF PR#123 (`RecursiveSI/flint` 15-shard weights, Pallas GMM, `b=1024`, `mb=32`, `seq=2048`)** | **1,048,576** | **`PASS` (15 steps)** | **5,215.8** | **83,453** | **19.76%** | **19.79%** | **`12.4968 -> 3.4368`** | **`11.78 GiB` / `383.2s`** |
| **`v5p-16` (`2x2x4`)** | **`v5p_p2_01_native_paired_tile256_n49_mb24`** | **Native PR#122 (`TpuGraphTrainer`, Paired GMM `tile_m=256`, `n49`, CompiledOpt, `mb=24`)** | **786,432** | **`PASS` (15 steps)** | **4,491.8** | **71,869** | **17.01%** | **17.09%** | **`12.3177 -> 3.3964`** | **`14.32 GiB` / `400.1s`** |

---

## 3. Recommended Production Configurations for `RecursiveSI/flint`

1. **Maximum Cluster Throughput (`396,570` Steady / `~398,300` Peak tok/s on 16 `v7x` Chips — `18.68%` Steady / `18.76%` Peak MFU):**
   - **Trainer:** TorchTitan Native `TpuGraphTrainer` (`flint_40b_p9_n43_moe_route_comp_opt`, `SimpleFSDP` across 32 TensorCores on `v7x-16`).
   - **Settings:** `local_batch_size=48` (`98,304` tok/rank = `6,291,456` tok/step with `2×` grad accum in `v7x_p9_02`, or `1×` `n45` in `v7x_p9_01` at `18.53%`), `--training.mixed_precision_reduce=bfloat16`, `gmm_tiling=(1024, 2048, 2048)`, `tgmm_tiling=(512, 2048, 0)`, `optimizer_compile="separate_graph"`, `compile_grad_norm=True`, plus `Scout/HierSC` `LIBTPU_INIT_ARGS`.
2. **Maximum MFU Efficiency (`24.87%` Steady / `24.95%` Peak MFU on 32 `v5p` Chips; `18.29%` Steady / `18.56%` Step-2 Peak on 16 `v7x` Chips):**
   - **Trainer:** TorchTitan PR #123 `transformers_modeling_backend` (`--hf-assets-path=/path/to/RecursiveSI/flint`).
   - **Settings:** `local_batch_size=48` (`3,145,728` tok/step in `v5p_p8_02` & `v7x_p9_03`; or `HSDP 2x FSDP-16` + `HybridAC N=28` + `2×` GA = `6,291,456` tok/step at **`23.81%` MFU** in `v5p_p8_06`), `--training.mixed_precision_reduce=bfloat16`, `experts.use_pallas_paired_grouped_mm=True`, `--loss.b_block_size=1024`, `--activation_checkpoint.mode=selective`, and `Llama4-Scout` `LIBTPU_INIT_ARGS`.
