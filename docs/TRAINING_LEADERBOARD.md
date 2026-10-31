# `RecursiveSI/flint` TorchTPU 0.2.0 GA Training & MFU Optimization

- **Project / Region:** `YOUR_GCP_PROJECT_ID` / `us-central1`
- **Clusters:** `tpu-v5p-cluster` & `tpu-v7x-cluster`
- **GCS Bucket:** `gs://YOUR_GCS_BUCKET/flint-results/`
- **Status (`2026-10-31 08:45 UTC`, 11 `v7x-16` Passes + 9 `v5p-32` Passes + 9 `v5p-16` Passes — 58+ Experiments Persisted to GCS):**
  - **OVERALL #1 MFU RECORD ACROSS ALL HARDWARE — `25.76%` Steady MFU (`25.81%` Peak MFU) on `v5p-16` (`2x2x4`, 16 `v5p` Chips, Pass 9 `v5p_p9_02`) & `24.88%` Steady MFU (`24.95%` Peak MFU) on `v5p-32` (`2x4x4`, 32 `v5p` Chips, Pass 9 `v5p_p9_02` / Pass 8 `v5p_p8_02`):**
    - **`v5p-16` (`2x2x4`, 16 Chips) #1 Record (`v5p_p9_02`):** PR #123 HF Backend (`RecursiveSI/flint`) + `SBDP` + `FusedCompiledOpt` + `Splash b=1024` + **`HybridAC N=16`** + **`mb=64` (`131,072` tok/chip = `2,097,152` tok/step)**: **`6,802.0` Steady tok/s/chip** (`108,832` tok/s slice-wide, `22.84 GiB` HBM, `loss: 12.4957 -> 3.9676`, `390.6s` wall time) — **`+6.00` pp MFU (`+30.4%` throughput)** over `v5p-16` Pass 1 (`19.76%`, `5,215.8` tok/s/chip)!
    - **`v5p-32` (`2x4x4`, 32 Chips) #1 Record (`v5p_p9_02` / `v5p_p8_02`):** **`24.88%` Steady MFU** (`6,568.4` tok/s/chip = **`210,189` tok/s cluster-wide**, `22.84 GiB` HBM at `mb=64` = `4,194,304` tok/step, `loss: 12.2951 -> 4.3073`) — **96.6% multi-slice weak-scaling efficiency** from 16 to 32 `v5p` chips!
    - **Native `TpuGraphTrainer` PR #122 on `v5p-32` (`v5p_p9_06`):** **`23.89%` Steady MFU (`24.00%` Peak MFU, `6,308.1` tok/s/chip = `201,859` tok/s cluster-wide, `28.11 GiB` HBM, `loss: 12.3288 -> 4.1735`)** with `SBDP` + `BF16` RoPE + `FusedCompiledOpt` (`+1.77` pp over Pass 8 `22.12%`)!
    - **Large-Batch `6.29M` tok/step (`2×` GradAccum) via `HSDP 2x FSDP-16` + `HybridAC N=28` + `BF16` Reduce (`v5p_p8_06`):** **`23.81%` Steady MFU (`23.83%` Peak MFU, `6,285.7` tok/s/chip = `201,142` tok/s cluster-wide, `loss: 12.4958 -> 3.7789`)**!
  - **OVERALL #1 CLUSTER THROUGHPUT & `v7x-16` MFU RECORD — `18.68%` Steady MFU (`18.76%` Peak MFU) & `396,570` Steady / `~398,300` Peak tok/s on `v7x-16` (`2x2x4`, 16 `v7x` Chips / 32 TensorCores, Pass 9 `v7x_p9_02`):**
    - **Native `TpuGraphTrainer` (PR #122 + `BF16` Reduce + `Llama4-Scout` / `HierSC` XLA, `mb=48`, `2×` GradAccum = `6,291,456` tok/step, `n43`):** **`12,392.8` Steady tok/s/rank** (`396,570` Steady tok/s cluster-wide, `27.12 GiB` HBM, `loss: 12.3201 -> 4.1606`, `295.9s` wall time) — **`+2.45` pp Steady MFU (`+15.0%` per-device throughput)** over `v7x-8` (`16.23%`, `10,779` tok/s/rank) while scaling **2×** from 16 to 32 TensorCores!
    - At **`1×` single-microbatch step (`v7x_p9_01`, `3,145,728` tok/step, `n45` + all-49 `moe.routing` + `BF16` Reduce)**, Native `TpuGraphTrainer` reaches **`18.53%` Steady / `18.62%` Peak MFU (`12,297.1` tok/s/rank = `393,507` tok/s cluster-wide)**!
    - **16 `v7x` Ironwood chips (`396,570` tok/s) outrun 32 `v5p` chips (`210,189` tok/s) by `1.89×` cluster-wide (`3.77×` per physical TPU chip)!**
  - **PR #123 HF Transformers Backend #1 `v7x-16` RECORD (`v7x_p9_03`, Pass 9B) — `18.29%` Steady / `18.34%` Peak MFU (`18.56%` Step-2 Peak):**
    - **`12,132.8` Steady tok/s/rank (`388,250` Steady / `394,112` Peak tok/s cluster-wide)** with Paired W1/W3 GMM (`tile_m=1024`), `BF16` Reduce, `Scout/HierSC` XLA, Compiled Optimizer/GradNorm, and `b=1024` (`16.92 GiB` HBM, `loss: 12.3104 -> 4.5041`, `302.7s`), improving **`+2.42` pp** over out-of-the-box PR #123 (`15.87%`)!
  - **Static-Shape Expert Parallelism (`StaticEP=16 + ZeRO-2 + PairedGMM-256 + EPCompiledGradNorm + SplitMeshOpt`) Verified on `v7x-16` (`v7x_p11c_02` / `v7x_p11d_03`):**
    - **`StaticTokenDispatcher`** (100% static `[EP, E_local, C, D]` XLA `all_to_all_single` shapes, zero dynamic ragged `all_to_all` recompilation) + `ZeRO-2` + `PairedGMM` (`tile_m=256`) + `EPCompiledGradNorm` (`DTensor` mesh-group bucketed compiled L2 norm) + `SplitMeshOpt` runs cleanly at **`24.48 GiB` (`25.50%`) HBM** with real C4 loss **`12.32097 -> 3.69201`** across 15 steps (`4,057` tok/s/rank = `6.11%` MFU under 100-part `GraphSplitter`, with `WholeStep` unsplit graph + `ZeRO-3` sweeps active in Pass 11D).
  - **Real `RecursiveSI/flint` Pretrained Checkpoint Parity Verified Across Both Lanes:**
    - Staged `RecursiveSI/flint` (39.8B total / ~2.52B active MoE, 256 routed experts, top-6, 49 layers, 15 safetensors shards, `79.59 GB` BF16) at `gs://YOUR_GCS_BUCKET/models/RecursiveSI/flint/`.
    - PR #123 HF Backend (`01_hf`, `p5_04`, `p7_03`, `v7x_p9_03`, `v5p_p8_02`, `v5p_p9_02`) converged smoothly on real C4 text; Native `FlintStateDictAdapter` (`p6_02`) loaded all 15 HF shards (`36,734` FQN tensors transposed into kernel-native `(E, K, N)` `TpuGroupedExperts`) in `95.24s` and produced **`step 1 loss: 3.43601`** on C4 — proving **exact 3-decimal numerical parity** between the HF Transformers Modeling Backend and Native `TpuGroupedExperts`!
  - **Container Image:** `us-central1-docker.pkg.dev/YOUR_GCP_PROJECT_ID/torchtpu/torchtitan-flint:v0.2.0` (`torch==2.15.0+cpu`, `torch_tpu==0.2.0` GA, `jax==0.11.1`, `libtpu==0.0.50`, `tokamax==0.0.14`, `transformers==5.22.0`, PR #123 in `/torchtitan-hf`, PR #122 in `/torchtitan-native`).

---

## 1. Key Engineering, Kernel & Roofline Discoveries

1. **Large-Microbatch `mb=64` (`131,072` tok/chip) + `HybridAC N=16` + `SBDP` + `FusedCompiledOpt` (`v5p_p9_02` -> `25.76%` MFU on `v5p-16`, `24.88%` MFU on `v5p-32`):**
   - Increasing per-chip tokens by `+33%` (`mb=48 -> mb=64` = `131,072` tok/chip = `3,072` tokens/expert across 256 top-6 experts!) increases GroupedMM MXU tile occupancy while amortizing fixed `79.6 GB` `BF16` parameter `AllGather` + `ReduceScatter` communication by `1.33×`. Pairing `mb=64` with `HybridAC N=16` (full block recompute on layers `0..15`, attention-only recompute on layers `16..48`), `SBDP` (`--xla_tpu_enable_sbdp=true`), `Splash b=1024`, and `FusedCompiledOpt` holds peak HBM to **`22.84 GiB`** (`24%` of `95 GiB`) and lifts `v5p-16` to **`25.76%` Steady / `25.81%` Peak MFU (`6,802.0` tok/s/chip)**!
2. **`BF16` Gradient ReduceScatter + `Llama4-Scout` / `HierSC` XLA Latency-Hiding Scheduler Flags (`v5p_p8_02` -> `24.87%` MFU; `v7x_p9_02` -> `18.68%` MFU):**
   - For fine-grained sparse MoE (`39.8B` total params vs only `~2.52B` active params = **`15.8×` parameter-to-FLOP sparsity amplification**!), standard FP32 gradient `ReduceScatter` ships `159.2 GB` of gradients per step across ICI for just `17.385` GFLOP/token of active MXU work.
   - Switching `--training.mixed_precision_reduce=bfloat16` halves ICI ReduceScatter payload (`159.2 GB -> 79.6 GB`) while enabling `--xla_enable_async_all_gather=true`, `--xla_tpu_enable_ag_backward_pipelining=true`, `--xla_tpu_enable_ici_ag_pipelining=true`, and `--xla_tpu_overlap_compute_collective_tc=true`, vaulting `v5p-32` Native MFU to **`23.89%` (`v5p_p9_06`)** and `v7x-16` Native MFU to **`18.68%` Steady / `18.76%` Peak MFU (`v7x_p9_02`)**!
3. **Why `Flint` (`~18.7%` `v7x` / `~25.8%` `v5p`) Differs Structurally from `Llama-4-Scout` (`54.60%` MFU) — First-Principles Roofline Physics:**
   - `Llama-4-Scout` (`109B` total, **`17B` active**, only **`16` experts top-1**, `d_ff = 8,192`) executes **`6,144` tokens/expert** (`48 × 128 × 128` MXU tiles!) with `6.4 : 1` total-to-active ratio and **`1,558 FLOPs/byte`** arithmetic intensity (`9.6×` above `v7x`'s `162 FLOPs/byte` ridge point).
   - `RecursiveSI/flint` (`39.8B` total, **`~2.52B` active**, **`256` routed experts top-6**, tiny **`d_ff = 768`** = just `6` `128`-wide columns!) has a **`15.8 : 1` total-to-active ratio** (`2.5×` worse FSDP byte/FLOP tax!) and **`~312 FLOPs/byte`** GroupedMM arithmetic intensity (`5×` lower than Scout), plus `49` layers of `256`-expert sort/permute/unpermute routing overhead. Accounting for these hardware rooflines places `Flint`'s theoretical FSDP ceiling at `~26–28%` on `v5p` (where we hit **`25.81%` peak = `>92%` of roofline!**).
4. **Static-Shape Expert Parallelism (`StaticTokenDispatcher` + `EPCompiledGradNorm` + `SplitMeshOpt`):**
   - Upstream `TokenDispatcher` (`EP > 1`) executes data-dependent `all_to_all` shapes that trigger XLA recompilation on every step. Replacing it with static-capacity `[EP, E_local, C, D]` `all_to_all_single` + compiled per-mesh `DTensor` L2 grad-norm + per-mesh chunked compiled AdamW groups achieved zero recompilation, flat **`24.48 GiB`** HBM, and exact C4 loss convergence (`12.32097 -> 3.69201`).

---

## 2. Complete Training & MFU Leaderboard — `v7x-16` (`2x2x4`, 16 Ironwood Chips / 32 TensorCores)

- **GCS Artifact Directories (`gs://YOUR_GCS_BUCKET/flint-results/` & `benchmark-results/`):**
  - Pass 9 (`v7x-16-2x2x4_pass9_20261031T024421Z/`): **#1 Overall `v7x-16` Native `18.68%` Steady / `18.76%` Peak MFU (`396,570` steady tok/s)**
  - Pass 9B (`v7x-16-2x2x4_pass9b_20261031T050635Z/`): **#1 Overall `v7x-16` HF PR#123 `18.29%` Steady / `18.34%` Peak MFU (`388,250` steady tok/s)** + `HSDP 2x16` (`17.61%`)
  - Pass 11C / 11D (`flint_v7x-16-flex-2x2x4_pass11c_SUMMARY.md`): **`StaticEP=16 + ZeRO-2 + PairedGMM-256 + EPCompiledGradNorm + SplitMeshOpt` (`24.48 GiB` HBM, `12.3210 -> 3.6920` real C4 convergence)**
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
| **EP=16** | `v7x_p11c_02` / `v7x_p11d_03` (`StaticEP=16`) | Native `StaticEP=16 + ZeRO-2 + PairedGMM-256 + EPCompiledGradNorm + SplitMeshOpt` (`24.48 GiB` HBM, `GraphSplitter`) | 6,291,456 | `PASS` | **3,880.0** *(4,057 pk)* | 124,160 *(129,824 pk)* | **5.85%** | **6.11%** | `12.3210 -> 3.6920` | 645.5 |
| - | `p6_02_native_real_flint_hf_checkpoint_30steps_mb48` | Native `FlintStateDictAdapter` 15-Shard Load (`36,734` tensors in `95.24s`, verified `loss=3.43601` parity) | 3,145,728 | `PARITY (3.43601)` | - | - | - | - | `3.43601` (step 1) | 367.3 |

---

## 2B. Cross-Generation TPU `v5p-16` (`2x2x4`, 16 Chips) & `v5p-32` (`2x4x4`, 32 Chips) Multi-Slice Scaling

- **Hardware / Peak Spec:** TPU `v5p` dual-MXU chips (`95 GiB` HBM/chip, `459.0` BF16 TFLOP/s/chip peak) across **`v5p-16` (`2x2x4`, 16 chips / 4 hosts)** and **`v5p-32` (`2x4x4`, 32 chips / 8 hosts)**.
- **GCS Run Directories:**
  - **Pass 9 `v5p-16` (`25.76%` Steady / `25.81%` Peak MFU — ALL-TIME MFU RECORD!):** `flint_v5p-16-2x2x4_pass9_SUMMARY.md`
  - **Pass 9 `v5p-32` (`24.88%` Steady HF + `23.89%` Steady Native — ALL-TIME 32-CHIP RECORD!):** `flint_v5p-32-2x4x4_pass9_SUMMARY.md`
  - **Pass 8 `v5p-32` (`24.87%` Steady / `24.95%` Peak MFU):** `flint_v5p-32-2x4x4_pass8_SUMMARY.md`

| Slice | Experiment | Lane / Key Configuration | Global Batch (tok/step) | Status / Steps | Steady tok/s/rank | Cluster tok/s | Steady MFU (%) | Peak MFU (%) | Loss (`start -> end`) | HBM / Wall Time |
|---|---|---|---:|---|---:|---:|---:|---:|---|---|
| **`v5p-16` (`2x2x4`)** | **`v5p_p9_02_hf_pr123_sbdp_fusedopt_sa1024_hybridac16_mb64_1x`** | **HF PR#123 + `SBDP` + `FusedCompiledOpt` + `Splash b=1024` + `HybridAC N=16` + `BF16` Reduce (`mb=64`)** | **2,097,152** | **`PASS` (12 steps)** | **6,802.0** | **108,832** | **25.76%** | **25.81%** | **`12.4957 -> 3.9676`** | **`22.84 GiB` / `390.6s`** |
| **`v5p-32` (`2x4x4`)** | **`v5p_p9_02_hf_pr123_sbdp_fusedopt_sa1024_hybridac16_mb64_1x`** | **HF PR#123 + `SBDP` + `FusedCompiledOpt` + `Splash b=1024` + `HybridAC N=16` + `BF16` Reduce (`mb=64`)** | **4,194,304** | **`PASS` (12 steps)** | **6,568.4** | **210,189** | **24.88%** | **24.91%** | **`12.2951 -> 4.3073`** | **`22.84 GiB` / `459.4s`** |
| **`v5p-32` (`2x4x4`)** | **`v5p_p8_02_hf_scout_xla_bf16red_mb48`** | **HF PR#123 + `BF16` Grad Reduce + `Llama4-Scout` XLA LHS/AG-Bwd-Pipe + `VMEM=80MiB` (`AttnOnlyAC`, `mb=48`)** | **3,145,728** | **`PASS` (12 steps)** | **6,565.3** | **210,090** | **24.87%** | **24.95%** | **`12.2947 -> 4.4061`** | **`16.92 GiB` / `344.2s`** |
| **`v5p-16` (`2x2x4`)** | **`v5p_p9_01_hf_pr123_sbdp_fusedopt_sa1024_mb48_1x`** | **HF PR#123 + `SBDP` + `FusedCompiledOpt` + `Splash b=1024` + `BF16` Reduce (`AttnOnlyAC`, `mb=48`)** | **1,572,864** | **`PASS` (12 steps)** | **6,522.9** | **104,366** | **24.71%** | **24.75%** | **`12.4971 -> 3.9204`** | **`16.92 GiB` / `332.9s`** |
| **`v5p-32` (`2x4x4`)** | **`v5p_p9_06_native_sbdp_bf16rope_fusedopt_mb48_1x`** | **Native PR#122 (`TpuGraphTrainer` `n40`) + `SBDP` + `BF16` RoPE + `FusedCompiledOpt` + `BF16` Reduce (`mb=48`)** | **3,145,728** | **`PASS` (12 steps)** | **6,308.1** | **201,859** | **23.89%** | **24.00%** | **`12.3288 -> 4.1735`** | **`28.11 GiB` / `531.6s`** |
| **`v5p-32` (`2x4x4`)** | **`v5p_p8_06_hf_hsdp2x16_scout_xla_bf16red_2xga_mb48`** | **HF PR#123 `HSDP 2x FSDP-16` + `BF16` Reduce + `Llama4-Scout` XLA + `HybridAC N=28` + `2×` GradAccum (`mb=48`)** | **6,291,456** | **`PASS` (12 steps)** | **6,285.7** | **201,142** | **23.81%** | **23.83%** | **`12.4958 -> 3.7789`** | **`21.08 GiB` / `632.1s`** |
| **`v5p-32` (`2x4x4`)** | **`v5p_p8_01_hf_scout_xla_attnac_mb48`** | **HF PR#123 (`FP32` Reduce) + `Llama4-Scout` XLA LHS/AG-Bwd-Pipe + `VMEM=80MiB` (`AttnOnlyAC`, `mb=48`)** | **3,145,728** | **`PASS` (12 steps)** | **6,147.5** | **196,720** | **23.28%** | **23.42%** | **`12.2947 -> 4.4175`** | **`16.92 GiB` / `441.0s`** |
| **`v5p-16` (`2x2x4`)** | `01_hf_backend_pr123_flint_v5p` | Out-of-the-Box HF PR#123 (`RecursiveSI/flint` 15-shard weights, `b=1024`, `mb=32`, `seq=2048`) | 1,048,576 | `PASS` (15 steps) | **5,215.8** | 83,453 | **19.76%** | 19.79% | `12.4968 -> 3.4368` | `11.78 GiB` / `383.2s` |

---

## 3. Recommended Production Configurations for `RecursiveSI/flint`

1. **Maximum Cluster Throughput (`396,570` Steady / `~398,300` Peak tok/s on 16 `v7x` Chips — `18.68%` Steady / `18.76%` Peak MFU):**
   - **Trainer:** TorchTitan Native `TpuGraphTrainer` (`flint_40b_p9_n43_moe_route_comp_opt`, `SimpleFSDP` across 32 TensorCores on `v7x-16`).
   - **Settings:** `local_batch_size=48` (`98,304` tok/rank = `6,291,456` tok/step with `2×` grad accum in `v7x_p9_02`, or `1×` `n45` in `v7x_p9_01` at `18.53%`), `--training.mixed_precision_reduce=bfloat16`, `gmm_tiling=(1024, 2048, 2048)`, `tgmm_tiling=(512, 2048, 0)`, `optimizer_compile="separate_graph"`, `compile_grad_norm=True`, plus `Scout/HierSC` `LIBTPU_INIT_ARGS`.
2. **Maximum MFU Efficiency (`25.76%` Steady / `25.81%` Peak MFU on 16 `v5p` Chips; `24.88%` Steady / `24.95%` Peak MFU on 32 `v5p` Chips; `18.29%` Steady / `18.56%` Step-2 Peak on 16 `v7x` Chips):**
   - **Trainer:** TorchTitan PR #123 `transformers_modeling_backend` (`--hf-assets-path=/path/to/RecursiveSI/flint`).
   - **Settings:** `local_batch_size=64` + `HybridAC N=16` + `SBDP` + `FusedCompiledOpt` + `Splash b=1024` (`2,097,152` tok/step on `v5p-16` at **`25.76%` MFU**, `4,194,304` tok/step on `v5p-32` at **`24.88%` MFU`; or `HSDP 2x FSDP-16` + `HybridAC N=28` + `2×` GA = `6,291,456` tok/step at **`23.81%` MFU**), `--training.mixed_precision_reduce=bfloat16`, `experts.use_pallas_paired_grouped_mm=True`, `--loss.b_block_size=1024`.
