# Appendix: Historical Short-Window Tuning Passes (`Pass 1..14`)

> [!IMPORTANT]
> **Use the main [`README.md`](../README.md) for validated performance numbers.**
> The early exploratory tuning sweeps summarized in this appendix (`Pass 1..14`, preserved in [`archive/`](../archive/)) ran `10–15` steps with upstream `c4_test` End-of-Document (`EoD`) packing and the TorchTitan unmasked square-attention FLOPs counter.
>
> **Why `32K` / `64K` MFU in `Pass 11` / `Pass 14` (`47.92%` on `v7x-16` and `65.46%` on `v5p-16`) was inflated and is NOT used as a headline metric:**
> 1. **`c4_test` Short-Document EoD Block Skipping:** `c4_test` web documents average `~450` tokens. Packing `~145` documents into a `65,536`-token row causes SplashAttention's segment mask to skip `>98%` of global attention tiles across the `14` full-attention layers, making step time nearly as fast as short-context training while the FLOPs numerator still assumes full `65,536`-token attention (`+59%`–`+69%` throughput inflation at `64K`).
> 2. **Square-Attention FLOPs Counter:** Upstream TorchTitan counts full `S × S` attention (`61.07` GFLOP/tok at `64K`) rather than the lower-triangular causal `S × (S+1) / 2` FLOPs (`38.52` GFLOP/tok at `64K`), adding another `1.585×` multiplier at `64K`.
>
> Our 60-step validation suite ([`scripts/run_flint_customer_60step_validation.py`](../scripts/run_flint_customer_60step_validation.py)) explicitly includes a `64K` `c4_test` EoD control run (`07_ctrl_s65536_mb1_ga1_c4eod`) that reproduces this exact effect (`48.06%` TorchTitan Counter MFU on `v7x-16` and `64.81%` on `v5p-16`), alongside the **true contiguous full-sequence 60-step runs** (`17.90%–18.08%` Exact Causal MFU / `28.38%–28.67%` TorchTitan Counter MFU on `v7x-16`, and `26.51%–27.20%` Exact Causal MFU / `42.04%–43.12%` TorchTitan Counter MFU on `v5p-16` / `v5p-32`).

---

## 1. Where Archived Artifacts Live

All legacy exploratory scripts, Kubernetes manifests, and raw JSON/Markdown summaries from `Pass 1..14` have been moved out of the primary directories into [`archive/`](../archive/):

- **Archived tuning scripts (`Pass 1..14`):** [`archive/scripts/`](../archive/scripts/)
- **Archived Kubernetes Job manifests:** [`archive/k8s/`](../archive/k8s/)
- **Archived `10–15`-step summaries & JSON logs:** [`archive/results/`](../archive/results/)

---

## 2. Summary of Exploratory Findings (`2K`–`16K` Short-Window Sweeps & Parallelism Studies)

While the `32K`/`64K` `c4_test` numbers were superseded by `CUSTOMER_VAL60`, the `Pass 1..13` tuning sweeps identified the kernel, memory, and communication optimizations that form the validated recipe:

1. **Kernel & Compiler Optimizations Incorporated into the Validated Recipe:**
   - **Native `TpuGraphTrainer` on `v5p`:** Fixed the `16 MiB` Scoped VMEM compilation error on `v5p` by lowering Pallas grouped-MM `tile_m` from `512` to `256` (`flint_40b_v5p_n40` in [`docker/patch_flint_configs.py`](../docker/patch_flint_configs.py)).
   - **Zero Attention Recompute (`n49`) + `tile_m=1024` on `v7x`:** Saving SplashAttention outputs across all `49` layers (`flint_40b_v014_tile1024`) improved throughput while keeping per-rank HBM under `23.5 GiB` on `v7x-16`.
   - **Pre-Down-Projection Router Scaling (`scale_before_down_proj`), `BF16` RoPE, and Compiled Fused AdamW:** Delivered the main single-step compute speedup (`+11.1% tok/s` on `v7x-8`, verified over 60 steps in [`docs/BASELINE_AND_ABLATIONS.md`](BASELINE_AND_ABLATIONS.md)).
2. **Pretrained Checkpoint Loss Equivalence (`Pass 6`, `p6_02`):**
   - Native `FlintStateDictAdapter` loaded all `15` `RecursiveSI/flint` safetensors shards (`36,734` FQN tensors) in `95.24s` and matched the HuggingFace Modeling Backend step-1 C4 loss (`3.43601`).
3. **Async PCIe Gen5 Host Activation Offload (`Pass 11b` / `Pass 12e` on `v7x-16`, `2K`):**
   - Streaming layer activations asynchronously between HBM and host DRAM (`save_last_n=0, prefetch=2`) enabled `mb=64` (`394,966` slice tok/s, `34.91 GiB` HBM) and `mb=96` (`385,200` slice tok/s, `53.43 GiB` HBM) on `v7x-16`, though keeping activations in HBM (`n49`, `mb=48`) achieved higher throughput (`431,165`–`441,664` slice tok/s at `23.02 GiB` HBM).
4. **Static-Shape Expert Parallelism (`StaticEP=2..32` on `v7x-16` and `v5p-32`, `2K`):**
   - Replacing dynamic `TokenDispatcher` / `torch.bincount` with static-capacity `[EP, E_local, C, D]` `all_to_all_single` eliminated XLA recompilation across `EP = 2, 4, 8, 16, 32`.
   - Because `RecursiveSI/flint` activates `6` of `256` tiny experts (`d_ff = 512`) per token across `49` MoE layers (`98` `AllToAll` collectives per forward+backward microbatch), pure `SimpleFSDP` (`18.08%–19.36%` Exact Causal MFU at `2K` on `v7x-16`) substantially outperformed `StaticEP=2 × FSDP=16` (`11.85%` TorchTitan Counter MFU, intra-package `0`-hop `AllToAll`), `StaticEP=4` (`7.17%`), `StaticEP=8` (`6.73%`), `StaticEP=16` (`5.85%`), and `StaticEP=32` (`4.70%`).

---

## 3. Selected Historical `2K`–`16K` Short-Window Reference Rows (`10–15` Steps, `C4-EoD`, TorchTitan Counter MFU)

For traceability to archived logs in [`archive/results/`](../archive/results/), the table below lists representative `2K`–`16K` short-window tuning runs (`10–15` steps, `c4_test` EoD packing, TorchTitan Counter MFU):

| TPU Slice | Archived Run ID | Lane & Seq Length | Global Batch (tok/step) | Steady tok/s/rank | Slice tok/s | TorchTitan Counter MFU (`C4-EoD`) | Peak HBM (GiB) | Archived Script |
|---|---|---|---:|---:|---:|---:|---:|---|
| `v7x-16` (`32` ranks) | `02_native_pr122_v011_n40` | Native `2K` default baseline (`mb=32`) | `2,097,152` | `10,985.3` | `351,530` | `16.56%` | `19.65` | [`run_flint_sweep.py`](../archive/scripts/run_flint_sweep.py) |
| `v7x-16` (`32` ranks) | `01_hf_backend_pr123_flint_40b` | HF Backend `2K` default baseline (`mb=32`) | `2,097,152` | `10,532.0` | `337,024` | `15.87%` | `11.78` | [`run_flint_sweep.py`](../archive/scripts/run_flint_sweep.py) |
| `v7x-16` (`32` ranks) | `v7x_p12g_02_native_..._mb48_ga8` | Native `2K` tuned (`mb=48, ga=8`) | `25,165,824` | `13,802.0` | `441,664` | `20.80%` | `23.02` | [`run_flint_v7x_pass12.py`](../archive/scripts/run_flint_v7x_pass12.py) |
| `v7x-16` (`32` ranks) | `v7x_p13_03_native_s16384_mb6_ga8` | Native `16K` tuned (`mb=6, ga=8`, `C4-EoD`) | `25,165,824` | `11,494.0` | `367,808` | `27.15%` | `20.52` | [`run_flint_v7x_pass13_longctx.py`](../archive/scripts/run_flint_v7x_pass13_longctx.py) |
| `v7x-16` (`32` ranks) | `v7x_p13_07_hf_pr123_s16384_mb8_ga2` | HF Backend `16K` tuned (`mb=8, ga=2`, `C4-EoD`) | `8,388,608` | `10,713.5` | `342,832` | `25.31%` | `25.50` | [`run_flint_v7x_pass13_longctx.py`](../archive/scripts/run_flint_v7x_pass13_longctx.py) |
| `v5p-16` (`16` chips) | `01_hf_backend_pr123_flint_v5p` | HF Backend `2K` default baseline (`mb=32`) | `1,048,576` | `5,215.8` | `83,453` | `19.76%` | `11.78` | [`run_flint_sweep.py`](../archive/scripts/run_flint_sweep.py) |
| `v5p-16` (`16` chips) | `v5p_p9_02_hf_..._mb64_1x` | HF Backend `2K` tuned (`mb=64, ga=1`) | `2,097,152` | `6,802.0` | `108,832` | `25.76%` | `22.84` | [`run_flint_v5p_pass9.py`](../archive/scripts/run_flint_v5p_pass9.py) |
| `v5p-32` (`32` chips) | `v5p32_p10_03_hf_s16384_mb8_1x` | HF Backend `16K` tuned (`mb=8, ga=1`, `C4-EoD`) | `4,194,304` | `5,951.6` | `190,451` | `35.33%` | `22.84` | [`run_flint_v5p_pass10_longctx.py`](../archive/scripts/run_flint_v5p_pass10_longctx.py) |
| `v5p-32` (`32` chips) | `v5p32_p10_07_native_s16384_mb6_1x` | Native `16K` tuned (`mb=6, ga=1`, `C4-EoD`) | `3,145,728` | `5,670.7` | `181,462` | `33.67%` | `29.21` | [`run_flint_v5p_pass10_longctx.py`](../archive/scripts/run_flint_v5p_pass10_longctx.py) |
