# `scripts/` — Script Index by Optimization Pass

This directory contains the benchmark and ablation scripts progressing from the initial baselines (`15.87%`–`19.76%` MFU) to `35.33%` MFU on `v5p-32`, `35.21%` MFU on `v5p-16`, and `27.15%` MFU on `v7x-16`.

## 1. Primary Scripts

| Script | Target Hardware | Measured Results | Matching K8s Manifest | Result Files |
|---|---|---|---|---|
| **[`run_flint_v5p_pass10_longctx.py`](./run_flint_v5p_pass10_longctx.py)** | `v5p-32` (`2x4x4`) & `v5p-16` (`2x2x4`) | `35.33%` / `36.20%` MFU (`v5p32_p10_03` HF `16K`), `35.21%` / `35.92%` MFU (`v5p16_p10_03` HF `16K`), & `33.67%` / `34.63%` MFU (`v5p32_p10_07` Native `16K`) | [`v5p-32-flint-pass10.yaml`](../k8s/v5p-32-flint-pass10.yaml) · [`v5p-16-flint-pass10.yaml`](../k8s/v5p-16-flint-pass10.yaml) | [`results/v5p32_pass10/`](../results/v5p32_pass10/SUMMARY.md) · [`results/v5p16_pass10/`](../results/v5p16_pass10/SUMMARY.md) |
| **[`run_flint_v7x_pass13_longctx.py`](./run_flint_v7x_pass13_longctx.py)** | `v7x-16` (`2x2x4`, 32 TC) | `27.15%` / `27.20%` MFU (`v7x_p13_03` Native `16K`) & `25.31%` / `25.35%` MFU (`v7x_p13_07` HF `16K`) | [`v7x-16-flint-pass13.yaml`](../k8s/v7x-16-flint-pass13.yaml) | [`results/v7x16_pass13/`](../results/v7x16_pass13/SUMMARY.md) |
| **[`run_flint_v7x_pass12.py`](./run_flint_v7x_pass12.py)** | `v7x-16` (`2x2x4`, 32 TC) | `441,664` tok/s / `20.80%` MFU (`v7x_p12g_02` Native `2K`), `18.58%` MFU (`v7x_p12_04` HF `2K`), `18.60%` MFU (`v7x_p12e_04` Host Offload), & `11.85%` / `12.24%` MFU (`v7x_p12_08` `StaticEP=2`) | [`v7x-16-flint-pass12g.yaml`](../k8s/v7x-16-flint-pass12g.yaml) · [`v7x-16-flint-pass12b.yaml`](../k8s/v7x-16-flint-pass12b.yaml) | [`pass12g SUMMARY.md`](../results/v7x-16-flex-2x2x4_pass12g_20261101T114752Z/SUMMARY.md) · [`pass12 SUMMARY.md`](../results/flint_v7x-16-flex-2x2x4_pass12_SUMMARY.md) |
| **[`run_flint_v5p_pass9.py`](./run_flint_v5p_pass9.py)** | `v5p-16` & `v5p-32` | `25.76%` / `25.81%` MFU (`v5p-16` `2K`) & `24.88%` MFU (`v5p-32` `2K`, `210,189` tok/s) in `v5p_p9_02`; `23.89%` MFU Native `2K` (`v5p_p9_06`) | [`v5p-32-and-16-flint-pass9.yaml`](../k8s/v5p-32-and-16-flint-pass9.yaml) | [`v5p-16 pass9 MD`](../results/flint_v5p-16-2x2x4_pass9_SUMMARY.md) · [`v5p-32 pass9 MD`](../results/flint_v5p-32-2x4x4_pass9_SUMMARY.md) |

## 2. Iterative Ablation Scripts

- [`stage_hf_to_gcs.py`](./stage_hf_to_gcs.py): Streams the 15 `RecursiveSI/flint` Hugging Face safetensors shards (`79.59 GB`) into regional GCS.
- [`run_flint_sweep.py`](./run_flint_sweep.py): **Pass 1** default PR #122 (`16.56%` `v7x-16`) & PR #123 (`15.87%` `v7x-16`, `19.76%` `v5p-16`) baselines + `HierSC` / `n49` / `tile_m=1024`.
- [`run_flint_pass2.py`](./run_flint_pass2.py) .. [`run_flint_pass7_v7x_next.py`](./run_flint_pass7_v7x_next.py): **Passes 2–7 (`v7x-16`)** — compiled AdamW + GradNorm (`p3_01`), saved `moe.routing` (`p3_02`), 15-shard numerical parity `loss=3.43601` (`p6_02` in [`run_flint_pass6.py`](./run_flint_pass6.py)), and `mb=48` + `2×` GA (`p7_01` -> `18.36%`).
- [`run_flint_v7x_pass8.py`](./run_flint_v7x_pass8.py) .. [`run_flint_v7x_pass11.py`](./run_flint_v7x_pass11.py): **Passes 8–11 (`v7x-16`)** — `BF16` Reduce + Scout XLA (`v7x_p9_02` -> `18.68%`), `tile_m=256` zero-padding (`pass10b`), `mb=96` host offload (`pass11b`), and `StaticEP=8/16/32` (`pass11`).
- [`run_flint_v5p_pass2.py`](./run_flint_v5p_pass2.py) .. [`run_flint_v5p_pass8.py`](./run_flint_v5p_pass8.py) & [`run_flint_v5p_pass9b.py`](./run_flint_v5p_pass9b.py): **Passes 2–8 & 9B (`v5p`)** — `16 MiB` Scoped VMEM `tile_m=256` fix, `BF16` Reduce + XLA LHS (`v5p_p8_02` -> `24.87%`), `HSDP 2×16` (`v5p_p8_06` -> `23.81%`), and `StaticEP=8/16` (`pass9b` -> `7.30%`).
