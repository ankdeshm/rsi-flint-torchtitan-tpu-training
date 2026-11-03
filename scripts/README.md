# Validated Benchmark & Staging Scripts (`scripts/`)

This directory contains only the active, validated scripts used to stage `RecursiveSI/flint` weights and run the 60-step TorchTitan TPU suites across TPU `v7x` and `v5p`.

| Script | Purpose | Kubernetes Manifests | Results Directory |
|---|---|---|---|
| [`stage_hf_to_gcs.py`](stage_hf_to_gcs.py) | Downloads all `15` `RecursiveSI/flint` safetensors shards + tokenizer files from Hugging Face into GCS (`gs://YOUR_GCS_BUCKET/models/RecursiveSI/flint`) so multi-host TPU pods read directly via GCSFuse | [`../k8s/stage-flint-job.yaml`](../k8s/stage-flint-job.yaml) | — |
| [`run_flint_customer_60step_validation.py`](run_flint_customer_60step_validation.py) | Multi-host 60-step validation and baseline/ablation runner with lockstep rendezvous coordination, contiguous full-sequence causal attention (`TT_CONTIGUOUS_FULL_SEQ=1`), and dual MFU accounting (Exact Causal FLOPs + TorchTitan Counter FLOPs) | [`../k8s/v7x-8-flint-baseline-2k.yaml`](../k8s/v7x-8-flint-baseline-2k.yaml)<br>[`../k8s/v7x-16-flint-val60.yaml`](../k8s/v7x-16-flint-val60.yaml)<br>[`../k8s/v7x-16-flint-val60-rerun.yaml`](../k8s/v7x-16-flint-val60-rerun.yaml)<br>[`../k8s/v5p-16-flint-val60.yaml`](../k8s/v5p-16-flint-val60.yaml)<br>[`../k8s/v5p-32-flint-val60.yaml`](../k8s/v5p-32-flint-val60.yaml)<br>[`../k8s/v5p-16-flint-val60-longctx.yaml`](../k8s/v5p-16-flint-val60-longctx.yaml)<br>[`../k8s/v5p-32-flint-val60-longctx.yaml`](../k8s/v5p-32-flint-val60-longctx.yaml) | [`../results/baseline_2k/`](../results/baseline_2k/)<br>[`../results/customer_val60/`](../results/customer_val60/)<br>[`../results/customer_val60_longctx/`](../results/customer_val60_longctx/) |

## Suite Modes in `run_flint_customer_60step_validation.py`

1. **`--suite=baseline_2k` (`v7x-8`, `16` TensorCore ranks):**
   - Reproduces the unmodified `google-pytorch/torchtitan` Native `TpuGraphTrainer` ([PR #122](https://github.com/google-pytorch/torchtitan/pull/122)) and HuggingFace Modeling Backend ([PR #123](https://github.com/google-pytorch/torchtitan/pull/123)) `2K` recipes, followed by 4 incremental optimization steps and 2 batch-scaling HBM probes (`60` steps each, steady mean of steps `11..60`).
2. **`--suite=customer_val60` (`v7x-16`, `v5p-16`, `v5p-32`):**
   - Evaluates context-length scaling (`2K`, `4K`, `8K`, `16K`, `32K`, `64K`) at `60` optimizer steps per configuration with true contiguous full-sequence causal attention (`TT_CONTIGUOUS_FULL_SEQ=1`), plus a `64K` `c4_test` End-of-Document (`EoD`) control run (`07_ctrl_s65536_mb1_ga1_c4eod`) that quantifies how short-document packing skips global attention blocks.
3. **`--suite=customer_val60_longctx` (`v5p-16`, `v5p-32`):**
   - Probes `128K` (`seq_len=131,072`) single-sequence-per-rank (`mb=1`) HBM limits across progressive attention-activation checkpointing levels (`attn49 -> attn32 -> attn16`) without Context/Tensor Parallelism.

> **Historical Tuning Scripts:** Earlier `10–15`-step exploratory scripts (`Pass 1..14`) are archived under [`../archive/scripts/`](../archive/scripts/) and documented in [`../docs/APPENDIX_HISTORICAL_RUNS.md`](../docs/APPENDIX_HISTORICAL_RUNS.md).
