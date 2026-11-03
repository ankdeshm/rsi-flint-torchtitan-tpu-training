# RSI Flint (`RecursiveSI/flint` 40B-A2.5B MoE) 60-Step Customer Validation (`CUSTOMER_VAL60_LONGCTX`) — `v5p-32-8n-2x4x4`

- **Updated (UTC):** `2026-11-03T03:22:13.322366+00:00`
- **Hardware:** `32` × `v5p` chips (`32` TensorCore ranks, `459.0` BF16 TFLOP/s/chip)
- **Methodology:** `60` optimizer steps per run (`log_freq=1`), Steady-State Mean over **50 post-warmup steps (`11..60`)**; True Unskipped Contiguous `seq_len` Causal Attention (`TT_CONTIGUOUS_FULL_SEQ=1`) unless marked `c4eod`.
- **Long-context extension (`128K` / `256K`):** Flint in this TorchTitan TPU stack rejects context and tensor parallelism, so each rank trains on one full `seq_len` sequence per microbatch (`mb=1` -> `131,072` / `262,144` tokens/rank/microbatch vs a constant `65,536` in the `2K..64K` rows); global batch = ranks x `seq_len` tokens/step.
- `_attn{k}` = layers whose Splash-attention output is kept for backward (`49` = the `2K..64K` recipe). A smaller `k` is used only after an HBM OOM at the larger one; recomputed attention FLOPs are **not** counted in any MFU column.

| Experiment | Seq | Seq Mode | Steady Steps | **tok/s/chip** | tok/s/rank (±std) | Slice tok/s | Causal GF/tok | **Steady Causal MFU** | PaLM GF/tok | Steady PaLM MFU | `6*N_act` MFU | HBM (GiB) | Loss (`1 -> 60`) | Status |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| `v5p_val60_12_long_s131072_mb1_ga1_fullseq_attn49` | `131072` | `Full-Seq` | `0` (`-`) | **0.0** | 0.0 (±0.0) | 0 | `61.0715` | **0.00%** | `106.17` | 0.00% | 0.00% | - | - | `OOM(rc=1) -> retry attn32` |
| `v5p_val60_12_long_s131072_mb1_ga1_fullseq_attn32` | `131072` | `Full-Seq` | `0` (`-`) | **0.0** | 0.0 (±0.0) | 0 | `61.0715` | **0.00%** | `106.17` | 0.00% | 0.00% | - | - -> - | `OOM(rc=1) -> retry attn16` |
| `v5p_val60_12_long_s131072_mb1_ga1_fullseq_attn16` | `131072` | `Full-Seq` | `0` (`-`) | **0.0** | 0.0 (±0.0) | 0 | `61.0715` | **0.00%** | `106.17` | 0.00% | 0.00% | - | - | `FAIL(rc=-9)` |
