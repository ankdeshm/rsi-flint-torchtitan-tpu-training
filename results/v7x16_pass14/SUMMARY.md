# `RecursiveSI/flint` (40B MoE) TorchTitan Long-Context `32K` & `64K` MFU Frontier (`Pass 14`) — TPU `v7x-16` (`2x2x4`, `32` TensorCores)

- **Chip / World Size:** Cloud TPU `v7x` (`32` TensorCore ranks, `1,153.5` BF16 TFLOP/s/rank = `36.91` PFLOP/s slice peak)
- **Script & Manifest:** [`scripts/run_flint_v7x_pass14_mfu_frontier.py`](../../scripts/run_flint_v7x_pass14_mfu_frontier.py) · [`k8s/v7x-16-flint-pass14.yaml`](../../k8s/v7x-16-flint-pass14.yaml)

| Experiment ID | Seq Len | GFLOP/tok | Steady tok/s/rank | Cluster tok/s | Sustained PFLOP/s | Steady MFU | Peak MFU | TorchTitan Logged MFU | Peak HBM (GiB) | Real C4 Loss (`Start -> End`) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `v7x_p14_01_native_s65536_mb1_ga8_sbdp_bf16rope_fusedopt_n49` | `65,536` | `61.073` | **`9,051.7`** | **`289,654`** | **`17.69 PFLOP/s`** | **`47.92%`** | **`48.57%`** | `47.93%` | **`14.02`** | `12.32456 -> 4.53533` |
| `v7x_p14_09_native_s32768_mb3_ga8_sbdp_bf16rope_fusedopt_n49` | `32,768` | `38.524` | **`10,262.2`** | **`328,390`** | **`12.65 PFLOP/s`** | **`34.27%`** | **`34.60%`** | `34.27%` | **`20.52`** | `12.32438 -> 4.18028` |
