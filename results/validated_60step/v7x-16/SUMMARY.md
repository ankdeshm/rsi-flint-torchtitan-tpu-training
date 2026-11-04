# RSI Flint (`RecursiveSI/flint` 40B-A2.5B MoE) 60-Step Validated Runs (`val60`) — `v7x-16-flex-2x2x4`

- **Updated (UTC):** `2026-11-03T03:51:06.845631+00:00`
- **Hardware:** `16` × `v7x` chips (`32` TensorCore ranks, `2307.0` BF16 TFLOP/s/chip)
- **Methodology:** `60` optimizer steps per run (`log_freq=1`), Steady-State Mean over **50 post-warmup steps (`11..60`)**; True Unskipped Contiguous `seq_len` Causal Attention (`TT_CONTIGUOUS_FULL_SEQ=1`) unless marked `c4eod`.
- Rows `04`, `05`, `06`, `07`, `08` were measured with the lockstep multi-host runner (`k8s/v7x-16-flint-val60-lockstep.yaml`, `--lockstep`), which keeps every host on the same experiment and rendezvous port. Same image, recipe and 60-step method as the other rows.

| Experiment | Seq | Seq Mode | Steady Steps | **tok/s/chip** | tok/s/rank (±std) | Slice tok/s | Causal GF/tok | **Steady Causal MFU** | PaLM GF/tok | Steady PaLM MFU | `6*N_act` MFU | HBM (GiB) | Loss (`1 -> 60`) | Status |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| `v7x_val60_01_iso65k_s2048_mb32_ga1_fullseq` | `2048` | `Full-Seq` | `50` (`11..60`) | **25,168.4** | 12,584.2 (±579.5) | 402,694 | `16.5704` | **18.08%** | `17.385` | 18.97% | 16.47% | 15.52 | 12.31906 -> 2.82746 | `PASS` |
| `v7x_val60_02_iso65k_s16384_mb4_ga1_fullseq` | `16384` | `Full-Seq` | `50` (`11..60`) | **18,958.8** | 9,479.4 (±365.0) | 303,340 | `21.5994` | **17.75%** | `27.25` | 22.39% | 12.40% | 14.02 | 12.33117 -> 2.78966 | `PASS` |
| `v7x_val60_03_iso65k_s32768_mb2_ga1_fullseq` | `32768` | `Full-Seq` | `50` (`11..60`) | **14,866.0** | 7,433.0 (±237.1) | 237,855 | `27.2435` | **17.56%** | `38.5243` | 24.82% | 9.73% | 14.02 | 12.33477 -> 2.78622 | `PASS` |
| `v7x_val60_04_iso65k_s65536_mb1_ga1_fullseq` | `65536` | `Full-Seq` | `50` (`11..60`) | **10,722.3** | 5,361.1 (±139.9) | 171,556 | `38.5212` | **17.90%** | `61.0729` | 28.38% | 7.01% | 14.02 | 12.33406 -> 2.78358 | `PASS` |
| `v7x_val60_05_iso65k_s4096_mb16_ga1_fullseq` | `4096` | `Full-Seq` | `50` (`11..60`) | **24,552.1** | 12,276.0 (±581.6) | 392,833 | `17.3302` | **18.44%** | `18.7943` | 20.00% | 16.06% | 15.52 | 12.32523 -> 2.80467 | `PASS` |
| `v7x_val60_06_iso65k_s8192_mb8_ga1_fullseq` | `8192` | `Full-Seq` | `50` (`11..60`) | **22,326.5** | 11,163.2 (±494.6) | 357,224 | `18.7671` | **18.16%** | `21.6129` | 20.92% | 14.61% | 16.02 | 12.32958 -> 2.78572 | `PASS` |
| `v7x_val60_07_ctrl_s65536_mb1_ga1_c4eod` | `65536` | `C4-EoD` | `50` (`11..60`) | **18,155.0** | 9,077.5 (±1430.7) | 290,481 | `38.5212` | **30.31%** | `61.0729` | 48.06% | 11.88% | 14.02 | 12.32568 -> 2.822 | `PASS` |
| `v7x_val60_08_opt_s65536_mb1_ga4_fullseq` | `65536` | `Full-Seq` | `50` (`11..60`) | **10,829.4** | 5,414.7 (±38.0) | 173,271 | `38.5212` | **18.08%** | `61.0729` | 28.67% | 7.08% | 14.02 | 12.33261 -> 2.85837 | `PASS` |
| `v7x_val60_09_opt_s2048_mb48_ga2_fullseq` | `2048` | `Full-Seq` | `50` (`11..60`) | **26,947.8** | 13,473.9 (±268.9) | 431,165 | `16.5704` | **19.36%** | `17.385` | 20.31% | 17.63% | 23.02 | 12.31829 -> 2.86263 | `PASS` |
| `v7x_val60_10_opt_s16384_mb6_ga2_fullseq` | `16384` | `Full-Seq` | `50` (`11..60`) | **19,323.3** | 9,661.6 (±148.5) | 309,172 | `21.5994` | **18.09%** | `27.25` | 22.82% | 12.64% | 20.52 | 12.32923 -> 2.81084 | `PASS` |
| `v7x_val60_11_opt_s32768_mb3_ga2_fullseq` | `32768` | `Full-Seq` | `50` (`11..60`) | **15,202.8** | 7,601.4 (±90.6) | 243,245 | `27.2435` | **17.95%** | `38.5243` | 25.39% | 9.95% | 20.52 | 12.33113 -> 2.80935 | `PASS` |
