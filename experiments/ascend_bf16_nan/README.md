# Ascend BF16 Fixed-FKL NaN Investigation

Artifacts from Ascend 910B NaN debugging of RPMem Fixed-FKL (`offline_fkl`) training on the 20k Ultrachat subset (and 500-sample repros).

## Key finding

Training collapses to irreversible `loss=nan` after the loss enters roughly the **0.5–0.7** plateau. Reducing NPU count / global batch only **delays** the failure; **1-NPU full 20k still NaNs**.

Earlier short 500-sample runs (≤200 steps) can look stable because they never enter this regime on the full corpus trajectory.

## Experiment matrix

| Job / log | NPUs | Data | Global batch | Result |
|-----------|------|------|--------------|--------|
| `rpmem_nan_short_1npu` | 1 | 500 | 2 | OK (40 steps) |
| `rpmem_nan_long_1npu` | 1 | 500 | 2 | OK (200 steps) |
| `rpmem_nan_2npu` | 2 | 500 | 4 | OK (50 steps) |
| `rpmem_nan_2npu_bigbatch` | 2 | 500 | 16 | NaN @ step 46 |
| `rpmem_kv_20k_train_fp32` | 8 | 20k | 64 | NaN by first log @ 25 |
| `rpmem_kv_20k_4npu_accum4` | 4 | 20k | 16 | NaN @ ~60 (loss≈0.76) |
| `rpmem_kv_20k_4npu_accum2` | 4 | 20k | 8 | NaN @ ~115 (loss≈0.70) |
| `rpmem_kv_20k_4npu_gbatch4` | 4 | 20k | 4 | NaN @ ~370 (loss≈0.60) |
| `rpmem_kv_20k_1npu` | 1 | 20k | 2 | NaN @ ~460 (loss≈0.70) |

Notes:

- Mixed precision remains **bf16** for compute. Log names containing `fp32` refer to float32 **metric/grad reduce** workarounds for HCCL, not full fp32 training.
- See `metrics_summary.csv` for parsed first-NaN steps.
- Curve plots under `plots/`.

## Bundle layout

```
experiments/ascend_bf16_nan/
  README.md
  metrics_summary.csv
  logs/          # raw vc job logs
  configs/       # accelerate + yaml used in these runs
  scripts/       # submit/run helpers used for repro
  plots/         # loss / lr / throughput figures
```

Local archive also stored at `experiments/ascend_bf16_nan_bundle.tar.gz`.


## NaN probe localization (1-NPU)

Run `rpmem_kv_20k_1npu_nan_probe` with `--nan_probe` from step 0.

- Forward on sample `16890` remained finite (`loss≈1.50`, head norms finite).
- First non-finite tensor was `trainable[0].grad` (shape `[3072, 768]`, all NaNs) at `global_step=53`, phase `before_clip`.
- Artifacts: `nan_probe/nan_probe.jsonl`, `nan_probe/first_bad_grad.json`, `logs/rpmem_kv_20k_1npu_nan_probe.1.log`.

This supports backward-first NaN gradients rather than a fixed loss≈0.6 threshold or an immediately corrupt teacher forward.

## Current hypothesis

Primary issue is numerical instability of **bf16 Fixed-FKL on Ascend** once the student approaches teacher (loss ~0.6), not multi-NPU communication alone. Next levers: fp32 critical path for FKL/logits, lower LR, stronger clipping, or NaN-skip/rollback.
