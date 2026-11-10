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

## Per-microbatch probe + boundary hooks (`nan_probe_s53`)

Follow-up with probe after every `accelerator.backward`, `NAN_PROBE_START_STEP=53`, and hooks on `student_logits` / LoRA A/B / `lora_emb`.

- Saved `checkpoint-53` (local only; not in git), then instrumented the next accum window `12027 → 16890`.
- micro 106 (`12027`): after_backward **finite** (params + all boundary grads).
- micro 107 (`16890`): forward finite; after_backward `perceiver.modality_projection.gate_proj.weight.grad` all NaN; **boundary grads still finite**.
- Artifacts: `nan_probe_s53/` (`nan_probe.jsonl`, `nan_probe_stop/first_bad_grad.json`, `train.log`, `args.yaml`).

Implication: first bad sample in this window is `16890`; FKL and HyperLoRA head are not the first-fault site; NaN appears inside Perceiver backward (`lora_emb` → `modality_projection`).

## Current hypothesis

Primary issue is numerical instability on the **Perceiver backward** path under Ascend bf16 for sample `16890` at step-53 weights—not FKL loss or HyperLoRA head first. Next levers: single-sample replay from `checkpoint-53`, deeper Perceiver hooks, then local fp32/stability fixes on that subgraph.

## Perceiver isolate from checkpoint-53 (`nan_probe_perceiver_isolate`)

Resume `checkpoint-53`, replay `12027 → 16890`, dump `ctx_features` + `dL/dlora_emb`, then isolated Perceiver `autograd.backward`.

| Setting | Training window result | Isolated Perceiver replay |
|---|---|---|
| Accumulated grads (default) | `lora_emb`/`latents` tiny & finite; `projected.grad` absmax `~2.5e31` (bf16) then `grad_norm=inf` | No NaN once (`projected` absmax `~6e-8`) |
| Zero-grad before `16890` | `projected.grad` has NaNs (`nan_count` 1024); `gate_proj.weight.grad` all NaN | No NaN flag, but `projected.grad` absmax `~2.8e29` |

Artifacts: `nan_probe_perceiver_isolate/`, `nan_probe_perceiver_isolate_zgrad/`.

Implication: fault is **inside encoder backward** (`latents → projected`), not FKL/head, and **not** from accumulating `12027`. `ctx_features` for `16890` already has absmax `~3.9e4`. Isolated replay is **nondeterministic** at this scale (1e-8 vs 1e29); next step is capture forward `projected`/`latents`, force deterministic SDPA, and/or FP32 encoder backward on the saved activations.

