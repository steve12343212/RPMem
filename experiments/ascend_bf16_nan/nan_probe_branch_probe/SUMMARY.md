# Encoder branch probe (Experiment 1)

Checkpoint-53 + `sample_16890_perceiver.pt`, 3 isolated Perceiver replays with per-block hooks.

## Headline

- Repeats: 2/3 magnitude-ok (`projected.grad` absmax ~1e-8); **1/3 catastrophic**.
- Failing repeat first anomalous cross-attn block: **block 6** (0-based).
- Within block 6, backward hook-fire order is `sdpa_v → sdpa_k → context_rms_out → projected_branch`. **First catastrophic signal is `sdpa_v` (~8.6e30)** while **`sdpa_k` stays ~4e-8**; RMSNorm/`projected_branch` then inherit the V-path blow-up. Blocks 7–8 also bad; block 8 shows NaNs.
- `ctx_features` slices 12 and 13 are **bit-identical**.

## Artifacts

- `perceiver_branch_probe.json`
- `first_bad_block_repeat1.pt` (block-6 forwards + grads for Exp 2)

## Next

Experiment 2: BF16 SDPA vs explicit BF16 attention vs FP32 attention on the saved block-6 inputs/`dL/dV` (and full SDPA inputs).

## Local-only tensors
`first_bad_block_repeat*.pt` (~100MB+) stay on the training machine (GitHub 100MB limit); JSON reports above are enough to reproduce the conclusion.
