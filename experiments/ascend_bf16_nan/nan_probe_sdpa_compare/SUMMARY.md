# SDPA backend对照 (Experiment 2)

## Setup
- Capture failing encoder block SDPA I/O from checkpoint-53 + sample 16890 dump.
- Compare isolated backward: `sdpa_bf16` vs `explicit_bf16` vs `explicit_fp32` on identical `(q,k,v,dout)`.
- Stress isolated SDPA / explicit BF16 repeats on the same tensors.

## Results
- Failing capture landed on **block 3** (block id is intermittent; earlier branch probe saw block 6).
- Observed grads from the full encoder graph: `sdpa_v` absmax `~2e30` (catastrophic), while `sdpa_q` / `sdpa_k` / `dout` stay tiny (`~1e-8` / `~1e-6`).
- Isolated对照 on the saved tensors:
  - **`explicit_fp32`**: stable (`dv ~1e-7`)
  - **`explicit_bf16`**: stable (`dv ~1e-7`)
  - **`sdpa_bf16`**: intermittently catastrophic (one compare run exploded on the first SDPA call; a 30× stress run also saw `~1/30` absmax `~1e30`; another 40× stress stayed quiet)
- When SDPA explodes in isolation, explicit BF16/FP32 on the same inputs do not.

## Recommendation
**`replace_sdpa_backend`** for Perceiver cross-attn on Ascend: keep BF16, swap fused SDPA for explicit attention. Evidence does **not** require whole-encoder FP32 or RMSNorm changes as the first fix.

## Artifacts
- `perceiver_until_anomalous.json`
- `first_bad_block_repeat1.pt`
- `sdpa_backend_compare.json`
- `sdpa_backend_compare_stress.json`

## Local-only tensors
`first_bad_block_repeat*.pt` (~100MB+) stay on the training machine (GitHub 100MB limit); JSON reports above are enough to reproduce the conclusion.
