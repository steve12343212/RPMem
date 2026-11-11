# Explicit encoder cross-attn fix validation

## Change
On Ascend, Perceiver **encoder cross-attention** uses eager BF16 attention
(`QKᵀ/√d → softmax → PV`) instead of fused `scaled_dot_product_attention`.
Toggle: `RPMEM_PERCEIVER_EXPLICIT_ENCODER_CROSS_ATTN` (default `1` on NPU).

## Regression
From `checkpoint-53` + `sample_16890_perceiver.pt`, 20 isolated Perceiver replays
with branch probes:

| metric | value |
|---|---|
| passes | **20 / 20** |
| NaN / catastrophic | **0 / 0** |
| `projected.grad` absmax | `7.22e-8` (stable) |
| cross-attn `dV` absmax | `6.4e-8` … `9.5e-8` |

Previously the same dump intermittently produced `dV ~1e30` / NaNs under SDPA.

## Artifacts
- `explicit_attn_validate_summary.json`
- `explicit_attn_regression.json`
