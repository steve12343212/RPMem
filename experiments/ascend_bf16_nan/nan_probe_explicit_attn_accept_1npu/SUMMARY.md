# 1-NPU 20k acceptance past step ~460

Original `qwen3_8b_npu_20k_1npu.yaml` + explicit encoder cross-attn.
Preflight stop at step 1000 (save_steps-aligned).

- No NaN / non-finite loss through step 1000
- Step 460 loss ≈ 0.65; step 1000 loss ≈ 0.61; avg_loss ≈ 0.76
- Throughput ≈ 0.8 steps/s
