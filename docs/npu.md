# Ascend NPU adaptation (8×910B4)

This tree runs on the workshop Ascend login / `vc` cluster. Host `npu-smi`
sees eight healthy `910B4-1` cards (64 GB HBM each). Training uses the same
Ascend image as EDT/MRM:

`hub.szaic.com/sjtu/sjtu_shenliping-sunwenjie-strategy-lora-ascend:v1.0.0-cann8.2rc1-torch2.6.0`

## What changed

- `rpmem/device.py` — resolve NPU → CUDA → CPU; autocast / empty_cache / RNG / memory helpers
- CUDA hardcodes in training, inference, and benchmark scripts now call those helpers
- `configs/accelerate_8npu.yaml` — 8-process `MULTI_NPU` + bf16
- `configs/compiler/qwen3_8b_npu.yaml` — local Qwen3-8B + ModernBERT paths
- `scripts/env_rpmem_npu.sh`, `setup_container_deps.sh`, `run_*_8npu.sh`, `submit_compiler_train_8npu.sh`
- Flash-attn stays off on NPU (`--no-use_flash_attn` / SDPA)

## Host vs container

| Location | Role |
| --- | --- |
| Login node | `npu-smi`, downloads, conda `rpmem` CPU tooling. Do **not** expect a working `torch_npu` here unless CANN+torch_npu are installed into the env. |
| `vc` job (`-g 8`) | Real training. Image provides `torch`+`torch_npu`; `setup_container_deps.sh` installs the rest into user-site. |

Unset `ASCEND_RT_VISIBLE_DEVICES` for Accelerate DDP (vc already sets
`ASCEND_VISIBLE_DEVICES`). Teacher sharding intentionally pins one card per
process via `ASCEND_RT_VISIBLE_DEVICES=$shard_id`.

## Quick start

```bash
# data (HF mirror)
HF_ENDPOINT=https://hf-mirror.com bash scripts/download_resources.sh

# dry-run submit
DRY_RUN=1 bash scripts/submit_compiler_train_8npu.sh

# real 8-NPU job (after data + optional teacher cache)
DRY_RUN=0 CORPUS_ROOT=$PWD/data/rpmem/compiler \
  TEACHER_ROOT=$PWD/outputs/teacher \
  bash scripts/submit_compiler_train_8npu.sh
```

Teacher precompute (inside an 8-card job):

```bash
bash scripts/run_teacher_precompute_8npu.sh
```
