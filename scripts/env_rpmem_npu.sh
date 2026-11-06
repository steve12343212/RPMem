#!/usr/bin/env bash
# Environment for RPMem on Ascend (source inside vc job, or on a host with torch_npu).
# Usage: source scripts/env_rpmem_npu.sh

RPMEM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export RPMEM_ROOT
export RPMEM_MODELS_DIR="${RPMEM_MODELS_DIR:-/aistor/sjtu/hpc_stor01/home/sunwenjie/workshop/models}"
export RPMEM_BASE_MODEL="${RPMEM_BASE_MODEL:-${RPMEM_MODELS_DIR}/Qwen3-8B}"
export RPMEM_CTX_ENCODER="${RPMEM_CTX_ENCODER:-${RPMEM_MODELS_DIR}/ModernBERT-base}"
export RPMEM_DATA_ROOT="${RPMEM_DATA_ROOT:-${RPMEM_ROOT}/data/rpmem}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

# Prefer Ascend-bundled Python that already has torch_npu.
_ASCEND_PY=""
for cand in python python3 /usr/bin/python /usr/local/python3.10.15/bin/python /usr/local/python3.10.15/bin/python3; do
  if command -v "$cand" >/dev/null 2>&1 || [[ -x "$cand" ]]; then
    if "$cand" -c "import torch_npu" >/dev/null 2>&1; then
      _ASCEND_PY="$(command -v "$cand" 2>/dev/null || echo "$cand")"
      break
    fi
  fi
done
# Fallback: workshop conda env for host-side CPU tooling.
if [[ -z "$_ASCEND_PY" ]]; then
  _CONDA_PY="/aistor/sjtu/hpc_stor01/home/sunwenjie/miniconda3/envs/rpmem/bin/python"
  if [[ -x "$_CONDA_PY" ]]; then
    _ASCEND_PY="$_CONDA_PY"
  fi
fi
export PYTHON="${_ASCEND_PY:-python3}"
export PYTHON_BIN="${PYTHON_BIN:-$PYTHON}"

export DEVICE_KIND="${DEVICE_KIND:-npu}"
export PYTHONUNBUFFERED=1

# vc assigns ASCEND_VISIBLE_DEVICES; do NOT copy physical ids into
# ASCEND_RT_VISIBLE_DEVICES or torch.npu.is_available() becomes False.
unset ASCEND_RT_VISIBLE_DEVICES 2>/dev/null || true
unset MASTER_ADDR MASTER_PORT RANK LOCAL_RANK WORLD_SIZE NODE_RANK GROUP_RANK 2>/dev/null || true
unset LOCAL_WORLD_SIZE GROUP_WORLD_SIZE ROLE 2>/dev/null || true

# HCCL defaults for 8-card single-node jobs
export HCCL_CONNECT_TIMEOUT="${HCCL_CONNECT_TIMEOUT:-1800}"
export HCCL_EXEC_TIMEOUT="${HCCL_EXEC_TIMEOUT:-1800}"
export ASCEND_LAUNCH_BLOCKING="${ASCEND_LAUNCH_BLOCKING:-0}"

_USER_SITE="$("$PYTHON" -c 'import site; print(site.getusersitepackages())' 2>/dev/null || true)"
export PYTHONPATH="${_USER_SITE:+${_USER_SITE}:}${RPMEM_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

cd "$RPMEM_ROOT"
echo "RPMEM_ROOT=$RPMEM_ROOT"
echo "PYTHON=${PYTHON} ($("${PYTHON}" -V 2>&1 || true))"
echo "RPMEM_BASE_MODEL=$RPMEM_BASE_MODEL (exists=$([[ -d $RPMEM_BASE_MODEL ]] && echo yes || echo no))"
echo "RPMEM_CTX_ENCODER=$RPMEM_CTX_ENCODER (exists=$([[ -d $RPMEM_CTX_ENCODER ]] && echo yes || echo no))"
echo "RPMEM_DATA_ROOT=$RPMEM_DATA_ROOT (exists=$([[ -d $RPMEM_DATA_ROOT ]] && echo yes || echo no))"
echo "DEVICE_KIND=$DEVICE_KIND ASCEND_VISIBLE_DEVICES=${ASCEND_VISIBLE_DEVICES:-} ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-<unset>}"
