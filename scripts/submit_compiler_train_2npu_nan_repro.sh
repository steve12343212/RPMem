#!/usr/bin/env bash
# Submit 2-NPU NaN repro via vc only (no card pinning).
#   DRY_RUN=0 bash scripts/submit_compiler_train_2npu_nan_repro.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RPMEM_ROOT="$(dirname "$SCRIPT_DIR")"

DRY_RUN="${DRY_RUN:-1}"
PARTITION="${PARTITION:-pdgpu-sjtu-ai}"
TRAIN_IMAGE="${TRAIN_IMAGE:-hub.szaic.com/sjtu/sjtu_shenliping-sunwenjie-strategy-lora-ascend:v1.0.0-cann8.2rc1-torch2.6.0}"
GPUS=2
CPUS="${CPUS:-32}"
MEM="${MEM:-200G}"
JOB_NAME="${JOB_NAME:-rpmem_nan_2npu}"

CONFIG="${CONFIG:-$RPMEM_ROOT/configs/compiler/qwen3_8b_npu_nan_2npu.yaml}"
TRAIN_DATA="${TRAIN_DATA:-$RPMEM_ROOT/data/rpmem/compiler_nan_repro/train_500.jsonl}"
TEACHER_ROOT="${TEACHER_ROOT:-$RPMEM_ROOT/outputs/teacher_kvcache_500}"
OUTPUT_DIR="${OUTPUT_DIR:-$RPMEM_ROOT/outputs/compiler_nan_repro_2npu}"
LOG_NAME="compiler_train_2npu_nan_repro.log"

LOG_DIR="${LOG_DIR:-$RPMEM_ROOT/logs/vc}"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/${JOB_NAME}.JOB.log"

RUN_CMD=$(cat <<EOF
set -euo pipefail
cd '${RPMEM_ROOT}'
export CONFIG='${CONFIG}'
export TRAIN_DATA='${TRAIN_DATA}'
export TEACHER_ROOT='${TEACHER_ROOT}'
export OUTPUT_DIR='${OUTPUT_DIR}'
export LOG_NAME='${LOG_NAME}'
unset ASCEND_RT_VISIBLE_DEVICES
unset PYTHON
bash scripts/run_compiler_train_2npu_nan_repro.sh
EOF
)

echo "=== RPMem 2-NPU NaN repro submit ==="
echo "JOB=$JOB_NAME GPUS=$GPUS CONFIG=$CONFIG OUT=$OUTPUT_DIR DRY_RUN=$DRY_RUN"

if [[ "$DRY_RUN" == "1" ]]; then
  echo "DRY_RUN=1 — not submitting"
  echo "$RUN_CMD"
  exit 0
fi

vc submit -p "$PARTITION" \
  -i "$TRAIN_IMAGE" \
  -c "$CPUS" -m "$MEM" -g "$GPUS" \
  -j "$JOB_NAME" \
  -d "$RPMEM_ROOT" \
  --nopassenv \
  JOB=1:1 "$LOG_FILE" \
  --cmd "$RUN_CMD"

echo "Submitted $JOB_NAME"
echo "Logs: tail -f ${LOG_DIR}/${JOB_NAME}.1.log"
