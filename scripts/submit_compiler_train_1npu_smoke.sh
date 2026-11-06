#!/usr/bin/env bash
# Submit 1-NPU float32-reduce smoke train (reuse KV teacher; no teacher rebuild).
#   DRY_RUN=0 bash scripts/submit_compiler_train_1npu_smoke.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RPMEM_ROOT="$(dirname "$SCRIPT_DIR")"

DRY_RUN="${DRY_RUN:-1}"
PARTITION="${PARTITION:-pdgpu-sjtu-ai}"
TRAIN_IMAGE="${TRAIN_IMAGE:-hub.szaic.com/sjtu/sjtu_shenliping-sunwenjie-strategy-lora-ascend:v1.0.0-cann8.2rc1-torch2.6.0}"
GPUS="${GPUS:-1}"
# Queue limit: <=20 CPU and <=120G MEM per GPU.
CPUS="${CPUS:-16}"
MEM="${MEM:-120G}"
JOB_NAME="${JOB_NAME:-rpmem_smoke_1npu}"
LOG_DIR="${LOG_DIR:-$RPMEM_ROOT/logs/vc}"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/${JOB_NAME}.JOB.log"

TRAIN_DATA="${TRAIN_DATA:-$RPMEM_ROOT/data/rpmem/compiler_20k/train.jsonl}"
TEACHER_ROOT="${TEACHER_ROOT:-$RPMEM_ROOT/outputs/teacher_kvcache_20k}"
OUTPUT_DIR="${OUTPUT_DIR:-$RPMEM_ROOT/outputs/compiler_qwen3_8b_npu_smoke}"
CONFIG="${CONFIG:-$RPMEM_ROOT/configs/compiler/qwen3_8b_npu_smoke.yaml}"
PREFLIGHT_STOP_AFTER_STEP="${PREFLIGHT_STOP_AFTER_STEP:-2}"

RUN_CMD=$(cat <<EOF
set -euo pipefail
cd '${RPMEM_ROOT}'
export TRAIN_DATA='${TRAIN_DATA}'
export TEACHER_ROOT='${TEACHER_ROOT}'
export OUTPUT_DIR='${OUTPUT_DIR}'
export CONFIG='${CONFIG}'
export PREFLIGHT_STOP_AFTER_STEP='${PREFLIGHT_STOP_AFTER_STEP}'
unset PYTHON
echo '[smoke] 1-NPU Fixed-FKL short run'
bash scripts/run_compiler_train_1npu_smoke.sh
EOF
)

echo "=== RPMem 1-NPU smoke submit ==="
echo "JOB=$JOB_NAME GPUS=$GPUS CPUS=$CPUS MEM=$MEM CONFIG=$CONFIG"
echo "OUT=$OUTPUT_DIR PREFLIGHT_STOP=$PREFLIGHT_STOP_AFTER_STEP DRY_RUN=$DRY_RUN"

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
