#!/usr/bin/env bash
# Submit capture+isolated Perceiver replay from checkpoint-53 (1 NPU, no card pin).
#   DRY_RUN=0 bash scripts/submit_perceiver_nan_isolate.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RPMEM_ROOT="$(dirname "$SCRIPT_DIR")"

DRY_RUN="${DRY_RUN:-1}"
PARTITION="${PARTITION:-pdgpu-sjtu-ai}"
TRAIN_IMAGE="${TRAIN_IMAGE:-hub.szaic.com/sjtu/sjtu_shenliping-sunwenjie-strategy-lora-ascend:v1.0.0-cann8.2rc1-torch2.6.0}"
GPUS=1
CPUS="${CPUS:-16}"
MEM="${MEM:-120G}"
JOB_NAME="${JOB_NAME:-rpmem_perceiver_nan_isolate}"

CONFIG="${CONFIG:-$RPMEM_ROOT/configs/compiler/qwen3_8b_npu_20k_1npu_nan_probe.yaml}"
TRAIN_DATA="${TRAIN_DATA:-$RPMEM_ROOT/data/rpmem/compiler_20k/train.jsonl}"
TEACHER_ROOT="${TEACHER_ROOT:-$RPMEM_ROOT/outputs/teacher_kvcache_20k}"
RESUME_CKPT="${RESUME_CKPT:-$RPMEM_ROOT/outputs/compiler_qwen3_8b_npu_20k_1npu_nan_probe_s53/checkpoint-53}"
OUTPUT_DIR="${OUTPUT_DIR:-$RPMEM_ROOT/outputs/compiler_qwen3_8b_npu_perceiver_isolate}"
LOG_NAME="${LOG_NAME:-compiler_perceiver_nan_isolate.log}"
NAN_PROBE_CAPTURE_SAMPLE="${NAN_PROBE_CAPTURE_SAMPLE:-16890}"
NAN_PROBE_START_STEP="${NAN_PROBE_START_STEP:-53}"
NAN_PROBE_ZERO_GRAD_BEFORE_CAPTURE="${NAN_PROBE_ZERO_GRAD_BEFORE_CAPTURE:-0}"

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
export RESUME_CKPT='${RESUME_CKPT}'
export LOG_NAME='${LOG_NAME}'
export NAN_PROBE_CAPTURE_SAMPLE='${NAN_PROBE_CAPTURE_SAMPLE}'
export NAN_PROBE_START_STEP='${NAN_PROBE_START_STEP}'
export NAN_PROBE_ZERO_GRAD_BEFORE_CAPTURE='${NAN_PROBE_ZERO_GRAD_BEFORE_CAPTURE}'
unset ASCEND_RT_VISIBLE_DEVICES
unset PYTHON
bash scripts/run_perceiver_nan_isolate.sh
EOF
)

echo "=== RPMem Perceiver NaN isolate submit ==="
echo "JOB=$JOB_NAME RESUME=$RESUME_CKPT OUT=$OUTPUT_DIR CAPTURE=$NAN_PROBE_CAPTURE_SAMPLE DRY_RUN=$DRY_RUN"

if [[ "$DRY_RUN" == "1" ]]; then
  echo "DRY_RUN=1 — not submitting"
  echo "$RUN_CMD"
  exit 0
fi

if [[ ! -d "$RESUME_CKPT" ]]; then
  echo "ERROR: RESUME_CKPT missing: $RESUME_CKPT"
  exit 1
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
