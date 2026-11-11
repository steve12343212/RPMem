#!/usr/bin/env bash
# Submit Ascend explicit-encoder-cross-attn regression (1 NPU, no card pin).
#   DRY_RUN=0 bash scripts/submit_perceiver_explicit_attn_validate.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RPMEM_ROOT="$(dirname "$SCRIPT_DIR")"

DRY_RUN="${DRY_RUN:-1}"
PARTITION="${PARTITION:-pdgpu-sjtu-ai}"
TRAIN_IMAGE="${TRAIN_IMAGE:-hub.szaic.com/sjtu/sjtu_shenliping-sunwenjie-strategy-lora-ascend:v1.0.0-cann8.2rc1-torch2.6.0}"
GPUS=1
CPUS="${CPUS:-16}"
MEM="${MEM:-120G}"
JOB_NAME="${JOB_NAME:-rpmem_perceiver_explicit_attn_validate}"

RESUME_CKPT="${RESUME_CKPT:-$RPMEM_ROOT/outputs/compiler_qwen3_8b_npu_20k_1npu_nan_probe_s53/checkpoint-53}"
DUMP="${DUMP:-$RPMEM_ROOT/experiments/ascend_bf16_nan/nan_probe_perceiver_isolate/sample_16890_perceiver.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-$RPMEM_ROOT/outputs/compiler_qwen3_8b_npu_perceiver_explicit_attn_validate}"
LOG_NAME="${LOG_NAME:-compiler_perceiver_explicit_attn_validate.log}"
REPEATS="${REPEATS:-20}"

LOG_DIR="${LOG_DIR:-$RPMEM_ROOT/logs/vc}"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/${JOB_NAME}.JOB.log"

RUN_CMD=$(cat <<EOF
set -euo pipefail
cd '${RPMEM_ROOT}'
export RESUME_CKPT='${RESUME_CKPT}'
export DUMP='${DUMP}'
export OUTPUT_DIR='${OUTPUT_DIR}'
export LOG_NAME='${LOG_NAME}'
export REPEATS='${REPEATS}'
export RPMEM_PERCEIVER_EXPLICIT_ENCODER_CROSS_ATTN=1
unset ASCEND_RT_VISIBLE_DEVICES
unset PYTHON
bash scripts/run_perceiver_explicit_attn_validate.sh
EOF
)

echo "=== RPMem explicit encoder cross-attn validate ==="
echo "JOB=$JOB_NAME OUT=$OUTPUT_DIR REPEATS=$REPEATS DRY_RUN=$DRY_RUN"

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
