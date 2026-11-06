#!/usr/bin/env bash
# Submit 8-NPU compiler training via vc.
#   DRY_RUN=1 bash scripts/submit_compiler_train_8npu.sh
#   DRY_RUN=0 bash scripts/submit_compiler_train_8npu.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RPMEM_ROOT="$(dirname "$SCRIPT_DIR")"

DRY_RUN="${DRY_RUN:-1}"
PARTITION="${PARTITION:-pdgpu-sjtu-ai}"
TRAIN_IMAGE="${TRAIN_IMAGE:-hub.szaic.com/sjtu/sjtu_shenliping-sunwenjie-strategy-lora-ascend:v1.0.0-cann8.2rc1-torch2.6.0}"
GPUS="${GPUS:-8}"
CPUS="${CPUS:-64}"
MEM="${MEM:-640G}"
JOB_NAME="${JOB_NAME:-rpmem_compiler_8npu}"
LOG_DIR="${LOG_DIR:-$RPMEM_ROOT/logs/vc}"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/${JOB_NAME}.JOB.log"

CORPUS_ROOT="${CORPUS_ROOT:-$RPMEM_ROOT/data/rpmem/compiler}"
TEACHER_ROOT="${TEACHER_ROOT:-$RPMEM_ROOT/outputs/teacher}"
OUTPUT_DIR="${OUTPUT_DIR:-$RPMEM_ROOT/outputs/compiler_qwen3_8b_npu}"
CONFIG="${CONFIG:-$RPMEM_ROOT/configs/compiler/qwen3_8b_npu.yaml}"
# offline_fkl needs teacher cache; default is to build it in-job when missing.
RUN_TEACHER="${RUN_TEACHER:-auto}"

RUN_CMD=$(cat <<EOF
set -euo pipefail
cd '${RPMEM_ROOT}'
export CORPUS_ROOT='${CORPUS_ROOT}'
export TEACHER_ROOT='${TEACHER_ROOT}'
export OUTPUT_DIR='${OUTPUT_DIR}'
export CONFIG='${CONFIG}'
unset PYTHON
if [[ '${RUN_TEACHER}' == '1' ]] || { [[ '${RUN_TEACHER}' == 'auto' ]] && [[ ! -d '${TEACHER_ROOT}/train' ]]; }; then
  echo '[pipeline] teacher precompute -> ${TEACHER_ROOT}'
  # Do not leak compiler OUTPUT_DIR into the teacher step.
  OUTPUT_DIR='${TEACHER_ROOT}' TEACHER_ROOT='${TEACHER_ROOT}' \
    bash scripts/run_teacher_precompute_8npu.sh
fi
echo '[pipeline] compiler train -> ${OUTPUT_DIR}'
bash scripts/run_compiler_train_8npu.sh
EOF
)

echo "=== RPMem compiler 8-NPU vc submit ==="
echo "JOB=$JOB_NAME GPUS=$GPUS CONFIG=$CONFIG CORPUS=$CORPUS_ROOT OUT=$OUTPUT_DIR DRY_RUN=$DRY_RUN"

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
