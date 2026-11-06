#!/usr/bin/env bash
# Submit 8-NPU pipeline: KV-cache teacher (20k) -> Fixed-FKL train (20k).
#   DRY_RUN=0 bash scripts/submit_compiler_train_8npu_kvcache_20k.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RPMEM_ROOT="$(dirname "$SCRIPT_DIR")"

DRY_RUN="${DRY_RUN:-1}"
PARTITION="${PARTITION:-pdgpu-sjtu-ai}"
TRAIN_IMAGE="${TRAIN_IMAGE:-hub.szaic.com/sjtu/sjtu_shenliping-sunwenjie-strategy-lora-ascend:v1.0.0-cann8.2rc1-torch2.6.0}"
GPUS="${GPUS:-8}"
CPUS="${CPUS:-64}"
MEM="${MEM:-640G}"
JOB_NAME="${JOB_NAME:-rpmem_kv_20k_8npu}"
LOG_DIR="${LOG_DIR:-$RPMEM_ROOT/logs/vc}"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/${JOB_NAME}.JOB.log"

TRAIN_DATA="${TRAIN_DATA:-$RPMEM_ROOT/data/rpmem/compiler_20k/train.jsonl}"
TEACHER_ROOT="${TEACHER_ROOT:-$RPMEM_ROOT/outputs/teacher_kvcache_20k}"
OUTPUT_DIR="${OUTPUT_DIR:-$RPMEM_ROOT/outputs/compiler_qwen3_8b_npu_20k}"
CONFIG="${CONFIG:-$RPMEM_ROOT/configs/compiler/qwen3_8b_npu_20k.yaml}"
RUN_TEACHER="${RUN_TEACHER:-1}"

if [[ ! -f "$TRAIN_DATA" ]]; then
  echo "ERROR: TRAIN_DATA missing: $TRAIN_DATA"
  echo "Export first: python -m rpmem.training.corpus.export_subset --inputs data/rpmem/compiler/train.corpus.json --output data/rpmem/compiler_20k/train.jsonl --max_samples 20000"
  exit 1
fi

RUN_CMD=$(cat <<EOF
set -euo pipefail
cd '${RPMEM_ROOT}'
export TRAIN_DATA='${TRAIN_DATA}'
export TEACHER_ROOT='${TEACHER_ROOT}'
export OUTPUT_DIR='${OUTPUT_DIR}'
export CONFIG='${CONFIG}'
unset PYTHON
if [[ '${RUN_TEACHER}' == '1' ]]; then
  echo '[pipeline] KV teacher 20k -> ${TEACHER_ROOT}'
  # Keep teacher OUTPUT_DIR isolated from compiler checkpoints.
  OUTPUT_DIR='${TEACHER_ROOT}' TEACHER_ROOT='${TEACHER_ROOT}' TRAIN_DATA='${TRAIN_DATA}' \
    bash scripts/run_teacher_precompute_8npu_kvcache_20k.sh
fi
echo '[pipeline] compiler train 20k -> ${OUTPUT_DIR}'
TRAIN_DATA='${TRAIN_DATA}' TEACHER_ROOT='${TEACHER_ROOT}' OUTPUT_DIR='${OUTPUT_DIR}' CONFIG='${CONFIG}' \
  bash scripts/run_compiler_train_8npu_20k.sh
EOF
)

echo "=== RPMem KV-cache 20k teacher+train 8-NPU vc submit ==="
echo "JOB=$JOB_NAME GPUS=$GPUS CONFIG=$CONFIG"
echo "TRAIN_DATA=$TRAIN_DATA"
echo "TEACHER=$TEACHER_ROOT OUT=$OUTPUT_DIR DRY_RUN=$DRY_RUN RUN_TEACHER=$RUN_TEACHER"

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
