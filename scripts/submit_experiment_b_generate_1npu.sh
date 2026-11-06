#!/usr/bin/env bash
# Submit Experiment-B LLM probe generation (1 NPU).
#   DRY_RUN=0 MAX_SESSIONS=500 bash scripts/submit_experiment_b_generate_1npu.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RPMEM_ROOT="$(dirname "$SCRIPT_DIR")"

DRY_RUN="${DRY_RUN:-1}"
PARTITION="${PARTITION:-pdgpu-sjtu-ai}"
TRAIN_IMAGE="${TRAIN_IMAGE:-hub.szaic.com/sjtu/sjtu_shenliping-sunwenjie-strategy-lora-ascend:v1.0.0-cann8.2rc1-torch2.6.0}"
GPUS="${GPUS:-1}"
CPUS="${CPUS:-16}"
MEM="${MEM:-128G}"
JOB_NAME="${JOB_NAME:-rpmem_expB_gen_1npu}"
LOG_DIR="${LOG_DIR:-$RPMEM_ROOT/logs/vc}"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/${JOB_NAME}.JOB.log"

SPLIT_DIR="${SPLIT_DIR:-$RPMEM_ROOT/data/rpmem/compiler_20k_event_split}"
OUTPUT_DIR="${OUTPUT_DIR:-$SPLIT_DIR/experiment_b}"
MODEL_PATH="${MODEL_PATH:-/aistor/sjtu/hpc_stor01/home/sunwenjie/workshop/models/Qwen2.5-7B-Instruct}"
MAX_SESSIONS="${MAX_SESSIONS:-500}"

RUN_CMD=$(cat <<EOF
set -euo pipefail
cd '${RPMEM_ROOT}'
export SPLIT_DIR='${SPLIT_DIR}'
export OUTPUT_DIR='${OUTPUT_DIR}'
export MODEL_PATH='${MODEL_PATH}'
export MAX_SESSIONS='${MAX_SESSIONS}'
unset PYTHON
bash scripts/run_experiment_b_generate_1npu.sh
EOF
)

echo "=== Experiment B generate 1-NPU submit ==="
echo "JOB=$JOB_NAME MAX_SESSIONS=$MAX_SESSIONS OUT=$OUTPUT_DIR DRY_RUN=$DRY_RUN"

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
