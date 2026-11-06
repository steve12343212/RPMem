#!/usr/bin/env bash
# Submit the optional context-KV teacher path (separate from the running baseline).
#   DRY_RUN=1 bash scripts/submit_teacher_kvcache_8npu.sh
#   DRY_RUN=0 bash scripts/submit_teacher_kvcache_8npu.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RPMEM_ROOT="$(dirname "$SCRIPT_DIR")"

DRY_RUN="${DRY_RUN:-1}"
PARTITION="${PARTITION:-pdgpu-sjtu-ai}"
TRAIN_IMAGE="${TRAIN_IMAGE:-hub.szaic.com/sjtu/sjtu_shenliping-sunwenjie-strategy-lora-ascend:v1.0.0-cann8.2rc1-torch2.6.0}"
GPUS="${GPUS:-8}"
CPUS="${CPUS:-64}"
MEM="${MEM:-640G}"
JOB_NAME="${JOB_NAME:-rpmem_teacher_kvcache_8npu}"
LOG_DIR="${LOG_DIR:-$RPMEM_ROOT/logs/vc}"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/${JOB_NAME}.JOB.log"

TEACHER_ROOT="${TEACHER_ROOT:-$RPMEM_ROOT/outputs/teacher_kvcache}"
CORPUS_ROOT="${CORPUS_ROOT:-$RPMEM_ROOT/data/rpmem/compiler}"

RUN_CMD=$(cat <<EOF
set -euo pipefail
cd '${RPMEM_ROOT}'
export TEACHER_ROOT='${TEACHER_ROOT}'
export CORPUS_ROOT='${CORPUS_ROOT}'
unset PYTHON
bash scripts/run_teacher_precompute_8npu_kvcache.sh
EOF
)

echo "=== RPMem teacher KV-cache 8-NPU vc submit ==="
echo "JOB=$JOB_NAME OUT=$TEACHER_ROOT DRY_RUN=$DRY_RUN"

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
