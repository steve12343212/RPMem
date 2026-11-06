#!/usr/bin/env bash
# Train the Qwen3-8B session compiler on 8 Ascend NPUs.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/env_rpmem_npu.sh"
bash "$SCRIPT_DIR/setup_container_deps.sh"

CONFIG="${CONFIG:-$RPMEM_ROOT/configs/compiler/qwen3_8b_npu.yaml}"
ACCEL_CFG="${ACCEL_CFG:-$RPMEM_ROOT/configs/accelerate_8npu.yaml}"
CORPUS_ROOT="${CORPUS_ROOT:-$RPMEM_DATA_ROOT/compiler}"
TEACHER_ROOT="${TEACHER_ROOT:-$RPMEM_ROOT/outputs/teacher}"
OUTPUT_DIR="${OUTPUT_DIR:-$RPMEM_ROOT/outputs/compiler_qwen3_8b_npu}"
mkdir -p "$OUTPUT_DIR" "$RPMEM_ROOT/logs"

# Recover if an earlier pipeline step wrote teacher shards into OUTPUT_DIR
# (OUTPUT_DIR was shared with the compiler checkpoint path).
if [[ ! -d "$TEACHER_ROOT/train" ]] && [[ -f "$OUTPUT_DIR/meta.json" ]] \
  && grep -q 'teacher_topk' "$OUTPUT_DIR/meta.json" 2>/dev/null; then
  echo "[recover] moving teacher cache $OUTPUT_DIR -> $TEACHER_ROOT"
  mkdir -p "$TEACHER_ROOT"
  # Move teacher artifacts out so OUTPUT_DIR can hold compiler checkpoints.
  shopt -s nullglob
  for item in "$OUTPUT_DIR"/*; do
    mv "$item" "$TEACHER_ROOT"/
  done
  shopt -u nullglob
fi

NPU_COUNT="$("$PYTHON" - <<'PY'
from rpmem.device import device_count
print(device_count())
PY
)"
echo "NPU_COUNT=$NPU_COUNT CONFIG=$CONFIG CORPUS_ROOT=$CORPUS_ROOT"
if [[ "$NPU_COUNT" -lt 8 ]]; then
  echo "WARNING: expected 8 NPUs, visible=$NPU_COUNT — adjust accelerate num_processes if needed"
fi

TRAIN_ARGS=(
  --config "$CONFIG"
  --train_data "$CORPUS_ROOT/train.corpus.json"
  --output_dir "$OUTPUT_DIR"
  --base_model_path "$RPMEM_BASE_MODEL"
  --ctx_encoder_path "$RPMEM_CTX_ENCODER"
  --no-use_flash_attn
)
if [[ -n "$TEACHER_ROOT" ]]; then
  TRAIN_ARGS+=(--teacher_logprobs_dir "$TEACHER_ROOT/train")
fi
if [[ -n "${VAL_DATA:-}" ]]; then
  TRAIN_ARGS+=(--val_data "$VAL_DATA")
fi
if [[ -n "${VAL_TEACHER_ROOT:-}" ]]; then
  TRAIN_ARGS+=(--val_teacher_logprobs_dir "$VAL_TEACHER_ROOT")
fi

export DEVICE_KIND=npu
"$PYTHON" -m accelerate.commands.launch \
  --config_file "$ACCEL_CFG" \
  -m rpmem.training.train_hypernet \
  "${TRAIN_ARGS[@]}" \
  2>&1 | tee "$RPMEM_ROOT/logs/compiler_train_8npu.log"

echo "DONE -> $OUTPUT_DIR"
