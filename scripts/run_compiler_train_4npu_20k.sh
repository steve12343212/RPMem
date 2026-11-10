#!/usr/bin/env bash
# 4-NPU Fixed-FKL train on 20k (expects teacher already built). No card pinning.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/env_rpmem_npu.sh"
bash "$SCRIPT_DIR/setup_container_deps.sh"

CONFIG="${CONFIG:-$RPMEM_ROOT/configs/compiler/qwen3_8b_npu_20k_4npu_accum4.yaml}"
ACCEL_CFG="${ACCEL_CFG:-$RPMEM_ROOT/configs/accelerate_4npu.yaml}"
TRAIN_DATA="${TRAIN_DATA:-$RPMEM_ROOT/data/rpmem/compiler_20k/train.jsonl}"
TEACHER_ROOT="${TEACHER_ROOT:-$RPMEM_ROOT/outputs/teacher_kvcache_20k}"
OUTPUT_DIR="${OUTPUT_DIR:-$RPMEM_ROOT/outputs/compiler_qwen3_8b_npu_20k_4npu_accum4}"
LOG_NAME="${LOG_NAME:-compiler_train_4npu_20k_accum4.log}"
mkdir -p "$OUTPUT_DIR" "$RPMEM_ROOT/logs"

unset ASCEND_RT_VISIBLE_DEVICES 2>/dev/null || true

TEACHER_DIR="$TEACHER_ROOT"
if [[ ! -f "$TEACHER_DIR/meta.json" ]] && [[ -f "$TEACHER_ROOT/train/meta.json" ]]; then
  TEACHER_DIR="$TEACHER_ROOT/train"
fi
if [[ ! -f "$TEACHER_DIR/meta.json" ]]; then
  echo "ERROR: teacher cache missing at $TEACHER_ROOT"
  exit 1
fi
if [[ ! -f "$TRAIN_DATA" ]]; then
  echo "ERROR: TRAIN_DATA missing: $TRAIN_DATA"
  exit 1
fi

NPU_COUNT="$("$PYTHON" - <<'PY'
from rpmem.device import device_count
print(device_count())
PY
)"
echo "TRAIN 4NPU CONFIG=$CONFIG DATA=$TRAIN_DATA TEACHER=$TEACHER_DIR OUT=$OUTPUT_DIR NPU_COUNT=$NPU_COUNT"
echo "visible ASCEND_VISIBLE_DEVICES=${ASCEND_VISIBLE_DEVICES:-<unset>} ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-<unset>}"
export DEVICE_KIND=npu

"$PYTHON" -m accelerate.commands.launch \
  --config_file "$ACCEL_CFG" \
  --num_processes 4 \
  -m rpmem.training.train_hypernet \
  --config "$CONFIG" \
  --train_data "$TRAIN_DATA" \
  --output_dir "$OUTPUT_DIR" \
  --base_model_path "$RPMEM_BASE_MODEL" \
  --ctx_encoder_path "$RPMEM_CTX_ENCODER" \
  --teacher_logprobs_dir "$TEACHER_DIR" \
  --no-use_flash_attn \
  2>&1 | tee "$RPMEM_ROOT/logs/$LOG_NAME"

echo "DONE -> $OUTPUT_DIR"
