#!/usr/bin/env bash
# 2-NPU Fixed-FKL NaN repro. Relies on vc-assigned devices only; never pins card ids.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/env_rpmem_npu.sh"
bash "$SCRIPT_DIR/setup_container_deps.sh"

CONFIG="${CONFIG:-$RPMEM_ROOT/configs/compiler/qwen3_8b_npu_nan_2npu.yaml}"
ACCEL_CFG="${ACCEL_CFG:-$RPMEM_ROOT/configs/accelerate_2npu.yaml}"
TRAIN_DATA="${TRAIN_DATA:-$RPMEM_ROOT/data/rpmem/compiler_nan_repro/train_500.jsonl}"
TEACHER_ROOT="${TEACHER_ROOT:-$RPMEM_ROOT/outputs/teacher_kvcache_500}"
OUTPUT_DIR="${OUTPUT_DIR:-$RPMEM_ROOT/outputs/compiler_nan_repro_2npu}"
LOG_NAME="${LOG_NAME:-compiler_train_2npu_nan_repro.log}"
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
echo "NAN_REPRO 2NPU CONFIG=$CONFIG DATA=$TRAIN_DATA TEACHER=$TEACHER_DIR OUT=$OUTPUT_DIR NPU_COUNT=$NPU_COUNT"
echo "visible ASCEND_VISIBLE_DEVICES=${ASCEND_VISIBLE_DEVICES:-<unset>} ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-<unset>}"
export DEVICE_KIND=npu

"$PYTHON" -m accelerate.commands.launch \
  --config_file "$ACCEL_CFG" \
  --num_processes 2 \
  -m rpmem.training.train_hypernet \
  --config "$CONFIG" \
  --train_data "$TRAIN_DATA" \
  --output_dir "$OUTPUT_DIR" \
  --base_model_path "$RPMEM_BASE_MODEL" \
  --ctx_encoder_path "$RPMEM_CTX_ENCODER" \
  --teacher_logprobs_dir "$TEACHER_DIR" \
  --no-use_flash_attn \
  2>&1 | tee "$RPMEM_ROOT/logs/$LOG_NAME"

echo "NAN_REPRO DONE -> $OUTPUT_DIR"
