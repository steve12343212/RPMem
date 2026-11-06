#!/usr/bin/env bash
# 1-NPU Fixed-FKL smoke (<~20 min): reuse existing 20k teacher cache.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/env_rpmem_npu.sh"
bash "$SCRIPT_DIR/setup_container_deps.sh"

CONFIG="${CONFIG:-$RPMEM_ROOT/configs/compiler/qwen3_8b_npu_smoke.yaml}"
ACCEL_CFG="${ACCEL_CFG:-$RPMEM_ROOT/configs/accelerate_1npu.yaml}"
TRAIN_DATA="${TRAIN_DATA:-$RPMEM_ROOT/data/rpmem/compiler_20k/train.jsonl}"
TEACHER_ROOT="${TEACHER_ROOT:-$RPMEM_ROOT/outputs/teacher_kvcache_20k}"
OUTPUT_DIR="${OUTPUT_DIR:-$RPMEM_ROOT/outputs/compiler_qwen3_8b_npu_smoke}"
PREFLIGHT_STOP_AFTER_STEP="${PREFLIGHT_STOP_AFTER_STEP:-2}"
mkdir -p "$OUTPUT_DIR" "$RPMEM_ROOT/logs"

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

echo "SMOKE 1NPU CONFIG=$CONFIG TRAIN_DATA=$TRAIN_DATA TEACHER=$TEACHER_DIR OUT=$OUTPUT_DIR"
export DEVICE_KIND=npu

EXTRA_ARGS=()
if [[ -n "$PREFLIGHT_STOP_AFTER_STEP" ]]; then
  EXTRA_ARGS+=(--preflight_stop_after_step "$PREFLIGHT_STOP_AFTER_STEP")
fi

"$PYTHON" -m accelerate.commands.launch \
  --config_file "$ACCEL_CFG" \
  --num_processes 1 \
  -m rpmem.training.train_hypernet \
  --config "$CONFIG" \
  --train_data "$TRAIN_DATA" \
  --output_dir "$OUTPUT_DIR" \
  --base_model_path "$RPMEM_BASE_MODEL" \
  --ctx_encoder_path "$RPMEM_CTX_ENCODER" \
  --teacher_logprobs_dir "$TEACHER_DIR" \
  --no-use_flash_attn \
  "${EXTRA_ARGS[@]}" \
  2>&1 | tee "$RPMEM_ROOT/logs/compiler_train_1npu_smoke.log"

echo "SMOKE DONE -> $OUTPUT_DIR"
