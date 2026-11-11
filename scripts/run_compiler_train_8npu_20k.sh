#!/usr/bin/env bash
# Train Fixed-FKL compiler on the 20k subset (expects teacher already built).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/env_rpmem_npu.sh"
bash "$SCRIPT_DIR/setup_container_deps.sh"

CONFIG="${CONFIG:-$RPMEM_ROOT/configs/compiler/qwen3_8b_npu_20k.yaml}"
ACCEL_CFG="${ACCEL_CFG:-$RPMEM_ROOT/configs/accelerate_8npu.yaml}"
TRAIN_DATA="${TRAIN_DATA:-$RPMEM_ROOT/data/rpmem/compiler_20k/train.jsonl}"
TEACHER_ROOT="${TEACHER_ROOT:-$RPMEM_ROOT/outputs/teacher_kvcache_20k}"
OUTPUT_DIR="${OUTPUT_DIR:-$RPMEM_ROOT/outputs/compiler_qwen3_8b_npu_20k}"
LOG_NAME="${LOG_NAME:-compiler_train_8npu_20k.log}"
mkdir -p "$OUTPUT_DIR" "$RPMEM_ROOT/logs"

# Resolve teacher dir: prefer TEACHER_ROOT itself, else TEACHER_ROOT/train.
TEACHER_DIR="$TEACHER_ROOT"
if [[ ! -f "$TEACHER_DIR/meta.json" ]] && [[ -f "$TEACHER_ROOT/train/meta.json" ]]; then
  TEACHER_DIR="$TEACHER_ROOT/train"
fi
if [[ ! -f "$TEACHER_DIR/meta.json" ]]; then
  echo "ERROR: teacher cache missing at $TEACHER_ROOT (no meta.json)"
  exit 1
fi

NPU_COUNT="$("$PYTHON" - <<'PY'
from rpmem.device import device_count
print(device_count())
PY
)"
echo "NPU_COUNT=$NPU_COUNT CONFIG=$CONFIG TRAIN_DATA=$TRAIN_DATA TEACHER=$TEACHER_DIR"
if [[ "$NPU_COUNT" -lt 8 ]]; then
  echo "WARNING: expected 8 NPUs, visible=$NPU_COUNT"
fi

export DEVICE_KIND=npu
# Ascend encoder cross-attn fix (default on when unset).
export RPMEM_PERCEIVER_EXPLICIT_ENCODER_CROSS_ATTN="${RPMEM_PERCEIVER_EXPLICIT_ENCODER_CROSS_ATTN:-1}"

EXTRA_ARGS=()
if [[ "${NAN_PROBE:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--nan_probe)
  EXTRA_ARGS+=(--nan_probe_start_step "${NAN_PROBE_START_STEP:-0}")
  if [[ -n "${NAN_PROBE_JSONL:-}" ]]; then
    EXTRA_ARGS+=(--nan_probe_jsonl "$NAN_PROBE_JSONL")
  fi
fi

echo "EXPLICIT_ENCODER_CROSS_ATTN=$RPMEM_PERCEIVER_EXPLICIT_ENCODER_CROSS_ATTN NAN_PROBE=${NAN_PROBE:-0} START=${NAN_PROBE_START_STEP:-0}"

"$PYTHON" -m accelerate.commands.launch \
  --config_file "$ACCEL_CFG" \
  -m rpmem.training.train_hypernet \
  --config "$CONFIG" \
  --train_data "$TRAIN_DATA" \
  --output_dir "$OUTPUT_DIR" \
  --base_model_path "$RPMEM_BASE_MODEL" \
  --ctx_encoder_path "$RPMEM_CTX_ENCODER" \
  --teacher_logprobs_dir "$TEACHER_DIR" \
  --no-use_flash_attn \
  "${EXTRA_ARGS[@]}" \
  2>&1 | tee "$RPMEM_ROOT/logs/$LOG_NAME"

echo "DONE -> $OUTPUT_DIR"
