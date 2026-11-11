#!/usr/bin/env bash
# 1-NPU Fixed-FKL NaN repro. Relies on vc-assigned devices only; never pins card ids.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/env_rpmem_npu.sh"
bash "$SCRIPT_DIR/setup_container_deps.sh"

CONFIG="${CONFIG:?CONFIG required}"
ACCEL_CFG="${ACCEL_CFG:-$RPMEM_ROOT/configs/accelerate_1npu.yaml}"
TRAIN_DATA="${TRAIN_DATA:-$RPMEM_ROOT/data/rpmem/compiler_nan_repro/train_500.jsonl}"
TEACHER_ROOT="${TEACHER_ROOT:-$RPMEM_ROOT/outputs/teacher_kvcache_500}"
OUTPUT_DIR="${OUTPUT_DIR:?OUTPUT_DIR required}"
LOG_NAME="${LOG_NAME:-compiler_train_1npu_nan_repro.log}"
mkdir -p "$OUTPUT_DIR" "$RPMEM_ROOT/logs"

# Do not set ASCEND_RT_VISIBLE_DEVICES. vc provides ASCEND_VISIBLE_DEVICES.
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

echo "NAN_REPRO 1NPU CONFIG=$CONFIG DATA=$TRAIN_DATA TEACHER=$TEACHER_DIR OUT=$OUTPUT_DIR"
echo "visible ASCEND_VISIBLE_DEVICES=${ASCEND_VISIBLE_DEVICES:-<unset>} ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-<unset>}"
echo "RPMEM_PERCEIVER_EXPLICIT_ENCODER_CROSS_ATTN=${RPMEM_PERCEIVER_EXPLICIT_ENCODER_CROSS_ATTN:-<unset>}"
export DEVICE_KIND=npu

EXTRA_ARGS=()
if [[ "${NAN_PROBE:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--nan_probe)
  EXTRA_ARGS+=(--nan_probe_start_step "${NAN_PROBE_START_STEP:-430}")
  if [[ -n "${NAN_PROBE_JSONL:-}" ]]; then
    EXTRA_ARGS+=(--nan_probe_jsonl "$NAN_PROBE_JSONL")
  fi
fi
if [[ -n "${PREFLIGHT_STOP_AFTER_STEP:-}" ]]; then
  EXTRA_ARGS+=(--preflight_stop_after_step "${PREFLIGHT_STOP_AFTER_STEP}")
fi
if [[ -n "${SELECTED_ENDPOINT_STEP:-}" ]]; then
  EXTRA_ARGS+=(--selected_endpoint_step "${SELECTED_ENDPOINT_STEP}")
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
  2>&1 | tee "$RPMEM_ROOT/logs/$LOG_NAME"

echo "NAN_REPRO DONE -> $OUTPUT_DIR"
