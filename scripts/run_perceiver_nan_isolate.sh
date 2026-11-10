#!/usr/bin/env bash
# Resume checkpoint-53 → capture sample 16890 dump → isolated Perceiver replay.
# Relies on vc-assigned devices only; never pins card ids.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/env_rpmem_npu.sh"
bash "$SCRIPT_DIR/setup_container_deps.sh"

CONFIG="${CONFIG:?CONFIG required}"
ACCEL_CFG="${ACCEL_CFG:-$RPMEM_ROOT/configs/accelerate_1npu.yaml}"
TRAIN_DATA="${TRAIN_DATA:-$RPMEM_ROOT/data/rpmem/compiler_20k/train.jsonl}"
TEACHER_ROOT="${TEACHER_ROOT:-$RPMEM_ROOT/outputs/teacher_kvcache_20k}"
OUTPUT_DIR="${OUTPUT_DIR:?OUTPUT_DIR required}"
RESUME_CKPT="${RESUME_CKPT:?RESUME_CKPT required}"
LOG_NAME="${LOG_NAME:-compiler_perceiver_nan_isolate.log}"
CAPTURE_SAMPLE="${NAN_PROBE_CAPTURE_SAMPLE:-16890}"
NAN_PROBE_START_STEP="${NAN_PROBE_START_STEP:-53}"
ZERO_GRAD_BEFORE_CAPTURE="${NAN_PROBE_ZERO_GRAD_BEFORE_CAPTURE:-0}"
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
if [[ ! -d "$RESUME_CKPT" ]]; then
  echo "ERROR: RESUME_CKPT missing: $RESUME_CKPT"
  exit 1
fi

echo "PERCEIVER_ISOLATE CONFIG=$CONFIG RESUME=$RESUME_CKPT OUT=$OUTPUT_DIR CAPTURE=$CAPTURE_SAMPLE"
echo "visible ASCEND_VISIBLE_DEVICES=${ASCEND_VISIBLE_DEVICES:-<unset>}"
export DEVICE_KIND=npu

CAPTURE_ARGS=(--nan_probe --nan_probe_start_step "$NAN_PROBE_START_STEP" --nan_probe_capture_sample "$CAPTURE_SAMPLE")
if [[ "$ZERO_GRAD_BEFORE_CAPTURE" == "1" ]]; then
  CAPTURE_ARGS+=(--nan_probe_zero_grad_before_capture)
else
  CAPTURE_ARGS+=(--no-nan_probe_zero_grad_before_capture)
fi

set +e
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
  --resume_from_checkpoint "$RESUME_CKPT" \
  --no-use_flash_attn \
  "${CAPTURE_ARGS[@]}" \
  2>&1 | tee "$RPMEM_ROOT/logs/$LOG_NAME"
train_rc=${PIPESTATUS[0]}
set -e

DUMP="$OUTPUT_DIR/nan_probe_dump/sample_${CAPTURE_SAMPLE}_perceiver.pt"
if [[ ! -f "$DUMP" ]]; then
  echo "ERROR: expected dump missing after capture run: $DUMP (train_rc=$train_rc)"
  exit 1
fi

echo "=== Isolated Perceiver replay ==="
"$PYTHON" -m rpmem.training.replay_perceiver_nan \
  --checkpoint "$RESUME_CKPT" \
  --dump "$DUMP" \
  --output_json "$OUTPUT_DIR/nan_probe_dump/perceiver_replay.json" \
  2>&1 | tee -a "$RPMEM_ROOT/logs/$LOG_NAME"

echo "PERCEIVER_ISOLATE DONE -> $OUTPUT_DIR (capture train_rc=$train_rc)"
