#!/usr/bin/env bash
# Experiment 2: capture a failing block SDPA I/O dump, then BF16/FP32 backend对照.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/env_rpmem_npu.sh"
bash "$SCRIPT_DIR/setup_container_deps.sh"

RESUME_CKPT="${RESUME_CKPT:?RESUME_CKPT required}"
DUMP="${DUMP:?DUMP required}"
OUTPUT_DIR="${OUTPUT_DIR:?OUTPUT_DIR required}"
LOG_NAME="${LOG_NAME:-compiler_perceiver_sdpa_compare.log}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-20}"
mkdir -p "$OUTPUT_DIR" "$RPMEM_ROOT/logs"

unset ASCEND_RT_VISIBLE_DEVICES 2>/dev/null || true
export DEVICE_KIND=npu

echo "SDPA_COMPARE CKPT=$RESUME_CKPT DUMP=$DUMP OUT=$OUTPUT_DIR MAX_ATTEMPTS=$MAX_ATTEMPTS"
echo "visible ASCEND_VISIBLE_DEVICES=${ASCEND_VISIBLE_DEVICES:-<unset>}"

CAPTURE_JSON="$OUTPUT_DIR/perceiver_until_anomalous.json"
set +e
"$PYTHON" -m rpmem.training.replay_perceiver_nan \
  --checkpoint "$RESUME_CKPT" \
  --dump "$DUMP" \
  --output_json "$CAPTURE_JSON" \
  --branch_probe \
  --until_anomalous \
  --max_attempts "$MAX_ATTEMPTS" \
  2>&1 | tee "$RPMEM_ROOT/logs/$LOG_NAME"
capture_rc=${PIPESTATUS[0]}
set -e

BAD_DUMP=$(ls -1t "$OUTPUT_DIR"/first_bad_block_repeat*.pt 2>/dev/null | head -1 || true)
if [[ -z "${BAD_DUMP}" ]]; then
  echo "ERROR: no first_bad_block dump after $MAX_ATTEMPTS attempts (capture_rc=$capture_rc)"
  exit 1
fi
echo "Using failing block dump: $BAD_DUMP"

"$PYTHON" -m rpmem.training.compare_sdpa_backends \
  --dump "$BAD_DUMP" \
  --output_json "$OUTPUT_DIR/sdpa_backend_compare.json" \
  --stress_repeats "${STRESS_REPEATS:-20}" \
  2>&1 | tee -a "$RPMEM_ROOT/logs/$LOG_NAME"

echo "SDPA_COMPARE DONE -> $OUTPUT_DIR"
