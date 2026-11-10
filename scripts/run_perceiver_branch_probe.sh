#!/usr/bin/env bash
# Experiment 1: per-block encoder branch probe from checkpoint-53 + sample dump.
# Relies on vc-assigned devices only; never pins card ids.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/env_rpmem_npu.sh"
bash "$SCRIPT_DIR/setup_container_deps.sh"

RESUME_CKPT="${RESUME_CKPT:?RESUME_CKPT required}"
DUMP="${DUMP:?DUMP required}"
OUTPUT_DIR="${OUTPUT_DIR:?OUTPUT_DIR required}"
LOG_NAME="${LOG_NAME:-compiler_perceiver_branch_probe.log}"
REPEATS="${REPEATS:-3}"
mkdir -p "$OUTPUT_DIR" "$RPMEM_ROOT/logs"

unset ASCEND_RT_VISIBLE_DEVICES 2>/dev/null || true
export DEVICE_KIND=npu

if [[ ! -d "$RESUME_CKPT" && ! -f "$RESUME_CKPT" ]]; then
  echo "ERROR: RESUME_CKPT missing: $RESUME_CKPT"
  exit 1
fi
if [[ ! -f "$DUMP" ]]; then
  echo "ERROR: DUMP missing: $DUMP"
  exit 1
fi

echo "BRANCH_PROBE CKPT=$RESUME_CKPT DUMP=$DUMP OUT=$OUTPUT_DIR REPEATS=$REPEATS"
echo "visible ASCEND_VISIBLE_DEVICES=${ASCEND_VISIBLE_DEVICES:-<unset>}"

"$PYTHON" -m rpmem.training.replay_perceiver_nan \
  --checkpoint "$RESUME_CKPT" \
  --dump "$DUMP" \
  --output_json "$OUTPUT_DIR/perceiver_branch_probe.json" \
  --branch_probe \
  --repeats "$REPEATS" \
  2>&1 | tee "$RPMEM_ROOT/logs/$LOG_NAME"

echo "BRANCH_PROBE DONE -> $OUTPUT_DIR"
