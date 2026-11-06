#!/usr/bin/env bash
# Generate Experiment-B evaluation probes on 1 Ascend NPU.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/env_rpmem_npu.sh"
bash "$SCRIPT_DIR/setup_container_deps.sh"

SPLIT_DIR="${SPLIT_DIR:-$RPMEM_ROOT/data/rpmem/compiler_20k_event_split}"
OUTPUT_DIR="${OUTPUT_DIR:-$SPLIT_DIR/experiment_b}"
MODEL_PATH="${MODEL_PATH:-/aistor/sjtu/hpc_stor01/home/sunwenjie/workshop/models/Qwen2.5-7B-Instruct}"
MAX_SESSIONS="${MAX_SESSIONS:-500}"
SEED="${SEED:-42}"

mkdir -p "$OUTPUT_DIR" "$RPMEM_ROOT/logs"
export DEVICE_KIND=npu

"$PYTHON" -m rpmem.training.corpus.generate_experiment_b \
  --split_dir "$SPLIT_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --model_path "$MODEL_PATH" \
  --max_sessions "$MAX_SESSIONS" \
  --seed "$SEED" \
  --require_unseen \
  2>&1 | tee "$RPMEM_ROOT/logs/experiment_b_generate.log"

echo "DONE -> $OUTPUT_DIR"
