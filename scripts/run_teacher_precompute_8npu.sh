#!/usr/bin/env bash
# Shard teacher logprob precompute across 8 NPUs (one process per card).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/env_rpmem_npu.sh"
bash "$SCRIPT_DIR/setup_container_deps.sh"

CORPUS_ROOT="${CORPUS_ROOT:-$RPMEM_DATA_ROOT/compiler}"
# Prefer TEACHER_ROOT so a shared OUTPUT_DIR (compiler ckpt path) cannot clobber this.
OUTPUT_DIR="${TEACHER_ROOT:-${TEACHER_OUTPUT_DIR:-$RPMEM_ROOT/outputs/teacher}}"
NUM_SHARDS="${NUM_SHARDS:-8}"
mkdir -p "$OUTPUT_DIR" "$RPMEM_ROOT/logs"

# One visible card per shard process (logical npu:0 inside each worker).
# Intentionally pin ASCEND_RT_VISIBLE_DEVICES here; leave unset for DDP training.

NPU_COUNT="$("$PYTHON" - <<'PY'
from rpmem.device import device_count
print(device_count())
PY
)"
if [[ "$NUM_SHARDS" -gt "$NPU_COUNT" ]]; then
  NUM_SHARDS="$NPU_COUNT"
fi
echo "teacher precompute shards=$NUM_SHARDS corpus=$CORPUS_ROOT out=$OUTPUT_DIR"

# Do not set ASCEND_RT_VISIBLE_DEVICES; pin via CUDA/NPU device index inside each shard.
unset ASCEND_RT_VISIBLE_DEVICES 2>/dev/null || true
PIDS=()
for ((s=0; s<NUM_SHARDS; s++)); do
  (
    export ASCEND_RT_VISIBLE_DEVICES="$s"
    export DEVICE_KIND=npu
    "$PYTHON" -m rpmem.training.precompute_teacher \
      --base_model_path "$RPMEM_BASE_MODEL" \
      --train_data "$CORPUS_ROOT/train.corpus.json" \
      --output_dir "$OUTPUT_DIR" \
      --top_k 32 --max_seq_len 640 \
      --max_teacher_ctx_tokens 4096 --max_teacher_seq_len 4864 \
      --num_shards "$NUM_SHARDS" --shard_id "$s" \
      --no-use_flash_attn
  ) > "$RPMEM_ROOT/logs/teacher_shard${s}.log" 2>&1 &
  PIDS+=($!)
  echo "started shard $s pid ${PIDS[-1]}"
done

FAIL=0
for pid in "${PIDS[@]}"; do
  if ! wait "$pid"; then FAIL=1; echo "shard pid $pid failed"; fi
done
if [[ "$FAIL" -ne 0 ]]; then
  echo "ERROR: teacher shard failure; see logs/teacher_shard*.log"
  exit 1
fi
echo "DONE -> $OUTPUT_DIR"
