#!/usr/bin/env bash
# KV-cache teacher precompute for the 20k JSONL subset (8 Ascend NPUs).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/env_rpmem_npu.sh"
bash "$SCRIPT_DIR/setup_container_deps.sh"

TRAIN_DATA="${TRAIN_DATA:-$RPMEM_ROOT/data/rpmem/compiler_20k/train.jsonl}"
OUTPUT_DIR="${TEACHER_ROOT:-$RPMEM_ROOT/outputs/teacher_kvcache_20k}"
NUM_SHARDS="${NUM_SHARDS:-8}"
KV_MIN_PREFIX_LEN="${KV_MIN_PREFIX_LEN:-64}"
mkdir -p "$OUTPUT_DIR" "$RPMEM_ROOT/logs"

if [[ ! -f "$TRAIN_DATA" ]]; then
  echo "ERROR: missing TRAIN_DATA=$TRAIN_DATA"
  exit 1
fi

NPU_COUNT="$("$PYTHON" - <<'PY'
from rpmem.device import device_count
print(device_count())
PY
)"
if [[ "$NUM_SHARDS" -gt "$NPU_COUNT" ]]; then
  NUM_SHARDS="$NPU_COUNT"
fi
echo "teacher KV-cache 20k shards=$NUM_SHARDS data=$TRAIN_DATA out=$OUTPUT_DIR"

unset ASCEND_RT_VISIBLE_DEVICES 2>/dev/null || true
PIDS=()
for ((s=0; s<NUM_SHARDS; s++)); do
  (
    export ASCEND_RT_VISIBLE_DEVICES="$s"
    export DEVICE_KIND=npu
    "$PYTHON" -m rpmem.training.precompute_teacher \
      --base_model_path "$RPMEM_BASE_MODEL" \
      --train_data "$TRAIN_DATA" \
      --output_dir "$OUTPUT_DIR" \
      --top_k 32 --max_seq_len 640 \
      --max_teacher_ctx_tokens 4096 --max_teacher_seq_len 4864 \
      --num_shards "$NUM_SHARDS" --shard_id "$s" \
      --reuse_context_kv \
      --kv_min_prefix_len "$KV_MIN_PREFIX_LEN" \
      --no-use_flash_attn
  ) > "$RPMEM_ROOT/logs/teacher_kvcache_20k_shard${s}.log" 2>&1 &
  PIDS+=($!)
  echo "started kvcache_20k shard $s pid ${PIDS[-1]}"
done

FAIL=0
for pid in "${PIDS[@]}"; do
  if ! wait "$pid"; then FAIL=1; echo "kvcache_20k shard pid $pid failed"; fi
done
if [[ "$FAIL" -ne 0 ]]; then
  echo "ERROR: teacher kvcache_20k shard failure; see logs/teacher_kvcache_20k_shard*.log"
  exit 1
fi
echo "DONE -> $OUTPUT_DIR"
