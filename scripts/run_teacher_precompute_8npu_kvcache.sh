#!/usr/bin/env bash
# Optional teacher path: reuse shared context KV prefix across probes.
# Does NOT touch the default teacher job output; writes to a separate directory.
#
#   bash scripts/run_teacher_precompute_8npu_kvcache.sh
#
# Or submit via vc with this script instead of run_teacher_precompute_8npu.sh.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/env_rpmem_npu.sh"
bash "$SCRIPT_DIR/setup_container_deps.sh"

CORPUS_ROOT="${CORPUS_ROOT:-$RPMEM_DATA_ROOT/compiler}"
# Keep default teacher cache for the running baseline job untouched.
OUTPUT_DIR="${TEACHER_ROOT:-$RPMEM_ROOT/outputs/teacher_kvcache}"
NUM_SHARDS="${NUM_SHARDS:-8}"
KV_MIN_PREFIX_LEN="${KV_MIN_PREFIX_LEN:-64}"
mkdir -p "$OUTPUT_DIR" "$RPMEM_ROOT/logs"

NPU_COUNT="$("$PYTHON" - <<'PY'
from rpmem.device import device_count
print(device_count())
PY
)"
if [[ "$NUM_SHARDS" -gt "$NPU_COUNT" ]]; then
  NUM_SHARDS="$NPU_COUNT"
fi
echo "teacher KV-cache precompute shards=$NUM_SHARDS corpus=$CORPUS_ROOT out=$OUTPUT_DIR"

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
      --reuse_context_kv \
      --kv_min_prefix_len "$KV_MIN_PREFIX_LEN" \
      --no-use_flash_attn
  ) > "$RPMEM_ROOT/logs/teacher_kvcache_shard${s}.log" 2>&1 &
  PIDS+=($!)
  echo "started kvcache shard $s pid ${PIDS[-1]}"
done

FAIL=0
for pid in "${PIDS[@]}"; do
  if ! wait "$pid"; then FAIL=1; echo "kvcache shard pid $pid failed"; fi
done
if [[ "$FAIL" -ne 0 ]]; then
  echo "ERROR: teacher kvcache shard failure; see logs/teacher_kvcache_shard*.log"
  exit 1
fi
echo "DONE -> $OUTPUT_DIR"
