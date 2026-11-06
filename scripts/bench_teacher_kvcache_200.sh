#!/usr/bin/env bash
# Timed benchmark: 200 teacher samples with --reuse_context_kv on 1 NPU.
# Also runs the baseline full-forward path on the same 200 for comparison.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/env_rpmem_npu.sh"
bash "$SCRIPT_DIR/setup_container_deps.sh"

CORPUS_ROOT="${CORPUS_ROOT:-$RPMEM_DATA_ROOT/compiler}"
OUT_ROOT="${OUT_ROOT:-$RPMEM_ROOT/outputs/bench_teacher_kv_200}"
N_SAMPLES="${N_SAMPLES:-200}"
mkdir -p "$OUT_ROOT" "$RPMEM_ROOT/logs"
unset ASCEND_RT_VISIBLE_DEVICES 2>/dev/null || true

COMMON=(
  --base_model_path "$RPMEM_BASE_MODEL"
  --train_data "$CORPUS_ROOT/train.corpus.json"
  --top_k 32 --max_seq_len 640
  --max_teacher_ctx_tokens 4096 --max_teacher_seq_len 4864
  --num_shards 1 --shard_id 0
  --max_samples "$N_SAMPLES"
  --records_per_file 512
  --no-use_flash_attn
  --overwrite
)

echo "==== [1/2] KV-cache path: $N_SAMPLES samples ===="
rm -rf "$OUT_ROOT/kv"
mkdir -p "$OUT_ROOT/kv"
START_KV=$(date +%s)
"$PYTHON" -m rpmem.training.precompute_teacher \
  "${COMMON[@]}" \
  --output_dir "$OUT_ROOT/kv" \
  --reuse_context_kv \
  --kv_min_prefix_len 64 \
  2>&1 | tee "$RPMEM_ROOT/logs/bench_teacher_kv_200.log"
END_KV=$(date +%s)
KV_SEC=$((END_KV - START_KV))

echo "==== [2/2] baseline full-forward: $N_SAMPLES samples ===="
rm -rf "$OUT_ROOT/full"
mkdir -p "$OUT_ROOT/full"
START_FULL=$(date +%s)
"$PYTHON" -m rpmem.training.precompute_teacher \
  "${COMMON[@]}" \
  --output_dir "$OUT_ROOT/full" \
  --no-reuse_context_kv \
  2>&1 | tee "$RPMEM_ROOT/logs/bench_teacher_full_200.log"
END_FULL=$(date +%s)
FULL_SEC=$((END_FULL - START_FULL))

"$PYTHON" - <<PY
import json, time
from pathlib import Path
out = Path("$OUT_ROOT")
summary = {
  "n_samples": int("$N_SAMPLES"),
  "kv_seconds": int("$KV_SEC"),
  "full_seconds": int("$FULL_SEC"),
  "kv_samples_per_sec": (int("$N_SAMPLES") / max(int("$KV_SEC"), 1)),
  "full_samples_per_sec": (int("$N_SAMPLES") / max(int("$FULL_SEC"), 1)),
  "speedup_vs_full": (int("$FULL_SEC") / max(int("$KV_SEC"), 1)),
  "device": "npu",
  "finished_unix": int(time.time()),
}
(out / "bench_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print("==== BENCH SUMMARY ====")
print(json.dumps(summary, indent=2))
PY
