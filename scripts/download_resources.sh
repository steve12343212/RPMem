#!/usr/bin/env bash
# Download RPMem-data via HF mirror. Prefer git+LFS (avoids Hub API rate limits).
set -euo pipefail
RPMEM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
MODELS_DIR="${RPMEM_MODELS_DIR:-/aistor/sjtu/hpc_stor01/home/sunwenjie/workshop/models}"
DATA_DIR="${RPMEM_DATA_ROOT:-$RPMEM_ROOT/data/rpmem}"
MIRROR_GIT="${HF_GIT_URL:-https://hf-mirror.com/datasets/PolarSnowLeopard/RPMem-data}"
mkdir -p "$(dirname "$DATA_DIR")" "$MODELS_DIR" "$RPMEM_ROOT/logs"

complete_ok() {
  [[ -f "$DATA_DIR/compiler/train.corpus.json" ]] || return 1
  local bytes
  bytes=$(du -sb "$DATA_DIR" 2>/dev/null | awk '{print $1}')
  [[ -n "$bytes" && "$bytes" -gt 3000000000 ]]
}

if complete_ok; then
  echo "RPMem-data already complete at $DATA_DIR"
  du -sh "$DATA_DIR"
  exit 0
fi

TMP="${DATA_DIR}.gitclone.$$"
rm -rf "$TMP"
echo "git clone $MIRROR_GIT -> $TMP"
git -c http.version=HTTP/1.1 clone --depth 1 "$MIRROR_GIT" "$TMP"
if [[ -f "$TMP/compiler/train.corpus.json" ]]; then
  rm -rf "${DATA_DIR}.bak"
  if [[ -e "$DATA_DIR" ]]; then
    mv "$DATA_DIR" "${DATA_DIR}.bak"
  fi
  mv "$TMP" "$DATA_DIR"
fi

if complete_ok; then
  echo "RPMem-data ready at $DATA_DIR"
  du -sh "$DATA_DIR"
  exit 0
fi
echo "ERROR: download incomplete"
du -sh "$DATA_DIR" "$TMP" 2>/dev/null || true
exit 1
