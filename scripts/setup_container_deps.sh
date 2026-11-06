#!/usr/bin/env bash
# Install RPMem Python deps into the Ascend vc image user-site (torch_npu stays image-native).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/env_rpmem_npu.sh"

PY="${PYTHON}"
echo "[rpmem deps] using $PY"
"$PY" -m pip install --user -q -U pip setuptools wheel
"$PY" -m pip install --user -q \
  "numpy>=1.26,<3" \
  "transformers>=4.51,<5" \
  "accelerate>=1.0" \
  "tensorboard>=2.16" \
  "pyarrow>=14" \
  "PyYAML>=6" \
  "huggingface-hub>=0.30" \
  "safetensors" \
  "sentencepiece" \
  "protobuf"

cd "$RPMEM_ROOT"
"$PY" -m pip install --user -q -e ".[train,experiments]" --no-deps

"$PY" - <<'PY'
import torch
import torch_npu  # noqa: F401
import transformers
import accelerate
import rpmem
from rpmem.device import device_count, npu_available, resolve_device

print("torch", torch.__version__)
print("transformers", transformers.__version__)
print("accelerate", accelerate.__version__)
print("rpmem", rpmem.__version__)
assert npu_available(), "NPU not available after torch_npu import"
print("device", resolve_device(), "count", device_count())
if device_count() < 1:
    raise SystemExit("no visible NPU devices")
PY
echo "[rpmem deps] ok"
