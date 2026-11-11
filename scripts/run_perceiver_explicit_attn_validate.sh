#!/usr/bin/env bash
# Validate Ascend encoder cross-attn explicit-BF16 fix on checkpoint-53 dump.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/env_rpmem_npu.sh"
bash "$SCRIPT_DIR/setup_container_deps.sh"

RESUME_CKPT="${RESUME_CKPT:?RESUME_CKPT required}"
DUMP="${DUMP:?DUMP required}"
OUTPUT_DIR="${OUTPUT_DIR:?OUTPUT_DIR required}"
LOG_NAME="${LOG_NAME:-compiler_perceiver_explicit_attn_validate.log}"
REPEATS="${REPEATS:-20}"
mkdir -p "$OUTPUT_DIR" "$RPMEM_ROOT/logs"

unset ASCEND_RT_VISIBLE_DEVICES 2>/dev/null || true
export DEVICE_KIND=npu
# Ensure the fix path is active even if detection quirks appear.
export RPMEM_PERCEIVER_EXPLICIT_ENCODER_CROSS_ATTN="${RPMEM_PERCEIVER_EXPLICIT_ENCODER_CROSS_ATTN:-1}"

echo "EXPLICIT_ATTN_VALIDATE CKPT=$RESUME_CKPT DUMP=$DUMP OUT=$OUTPUT_DIR REPEATS=$REPEATS"
echo "RPMEM_PERCEIVER_EXPLICIT_ENCODER_CROSS_ATTN=$RPMEM_PERCEIVER_EXPLICIT_ENCODER_CROSS_ATTN"
echo "visible ASCEND_VISIBLE_DEVICES=${ASCEND_VISIBLE_DEVICES:-<unset>}"

"$PYTHON" -m rpmem.training.replay_perceiver_nan \
  --checkpoint "$RESUME_CKPT" \
  --dump "$DUMP" \
  --output_json "$OUTPUT_DIR/explicit_attn_regression.json" \
  --branch_probe \
  --repeats "$REPEATS" \
  2>&1 | tee "$RPMEM_ROOT/logs/$LOG_NAME"

"$PYTHON" - <<'PY' "$OUTPUT_DIR/explicit_attn_regression.json" "$OUTPUT_DIR/explicit_attn_validate_summary.json"
import json, sys
from pathlib import Path
src, dst = Path(sys.argv[1]), Path(sys.argv[2])
data = json.loads(src.read_text())
repeats = data.get("repeats") or []
n = len(repeats)
n_nan = sum(1 for r in repeats if r.get("reproduced_nan"))
n_cat = sum(1 for r in repeats if r.get("reproduced_catastrophic"))
n_ok = sum(1 for r in repeats if r.get("magnitude_ok"))
# Collect projected / max sdpa_v absmax across repeats when present.
proj = []
sdpa_v = []
for r in repeats:
    ba = r.get("boundary_absmax") or {}
    if "perceiver.projected" in ba and ba["perceiver.projected"].get("absmax") is not None:
        proj.append(ba["perceiver.projected"]["absmax"])
    bp = r.get("branch_probe") or {}
    absmap = bp.get("absmax_by_name") or {}
    for name, rep in absmap.items():
        if name.endswith(".sdpa_v") and rep.get("absmax") is not None:
            sdpa_v.append(rep["absmax"])
summary = {
    "repeats": n,
    "reproduced_nan_count": n_nan,
    "reproduced_catastrophic_count": n_cat,
    "magnitude_ok_count": n_ok,
    "all_magnitude_ok": n > 0 and n_ok == n and n_nan == 0 and n_cat == 0,
    "projected_absmax_min": min(proj) if proj else None,
    "projected_absmax_max": max(proj) if proj else None,
    "sdpa_v_absmax_min": min(sdpa_v) if sdpa_v else None,
    "sdpa_v_absmax_max": max(sdpa_v) if sdpa_v else None,
    "explicit_encoder_cross_attn_env": __import__("os").environ.get(
        "RPMEM_PERCEIVER_EXPLICIT_ENCODER_CROSS_ATTN"
    ),
    "pass": n > 0 and n_ok == n and n_nan == 0 and n_cat == 0,
}
dst.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print(json.dumps(summary, indent=2, sort_keys=True))
if not summary["pass"]:
    raise SystemExit(2)
PY

echo "EXPLICIT_ATTN_VALIDATE DONE -> $OUTPUT_DIR"
