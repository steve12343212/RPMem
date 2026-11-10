"""Experiment 2: compare SDPA BF16 vs explicit attention BF16/FP32 on one block.

Expects a dump produced by EncoderBranchProbe with:
  forwards: block{N}.sdpa_{q,k,v,out}
  grads:    block{N}.sdpa_out  (incoming dL/d(attn_out))

Replays three backward implementations with identical inputs and dout:
  1) torch SDPA in bf16
  2) explicit matmul+softmax attention in bf16
  3) explicit attention in fp32 (reference)
"""

from __future__ import annotations

import argparse
import json
import math
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from rpmem.device import resolve_device
from rpmem.training.nan_probe import (
    CATASTROPHIC_ABSMAX,
    report_is_catastrophic,
    tensor_finite_report,
)


def _jsonable(report: dict[str, Any] | None) -> dict[str, Any] | None:
    if report is None:
        return None
    out: dict[str, Any] = {}
    for key, value in report.items():
        if isinstance(value, float) and (
            value != value or value in (float("inf"), float("-inf"))
        ):
            out[key] = None if value != value else str(value)
        else:
            out[key] = value
    return out


def explicit_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Eager attention matching SDPA math (no dropout, non-causal)."""

    head_dim = q.shape[-1]
    scale = 1.0 / math.sqrt(head_dim)
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    weights = torch.softmax(scores, dim=-1)
    return torch.matmul(weights, v)


def _run_backend(
    *,
    name: str,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dout: torch.Tensor,
    dtype: torch.dtype,
    explicit: bool,
    device: torch.device,
) -> dict[str, Any]:
    def _prep(t: torch.Tensor) -> torch.Tensor:
        return t.detach().to(device=device, dtype=dtype).requires_grad_(True)

    q_v = _prep(q)
    k_v = _prep(k)
    v_v = _prep(v)
    dout_v = dout.detach().to(device=device, dtype=dtype)

    # Keep math dtype exact; avoid nested autocast surprises on Ascend.
    with nullcontext():
        if explicit:
            out = explicit_attention(q_v, k_v, v_v)
        else:
            out = F.scaled_dot_product_attention(q_v, k_v, v_v, dropout_p=0.0)

    torch.autograd.backward(out, grad_tensors=dout_v)

    grads = {
        "q": None if q_v.grad is None else q_v.grad.detach(),
        "k": None if k_v.grad is None else k_v.grad.detach(),
        "v": None if v_v.grad is None else v_v.grad.detach(),
        "out": out.detach(),
    }
    reports = {
        key: _jsonable(tensor_finite_report(f"{name}.{key}", value))
        for key, value in grads.items()
    }
    catastrophic = any(
        value is not None
        and report_is_catastrophic(tensor_finite_report(f"{name}.{key}", value))
        for key, value in grads.items()
    )
    return {
        "name": name,
        "explicit": explicit,
        "dtype": str(dtype),
        "reports": reports,
        "catastrophic": catastrophic,
        "grads": grads,
    }


def _rel_err(candidate: torch.Tensor | None, reference: torch.Tensor | None) -> dict[str, Any]:
    if candidate is None or reference is None:
        return {"present": False}
    a = candidate.detach().float().cpu()
    b = reference.detach().float().cpu()
    if a.shape != b.shape:
        return {"present": True, "shape_mismatch": [list(a.shape), list(b.shape)]}
    diff = (a - b).abs()
    denom = b.abs().clamp_min(1e-12)
    rel = diff / denom
    finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
    cand_absmax = float(a.abs().max().item())
    return {
        "present": True,
        "finite": finite,
        "abs_max": float(diff.max().item()),
        "abs_mean": float(diff.mean().item()),
        "rel_max": float(rel.max().item()) if finite else None,
        "rel_mean": float(rel.mean().item()) if finite else None,
        "ref_absmax": float(b.abs().max().item()),
        "cand_absmax": cand_absmax,
        "catastrophic_candidate": bool(
            (not bool(torch.isfinite(a).all())) or cand_absmax >= CATASTROPHIC_ABSMAX
        ),
    }


def compare_from_block_dump(
    dump: dict[str, Any],
    *,
    block_id: int | None = None,
    device: torch.device | None = None,
) -> dict[str, Any]:
    device = device or resolve_device()
    forwards = dump.get("forwards") or {}
    grads = dump.get("grads") or {}
    if block_id is None:
        block_id = int(dump.get("block_id", -1))
    prefix = f"block{block_id}."

    required = {
        "q": f"{prefix}sdpa_q",
        "k": f"{prefix}sdpa_k",
        "v": f"{prefix}sdpa_v",
        "out": f"{prefix}sdpa_out",
    }
    missing = [key for key, name in required.items() if name not in forwards]
    if f"{prefix}sdpa_out" not in grads:
        missing.append("grad:sdpa_out")
    if missing:
        raise RuntimeError(
            f"block dump missing SDPA I/O fields {missing}; re-capture with updated probe"
        )

    q = forwards[required["q"]]
    k = forwards[required["k"]]
    v = forwards[required["v"]]
    dout = grads[f"{prefix}sdpa_out"]

    input_reports = {
        "q": _jsonable(tensor_finite_report("sdpa_q", q)),
        "k": _jsonable(tensor_finite_report("sdpa_k", k)),
        "v": _jsonable(tensor_finite_report("sdpa_v", v)),
        "dout": _jsonable(tensor_finite_report("sdpa_out.grad", dout)),
    }

    observed = {
        "q": grads.get(f"{prefix}sdpa_q"),
        "k": grads.get(f"{prefix}sdpa_k"),
        "v": grads.get(f"{prefix}sdpa_v"),
    }
    observed_reports = {
        key: _jsonable(tensor_finite_report(f"observed.sdpa_{key}", value))
        for key, value in observed.items()
        if value is not None
    }

    backends = [
        _run_backend(
            name="sdpa_bf16",
            q=q,
            k=k,
            v=v,
            dout=dout,
            dtype=torch.bfloat16,
            explicit=False,
            device=device,
        ),
        _run_backend(
            name="explicit_bf16",
            q=q,
            k=k,
            v=v,
            dout=dout,
            dtype=torch.bfloat16,
            explicit=True,
            device=device,
        ),
        _run_backend(
            name="explicit_fp32",
            q=q,
            k=k,
            v=v,
            dout=dout,
            dtype=torch.float32,
            explicit=True,
            device=device,
        ),
    ]

    ref = next(item for item in backends if item["name"] == "explicit_fp32")
    comparisons = {}
    for item in backends:
        comparisons[item["name"]] = {
            "catastrophic": item["catastrophic"],
            "vs_fp32": {
                key: _rel_err(item["grads"][key], ref["grads"][key])
                for key in ("q", "k", "v", "out")
            },
            "vs_observed": {
                key: _rel_err(item["grads"][key], observed.get(key))
                for key in ("q", "k", "v")
            },
            "reports": item["reports"],
        }

    sdpa_cat = comparisons["sdpa_bf16"]["catastrophic"]
    exp_bf_cat = comparisons["explicit_bf16"]["catastrophic"]
    exp_fp_cat = comparisons["explicit_fp32"]["catastrophic"]
    observed_v_cat = bool(
        observed.get("v") is not None
        and report_is_catastrophic(tensor_finite_report("observed.v", observed["v"]))
    )
    if sdpa_cat and (not exp_bf_cat) and (not exp_fp_cat):
        recommendation = "replace_sdpa_backend"
    elif (sdpa_cat or exp_bf_cat) and (not exp_fp_cat):
        recommendation = "local_fp32_attention"
    elif exp_fp_cat:
        recommendation = "inspect_forward_values_or_formula"
    elif observed_v_cat and (not sdpa_cat) and (not exp_bf_cat) and (not exp_fp_cat):
        # Original graph wrote a catastrophic dv, but the same (q,k,v,dout)
        # does not reproduce under any isolated backend.
        recommendation = "sdpa_runtime_nondeterminism_or_corruption"
    else:
        recommendation = "no_backend_divergence_on_this_dump"

    for item in backends:
        item.pop("grads", None)

    return {
        "block_id": block_id,
        "device": str(device),
        "catastrophic_absmax_threshold": CATASTROPHIC_ABSMAX,
        "input_reports": input_reports,
        "observed_grad_reports": observed_reports,
        "backends": [
            {
                "name": item["name"],
                "explicit": item["explicit"],
                "dtype": item["dtype"],
                "catastrophic": item["catastrophic"],
                "reports": item["reports"],
            }
            for item in backends
        ],
        "comparisons": comparisons,
        "recommendation": recommendation,
    }


def stress_backend(
    dump: dict[str, Any],
    *,
    name: str,
    explicit: bool,
    dtype: torch.dtype,
    block_id: int | None = None,
    repeats: int = 20,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Re-run one attention backend many times on the fixed dump tensors."""

    device = device or resolve_device()
    if block_id is None:
        block_id = int(dump.get("block_id", -1))
    prefix = f"block{block_id}."
    forwards = dump["forwards"]
    grads = dump["grads"]
    q = forwards[f"{prefix}sdpa_q"]
    k = forwards[f"{prefix}sdpa_k"]
    v = forwards[f"{prefix}sdpa_v"]
    dout = grads[f"{prefix}sdpa_out"]
    rows = []
    for idx in range(max(1, repeats)):
        result = _run_backend(
            name=f"{name}_stress_{idx}",
            q=q,
            k=k,
            v=v,
            dout=dout,
            dtype=dtype,
            explicit=explicit,
            device=device,
        )
        rows.append(
            {
                "repeat": idx,
                "catastrophic": result["catastrophic"],
                "v_absmax": result["reports"]["v"]["absmax"]
                if result["reports"]["v"]
                else None,
                "k_absmax": result["reports"]["k"]["absmax"]
                if result["reports"]["k"]
                else None,
                "q_absmax": result["reports"]["q"]["absmax"]
                if result["reports"]["q"]
                else None,
            }
        )
    v_vals = [row["v_absmax"] for row in rows if row["v_absmax"] is not None]
    return {
        "name": name,
        "explicit": explicit,
        "dtype": str(dtype),
        "repeats": rows,
        "any_catastrophic": any(row["catastrophic"] for row in rows),
        "v_absmax_min": min(v_vals) if v_vals else None,
        "v_absmax_max": max(v_vals) if v_vals else None,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump", required=True, help="first_bad_block_*.pt with SDPA I/O")
    parser.add_argument("--block_id", type=int, default=None)
    parser.add_argument("--output_json", default=None)
    parser.add_argument(
        "--stress_repeats",
        type=int,
        default=20,
        help="Extra isolated SDPA bf16 repeats on the same tensors.",
    )
    args = parser.parse_args(argv)

    dump_path = Path(args.dump)
    dump = torch.load(dump_path, map_location="cpu")
    summary = compare_from_block_dump(dump, block_id=args.block_id)
    if int(args.stress_repeats) > 0:
        summary["sdpa_bf16_stress"] = stress_backend(
            dump,
            name="sdpa_bf16",
            explicit=False,
            dtype=torch.bfloat16,
            block_id=args.block_id,
            repeats=int(args.stress_repeats),
        )
        summary["explicit_bf16_stress"] = stress_backend(
            dump,
            name="explicit_bf16",
            explicit=True,
            dtype=torch.bfloat16,
            block_id=args.block_id,
            repeats=int(args.stress_repeats),
        )
        sdpa_stress_cat = bool(summary["sdpa_bf16_stress"]["any_catastrophic"])
        explicit_stress_cat = bool(summary["explicit_bf16_stress"]["any_catastrophic"])
        if sdpa_stress_cat and not explicit_stress_cat:
            summary["recommendation"] = "replace_sdpa_backend"
            summary["recommendation_note"] = (
                "Isolated SDPA bf16 backward is intermittently catastrophic on this "
                "(q,k,v,dout), while explicit bf16 attention stays stable."
            )
    out_path = (
        Path(args.output_json)
        if args.output_json
        else dump_path.parent / "sdpa_backend_compare.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Wrote {out_path}")
    return 0 if summary["recommendation"] != "inspect_forward_values_or_formula" else 2


if __name__ == "__main__":
    raise SystemExit(main())
