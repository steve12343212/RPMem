"""Isolated Perceiver backward replay for Ascend NaN localization.

Loads Perceiver weights from a Hypernet checkpoint plus a dump produced by
``--nan_probe_capture_sample``, then runs only:

    lora_emb = perceiver(ctx_features, mask, pos)
    autograd.backward(lora_emb, grad_tensors=saved_dL_dlora_emb)

under the same bf16 autocast path used in training.

Reports distinguish:
  - non-finite grads (NaN/Inf)
  - catastrophic-but-finite absmax (>= CATASTROPHIC_ABSMAX)
  - magnitude-normal finite grads

With ``--branch_probe``, instruments each encoder cross-attn block with an
independent ``projected`` view plus RMSNorm / SDPA K/V hooks.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from rpmem.checkpoint.compat import load_compiler_checkpoint
from rpmem.device import autocast_context, resolve_device
from rpmem.training.hypernet_model import HypernetModel
from rpmem.training.nan_probe import (
    CATASTROPHIC_ABSMAX,
    BoundaryGradRecorder,
    EncoderBranchProbe,
    first_anomalous_report,
    named_param_grad_snapshot,
    report_is_catastrophic,
    tensor_finite_report,
)


def _to_device(value: torch.Tensor | None, device: torch.device) -> torch.Tensor | None:
    if value is None:
        return None
    return value.to(device=device, non_blocking=False)


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


def _load_perceiver(checkpoint: Path) -> torch.nn.Module:
    state = load_compiler_checkpoint(str(checkpoint), map_location="cpu")
    config = HypernetModel._checkpoint_config(state)
    perceiver = HypernetModel._build_perceiver(config)
    weights = {}
    for key, value in state.items():
        if not isinstance(value, torch.Tensor):
            continue
        name = key[len("_orig_mod.") :] if key.startswith("_orig_mod.") else key
        if name.startswith("perceiver."):
            weights[name[len("perceiver.") :]] = value
    missing, unexpected = perceiver.load_state_dict(weights, strict=False)
    if missing:
        raise RuntimeError(f"missing Perceiver weights: {missing[:8]}")
    if unexpected:
        raise RuntimeError(f"unexpected Perceiver weights: {unexpected[:8]}")
    return perceiver


def _boundary_absmax_table(events: list[dict[str, Any]]) -> dict[str, Any]:
    table: dict[str, Any] = {}
    for event in events:
        name = event.get("name")
        if not name:
            continue
        table[name] = {
            "finite": event.get("finite"),
            "absmax": event.get("absmax"),
            "absmax_coord": event.get("absmax_coord"),
            "catastrophic": report_is_catastrophic(event),
            "nan_count": event.get("nan_count"),
            "inf_count": event.get("inf_count"),
            "dtype": event.get("dtype"),
        }
    return table


def _slice_feature_compare(ctx_features: torch.Tensor) -> dict[str, Any]:
    """Compare layer slices mentioned in the investigation notes (12 vs 13)."""

    if ctx_features.ndim != 4 or ctx_features.shape[1] <= 13:
        return {"available": False}
    s12 = ctx_features[0, 12]
    s13 = ctx_features[0, 13]
    equal = bool(torch.equal(s12, s13))
    diff = (s12.float() - s13.float()).abs()
    return {
        "available": True,
        "slice_12_13_equal": equal,
        "slice_12_absmax": float(s12.float().abs().max().item()),
        "slice_13_absmax": float(s13.float().abs().max().item()),
        "slice_diff_absmax": float(diff.max().item()),
        "slice_diff_mean": float(diff.mean().item()),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to checkpoint dir or pytorch_model.bin from step 53.",
    )
    parser.add_argument(
        "--dump",
        required=True,
        help="sample_*_perceiver.pt dump from nan_probe_capture_sample.",
    )
    parser.add_argument(
        "--output_json",
        default=None,
        help="Write summary JSON here (default: <dump_dir>/perceiver_replay.json).",
    )
    parser.add_argument(
        "--branch_probe",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable per-block encoder branch probes (default: on).",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Repeat the same replay N times to expose nondeterminism.",
    )
    parser.add_argument(
        "--until_anomalous",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Stop early after the first catastrophic/NaN repeat.",
    )
    parser.add_argument(
        "--max_attempts",
        type=int,
        default=None,
        help="With --until_anomalous, cap attempts (default: --repeats).",
    )
    parser.add_argument(
        "--eval_mode",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run perceiver.eval() instead of train() (disables dropout).",
    )
    args = parser.parse_args(argv)

    ckpt_path = Path(args.checkpoint)
    if ckpt_path.is_dir():
        ckpt_path = ckpt_path / "pytorch_model.bin"
    dump_path = Path(args.dump)
    out_path = (
        Path(args.output_json)
        if args.output_json
        else dump_path.parent / "perceiver_replay.json"
    )
    out_dir = out_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device()
    dump = torch.load(dump_path, map_location="cpu")
    inputs = dump.get("inputs") or {}
    grad_lora_emb = dump.get("grad_lora_emb")
    if inputs.get("ctx_features") is None:
        raise RuntimeError("dump missing inputs.ctx_features")
    if grad_lora_emb is None:
        raise RuntimeError("dump missing grad_lora_emb")

    ctx_features_cpu = inputs["ctx_features"]
    slice_compare = _slice_feature_compare(ctx_features_cpu)

    repeat_summaries: list[dict[str, Any]] = []
    last_branch_payload = None
    max_attempts = int(args.max_attempts or args.repeats)
    max_attempts = max(1, max_attempts)

    for repeat_idx in range(max_attempts):
        perceiver = _load_perceiver(ckpt_path).to(device)
        if args.eval_mode:
            perceiver.eval()
        else:
            perceiver.train()
        for param in perceiver.parameters():
            param.requires_grad_(True)
        perceiver.zero_grad(set_to_none=True)

        hooks = BoundaryGradRecorder(
            capture_names={"perceiver.projected", "perceiver.latents", "lora_emb"}
        )
        perceiver.nan_probe_boundary_hooks = hooks
        branch_probe = None
        if args.branch_probe:
            branch_probe = EncoderBranchProbe(capture_tensors=True)
            perceiver.encoder.encoder_branch_probe = branch_probe

        ctx_features = _to_device(ctx_features_cpu, device)
        ctx_attn_mask = _to_device(inputs.get("ctx_attn_mask"), device)
        ctx_position_ids = _to_device(inputs.get("ctx_position_ids"), device)
        grad_tensor = _to_device(grad_lora_emb, device)
        assert ctx_features is not None and grad_tensor is not None

        input_reports = {
            "ctx_features": tensor_finite_report("ctx_features", ctx_features),
            "grad_lora_emb": tensor_finite_report("grad_lora_emb", grad_tensor),
        }
        if ctx_attn_mask is not None:
            input_reports["ctx_attn_mask"] = tensor_finite_report(
                "ctx_attn_mask", ctx_attn_mask
            )

        with autocast_context(device, dtype=torch.bfloat16):
            lora_emb, _ = perceiver(ctx_features, ctx_attn_mask, ctx_position_ids)
        hooks.register("lora_emb", lora_emb)
        torch.autograd.backward(lora_emb, grad_tensors=grad_tensor)

        named_params = list(perceiver.named_parameters())
        first_bad_param = None
        for name, param in named_params:
            if param.grad is None:
                continue
            if bool(torch.isfinite(param.grad).all().item()):
                continue
            first_bad_param = tensor_finite_report(f"{name}.grad", param.grad)
            first_bad_param["param_name"] = name
            break

        boundary_events = list(hooks.events)
        first_nonfinite_boundary = hooks.first_bad()
        first_anomalous_boundary = hooks.first_anomalous()
        absmax_table = _boundary_absmax_table(boundary_events)

        branch_summary = None
        if branch_probe is not None:
            block_reports = branch_probe.finalize_block_reports()
            shared = next(
                (e for e in branch_probe.events if e.get("name") == "encoder.projected_shared"),
                None,
            )
            first_bad_block = branch_probe.first_anomalous_block()
            branch_summary = {
                "shared_projected_grad": _jsonable(shared),
                "block_reports": [
                    {
                        "block_id": item["block_id"],
                        "anomalous": item["anomalous"],
                        "first_anomalous": _jsonable(item.get("first_anomalous")),
                        "events": [_jsonable(e) for e in item.get("events") or []],
                    }
                    for item in block_reports
                ],
                "first_anomalous_block_id": (
                    None
                    if first_bad_block is None
                    else int(first_bad_block["block_id"])
                ),
                "absmax_by_name": _boundary_absmax_table(branch_probe.events),
            }
            if first_bad_block is not None:
                last_branch_payload = branch_probe.dump_first_anomalous_block()
                bad_path = out_dir / f"first_bad_block_repeat{repeat_idx}.pt"
                torch.save(last_branch_payload, bad_path)
                branch_summary["first_bad_block_dump"] = str(bad_path)

        reproduced_nan = (
            first_bad_param is not None or first_nonfinite_boundary is not None
        )
        reproduced_catastrophic = (
            report_is_catastrophic(first_anomalous_boundary)
            or (
                branch_summary is not None
                and branch_summary.get("first_anomalous_block_id") is not None
            )
        )
        magnitude_ok = (not reproduced_nan) and (not reproduced_catastrophic)

        repeat_summaries.append(
            {
                "repeat": repeat_idx,
                "eval_mode": bool(args.eval_mode),
                "input_reports": {k: _jsonable(v) for k, v in input_reports.items()},
                "lora_emb_forward": _jsonable(
                    tensor_finite_report("lora_emb", lora_emb)
                ),
                "boundary_grad_events": [_jsonable(e) for e in boundary_events],
                "boundary_absmax": absmax_table,
                "first_nonfinite_boundary_grad": _jsonable(first_nonfinite_boundary),
                "first_anomalous_boundary_grad": _jsonable(first_anomalous_boundary),
                "first_bad_param_grad": _jsonable(first_bad_param),
                "perceiver_param_grads_nonfinite": {
                    k: _jsonable(v)
                    for k, v in named_param_grad_snapshot(
                        named_params, only_nonfinite=True
                    ).items()
                },
                "branch_probe": branch_summary,
                "reproduced_nan": reproduced_nan,
                "reproduced_catastrophic": reproduced_catastrophic,
                "magnitude_ok": magnitude_ok,
            }
        )

        # Drop large graph before next repeat.
        del perceiver, hooks, branch_probe, lora_emb
        if hasattr(torch, "npu") and torch.npu.is_available():
            torch.npu.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()

        if args.until_anomalous and (
            repeat_summaries[-1]["reproduced_nan"]
            or repeat_summaries[-1]["reproduced_catastrophic"]
        ):
            break

    any_nan = any(item["reproduced_nan"] for item in repeat_summaries)
    any_cat = any(item["reproduced_catastrophic"] for item in repeat_summaries)
    # Prefer a failing repeat for the convenience mirrors when present.
    mirror = next(
        (
            item
            for item in repeat_summaries
            if item["reproduced_nan"] or item["reproduced_catastrophic"]
        ),
        repeat_summaries[-1],
    )
    summary = {
        "checkpoint": str(ckpt_path),
        "dump": str(dump_path),
        "device": str(device),
        "sample_index": dump.get("sample_index"),
        "source_global_step": dump.get("global_step"),
        "source_micro_step": dump.get("micro_step"),
        "catastrophic_absmax_threshold": CATASTROPHIC_ABSMAX,
        "slice_12_13_compare": slice_compare,
        "repeats": repeat_summaries,
        "any_repeat_reproduced_nan": any_nan,
        "any_repeat_reproduced_catastrophic": any_cat,
        "all_repeats_magnitude_ok": (not any_nan) and (not any_cat),
        # Convenience mirrors: first failing repeat if any, else last.
        "reproduced_nan": mirror["reproduced_nan"],
        "reproduced_catastrophic": mirror["reproduced_catastrophic"],
        "magnitude_ok": mirror["magnitude_ok"],
        "boundary_absmax": mirror["boundary_absmax"],
        "branch_probe": mirror.get("branch_probe"),
    }

    out_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Wrote {out_path}")
    if summary["reproduced_nan"]:
        return 2
    if summary["reproduced_catastrophic"]:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
