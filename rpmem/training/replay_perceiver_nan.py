"""Isolated Perceiver backward replay for Ascend NaN localization.

Loads Perceiver weights from a Hypernet checkpoint plus a dump produced by
``--nan_probe_capture_sample``, then runs only:

    lora_emb = perceiver(ctx_features, mask, pos)
    autograd.backward(lora_emb, grad_tensors=saved_dL_dlora_emb)

under the same bf16 autocast path used in training. Reports finiteness of
``perceiver.projected`` / ``perceiver.latents`` grads and Perceiver param grads.
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
    BoundaryGradRecorder,
    named_param_grad_snapshot,
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

    device = resolve_device()
    dump = torch.load(dump_path, map_location="cpu")
    inputs = dump.get("inputs") or {}
    grad_lora_emb = dump.get("grad_lora_emb")
    if inputs.get("ctx_features") is None:
        raise RuntimeError("dump missing inputs.ctx_features")
    if grad_lora_emb is None:
        raise RuntimeError("dump missing grad_lora_emb")

    perceiver = _load_perceiver(ckpt_path).to(device)
    perceiver.train()
    for param in perceiver.parameters():
        param.requires_grad_(True)
    perceiver.zero_grad(set_to_none=True)

    hooks = BoundaryGradRecorder(
        capture_names={"perceiver.projected", "perceiver.latents", "lora_emb"}
    )
    perceiver.nan_probe_boundary_hooks = hooks

    ctx_features = _to_device(inputs["ctx_features"], device)
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
    first_bad = None
    for name, param in named_params:
        if param.grad is None:
            continue
        if bool(torch.isfinite(param.grad).all().item()):
            continue
        first_bad = tensor_finite_report(f"{name}.grad", param.grad)
        first_bad["param_name"] = name
        break

    summary = {
        "checkpoint": str(ckpt_path),
        "dump": str(dump_path),
        "device": str(device),
        "sample_index": dump.get("sample_index"),
        "source_global_step": dump.get("global_step"),
        "source_micro_step": dump.get("micro_step"),
        "input_reports": {k: _jsonable(v) for k, v in input_reports.items()},
        "lora_emb_forward": _jsonable(tensor_finite_report("lora_emb", lora_emb)),
        "boundary_grad_events": [_jsonable(e) for e in hooks.events],
        "first_bad_boundary_grad": _jsonable(hooks.first_bad()),
        "first_bad_param_grad": _jsonable(first_bad),
        "perceiver_param_grads_nonfinite": {
            k: _jsonable(v)
            for k, v in named_param_grad_snapshot(
                named_params, only_nonfinite=True
            ).items()
        },
        "reproduced_nan": first_bad is not None or hooks.first_bad() is not None,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Wrote {out_path}")
    return 0 if not summary["reproduced_nan"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
