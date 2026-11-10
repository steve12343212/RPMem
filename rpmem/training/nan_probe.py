"""Helpers for Ascend Fixed-FKL NaN localization."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Callable, Iterable

import torch

# Finite grads at/above this absmax are treated as catastrophic for probes/replay.
CATASTROPHIC_ABSMAX = 1.0e20


def tensor_finite_report(
    name: str,
    value: torch.Tensor | None,
    *,
    with_absmax_coord: bool = True,
) -> dict[str, Any]:
    if value is None:
        return {"name": name, "present": False}
    flat = value.detach()
    finite = bool(torch.isfinite(flat).all().item())
    report: dict[str, Any] = {
        "name": name,
        "present": True,
        "finite": finite,
        "dtype": str(flat.dtype),
        "shape": list(flat.shape),
    }
    if flat.numel() == 0:
        report["numel"] = 0
        return report
    report["numel"] = int(flat.numel())
    as_float = flat.float()
    report["min"] = float(as_float.min().item())
    report["max"] = float(as_float.max().item())
    report["mean"] = float(as_float.mean().item())
    absmax = float(as_float.abs().max().item())
    report["absmax"] = absmax
    if with_absmax_coord and math.isfinite(absmax):
        flat_index = int(as_float.abs().argmax().item())
        report["absmax_flat_index"] = flat_index
        coord = torch.unravel_index(
            torch.tensor(flat_index, device=as_float.device),
            flat.shape,
        )
        report["absmax_coord"] = [int(x.item()) for x in coord]
    if not finite:
        report["nan_count"] = int(torch.isnan(as_float).sum().item())
        report["inf_count"] = int(torch.isinf(as_float).sum().item())
    report["catastrophic"] = bool(
        (not finite)
        or (
            math.isfinite(absmax)
            and absmax >= CATASTROPHIC_ABSMAX
        )
    )
    return report


def report_is_catastrophic(report: dict[str, Any] | None) -> bool:
    if report is None or not report.get("present"):
        return False
    if report.get("finite") is False:
        return True
    if report.get("catastrophic"):
        return True
    absmax = report.get("absmax")
    return (
        absmax is not None
        and isinstance(absmax, (int, float))
        and math.isfinite(float(absmax))
        and float(absmax) >= CATASTROPHIC_ABSMAX
    )


def first_anomalous_report(
    reports: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Prefer non-finite; otherwise first catastrophic-magnitude report."""

    first_cat = None
    for report in reports:
        if not report.get("present"):
            continue
        if report.get("finite") is False:
            return report
        if first_cat is None and report_is_catastrophic(report):
            first_cat = report
    return first_cat


def generated_lora_reports(
    generated_loras: dict[str, dict[str, torch.Tensor]],
) -> list[dict[str, Any]]:
    reports = []
    for module, matrices in generated_loras.items():
        for key, tensor in matrices.items():
            reports.append(tensor_finite_report(f"lora.{module}.{key}", tensor))
    return reports


def first_nonfinite_named_grad(
    named_params: Iterable[tuple[str, torch.nn.Parameter]],
) -> dict[str, Any] | None:
    """Cheap scan: isfinite only; detail counts only for the first bad grad."""

    for name, param in named_params:
        grad = param.grad
        if grad is None:
            continue
        if bool(torch.isfinite(grad).all().item()):
            continue
        report = tensor_finite_report(f"{name}.grad", grad)
        report["param_name"] = name
        return report
    return None


def param_grad_reports(
    named_params: Iterable[tuple[str, torch.nn.Parameter]],
) -> list[dict[str, Any]]:
    reports = []
    for name, param in named_params:
        reports.append(tensor_finite_report(f"{name}.param", param.data))
        if param.grad is not None:
            reports.append(tensor_finite_report(f"{name}.grad", param.grad))
    return reports


def first_nonfinite(reports: list[dict[str, Any]]) -> dict[str, Any] | None:
    for report in reports:
        if report.get("present") and report.get("finite") is False:
            return report
    return None


class BoundaryGradRecorder:
    """Record incoming grads at selected autograd boundaries (observe only)."""

    def __init__(self, capture_names: Iterable[str] | None = None) -> None:
        self.events: list[dict[str, Any]] = []
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self.capture_names = set(capture_names or [])
        self.captured: dict[str, torch.Tensor] = {}

    def clear(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self.events.clear()
        self.captured.clear()

    def make_hook(self, name: str) -> Callable[[torch.Tensor], None]:
        def _hook(grad: torch.Tensor):
            report = tensor_finite_report(name, grad)
            self.events.append(report)
            if name in self.capture_names:
                self.captured[name] = grad.detach().clone()

        return _hook

    def register(self, name: str, tensor: torch.Tensor | None) -> None:
        if tensor is None or not getattr(tensor, "requires_grad", False):
            return
        self._handles.append(tensor.register_hook(self.make_hook(name)))

    def first_bad(self) -> dict[str, Any] | None:
        return first_nonfinite(self.events)

    def first_anomalous(self) -> dict[str, Any] | None:
        return first_anomalous_report(self.events)


class EncoderBranchProbe:
    """Per cross-attn block gradient probe with independent context views.

    Sharing one ``projected`` tensor across blocks makes a single hook see only
    the summed gradient. Each block receives ``context + 0`` so its hook records
    that branch's contribution; a separate hook on the shared tensor records the
    aggregate.
    """

    def __init__(self, *, capture_tensors: bool = True) -> None:
        self.capture_tensors = capture_tensors
        self.events: list[dict[str, Any]] = []
        self.block_reports: list[dict[str, Any]] = []
        self.forwards: dict[str, torch.Tensor] = {}
        self.captured_grads: dict[str, torch.Tensor] = {}
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self._block_event_names: dict[int, list[str]] = {}

    def clear(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self.events.clear()
        self.block_reports.clear()
        self.forwards.clear()
        self.captured_grads.clear()
        self._block_event_names.clear()

    def _stash_forward(self, name: str, tensor: torch.Tensor) -> None:
        if self.capture_tensors:
            self.forwards[name] = tensor.detach().cpu().clone()

    def _make_hook(self, name: str) -> Callable[[torch.Tensor], None]:
        def _hook(grad: torch.Tensor):
            report = tensor_finite_report(name, grad)
            self.events.append(report)
            if self.capture_tensors:
                self.captured_grads[name] = grad.detach().cpu().clone()

        return _hook

    def register(self, name: str, tensor: torch.Tensor | None, *, block_id: int | None = None) -> None:
        if tensor is None or not getattr(tensor, "requires_grad", False):
            return
        self._stash_forward(name, tensor)
        self._handles.append(tensor.register_hook(self._make_hook(name)))
        if block_id is not None:
            self._block_event_names.setdefault(block_id, []).append(name)

    def wrap_context(self, context: torch.Tensor, block_id: int) -> torch.Tensor:
        """Return a graph-connected view unique to this cross-attn block."""

        view = context + 0
        self.register(f"block{block_id}.projected_branch", view, block_id=block_id)
        return view

    def register_shared_context(self, context: torch.Tensor) -> None:
        self.register("encoder.projected_shared", context)

    def register_context_rms_out(self, block_id: int, tensor: torch.Tensor) -> None:
        self.register(f"block{block_id}.context_rms_out", tensor, block_id=block_id)

    def register_sdpa_kv(self, block_id: int, key: torch.Tensor, value: torch.Tensor) -> None:
        self.register(f"block{block_id}.sdpa_k", key, block_id=block_id)
        self.register(f"block{block_id}.sdpa_v", value, block_id=block_id)

    def register_sdpa_io(
        self,
        block_id: int,
        *,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_out: torch.Tensor,
    ) -> None:
        """Hook full SDPA I/O for backend对照 (q/k/v + pre-projection attn out)."""

        self.register(f"block{block_id}.sdpa_q", query, block_id=block_id)
        self.register(f"block{block_id}.sdpa_k", key, block_id=block_id)
        self.register(f"block{block_id}.sdpa_v", value, block_id=block_id)
        self.register(f"block{block_id}.sdpa_out", attn_out, block_id=block_id)

    def finalize_block_reports(self) -> list[dict[str, Any]]:
        # Prefer backward hook-fire order for "first", not forward registration order.
        events_by_block: dict[int, list[dict[str, Any]]] = {
            block_id: [] for block_id in self._block_event_names
        }
        known = {
            name: block_id
            for block_id, names in self._block_event_names.items()
            for name in names
        }
        for event in self.events:
            name = event.get("name")
            if name in known:
                events_by_block[known[name]].append(event)
        reports: list[dict[str, Any]] = []
        for block_id in sorted(events_by_block):
            block_events = events_by_block[block_id]
            anomalous = first_anomalous_report(block_events)
            reports.append(
                {
                    "block_id": block_id,
                    "events": block_events,
                    "first_anomalous": anomalous,
                    "anomalous": anomalous is not None,
                }
            )
        self.block_reports = reports
        return reports

    def first_anomalous_block(self) -> dict[str, Any] | None:
        if not self.block_reports:
            self.finalize_block_reports()
        for report in self.block_reports:
            if report.get("anomalous"):
                return report
        return None

    def dump_first_anomalous_block(self) -> dict[str, Any] | None:
        bad = self.first_anomalous_block()
        if bad is None:
            return None
        block_id = int(bad["block_id"])
        prefix = f"block{block_id}."
        payload = {
            "block_id": block_id,
            "first_anomalous": bad.get("first_anomalous"),
            "events": bad.get("events"),
            "forwards": {
                key: value
                for key, value in self.forwards.items()
                if key.startswith(prefix)
            },
            "grads": {
                key: value
                for key, value in self.captured_grads.items()
                if key.startswith(prefix)
            },
        }
        return payload


def named_param_grad_snapshot(
    named_params: Iterable[tuple[str, torch.nn.Parameter]],
    *,
    only_nonfinite: bool = False,
) -> dict[str, dict[str, Any]]:
    """Finite-ness reports for parameter grads (cheap unless nonfinite)."""

    out: dict[str, dict[str, Any]] = {}
    for name, param in named_params:
        grad = param.grad
        if grad is None:
            continue
        finite = bool(torch.isfinite(grad).all().item())
        if only_nonfinite and finite:
            continue
        if finite:
            out[name] = {
                "name": f"{name}.grad",
                "finite": True,
                "dtype": str(grad.dtype),
                "shape": list(grad.shape),
                "numel": int(grad.numel()),
            }
        else:
            out[name] = tensor_finite_report(f"{name}.grad", grad)
    return out


class NanProbeLogger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("a", encoding="utf-8")

    def write(self, event: dict[str, Any]) -> None:
        self._fh.write(json.dumps(event, ensure_ascii=True, sort_keys=True) + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()
