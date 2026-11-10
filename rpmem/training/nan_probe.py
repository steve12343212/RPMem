"""Helpers for Ascend Fixed-FKL NaN localization."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Iterable

import torch


def tensor_finite_report(name: str, value: torch.Tensor | None) -> dict[str, Any]:
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
    if not finite:
        report["nan_count"] = int(torch.isnan(as_float).sum().item())
        report["inf_count"] = int(torch.isinf(as_float).sum().item())
    return report


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

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []
        self._handles: list[torch.utils.hooks.RemovableHandle] = []

    def clear(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self.events.clear()

    def make_hook(self, name: str) -> Callable[[torch.Tensor], None]:
        def _hook(grad: torch.Tensor):
            report = tensor_finite_report(name, grad)
            self.events.append(report)

        return _hook

    def register(self, name: str, tensor: torch.Tensor | None) -> None:
        if tensor is None or not getattr(tensor, "requires_grad", False):
            return
        self._handles.append(tensor.register_hook(self.make_hook(name)))

    def first_bad(self) -> dict[str, Any] | None:
        return first_nonfinite(self.events)


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
