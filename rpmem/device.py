"""Device helpers that prefer Ascend NPU, then CUDA, then CPU.

Training on this cluster normally runs inside the Ascend vc image
(``torch_npu``). On the login node these helpers still resolve safely to
CPU when NPU runtime is absent.
"""

from __future__ import annotations

import os
from contextlib import nullcontext
from typing import Any, Optional, Union

import torch

_DeviceLike = Union[str, torch.device, None]
_NPU_TRIED = False
_NPU_OK = False


def ensure_npu_backend() -> bool:
    """Import ``torch_npu`` once when available. Returns whether NPU works."""
    global _NPU_TRIED, _NPU_OK
    if _NPU_TRIED:
        return _NPU_OK
    _NPU_TRIED = True
    try:
        import torch_npu  # noqa: F401

        _NPU_OK = bool(hasattr(torch, "npu") and torch.npu.is_available())
    except Exception:
        _NPU_OK = False
    return _NPU_OK


def npu_available() -> bool:
    return ensure_npu_backend()


def cuda_available() -> bool:
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def accelerator_available() -> bool:
    return npu_available() or cuda_available()


def device_count() -> int:
    if npu_available():
        return int(torch.npu.device_count())
    if cuda_available():
        return int(torch.cuda.device_count())
    return 0


def _as_device(device: _DeviceLike) -> torch.device:
    if device is None:
        return resolve_device()
    if isinstance(device, torch.device):
        return device
    return torch.device(device)


def resolve_device(preferred: _DeviceLike = None) -> torch.device:
    """Pick the default compute device.

    Order: explicit ``preferred`` → ``DEVICE_KIND`` env → NPU → CUDA → CPU.
    """
    if preferred is not None:
        return _as_device(preferred)

    kind = os.environ.get("DEVICE_KIND", "").strip().lower()
    if kind in {"npu", "ascend"}:
        if npu_available():
            return torch.device("npu")
        raise RuntimeError("DEVICE_KIND=npu but torch.npu is not available")
    if kind in {"cuda", "gpu"}:
        if cuda_available():
            return torch.device("cuda")
        raise RuntimeError("DEVICE_KIND=cuda but torch.cuda is not available")
    if kind in {"cpu"}:
        return torch.device("cpu")

    if npu_available():
        return torch.device("npu")
    if cuda_available():
        return torch.device("cuda")
    return torch.device("cpu")


def device_type(device: _DeviceLike = None) -> str:
    return _as_device(device).type


def is_accelerator_device(device: _DeviceLike = None) -> bool:
    return device_type(device) in {"cuda", "npu"}


def autocast_context(
    device: _DeviceLike = None,
    dtype: torch.dtype = torch.bfloat16,
    enabled: Optional[bool] = None,
):
    """``torch.autocast`` for cuda/npu; no-op on CPU."""
    dt = device_type(device)
    if enabled is None:
        enabled = dt in {"cuda", "npu"}
    if not enabled or dt == "cpu":
        return nullcontext()
    return torch.autocast(device_type=dt, dtype=dtype, enabled=True)


def empty_cache() -> None:
    if npu_available():
        torch.npu.empty_cache()
    elif cuda_available():
        torch.cuda.empty_cache()


def synchronize(device: _DeviceLike = None) -> None:
    if device is None:
        if npu_available():
            for index in range(torch.npu.device_count()):
                torch.npu.synchronize(index)
        elif cuda_available():
            for index in range(torch.cuda.device_count()):
                torch.cuda.synchronize(index)
        return

    dev = _as_device(device)
    if dev.type == "npu":
        torch.npu.synchronize(dev)
    elif dev.type == "cuda":
        torch.cuda.synchronize(dev)


def get_device_name(index: int = 0) -> str:
    if npu_available():
        try:
            return str(torch.npu.get_device_name(index))
        except Exception:
            return f"npu:{index}"
    if cuda_available():
        return str(torch.cuda.get_device_name(index))
    return "cpu"


def memory_allocated(device: _DeviceLike = None) -> int:
    if npu_available():
        return int(torch.npu.memory_allocated(device))
    if cuda_available():
        return int(torch.cuda.memory_allocated(device))
    return 0


def max_memory_allocated(device: _DeviceLike = None) -> int:
    if npu_available():
        return int(torch.npu.max_memory_allocated(device))
    if cuda_available():
        return int(torch.cuda.max_memory_allocated(device))
    return 0


def max_memory_reserved(device: _DeviceLike = None) -> int:
    if npu_available() and hasattr(torch.npu, "max_memory_reserved"):
        return int(torch.npu.max_memory_reserved(device))
    if cuda_available():
        return int(torch.cuda.max_memory_reserved(device))
    return 0


def reset_peak_memory_stats(device: _DeviceLike = None) -> None:
    if npu_available() and hasattr(torch.npu, "reset_peak_memory_stats"):
        torch.npu.reset_peak_memory_stats(device)
    elif cuda_available():
        torch.cuda.reset_peak_memory_stats(device)


def get_rng_state(device: _DeviceLike = None) -> Any:
    dev = _as_device(device)
    if dev.type == "npu" and hasattr(torch.npu, "get_rng_state"):
        return torch.npu.get_rng_state(dev)
    if dev.type == "cuda" and cuda_available():
        return torch.cuda.get_rng_state(dev)
    return None


def set_rng_state(state: Any, device: _DeviceLike = None) -> None:
    if state is None:
        return
    dev = _as_device(device)
    if dev.type == "npu" and hasattr(torch.npu, "set_rng_state"):
        torch.npu.set_rng_state(state, device=dev)
    elif dev.type == "cuda" and cuda_available():
        torch.cuda.set_rng_state(state, device=dev)


def manual_seed_for_device(seed: int, device: _DeviceLike = None) -> None:
    torch.manual_seed(seed)
    dt = device_type(device)
    if dt == "npu" and hasattr(torch.npu, "manual_seed"):
        torch.npu.manual_seed(seed)
    elif dt == "cuda" and cuda_available():
        torch.cuda.manual_seed(seed)


def fork_rng_devices(device: _DeviceLike = None) -> list:
    """Device indices for ``torch.random.fork_rng(devices=...)``."""
    dev = _as_device(device)
    if is_accelerator_device(dev) and dev.index is not None:
        return [dev.index]
    if is_accelerator_device(dev):
        return [0]
    return []


def move_module_to_device(module: torch.nn.Module, device: _DeviceLike = None):
    """``.to(device)`` with default resolution (replaces bare ``.cuda()``)."""
    return module.to(resolve_device(device))


def accelerator_memory_profile(base_model_device_map: Optional[str] = None) -> dict:
    """Collect peak memory across visible accelerator devices."""
    kind = "npu" if npu_available() else "cuda" if cuda_available() else "cpu"
    devices = []
    for index in range(device_count()):
        synchronize(f"{kind}:{index}" if kind != "cpu" else "cpu")
        devices.append(
            {
                "index": index,
                "name": get_device_name(index),
                "peak_allocated_bytes": max_memory_allocated(index),
                "peak_reserved_bytes": max_memory_reserved(index),
            }
        )
    return {
        "format": f"rpmem_training_{kind}_memory_v1",
        "base_model_device_map": base_model_device_map,
        "devices": devices,
    }
