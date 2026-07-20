from __future__ import annotations

import torch
from importlib import import_module

try:
    _C = import_module(f"{__package__}._C")
except ImportError:
    _C = None

__all__ = [
    "adaptive_max_pool3d_backward",
    "is_extension_loaded",
]


def is_extension_loaded() -> bool:
    return _C is not None


def _require_cpp_op():
    if not is_extension_loaded():
        raise RuntimeError("aten_adaptive_max_pool3d_grad C++ extension is not loaded")
    try:
        return torch.ops.aten_adaptive_max_pool3d_grad.adaptive_max_pool3d_backward
    except Exception as exc:
        raise RuntimeError(
            "aten_adaptive_max_pool3d_grad::adaptive_max_pool3d_backward is not registered"
        ) from exc


def adaptive_max_pool3d_backward(
    grad_output: torch.Tensor,
    self: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    return _require_cpp_op()(grad_output, self, indices)
