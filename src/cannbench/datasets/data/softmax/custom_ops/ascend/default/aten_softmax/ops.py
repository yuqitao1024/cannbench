from __future__ import annotations

import torch
from importlib import import_module

try:
    _C = import_module(f"{__package__}._C")
except ImportError:
    _C = None

__all__ = [
    "is_extension_loaded",
    "has_spatial_softmax_forward_op",
    "spatial_softmax_forward",
]


def is_extension_loaded() -> bool:
    return _C is not None


def has_spatial_softmax_forward_op() -> bool:
    try:
        return hasattr(torch.ops.aten_softmax, "spatial_softmax_forward")
    except Exception:
        return False


def _require_cpp_op(op_name: str):
    if not is_extension_loaded():
        raise RuntimeError("aten_softmax C++ extension is not loaded")
    try:
        namespace = torch.ops.aten_softmax
        return getattr(namespace, op_name)
    except Exception as exc:
        raise RuntimeError(f"aten_softmax::{op_name} is not registered") from exc


def spatial_softmax_forward(
    input: torch.Tensor,
    dim: int,
    half_to_float: bool = False,
) -> torch.Tensor:
    op = _require_cpp_op("spatial_softmax_forward")
    return op(input, dim, half_to_float)
