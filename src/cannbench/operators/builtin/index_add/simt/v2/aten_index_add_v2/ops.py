from __future__ import annotations

from importlib import import_module

import torch

try:
    _C = import_module(f"{__package__}._C")
except ImportError:
    _C = None

__all__ = [
    "is_extension_loaded",
    "has_index_add_forward_op",
    "index_add_forward",
]


def is_extension_loaded() -> bool:
    return _C is not None


def has_index_add_forward_op() -> bool:
    try:
        return hasattr(torch.ops.aten_index_add_v2, "index_add_forward")
    except Exception:
        return False


def _require_cpp_op(op_name: str):
    if not is_extension_loaded():
        raise RuntimeError("aten_index_add_v2 C++ extension is not loaded")
    try:
        namespace = torch.ops.aten_index_add_v2
        return getattr(namespace, op_name)
    except Exception as exc:
        raise RuntimeError(f"aten_index_add_v2::{op_name} is not registered") from exc


def index_add_forward(
    self: torch.Tensor,
    dim: int,
    index: torch.Tensor,
    source: torch.Tensor,
    alpha: float = 1.0,
) -> torch.Tensor:
    op = _require_cpp_op("index_add_forward")
    return op(self, dim, index, source, alpha)
