from __future__ import annotations

from importlib import import_module

try:
    torch = import_module("torch")
except ImportError:
    torch = None

__all__ = [
    "lightning_indexer_forward",
    "_prefill_reference",
    "_fallback_reference",
]


def lightning_indexer_forward(
    query,
    keys,
    weights,
    *,
    top_k: int,
    phase: str,
    family: str,
):
    custom_op = _load_registered_op()
    if custom_op is not None and phase == "prefill" and family == "family_4x64":
        return custom_op(query, keys, weights, top_k, phase, family)
    if phase == "prefill" and family in {"family_64x128", "family_4x64"}:
        return _prefill_reference(query, keys, weights, top_k=top_k)
    if phase == "decode" and family in {"family_64x128", "family_4x64"}:
        return _decode_reference(query, keys, weights, top_k=top_k)
    return _fallback_reference(query, keys, weights, top_k=top_k)


def _load_registered_op():
    if torch is None:
        return None
    try:
        namespace = torch.ops.aten_dsa_lightning_indexer
        return getattr(namespace, "lightning_indexer_forward")
    except Exception:
        return None


def _prefill_reference(query, keys, weights, *, top_k: int):
    if torch is None:
        raise RuntimeError("torch is required for lightning_indexer reference wrapper")
    scores = torch.einsum("bqhd,bcd->bqhc", query, keys)
    scores = torch.relu(scores)
    scores = scores * weights.unsqueeze(-1)
    reduced = scores.sum(dim=2)
    return torch.topk(
        reduced,
        top_k,
        dim=-1,
        largest=True,
        sorted=True,
    ).indices.to(torch.int32)


def _fallback_reference(query, keys, weights, *, top_k: int):
    return _prefill_reference(query, keys, weights, top_k=top_k)


def _decode_reference(query, keys, weights, *, top_k: int):
    return _prefill_reference(query, keys, weights, top_k=top_k)
