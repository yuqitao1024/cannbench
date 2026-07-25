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
    valid_context_lengths=None,
    top_k: int,
    phase: str,
    family: str,
):
    custom_op = _load_registered_op()
    if custom_op is not None and phase in {"prefill", "decode"} and family in {
        "family_64x128",
        "family_32x128",
        "family_4x64",
    }:
        if valid_context_lengths is None:
            valid_context_lengths = torch.full(
                query.shape[:2],
                keys.shape[1],
                dtype=torch.int32,
                device=query.device,
            )
        custom_args = (query, keys, weights, valid_context_lengths)
        return custom_op(*custom_args, top_k, phase, family)
    reference_kwargs = {"top_k": top_k}
    if valid_context_lengths is not None:
        reference_kwargs["valid_context_lengths"] = valid_context_lengths
    if phase == "prefill" and family in {"family_64x128", "family_4x64"}:
        return _prefill_reference(query, keys, weights, **reference_kwargs)
    if phase == "decode" and family in {"family_64x128", "family_4x64"}:
        return _decode_reference(query, keys, weights, **reference_kwargs)
    return _fallback_reference(query, keys, weights, top_k=top_k)


def _load_registered_op():
    if torch is None:
        return None
    try:
        namespace = torch.ops.aten_dsa_lightning_indexer
        return getattr(namespace, "lightning_indexer_forward")
    except Exception:
        return None


def _prefill_reference(
    query, keys, weights, *, top_k: int, valid_context_lengths=None
):
    if torch is None:
        raise RuntimeError("torch is required for lightning_indexer reference wrapper")
    scores = torch.einsum("bqhd,bcd->bqhc", query, keys)
    scores = torch.relu(scores)
    scores = scores * weights.unsqueeze(-1)
    reduced = scores.sum(dim=2)
    if valid_context_lengths is not None:
        context_positions = torch.arange(
            keys.shape[1], device=getattr(keys, "device", None)
        ).reshape(1, 1, -1)
        reduced = reduced.masked_fill(
            context_positions >= valid_context_lengths.unsqueeze(-1),
            float("-inf"),
        )
    return torch.topk(
        reduced,
        top_k,
        dim=-1,
        largest=True,
        sorted=True,
    ).indices.to(torch.int32)


def _fallback_reference(query, keys, weights, *, top_k: int):
    return _prefill_reference(query, keys, weights, top_k=top_k)


def _decode_reference(
    query, keys, weights, *, top_k: int, valid_context_lengths=None
):
    return _prefill_reference(
        query,
        keys,
        weights,
        top_k=top_k,
        valid_context_lengths=valid_context_lengths,
    )
