from __future__ import annotations

import math
from importlib import import_module

try:
    torch = import_module("torch")
except ImportError:
    torch = None

__all__ = [
    "sparse_attention_forward",
    "_prefill_reference",
    "_fallback_reference",
]


def sparse_attention_forward(query, keys, values, indices, *, phase: str, family: str, causal: bool):
    custom_op = _load_registered_op()
    if custom_op is not None and phase == "prefill" and family == "family_hd512":
        return custom_op(query, keys, values, indices, phase, family, causal)
    if phase == "prefill" and family in {"family_hd512", "family_hd128"}:
        return _prefill_reference(query, keys, values, indices, causal=causal)
    if phase == "decode" and family in {"family_hd512", "family_hd128"}:
        return _decode_reference(query, keys, values, indices, causal=causal)
    return _fallback_reference(query, keys, values, indices, causal=causal)


def _load_registered_op():
    if torch is None:
        return None
    try:
        namespace = torch.ops.aten_dsa_sparse_attention
        return getattr(namespace, "sparse_attention_forward")
    except Exception:
        return None


def _prefill_reference(query, keys, values, indices, *, causal: bool):
    if torch is None:
        raise RuntimeError("torch is required for sparse_attention reference wrapper")
    expanded_keys, expanded_values = _expand_kv(keys, values, query.shape[1])
    selected_keys = _gather_selected(expanded_keys, indices)
    selected_values = _gather_selected(expanded_values, indices)
    scores = (query.unsqueeze(3) * selected_keys).sum(dim=-1) / math.sqrt(query.shape[-1])
    if causal and query.shape[2] > 1:
        positions = torch.arange(query.shape[2], device=getattr(query, "device", None)).reshape(
            1, 1, query.shape[2], 1
        )
        scores = scores.masked_fill(indices[:, None, :, :] > positions, float("-inf"))
    probabilities = torch.softmax(scores.float(), dim=-1)
    output = (probabilities.to(query.dtype).unsqueeze(-1) * selected_values).sum(dim=-2)
    lse = torch.logsumexp(scores.float(), dim=-1)
    return output, lse


def _fallback_reference(query, keys, values, indices, *, causal: bool):
    return _prefill_reference(query, keys, values, indices, causal=causal)


def _decode_reference(query, keys, values, indices, *, causal: bool):
    return _prefill_reference(query, keys, values, indices, causal=causal)


def _expand_kv(keys, values, query_heads: int):
    if keys.shape[1] == query_heads:
        return keys, values
    repeats = query_heads // keys.shape[1]
    return (
        _repeat_interleave_heads(keys, repeats),
        _repeat_interleave_heads(values, repeats),
    )


def _repeat_interleave_heads(tensor, repeats: int):
    if hasattr(tensor, "repeat_interleave"):
        return tensor.repeat_interleave(repeats, dim=1)
    expanded = []
    for batch in tensor.data:
        batch_heads = []
        for head in batch:
            for _ in range(repeats):
                batch_heads.append(_clone_nested(head))
        expanded.append(batch_heads)
    return tensor.__class__(expanded, dtype=tensor.dtype)


def _gather_selected(tensor, indices):
    if hasattr(torch, "gather"):
        batch, query_heads, _, head_dim = tensor.shape
        _, query_tokens, selected_tokens = indices.shape
        gather_index = indices[:, None, :, :, None].expand(
            batch, query_heads, query_tokens, selected_tokens, head_dim
        )
        source = tensor[:, :, None, :, :].expand(
            batch, query_heads, query_tokens, tensor.shape[2], head_dim
        )
        return torch.gather(source, 3, gather_index)
    gathered = []
    for batch_index, batch in enumerate(tensor.data):
        batch_heads = []
        for head in batch:
            query_rows = []
            for query_indices in indices.data[batch_index]:
                query_rows.append([_clone_nested(head[token_index]) for token_index in query_indices])
            batch_heads.append(query_rows)
        gathered.append(batch_heads)
    return tensor.__class__(gathered, dtype=tensor.dtype)


def _clone_nested(value):
    if isinstance(value, list):
        return [_clone_nested(item) for item in value]
    return value
