from __future__ import annotations

from importlib import import_module
import os

try:
    torch = import_module("torch")
except ImportError:
    torch = None

__all__ = [
    "sparse_attention_forward",
    "_prefill_reference",
    "_decode_reference",
]

_HEAD_TILE_ENV = "CANNBENCH_SPARSE_ATTENTION_HEAD_TILE"
_SELECTED_PARTITIONS_ENV = "CANNBENCH_SPARSE_ATTENTION_SELECTED_PARTITIONS"
_SUPPORTED_TUNING = {(1, 1), (64, 1)}


def _read_positive_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be an integer, got {raw!r}") from exc
    if value <= 0:
        raise RuntimeError(f"{name} must be positive, got {value}")
    return value


def _resolve_tuning() -> tuple[int, int]:
    tuning = (
        _read_positive_int(_HEAD_TILE_ENV, 1),
        _read_positive_int(_SELECTED_PARTITIONS_ENV, 1),
    )
    if tuning not in _SUPPORTED_TUNING:
        raise RuntimeError(
            "unsupported sparse_attention tuning: "
            f"head_tile={tuning[0]}, selected_partitions={tuning[1]}"
        )
    return tuning


def sparse_attention_forward(
    query,
    shared_kv,
    indices,
    *,
    value_head_dim: int,
    phase: str,
    family: str,
    causal: bool,
):
    if phase not in {"prefill", "decode"}:
        raise RuntimeError(f"unsupported sparse_attention phase for custom op wrapper: {phase}")
    if family not in {
        "family_hd128",
        "family_hd256",
        "family_hd512",
        "family_hd576",
    }:
        raise RuntimeError(f"unsupported sparse_attention family for custom op wrapper: {family}")
    head_tile, selected_partitions = _resolve_tuning()
    custom_op = _load_registered_op()
    if custom_op is None:
        raise RuntimeError("aten_dsa_sparse_attention custom op is not registered")
    result = custom_op(
        query,
        shared_kv,
        indices,
        value_head_dim,
        phase,
        family,
        causal,
        head_tile,
        selected_partitions,
    )
    if not isinstance(result, tuple):
        return result
    output, lse = result
    return output.permute(0, 2, 1, 3), lse.permute(0, 2, 1)


def _load_registered_op():
    if torch is None:
        return None
    try:
        namespace = torch.ops.aten_dsa_sparse_attention
        return getattr(namespace, "sparse_attention_forward")
    except Exception:
        return None


def _prefill_reference(
    query,
    shared_kv,
    indices,
    *,
    value_head_dim: int,
    causal: bool,
    softmax_scale: float | None = None,
    topk_lengths=None,
):
    if torch is None:
        raise RuntimeError("torch is required for sparse_attention reference wrapper")
    values = shared_kv[:, :, :, :value_head_dim]
    expanded_keys, expanded_values = _expand_kv(
        shared_kv, values, query.shape[1]
    )
    context_tokens = shared_kv.shape[2]
    invalid_mask = None
    safe_indices = indices
    if hasattr(indices, "clamp"):
        invalid_mask = (indices < 0) | (indices >= context_tokens)
        if topk_lengths is not None:
            topk_positions = torch.arange(
                indices.shape[2], device=getattr(indices, "device", None)
            ).reshape(1, 1, indices.shape[2])
            invalid_mask = invalid_mask | (
                topk_positions >= topk_lengths.unsqueeze(-1)
            )
        safe_indices = indices.clamp(min=0, max=context_tokens - 1)
    selected_keys = _gather_selected(expanded_keys, safe_indices)
    selected_values = _gather_selected(expanded_values, safe_indices)
    scores = (query.unsqueeze(3) * selected_keys).sum(dim=-1)
    resolved_scale = (
        query.shape[-1] ** -0.5 if softmax_scale is None else softmax_scale
    )
    scores = scores / (1.0 / resolved_scale)
    if causal:
        positions = (
            torch.arange(query.shape[2], device=getattr(query, "device", None))
            + (shared_kv.shape[2] - query.shape[2])
        ).reshape(1, 1, query.shape[2], 1)
        causal_mask = indices[:, None, :, :] > positions
        invalid_mask = (
            causal_mask
            if invalid_mask is None
            else invalid_mask[:, None, :, :] | causal_mask
        )
    elif invalid_mask is not None:
        invalid_mask = invalid_mask[:, None, :, :]
    if invalid_mask is not None:
        scores = scores.masked_fill(invalid_mask, float("-inf"))
    scores_float = scores.float()
    probabilities = torch.softmax(scores_float, dim=-1)
    lse = torch.logsumexp(scores_float, dim=-1)
    if invalid_mask is not None:
        all_invalid = invalid_mask.all(dim=-1)
        probabilities = probabilities.masked_fill(all_invalid.unsqueeze(-1), 0.0)
        lse = lse.masked_fill(all_invalid, float("-inf"))
    output = (probabilities.to(query.dtype).unsqueeze(-1) * selected_values).sum(dim=-2)
    return output.permute(0, 2, 1, 3), lse.permute(0, 2, 1)


def _decode_reference(
    query,
    shared_kv,
    indices,
    *,
    value_head_dim: int,
    causal: bool,
    softmax_scale: float | None = None,
    topk_lengths=None,
):
    return _prefill_reference(
        query,
        shared_kv,
        indices,
        value_head_dim=value_head_dim,
        causal=causal,
        softmax_scale=softmax_scale,
        topk_lengths=topk_lengths,
    )


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
