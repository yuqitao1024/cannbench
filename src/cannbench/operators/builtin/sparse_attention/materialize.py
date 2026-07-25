from __future__ import annotations

import random

from .cases import SparseAttentionCase

_MAX_HOST_MATERIALIZED_ELEMENTS = 64 * 1024 * 1024


def _requires_direct_device_inputs(case: SparseAttentionCase) -> bool:
    query_size = (
        case.batch * case.query_heads * case.query_tokens * case.qk_head_dim
    )
    shared_kv_size = (
        case.batch * case.kv_heads * case.context_tokens * case.qk_head_dim
    )
    indices_size = case.batch * case.query_tokens * case.selected_tokens
    return max(query_size, shared_kv_size, indices_size) >= (
        _MAX_HOST_MATERIALIZED_ELEMENTS
    )


def materialize_sparse_attention_inputs(
    case: SparseAttentionCase, *, dtype: str, seed: int
) -> dict[str, object]:
    generator = random.Random(seed)
    query_shape = (
        case.batch,
        case.query_heads,
        case.query_tokens,
        case.qk_head_dim,
    )
    shared_kv_shape = (
        case.batch,
        case.kv_heads,
        case.context_tokens,
        case.qk_head_dim,
    )
    indices_shape = (case.batch, case.query_tokens, case.selected_tokens)
    query_size = case.batch * case.query_heads * case.query_tokens * case.qk_head_dim
    shared_kv_size = (
        case.batch * case.kv_heads * case.context_tokens * case.qk_head_dim
    )
    indices_size = case.batch * case.query_tokens * case.selected_tokens

    query = tuple(round(generator.uniform(-1.0, 1.0), 6) for _ in range(query_size))
    shared_kv = tuple(
        round(generator.uniform(-1.0, 1.0), 6)
        for _ in range(shared_kv_size)
    )
    if case.causal:
        generated_indices = []
        for _batch in range(case.batch):
            for query_index in range(case.query_tokens):
                upper_bound = min(
                    case.context_tokens,
                    case.context_tokens - case.query_tokens + query_index + 1,
                )
                for _selected in range(case.selected_tokens):
                    generated_indices.append(generator.randrange(upper_bound))
        indices = tuple(generated_indices)
    else:
        indices = tuple(generator.randrange(case.context_tokens) for _ in range(indices_size))
    return {
        "query_shape": query_shape,
        "shared_kv_shape": shared_kv_shape,
        "indices_shape": indices_shape,
        "query_heads": case.query_heads,
        "kv_heads": case.kv_heads,
        "selected_tokens": case.selected_tokens,
        "qk_head_dim": case.qk_head_dim,
        "value_head_dim": case.value_head_dim,
        "causal": case.causal,
        "phase": case.phase,
        "dtype": dtype,
        "query": query,
        "shared_kv": shared_kv,
        "indices": indices,
    }
