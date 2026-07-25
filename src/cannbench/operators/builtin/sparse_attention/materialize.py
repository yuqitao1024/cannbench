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
    query = tuple(round(generator.uniform(-1.0, 1.0), 6) for _ in range(query_size))
    shared_kv = tuple(
        round(generator.uniform(-1.0, 1.0), 6)
        for _ in range(shared_kv_size)
    )
    generated_indices = []
    topk_lengths = case.resolved_topk_lengths
    for batch_index in range(case.batch):
        for query_index in range(case.query_tokens):
            row_index = batch_index * case.query_tokens + query_index
            if case.causal:
                upper_bound = min(
                    case.context_tokens,
                    case.context_tokens - case.query_tokens + query_index + 1,
                )
            else:
                upper_bound = case.context_tokens
            for selected_index in range(case.selected_tokens):
                if selected_index < topk_lengths[row_index]:
                    generated_indices.append(generator.randrange(upper_bound))
                else:
                    generated_indices.append(-1)
    indices = tuple(generated_indices)
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
        "softmax_scale": case.softmax_scale,
        "topk_lengths": topk_lengths,
        "dtype": dtype,
        "query": query,
        "shared_kv": shared_kv,
        "indices": indices,
    }
