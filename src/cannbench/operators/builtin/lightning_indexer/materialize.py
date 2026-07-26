from __future__ import annotations

import random

from .cases import LightningIndexerCase


def valid_context_lengths(case: LightningIndexerCase) -> tuple[int, ...]:
    row_lengths = []
    for query_len, context_len, query_start in zip(
        case.resolved_query_lens,
        case.resolved_context_lens,
        case.resolved_query_start_positions,
        strict=True,
    ):
        for query_index in range(case.query_tokens):
            if query_index >= query_len:
                row_lengths.append(0)
            elif case.causal:
                row_lengths.append(min(context_len, query_start + query_index + 1))
            else:
                row_lengths.append(context_len)
    return tuple(row_lengths)


def materialize_lightning_indexer_inputs(
    case: LightningIndexerCase, *, dtype: str, seed: int
) -> dict[str, object]:
    generator = random.Random(seed)
    query_shape = (
        case.batch,
        case.query_tokens,
        case.index_heads,
        case.index_dim,
    )
    key_shape = (case.batch, case.context_tokens, case.index_dim)
    weight_shape = (case.batch, case.query_tokens, case.index_heads)
    query_size = case.batch * case.query_tokens * case.index_heads * case.index_dim
    key_size = case.batch * case.context_tokens * case.index_dim
    weight_size = case.batch * case.query_tokens * case.index_heads

    query = tuple(round(generator.uniform(-1.0, 1.0), 6) for _ in range(query_size))
    keys = tuple(round(generator.uniform(-1.0, 1.0), 6) for _ in range(key_size))
    weights = tuple(round(generator.uniform(0.0, 1.0), 6) for _ in range(weight_size))
    payload = {
        "query_shape": query_shape,
        "key_shape": key_shape,
        "weight_shape": weight_shape,
        "index_heads": case.index_heads,
        "index_dim": case.index_dim,
        "top_k": case.top_k,
        "causal": case.causal,
        "score_scale": case.score_scale,
        "tie_policy": case.tie_policy,
        "query_lens": case.resolved_query_lens,
        "context_lens": case.resolved_context_lens,
        "query_start_positions": case.resolved_query_start_positions,
        "cu_seqlens_q": case.cu_seqlens_q,
        "cu_seqlens_kv": case.cu_seqlens_kv,
        "page_block_size": case.resolved_page_block_size,
        "block_tables": case.block_tables,
        "valid_context_lengths": valid_context_lengths(case),
        "dtype": dtype,
        "query": query,
        "keys": keys,
        "weights": weights,
    }
    if case.phase is not None:
        payload["phase"] = case.phase
    if case.parallelism is not None:
        payload["parallelism"] = case.parallelism
    return payload
