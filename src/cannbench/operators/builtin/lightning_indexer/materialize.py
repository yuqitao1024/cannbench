from __future__ import annotations

import random

from .cases import LightningIndexerCase


def valid_context_lengths(case: LightningIndexerCase) -> tuple[int, ...]:
    if not case.causal:
        return (case.context_tokens,) * (case.batch * case.query_tokens)
    first_length = case.context_tokens - case.query_tokens + 1
    row_lengths = tuple(
        first_length + query_index for query_index in range(case.query_tokens)
    )
    return row_lengths * case.batch


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
        "valid_context_lengths": valid_context_lengths(case),
        "dtype": dtype,
        "query": query,
        "keys": keys,
        "weights": weights,
    }
    if case.phase is not None:
        payload["phase"] = case.phase
    return payload
