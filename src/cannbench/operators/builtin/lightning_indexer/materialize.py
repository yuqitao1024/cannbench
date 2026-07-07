from __future__ import annotations

import random

from .cases import LightningIndexerCase


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
        "top_k": case.top_k,
        "dtype": dtype,
        "query": query,
        "keys": keys,
        "weights": weights,
    }
    if case.phase is not None:
        payload["phase"] = case.phase
    return payload
