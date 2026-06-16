from __future__ import annotations

import random
from array import array

from cannbench.datasets.embedding import EmbeddingCase
from cannbench.datasets.softmax import SoftmaxCase


def materialize_softmax_inputs(
    case: SoftmaxCase, *, dtype: str, seed: int
) -> dict[str, object]:
    generator = random.Random(seed)
    size = 1
    for dim in case.shape:
        size *= dim

    values = tuple(round(generator.uniform(-1.0, 1.0), 6) for _ in range(size))
    return {
        "shape": case.shape,
        "dim": case.dim,
        "dtype": dtype,
        "values": values,
    }


def materialized_values_to_buffer(values: tuple[float, ...]) -> array[float]:
    return array("f", values)


def materialize_embedding_inputs(
    case: EmbeddingCase, *, dtype: str, seed: int
) -> dict[str, object]:
    generator = random.Random(seed)
    num_indices = 1
    for dim in case.index_shape:
        num_indices *= dim

    indices = tuple(generator.randrange(case.num_embeddings) for _ in range(num_indices))
    weights = tuple(
        round(generator.uniform(-1.0, 1.0), 6)
        for _ in range(case.num_embeddings * case.embedding_dim)
    )
    return {
        "index_shape": case.index_shape,
        "dtype": dtype,
        "indices": indices,
        "weights": weights,
        "num_embeddings": case.num_embeddings,
        "embedding_dim": case.embedding_dim,
    }
