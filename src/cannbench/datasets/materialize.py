from __future__ import annotations

import random
from array import array

from cannbench.datasets.gather import GatherCase
from cannbench.datasets.index_select import IndexSelectCase
from cannbench.datasets.embedding import EmbeddingCase
from cannbench.datasets.masked_select import MaskedSelectCase
from cannbench.datasets.cross_entropy import CrossEntropyCase
from cannbench.datasets.softmax import SoftmaxCase
from cannbench.datasets.take_along_dim import TakeAlongDimCase


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


def materialize_gather_inputs(
    case: GatherCase, *, dtype: str, seed: int
) -> dict[str, object]:
    generator = random.Random(seed)
    input_size = 1
    for dim in case.input_shape:
        input_size *= dim
    index_size = 1
    for dim in case.index_shape:
        index_size *= dim

    values = tuple(round(generator.uniform(-1.0, 1.0), 6) for _ in range(input_size))
    indices = tuple(generator.randrange(case.input_shape[case.dim]) for _ in range(index_size))
    return {
        "input_shape": case.input_shape,
        "index_shape": case.index_shape,
        "dim": case.dim,
        "dtype": dtype,
        "values": values,
        "indices": indices,
    }


def materialize_index_select_inputs(
    case: IndexSelectCase, *, dtype: str, seed: int
) -> dict[str, object]:
    generator = random.Random(seed)
    input_size = 1
    for dim in case.input_shape:
        input_size *= dim
    index_size = 1
    for dim in case.index_shape:
        index_size *= dim

    values = tuple(round(generator.uniform(-1.0, 1.0), 6) for _ in range(input_size))
    indices = tuple(generator.randrange(case.input_shape[case.dim]) for _ in range(index_size))
    return {
        "input_shape": case.input_shape,
        "index_shape": case.index_shape,
        "dim": case.dim,
        "dtype": dtype,
        "values": values,
        "indices": indices,
    }


def materialize_take_along_dim_inputs(
    case: TakeAlongDimCase, *, dtype: str, seed: int
) -> dict[str, object]:
    generator = random.Random(seed)
    input_size = 1
    for dim in case.input_shape:
        input_size *= dim
    index_size = 1
    for dim in case.index_shape:
        index_size *= dim

    values = tuple(round(generator.uniform(-1.0, 1.0), 6) for _ in range(input_size))
    indices = tuple(generator.randrange(case.input_shape[case.dim]) for _ in range(index_size))
    return {
        "input_shape": case.input_shape,
        "index_shape": case.index_shape,
        "dim": case.dim,
        "dtype": dtype,
        "values": values,
        "indices": indices,
    }


def materialize_masked_select_inputs(
    case: MaskedSelectCase, *, dtype: str, seed: int
) -> dict[str, object]:
    generator = random.Random(seed)
    input_size = 1
    for dim in case.input_shape:
        input_size *= dim
    mask_size = 1
    for dim in case.mask_shape:
        mask_size *= dim

    values = tuple(round(generator.uniform(-1.0, 1.0), 6) for _ in range(input_size))
    mask = tuple(generator.random() < case.mask_density for _ in range(mask_size))
    return {
        "input_shape": case.input_shape,
        "mask_shape": case.mask_shape,
        "mask_density": case.mask_density,
        "dtype": dtype,
        "values": values,
        "mask": mask,
    }


def materialize_cross_entropy_inputs(
    case: CrossEntropyCase, *, dtype: str, seed: int
) -> dict[str, object]:
    generator = random.Random(seed)
    logits_size = 1
    for dim in case.logits_shape:
        logits_size *= dim
    target_size = 1
    for dim in case.target_shape:
        target_size *= dim

    logits = tuple(round(generator.uniform(-1.0, 1.0), 6) for _ in range(logits_size))
    targets = tuple(generator.randrange(case.num_classes) for _ in range(target_size))
    return {
        "logits_shape": case.logits_shape,
        "target_shape": case.target_shape,
        "num_classes": case.num_classes,
        "dtype": dtype,
        "logits": logits,
        "targets": targets,
    }
