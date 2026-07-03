from __future__ import annotations

import random
from array import array

from cannbench.datasets.gather import GatherCase
from cannbench.datasets.index_select import IndexSelectCase
from cannbench.datasets.index_add import IndexAddCase
from cannbench.datasets.index_put import IndexPutCase
from cannbench.datasets.embedding import EmbeddingCase
from cannbench.datasets.masked_select import MaskedSelectCase
from cannbench.datasets.cross_entropy import CrossEntropyCase
from cannbench.datasets.scatter_add import ScatterAddCase
from cannbench.datasets.scatter import ScatterCase
from cannbench.datasets.lightning_indexer import LightningIndexerCase
from cannbench.datasets.softmax import SoftmaxCase
from cannbench.datasets.sparse_attention import SparseAttentionCase
from cannbench.datasets.take_along_dim import TakeAlongDimCase
from cannbench.datasets.topk import TopKCase


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


def materialize_topk_inputs(
    case: TopKCase, *, dtype: str, seed: int
) -> dict[str, object]:
    generator = random.Random(seed)
    input_size = 1
    for dim in case.input_shape:
        input_size *= dim

    values = tuple(round(generator.uniform(-1.0, 1.0), 6) for _ in range(input_size))
    return {
        "input_shape": case.input_shape,
        "k": case.k,
        "dim": case.dim,
        "largest": case.largest,
        "sorted": case.sorted,
        "dtype": dtype,
        "values": values,
    }


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
    return {
        "query_shape": query_shape,
        "key_shape": key_shape,
        "weight_shape": weight_shape,
        "top_k": case.top_k,
        "dtype": dtype,
        "query": query,
        "keys": keys,
        "weights": weights,
    }


def materialize_sparse_attention_inputs(
    case: SparseAttentionCase, *, dtype: str, seed: int
) -> dict[str, object]:
    generator = random.Random(seed)
    query_shape = (
        case.batch,
        case.query_heads,
        case.query_tokens,
        case.head_dim,
    )
    key_shape = (
        case.batch,
        case.kv_heads,
        case.context_tokens,
        case.head_dim,
    )
    value_shape = key_shape
    indices_shape = (case.batch, case.query_tokens, case.selected_tokens)
    query_size = case.batch * case.query_heads * case.query_tokens * case.head_dim
    kv_size = case.batch * case.kv_heads * case.context_tokens * case.head_dim
    indices_size = case.batch * case.query_tokens * case.selected_tokens

    query = tuple(round(generator.uniform(-1.0, 1.0), 6) for _ in range(query_size))
    keys = tuple(round(generator.uniform(-1.0, 1.0), 6) for _ in range(kv_size))
    values = tuple(round(generator.uniform(-1.0, 1.0), 6) for _ in range(kv_size))
    if case.causal and case.phase == "prefill":
        generated_indices = []
        for _batch in range(case.batch):
            for query_index in range(case.query_tokens):
                upper_bound = min(case.context_tokens, query_index + 1)
                for _selected in range(case.selected_tokens):
                    generated_indices.append(generator.randrange(upper_bound))
        indices = tuple(generated_indices)
    else:
        indices = tuple(generator.randrange(case.context_tokens) for _ in range(indices_size))
    return {
        "query_shape": query_shape,
        "key_shape": key_shape,
        "value_shape": value_shape,
        "indices_shape": indices_shape,
        "causal": case.causal,
        "phase": case.phase,
        "dtype": dtype,
        "query": query,
        "keys": keys,
        "values": values,
        "indices": indices,
    }


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


def materialize_index_add_inputs(
    case: IndexAddCase, *, dtype: str, seed: int
) -> dict[str, object]:
    generator = random.Random(seed)
    input_size = 1
    for dim in case.input_shape:
        input_size *= dim
    index_size = 1
    for dim in case.index_shape:
        index_size *= dim
    src_size = 1
    for dim in case.src_shape:
        src_size *= dim

    values = tuple(round(generator.uniform(-1.0, 1.0), 6) for _ in range(input_size))
    indices = tuple(generator.randrange(case.input_shape[case.dim]) for _ in range(index_size))
    src = tuple(round(generator.uniform(-1.0, 1.0), 6) for _ in range(src_size))
    return {
        "input_shape": case.input_shape,
        "index_shape": case.index_shape,
        "src_shape": case.src_shape,
        "dim": case.dim,
        "dtype": dtype,
        "values": values,
        "indices": indices,
        "src": src,
    }


def materialize_index_put_inputs(
    case: IndexPutCase, *, dtype: str, seed: int
) -> dict[str, object]:
    generator = random.Random(seed)
    input_size = 1
    for dim in case.input_shape:
        input_size *= dim
    index_size = 1
    for dim in case.index_shapes[0]:
        index_size *= dim
    values_size = 1
    for dim in case.values_shape:
        values_size *= dim

    values = tuple(round(generator.uniform(-1.0, 1.0), 6) for _ in range(input_size))
    indices = tuple(
        tuple(generator.randrange(case.input_shape[axis]) for _ in range(index_size))
        for axis in range(len(case.index_shapes))
    )
    put_values = tuple(round(generator.uniform(-1.0, 1.0), 6) for _ in range(values_size))
    return {
        "input_shape": case.input_shape,
        "index_shapes": case.index_shapes,
        "values_shape": case.values_shape,
        "accumulate": case.accumulate,
        "dtype": dtype,
        "values": values,
        "indices": indices,
        "put_values": put_values,
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


def materialize_scatter_add_inputs(
    case: ScatterAddCase, *, dtype: str, seed: int
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
    src = tuple(round(generator.uniform(-1.0, 1.0), 6) for _ in range(index_size))
    return {
        "input_shape": case.input_shape,
        "index_shape": case.index_shape,
        "src_shape": case.src_shape,
        "dim": case.dim,
        "dtype": dtype,
        "values": values,
        "indices": indices,
        "src": src,
    }


def materialize_scatter_inputs(
    case: ScatterCase, *, dtype: str, seed: int
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
    src = tuple(round(generator.uniform(-1.0, 1.0), 6) for _ in range(index_size))
    return {
        "input_shape": case.input_shape,
        "index_shape": case.index_shape,
        "src_shape": case.src_shape,
        "dim": case.dim,
        "dtype": dtype,
        "values": values,
        "indices": indices,
        "src": src,
    }
