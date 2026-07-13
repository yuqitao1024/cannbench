from __future__ import annotations

import random

from .cases import IndexAddCase


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
    dim_size = case.input_shape[case.dim]
    if case.index_pattern == "unique_contiguous":
        if index_size > dim_size:
            raise ValueError("unique_contiguous index_pattern requires index_size <= dim_size")
        indices = tuple(range(index_size))
    elif case.index_pattern == "unique_random_permutation":
        if index_size > dim_size:
            raise ValueError(
                "unique_random_permutation index_pattern requires index_size <= dim_size"
            )
        indices = tuple(generator.sample(range(dim_size), index_size))
    else:
        indices = tuple(generator.randrange(dim_size) for _ in range(index_size))
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
