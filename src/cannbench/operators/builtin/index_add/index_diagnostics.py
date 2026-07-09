from __future__ import annotations

import random
from collections import Counter
from collections.abc import Sequence

from .cases import get_index_add_dataset


def _rounded_ratio(numerator: int | float, denominator: int | float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def summarize_index_distribution(
    indices: Sequence[int],
    *,
    dim_size: int,
    block_size: int = 1024,
) -> dict[str, float | int]:
    index_size = len(indices)
    if index_size == 0:
        raise ValueError("indices must not be empty")
    if dim_size <= 0:
        raise ValueError("dim_size must be positive")
    if block_size <= 0:
        raise ValueError("block_size must be positive")

    counts = Counter(int(value) for value in indices)
    unique_count = len(counts)
    duplicate_count = index_size - unique_count
    max_bucket_count = max(counts.values())

    block_duplicate_ratios: list[float] = []
    for start in range(0, index_size, block_size):
        block = indices[start : start + block_size]
        block_unique_count = len(set(int(value) for value in block))
        block_duplicate_ratios.append(
            _rounded_ratio(len(block) - block_unique_count, len(block))
        )

    adjacent_pairs = max(index_size - 1, 0)
    adjacent_equal = 0
    adjacent_non_decreasing = 0
    for left, right in zip(indices, indices[1:]):
        if right == left:
            adjacent_equal += 1
        if right >= left:
            adjacent_non_decreasing += 1

    return {
        "index_size": index_size,
        "dim_size": dim_size,
        "unique_count": unique_count,
        "duplicate_count": duplicate_count,
        "duplicate_ratio": _rounded_ratio(duplicate_count, index_size),
        "load_factor": _rounded_ratio(index_size, dim_size),
        "max_bucket_count": max_bucket_count,
        "max_bucket_ratio": _rounded_ratio(max_bucket_count, index_size),
        "block_size": block_size,
        "block_count": len(block_duplicate_ratios),
        "mean_block_duplicate_ratio": sum(block_duplicate_ratios)
        / len(block_duplicate_ratios),
        "max_block_duplicate_ratio": max(block_duplicate_ratios),
        "adjacent_equal_ratio": _rounded_ratio(adjacent_equal, adjacent_pairs),
        "adjacent_non_decreasing_ratio": _rounded_ratio(
            adjacent_non_decreasing,
            adjacent_pairs,
        ),
    }


def summarize_index_add_dataset(
    dataset_name: str,
    *,
    dtype: str,
    seed: int,
    block_size: int = 1024,
) -> list[dict[str, object]]:
    dataset = get_index_add_dataset(dataset_name)
    summaries: list[dict[str, object]] = []

    for case in dataset.cases:
        generator = random.Random(seed)
        index_size = case.index_shape[0]
        indices = tuple(
            generator.randrange(case.input_shape[case.dim])
            for _ in range(index_size)
        )

        wrapped_dim = case.dim if case.dim >= 0 else case.dim + len(case.input_shape)
        dim_size = case.input_shape[wrapped_dim]
        distribution = summarize_index_distribution(
            indices,
            dim_size=dim_size,
            block_size=block_size,
        )
        summaries.append(
            {
                "case_id": case.case_id,
                "family": case.family,
                "rank": len(case.input_shape),
                "wrapped_dim": wrapped_dim,
                "input_shape": case.input_shape,
                "index_shape": case.index_shape,
                **distribution,
            }
        )

    return summaries
