from __future__ import annotations

from cannbench.datasets.softmax import SoftmaxCase


def build_softmax_smoke_case(
    *,
    case_id: str,
    family: str,
    shape: tuple[int, ...],
    dim: int,
    source_model: str,
) -> SoftmaxCase:
    return SoftmaxCase(
        case_id=case_id,
        family=family,
        shape=shape,
        dim=dim,
        source_kind="synthetic_smoke",
        source_project="cannbench",
        source_model=source_model,
        source_file="built-in",
        source_op="softmax",
    )


def build_softmax_stress_case(
    *,
    case_id: str,
    family: str,
    shape: tuple[int, ...],
    dim: int,
    source_model: str,
) -> SoftmaxCase:
    return SoftmaxCase(
        case_id=case_id,
        family=family,
        shape=shape,
        dim=dim,
        source_kind="synthetic_boundary",
        source_project="cannbench",
        source_model=source_model,
        source_file="generated",
        source_op="softmax",
    )
