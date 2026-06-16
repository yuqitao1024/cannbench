from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OperatorSpec:
    name: str
    supported_dtypes: tuple[str, ...]
    dataset_namespace: str
    runner_name: str

