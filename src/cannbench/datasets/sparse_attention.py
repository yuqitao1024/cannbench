from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files


@dataclass(frozen=True)
class SparseAttentionCase:
    case_id: str
    family: str
    batch: int
    query_heads: int
    kv_heads: int
    query_tokens: int
    context_tokens: int
    selected_tokens: int
    head_dim: int
    causal: bool
    phase: str
    source_kind: str
    source_project: str
    source_model: str
    source_file: str
    source_op: str

    def __post_init__(self) -> None:
        for name in (
            "batch",
            "query_heads",
            "kv_heads",
            "query_tokens",
            "context_tokens",
            "selected_tokens",
            "head_dim",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.query_heads % self.kv_heads != 0:
            raise ValueError("query_heads must be divisible by kv_heads")
        if self.selected_tokens > self.context_tokens:
            raise ValueError("selected_tokens must not exceed context_tokens")
        if self.phase not in {"decode", "prefill"}:
            raise ValueError("phase must be decode or prefill")

    @property
    def payload(self) -> dict[str, object]:
        return {
            "batch": self.batch,
            "query_heads": self.query_heads,
            "kv_heads": self.kv_heads,
            "query_tokens": self.query_tokens,
            "context_tokens": self.context_tokens,
            "selected_tokens": self.selected_tokens,
            "head_dim": self.head_dim,
            "causal": self.causal,
            "phase": self.phase,
        }


@dataclass(frozen=True)
class SparseAttentionDataset:
    name: str
    cases: tuple[SparseAttentionCase, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "cases", tuple(self.cases))


@lru_cache(maxsize=None)
def get_sparse_attention_dataset(name: str) -> SparseAttentionDataset:
    resource = files("cannbench.datasets.data.sparse_attention").joinpath(f"{name}.json")
    if not resource.is_file():
        raise ValueError(f"Unknown sparse_attention dataset: {name}")

    payload = json.loads(resource.read_text())
    cases = tuple(SparseAttentionCase(**item) for item in payload["cases"])
    return SparseAttentionDataset(name=payload["name"], cases=cases)


def get_sparse_attention_case(dataset_name: str, case_id: str) -> SparseAttentionCase:
    dataset = get_sparse_attention_dataset(dataset_name)
    for case in dataset.cases:
        if case.case_id == case_id:
            return case
    raise ValueError(f"Unknown sparse_attention case: {case_id}")
