from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files


@dataclass(frozen=True)
class LightningIndexerCase:
    case_id: str
    family: str
    batch: int
    query_tokens: int
    context_tokens: int
    index_heads: int
    index_dim: int
    top_k: int
    source_kind: str
    source_project: str
    source_model: str
    source_file: str
    source_op: str

    def __post_init__(self) -> None:
        for name in (
            "batch",
            "query_tokens",
            "context_tokens",
            "index_heads",
            "index_dim",
            "top_k",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.top_k > self.context_tokens:
            raise ValueError("top_k must not exceed context_tokens")

    @property
    def payload(self) -> dict[str, object]:
        payload = {
            "batch": self.batch,
            "query_tokens": self.query_tokens,
            "context_tokens": self.context_tokens,
            "index_heads": self.index_heads,
            "index_dim": self.index_dim,
            "top_k": self.top_k,
        }
        if self.phase is not None:
            payload["phase"] = self.phase
        return payload

    @property
    def phase(self) -> str | None:
        if self.family.startswith("decode_") or "_decode_" in self.family:
            return "decode"
        if self.family.startswith("prefill_") or "_prefill_" in self.family:
            return "prefill"
        return None


@dataclass(frozen=True)
class LightningIndexerDataset:
    name: str
    cases: tuple[LightningIndexerCase, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "cases", tuple(self.cases))


@lru_cache(maxsize=None)
def get_lightning_indexer_dataset(name: str) -> LightningIndexerDataset:
    resource = files(__package__).joinpath("data", f"{name}.json")
    if not resource.is_file():
        raise ValueError(f"Unknown lightning_indexer dataset: {name}")

    payload = json.loads(resource.read_text())
    cases = tuple(LightningIndexerCase(**item) for item in payload["cases"])
    return LightningIndexerDataset(name=payload["name"], cases=cases)


def get_lightning_indexer_case(
    dataset_name: str, case_id: str
) -> LightningIndexerCase:
    dataset = get_lightning_indexer_dataset(dataset_name)
    for case in dataset.cases:
        if case.case_id == case_id:
            return case
    raise ValueError(f"Unknown lightning_indexer case: {case_id}")
