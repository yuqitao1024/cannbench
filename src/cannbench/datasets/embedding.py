from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files


@dataclass(frozen=True)
class EmbeddingCase:
    case_id: str
    family: str
    num_embeddings: int
    embedding_dim: int
    index_shape: tuple[int, ...]
    source_kind: str
    source_project: str
    source_model: str
    source_file: str
    source_op: str

    def __post_init__(self) -> None:
        index_shape = tuple(int(value) for value in self.index_shape)
        if self.num_embeddings <= 0:
            raise ValueError("num_embeddings must be positive")
        if self.embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        if not index_shape or any(value <= 0 for value in index_shape):
            raise ValueError("index_shape must contain only positive integers")
        object.__setattr__(self, "index_shape", index_shape)

    @property
    def payload(self) -> dict[str, object]:
        return {
            "num_embeddings": self.num_embeddings,
            "embedding_dim": self.embedding_dim,
            "index_shape": self.index_shape,
        }


@dataclass(frozen=True)
class EmbeddingDataset:
    name: str
    cases: tuple[EmbeddingCase, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "cases", tuple(self.cases))


@lru_cache(maxsize=None)
def get_embedding_dataset(name: str) -> EmbeddingDataset:
    resource = files("cannbench.datasets.data.embedding").joinpath(f"{name}.json")
    if not resource.is_file():
        raise ValueError(f"Unknown embedding dataset: {name}")

    payload = json.loads(resource.read_text())
    cases = tuple(
        EmbeddingCase(
            index_shape=tuple(item["index_shape"]),
            **{k: v for k, v in item.items() if k != "index_shape"},
        )
        for item in payload["cases"]
    )
    return EmbeddingDataset(name=payload["name"], cases=cases)


def get_embedding_case(dataset_name: str, case_id: str) -> EmbeddingCase:
    dataset = get_embedding_dataset(dataset_name)
    for case in dataset.cases:
        if case.case_id == case_id:
            return case
    raise ValueError(f"Unknown embedding case: {case_id}")
