from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files


@dataclass(frozen=True)
class TopKCase:
    case_id: str
    family: str
    input_shape: tuple[int, ...]
    k: int
    dim: int
    largest: bool
    sorted: bool
    source_kind: str
    source_project: str
    source_model: str
    source_file: str
    source_op: str

    def __post_init__(self) -> None:
        input_shape = tuple(int(value) for value in self.input_shape)
        if not input_shape or any(value <= 0 for value in input_shape):
            raise ValueError("input_shape must contain only positive integers")
        rank = len(input_shape)
        if self.dim < -rank or self.dim >= rank:
            raise ValueError("dim must address an axis in input_shape")
        normalized_dim = self.dim if self.dim >= 0 else rank + self.dim
        if self.k <= 0:
            raise ValueError("k must be positive")
        if self.k > input_shape[normalized_dim]:
            raise ValueError("k must not exceed the selected dimension")
        object.__setattr__(self, "input_shape", input_shape)

    @property
    def payload(self) -> dict[str, object]:
        return {
            "input_shape": self.input_shape,
            "k": self.k,
            "dim": self.dim,
            "largest": self.largest,
            "sorted": self.sorted,
        }


@dataclass(frozen=True)
class TopKDataset:
    name: str
    cases: tuple[TopKCase, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "cases", tuple(self.cases))


@lru_cache(maxsize=None)
def get_topk_dataset(name: str) -> TopKDataset:
    resource = files("cannbench.datasets.data.topk").joinpath(f"{name}.json")
    if not resource.is_file():
        raise ValueError(f"Unknown topk dataset: {name}")

    payload = json.loads(resource.read_text())
    cases = tuple(
        TopKCase(
            input_shape=tuple(item["input_shape"]),
            **{k: v for k, v in item.items() if k != "input_shape"},
        )
        for item in payload["cases"]
    )
    return TopKDataset(name=payload["name"], cases=cases)


def get_topk_case(dataset_name: str, case_id: str) -> TopKCase:
    dataset = get_topk_dataset(dataset_name)
    for case in dataset.cases:
        if case.case_id == case_id:
            return case
    raise ValueError(f"Unknown topk case: {case_id}")
