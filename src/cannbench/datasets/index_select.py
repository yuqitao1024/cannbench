from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files


@dataclass(frozen=True)
class IndexSelectCase:
    case_id: str
    family: str
    input_shape: tuple[int, ...]
    index_shape: tuple[int, ...]
    dim: int
    source_kind: str
    source_project: str
    source_model: str
    source_file: str
    source_op: str

    def __post_init__(self) -> None:
        input_shape = tuple(int(value) for value in self.input_shape)
        index_shape = tuple(int(value) for value in self.index_shape)
        if not input_shape or any(value <= 0 for value in input_shape):
            raise ValueError("input_shape must contain only positive integers")
        if not index_shape or any(value <= 0 for value in index_shape):
            raise ValueError("index_shape must contain only positive integers")
        if len(index_shape) != 1:
            raise ValueError("index_shape must be one-dimensional for torch.index_select")
        if self.dim < -len(input_shape) or self.dim >= len(input_shape):
            raise ValueError("dim must address an axis in input_shape")
        object.__setattr__(self, "input_shape", input_shape)
        object.__setattr__(self, "index_shape", index_shape)

    @property
    def payload(self) -> dict[str, object]:
        return {
            "input_shape": self.input_shape,
            "index_shape": self.index_shape,
            "dim": self.dim,
        }


@dataclass(frozen=True)
class IndexSelectDataset:
    name: str
    cases: tuple[IndexSelectCase, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "cases", tuple(self.cases))


@lru_cache(maxsize=None)
def get_index_select_dataset(name: str) -> IndexSelectDataset:
    resource = files("cannbench.datasets.data.index_select").joinpath(f"{name}.json")
    if not resource.is_file():
        raise ValueError(f"Unknown index_select dataset: {name}")

    payload = json.loads(resource.read_text())
    cases = tuple(
        IndexSelectCase(
            input_shape=tuple(item["input_shape"]),
            index_shape=tuple(item["index_shape"]),
            **{k: v for k, v in item.items() if k not in {"input_shape", "index_shape"}},
        )
        for item in payload["cases"]
    )
    return IndexSelectDataset(name=payload["name"], cases=cases)


def get_index_select_case(dataset_name: str, case_id: str) -> IndexSelectCase:
    dataset = get_index_select_dataset(dataset_name)
    for case in dataset.cases:
        if case.case_id == case_id:
            return case
    raise ValueError(f"Unknown index_select case: {case_id}")
