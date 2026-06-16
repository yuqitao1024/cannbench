from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files


@dataclass(frozen=True)
class ScatterAddCase:
    case_id: str
    family: str
    input_shape: tuple[int, ...]
    index_shape: tuple[int, ...]
    src_shape: tuple[int, ...]
    dim: int
    source_kind: str
    source_project: str
    source_model: str
    source_file: str
    source_op: str

    def __post_init__(self) -> None:
        input_shape = tuple(int(value) for value in self.input_shape)
        index_shape = tuple(int(value) for value in self.index_shape)
        src_shape = tuple(int(value) for value in self.src_shape)
        if not input_shape or any(value <= 0 for value in input_shape):
            raise ValueError("input_shape must contain only positive integers")
        if not index_shape or any(value <= 0 for value in index_shape):
            raise ValueError("index_shape must contain only positive integers")
        if not src_shape or any(value <= 0 for value in src_shape):
            raise ValueError("src_shape must contain only positive integers")
        if len(index_shape) != len(input_shape):
            raise ValueError("index_shape must have the same rank as input_shape")
        if src_shape != index_shape:
            raise ValueError("src_shape must match index_shape")
        if self.dim < -len(input_shape) or self.dim >= len(input_shape):
            raise ValueError("dim must address an axis in input_shape")
        object.__setattr__(self, "input_shape", input_shape)
        object.__setattr__(self, "index_shape", index_shape)
        object.__setattr__(self, "src_shape", src_shape)

    @property
    def payload(self) -> dict[str, object]:
        return {
            "input_shape": self.input_shape,
            "index_shape": self.index_shape,
            "src_shape": self.src_shape,
            "dim": self.dim,
        }


@dataclass(frozen=True)
class ScatterAddDataset:
    name: str
    cases: tuple[ScatterAddCase, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "cases", tuple(self.cases))


@lru_cache(maxsize=None)
def get_scatter_add_dataset(name: str) -> ScatterAddDataset:
    resource = files("cannbench.datasets.data.scatter_add").joinpath(f"{name}.json")
    if not resource.is_file():
        raise ValueError(f"Unknown scatter_add dataset: {name}")

    payload = json.loads(resource.read_text())
    cases = tuple(
        ScatterAddCase(
            input_shape=tuple(item["input_shape"]),
            index_shape=tuple(item["index_shape"]),
            src_shape=tuple(item["src_shape"]),
            **{
                k: v
                for k, v in item.items()
                if k not in {"input_shape", "index_shape", "src_shape"}
            },
        )
        for item in payload["cases"]
    )
    return ScatterAddDataset(name=payload["name"], cases=cases)


def get_scatter_add_case(dataset_name: str, case_id: str) -> ScatterAddCase:
    dataset = get_scatter_add_dataset(dataset_name)
    for case in dataset.cases:
        if case.case_id == case_id:
            return case
    raise ValueError(f"Unknown scatter_add case: {case_id}")
