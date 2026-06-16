from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files


@dataclass(frozen=True)
class SoftmaxCase:
    case_id: str
    family: str
    shape: tuple[int, ...]
    dim: int
    source_kind: str
    source_project: str
    source_model: str
    source_file: str
    source_op: str

    def __post_init__(self) -> None:
        shape = tuple(int(value) for value in self.shape)
        if not shape:
            raise ValueError("shape must not be empty")
        if any(value <= 0 for value in shape):
            raise ValueError("shape must contain only positive integers")
        object.__setattr__(self, "shape", shape)


@dataclass(frozen=True)
class SoftmaxDataset:
    name: str
    cases: tuple[SoftmaxCase, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "cases", tuple(self.cases))


@lru_cache(maxsize=None)
def get_softmax_dataset(name: str) -> SoftmaxDataset:
    resource = files("cannbench.datasets.data.softmax").joinpath(f"{name}.json")
    if not resource.is_file():
        raise ValueError(f"Unknown softmax dataset: {name}")

    payload = json.loads(resource.read_text())
    cases = tuple(SoftmaxCase(shape=tuple(item["shape"]), **{k: v for k, v in item.items() if k != "shape"}) for item in payload["cases"])
    return SoftmaxDataset(name=payload["name"], cases=cases)


def get_softmax_case(dataset_name: str, case_id: str) -> SoftmaxCase:
    dataset = get_softmax_dataset(dataset_name)
    for case in dataset.cases:
        if case.case_id == case_id:
            return case
    raise ValueError(f"Unknown softmax case: {case_id}")
