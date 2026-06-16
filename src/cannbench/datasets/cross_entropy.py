from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files


@dataclass(frozen=True)
class CrossEntropyCase:
    case_id: str
    family: str
    logits_shape: tuple[int, ...]
    target_shape: tuple[int, ...]
    num_classes: int
    source_kind: str
    source_project: str
    source_model: str
    source_file: str
    source_op: str

    def __post_init__(self) -> None:
        logits_shape = tuple(int(value) for value in self.logits_shape)
        target_shape = tuple(int(value) for value in self.target_shape)
        if not logits_shape or any(value <= 0 for value in logits_shape):
            raise ValueError("logits_shape must contain only positive integers")
        if not target_shape or any(value <= 0 for value in target_shape):
            raise ValueError("target_shape must contain only positive integers")
        if len(target_shape) != len(logits_shape) - 1:
            raise ValueError("target_shape must be one rank smaller than logits_shape")
        if self.num_classes <= 0:
            raise ValueError("num_classes must be positive")
        if logits_shape[-1] != self.num_classes:
            raise ValueError("num_classes must match the last logits dimension")
        object.__setattr__(self, "logits_shape", logits_shape)
        object.__setattr__(self, "target_shape", target_shape)

    @property
    def payload(self) -> dict[str, object]:
        return {
            "logits_shape": self.logits_shape,
            "target_shape": self.target_shape,
            "num_classes": self.num_classes,
        }


@dataclass(frozen=True)
class CrossEntropyDataset:
    name: str
    cases: tuple[CrossEntropyCase, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "cases", tuple(self.cases))


@lru_cache(maxsize=None)
def get_cross_entropy_dataset(name: str) -> CrossEntropyDataset:
    resource = files("cannbench.datasets.data.cross_entropy").joinpath(f"{name}.json")
    if not resource.is_file():
        raise ValueError(f"Unknown cross_entropy dataset: {name}")

    payload = json.loads(resource.read_text())
    cases = tuple(
        CrossEntropyCase(
            logits_shape=tuple(item["logits_shape"]),
            target_shape=tuple(item["target_shape"]),
            **{k: v for k, v in item.items() if k not in {"logits_shape", "target_shape"}},
        )
        for item in payload["cases"]
    )
    return CrossEntropyDataset(name=payload["name"], cases=cases)


def get_cross_entropy_case(dataset_name: str, case_id: str) -> CrossEntropyCase:
    dataset = get_cross_entropy_dataset(dataset_name)
    for case in dataset.cases:
        if case.case_id == case_id:
            return case
    raise ValueError(f"Unknown cross_entropy case: {case_id}")
