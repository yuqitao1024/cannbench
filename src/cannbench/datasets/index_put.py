from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files


@dataclass(frozen=True)
class IndexPutCase:
    case_id: str
    family: str
    input_shape: tuple[int, ...]
    index_shapes: tuple[tuple[int, ...], ...]
    values_shape: tuple[int, ...]
    accumulate: bool
    source_kind: str
    source_project: str
    source_model: str
    source_file: str
    source_op: str

    def __post_init__(self) -> None:
        input_shape = tuple(int(value) for value in self.input_shape)
        index_shapes = tuple(
            tuple(int(value) for value in shape) for shape in self.index_shapes
        )
        values_shape = tuple(int(value) for value in self.values_shape)
        if not input_shape or any(value <= 0 for value in input_shape):
            raise ValueError("input_shape must contain only positive integers")
        if not index_shapes:
            raise ValueError("index_shapes must contain at least one index tensor")
        if len(index_shapes) > len(input_shape):
            raise ValueError("index_shapes cannot index more axes than input_shape rank")
        if any(not shape or any(value <= 0 for value in shape) for shape in index_shapes):
            raise ValueError("index_shapes must contain only positive integers")
        if len(set(index_shapes)) != 1:
            raise ValueError("index_shapes must share one broadcast shape")
        expected_values_shape = index_shapes[0] + input_shape[len(index_shapes):]
        if values_shape != expected_values_shape:
            raise ValueError("values_shape must match index broadcast shape plus trailing input axes")
        object.__setattr__(self, "input_shape", input_shape)
        object.__setattr__(self, "index_shapes", index_shapes)
        object.__setattr__(self, "values_shape", values_shape)

    @property
    def payload(self) -> dict[str, object]:
        return {
            "input_shape": self.input_shape,
            "index_shapes": self.index_shapes,
            "values_shape": self.values_shape,
            "accumulate": self.accumulate,
        }


@dataclass(frozen=True)
class IndexPutDataset:
    name: str
    cases: tuple[IndexPutCase, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "cases", tuple(self.cases))


@lru_cache(maxsize=None)
def get_index_put_dataset(name: str) -> IndexPutDataset:
    resource = files("cannbench.datasets.data.index_put").joinpath(f"{name}.json")
    if not resource.is_file():
        raise ValueError(f"Unknown index_put dataset: {name}")

    payload = json.loads(resource.read_text())
    cases = tuple(
        IndexPutCase(
            input_shape=tuple(item["input_shape"]),
            index_shapes=tuple(tuple(shape) for shape in item["index_shapes"]),
            values_shape=tuple(item["values_shape"]),
            **{
                k: v
                for k, v in item.items()
                if k not in {"input_shape", "index_shapes", "values_shape"}
            },
        )
        for item in payload["cases"]
    )
    return IndexPutDataset(name=payload["name"], cases=cases)


def get_index_put_case(dataset_name: str, case_id: str) -> IndexPutCase:
    dataset = get_index_put_dataset(dataset_name)
    for case in dataset.cases:
        if case.case_id == case_id:
            return case
    raise ValueError(f"Unknown index_put case: {case_id}")
