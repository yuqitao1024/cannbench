from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files


@dataclass(frozen=True)
class AdaptiveMaxPool3DGradCase:
    case_id: str
    family: str
    input_shape: tuple[int, ...]
    output_size: tuple[int, int, int]
    source_kind: str
    source_project: str
    source_model: str
    source_file: str
    source_op: str

    def __post_init__(self) -> None:
        input_shape = tuple(int(value) for value in self.input_shape)
        output_size = tuple(int(value) for value in self.output_size)
        if len(input_shape) != 5 or any(value <= 0 for value in input_shape):
            raise ValueError("input_shape must be a positive [N, C, D, H, W] shape")
        if len(output_size) != 3 or any(value <= 0 for value in output_size):
            raise ValueError("output_size must be a positive [D_out, H_out, W_out] shape")
        if any(out > inp for out, inp in zip(output_size, input_shape[2:])):
            raise ValueError("output_size must not exceed input spatial dimensions")
        object.__setattr__(self, "input_shape", input_shape)
        object.__setattr__(self, "output_size", output_size)

    @property
    def output_shape(self) -> tuple[int, int, int, int, int]:
        return (
            self.input_shape[0],
            self.input_shape[1],
            self.output_size[0],
            self.output_size[1],
            self.output_size[2],
        )

    @property
    def payload(self) -> dict[str, object]:
        return {
            "input_shape": self.input_shape,
            "output_size": self.output_size,
        }


@dataclass(frozen=True)
class AdaptiveMaxPool3DGradDataset:
    name: str
    cases: tuple[AdaptiveMaxPool3DGradCase, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "cases", tuple(self.cases))


@lru_cache(maxsize=None)
def get_adaptivemaxpool3dgrad_dataset(name: str) -> AdaptiveMaxPool3DGradDataset:
    resource = files(__package__).joinpath("data", f"{name}.json")
    if not resource.is_file():
        raise ValueError(f"Unknown adaptivemaxpool3dgrad dataset: {name}")

    payload = json.loads(resource.read_text())
    cases = tuple(
        AdaptiveMaxPool3DGradCase(
            input_shape=tuple(item["input_shape"]),
            output_size=tuple(item["output_size"]),
            **{
                k: v
                for k, v in item.items()
                if k not in {"input_shape", "output_size"}
            },
        )
        for item in payload["cases"]
    )
    return AdaptiveMaxPool3DGradDataset(name=payload["name"], cases=cases)


def get_adaptivemaxpool3dgrad_case(
    dataset_name: str,
    case_id: str,
) -> AdaptiveMaxPool3DGradCase:
    dataset = get_adaptivemaxpool3dgrad_dataset(dataset_name)
    for case in dataset.cases:
        if case.case_id == case_id:
            return case
    raise ValueError(f"Unknown adaptivemaxpool3dgrad case: {case_id}")
