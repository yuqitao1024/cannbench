from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

AxisRole = Literal["preserved", "contracted", "reduced", "produced"]
DeviceStatus = Literal["available", "unavailable"]


@dataclass(frozen=True)
class ShapeAxis:
    symbol: str
    value: int
    meaning: str
    role: AxisRole

    def __post_init__(self) -> None:
        if not self.symbol or not self.meaning:
            raise ValueError("axis symbol and meaning must not be empty")
        if self.value <= 0:
            raise ValueError("axis value must be positive")


@dataclass(frozen=True)
class ShapeTensor:
    id: str
    label: str
    axes: tuple[ShapeAxis, ...]
    logical_only: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "axes", tuple(self.axes))
        if not self.id or not self.label or not 1 <= len(self.axes) <= 3:
            raise ValueError("tensor requires an id, label, and one to three axes")


@dataclass(frozen=True)
class ShapeStage:
    id: str
    component: str
    title: str
    operation: str
    formula: str
    scope: str
    tensors: tuple[ShapeTensor, ...]
    input_ids: tuple[str, ...]
    output_ids: tuple[str, ...]
    contracted_axes: tuple[str, ...]
    insight: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "tensors", tuple(self.tensors))
        tensor_ids = {tensor.id for tensor in self.tensors}
        references = set(self.input_ids) | set(self.output_ids)
        unknown = references - tensor_ids
        if unknown:
            raise ValueError(f"stage references unknown tensor id: {sorted(unknown)}")


@dataclass(frozen=True)
class DeviceKernelTrace:
    id: str
    title: str
    summary: str
    task_count: int
    used_core_count: int
    task_formula: str
    task_axes: tuple[ShapeAxis, ...]
    tile_tensors: tuple[ShapeTensor, ...]
    steps: tuple[str, ...]


@dataclass(frozen=True)
class DeviceExecutionTrace:
    status: DeviceStatus
    implementation: str
    version: str | None
    message: str | None
    kernels: tuple[DeviceKernelTrace, ...]


@dataclass(frozen=True)
class ShapeTraceKey:
    operator: str
    dataset: str
    case_id: str
    phase: str
    group: str


@dataclass(frozen=True)
class ShapeTrace:
    schema_version: int
    operator: str
    dataset: str
    case_id: str
    phase: str
    group: str
    symbols: tuple[ShapeAxis, ...]
    stages: tuple[ShapeStage, ...]
    device_execution: DeviceExecutionTrace

    def __post_init__(self) -> None:
        object.__setattr__(self, "symbols", tuple(self.symbols))
        object.__setattr__(self, "stages", tuple(self.stages))
        if self.schema_version != 1:
            raise ValueError("shape trace schema_version must be 1")
        stage_ids = [stage.id for stage in self.stages]
        if not stage_ids or len(stage_ids) != len(set(stage_ids)):
            raise ValueError("shape trace stage ids must be non-empty and unique")


def shape_trace_to_payload(value: ShapeTrace | ShapeTraceKey) -> dict[str, Any]:
    return asdict(value)
