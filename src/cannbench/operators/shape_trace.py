from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

AxisRole = Literal["preserved", "contracted", "reduced", "produced"]
DeviceStatus = Literal["available", "unavailable"]


def _require_nonempty(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must not be empty")
    return value


def _require_unique(values: list[str], field_name: str) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"{field_name} must be unique")


@dataclass(frozen=True)
class ShapeAxis:
    symbol: str
    value: int
    meaning: str
    role: AxisRole

    def __post_init__(self) -> None:
        _require_nonempty(self.symbol, "axis symbol")
        _require_nonempty(self.meaning, "axis meaning")
        if self.role not in {"preserved", "contracted", "reduced", "produced"}:
            raise ValueError(f"invalid axis role: {self.role}")
        if isinstance(self.value, bool) or not isinstance(self.value, int) or self.value <= 0:
            raise ValueError("axis value must be positive")


@dataclass(frozen=True)
class ShapeTensor:
    id: str
    label: str
    axes: tuple[ShapeAxis, ...]
    logical_only: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "axes", tuple(self.axes))
        _require_nonempty(self.id, "tensor id")
        _require_nonempty(self.label, "tensor label")
        if not 1 <= len(self.axes) <= 3 or not all(
            isinstance(axis, ShapeAxis) for axis in self.axes
        ):
            raise ValueError("tensor requires one to three shape axes")
        _require_unique(
            [axis.symbol for axis in self.axes], "tensor axis symbols"
        )
        if not isinstance(self.logical_only, bool):
            raise ValueError("tensor logical_only must be boolean")


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
        object.__setattr__(self, "input_ids", tuple(self.input_ids))
        object.__setattr__(self, "output_ids", tuple(self.output_ids))
        object.__setattr__(self, "contracted_axes", tuple(self.contracted_axes))
        for field_name in (
            "id",
            "component",
            "title",
            "operation",
            "formula",
            "scope",
            "insight",
        ):
            _require_nonempty(getattr(self, field_name), f"stage {field_name}")
        if not self.tensors or not all(
            isinstance(tensor, ShapeTensor) for tensor in self.tensors
        ):
            raise ValueError("stage tensors must contain shape tensors")
        tensor_ids = [tensor.id for tensor in self.tensors]
        if len(tensor_ids) != len(set(tensor_ids)):
            duplicate = next(value for value in tensor_ids if tensor_ids.count(value) > 1)
            raise ValueError(f"stage has duplicate tensor id: {duplicate}")
        seen = set(tensor_ids)
        for field_name, reference_ids in (
            ("input_ids", self.input_ids),
            ("output_ids", self.output_ids),
        ):
            if not all(
                isinstance(tensor_id, str) and tensor_id.strip()
                for tensor_id in reference_ids
            ):
                raise ValueError(f"{field_name} must contain non-empty strings")
            seen_references: set[str] = set()
            for tensor_id in reference_ids:
                if tensor_id in seen_references:
                    raise ValueError(
                        f"{field_name} contains duplicate tensor id: {tensor_id}"
                    )
                seen_references.add(tensor_id)
        references = set(self.input_ids) | set(self.output_ids)
        unknown = references - seen
        if unknown:
            raise ValueError(f"stage references unknown tensor id: {sorted(unknown)}")
        if not all(
            isinstance(symbol, str) and symbol.strip()
            for symbol in self.contracted_axes
        ):
            raise ValueError("contracted_axes must contain non-empty strings")
        _require_unique(list(self.contracted_axes), "contracted_axes")
        tensor_axis_symbols = {
            axis.symbol for tensor in self.tensors for axis in tensor.axes
        }
        unknown_axes = set(self.contracted_axes) - tensor_axis_symbols
        if unknown_axes:
            raise ValueError(
                f"contracted_axes reference unknown tensor axes: {sorted(unknown_axes)}"
            )


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

    def __post_init__(self) -> None:
        object.__setattr__(self, "task_axes", tuple(self.task_axes))
        object.__setattr__(self, "tile_tensors", tuple(self.tile_tensors))
        object.__setattr__(self, "steps", tuple(self.steps))
        for field_name in ("id", "title", "summary", "task_formula"):
            _require_nonempty(getattr(self, field_name), f"kernel {field_name}")
        if (
            isinstance(self.task_count, bool)
            or not isinstance(self.task_count, int)
            or isinstance(self.used_core_count, bool)
            or not isinstance(self.used_core_count, int)
            or self.task_count <= 0
            or self.used_core_count <= 0
            or self.used_core_count > self.task_count
        ):
            raise ValueError(
                "kernel task/core counts must be positive and used cores must not exceed tasks"
            )
        if not self.task_axes or not all(
            isinstance(axis, ShapeAxis) for axis in self.task_axes
        ):
            raise ValueError("kernel task_axes must contain shape axes")
        _require_unique(
            [axis.symbol for axis in self.task_axes], "kernel task axis symbols"
        )
        if not self.tile_tensors or not all(
            isinstance(tensor, ShapeTensor) for tensor in self.tile_tensors
        ):
            raise ValueError("kernel tile_tensors must contain shape tensors")
        _require_unique(
            [tensor.id for tensor in self.tile_tensors], "kernel tile tensor ids"
        )
        if not self.steps or not all(
            isinstance(step, str) and step.strip() for step in self.steps
        ):
            raise ValueError("kernel steps must contain non-empty strings")


@dataclass(frozen=True)
class DeviceExecutionTrace:
    status: DeviceStatus
    implementation: str
    version: str | None
    message: str | None
    kernels: tuple[DeviceKernelTrace, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "kernels", tuple(self.kernels))
        if self.status not in {"available", "unavailable"}:
            raise ValueError(f"invalid device execution status: {self.status}")
        _require_nonempty(self.implementation, "device execution implementation")
        if self.version is not None:
            _require_nonempty(self.version, "device execution version")
        if self.message is not None:
            _require_nonempty(self.message, "device execution message")
        if not all(isinstance(kernel, DeviceKernelTrace) for kernel in self.kernels):
            raise ValueError("device execution kernels must contain kernel traces")
        _require_unique(
            [kernel.id for kernel in self.kernels], "device execution kernel ids"
        )
        if self.status == "available":
            if self.version is None or self.message is not None or not self.kernels:
                raise ValueError(
                    "available device execution requires a version, no message, and kernels"
                )
        elif self.message is None or self.kernels:
            raise ValueError(
                "unavailable device execution requires a message and no kernels"
            )


@dataclass(frozen=True)
class ShapeTraceKey:
    operator: str
    dataset: str
    case_id: str
    phase: str
    group: str

    def __post_init__(self) -> None:
        for field_name in ("operator", "dataset", "case_id", "phase", "group"):
            _require_nonempty(getattr(self, field_name), f"shape trace {field_name}")


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
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("shape trace schema_version must be 1")
        for field_name in ("operator", "dataset", "case_id", "phase", "group"):
            _require_nonempty(getattr(self, field_name), f"shape trace {field_name}")
        if not self.symbols or not all(
            isinstance(axis, ShapeAxis) for axis in self.symbols
        ):
            raise ValueError("shape trace symbols must contain shape axes")
        _require_unique(
            [axis.symbol for axis in self.symbols], "shape trace symbol ids"
        )
        if not self.stages or not all(
            isinstance(stage, ShapeStage) for stage in self.stages
        ):
            raise ValueError("shape trace stages must contain shape stages")
        stage_ids = [stage.id for stage in self.stages]
        _require_unique(stage_ids, "shape trace stage ids")
        symbols = {axis.symbol: axis for axis in self.symbols}
        for stage in self.stages:
            for tensor in stage.tensors:
                for axis in tensor.axes:
                    if axis.symbol not in symbols:
                        raise ValueError(
                            f"stage references unknown shape trace axis: {axis.symbol}"
                        )
                    if axis != symbols[axis.symbol]:
                        raise ValueError(
                            f"stage axis does not match shape trace symbol: {axis.symbol}"
                        )
        if not isinstance(self.device_execution, DeviceExecutionTrace):
            raise ValueError("shape trace device_execution must be a device trace")


def shape_trace_to_payload(value: ShapeTrace | ShapeTraceKey) -> dict[str, Any]:
    return asdict(value)
