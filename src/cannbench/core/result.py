from __future__ import annotations

from dataclasses import dataclass

def _normalize_payload_value(value: object) -> object:
    if isinstance(value, tuple):
        return tuple(_normalize_payload_value(item) for item in value)
    if isinstance(value, list):
        return tuple(_normalize_payload_value(item) for item in value)
    if isinstance(value, dict):
        return {str(key): _normalize_payload_value(item) for key, item in value.items()}
    return value


def _json_payload_value(value: object) -> object:
    if isinstance(value, tuple):
        return [_json_payload_value(item) for item in value]
    if isinstance(value, list):
        return [_json_payload_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_payload_value(item) for key, item in value.items()}
    return value


def _payload_summary_value(value: object) -> str:
    if isinstance(value, tuple):
        return "x".join(str(item) for item in value)
    if isinstance(value, list):
        return "x".join(str(item) for item in value)
    if isinstance(value, dict):
        items = sorted((str(key), _payload_summary_value(item)) for key, item in value.items())
        return ", ".join(f"{key}={item}" for key, item in items)
    return str(value)


def _payload_key_order(
    key: str,
    payload_key_order: tuple[str, ...],
) -> tuple[int, str]:
    preferred = {
        "dimensions": 0,
        "dim": 1,
        "input_shape": 0,
        "index_shape": 1,
    }
    if key in payload_key_order:
        return (payload_key_order.index(key), key)
    return (preferred.get(key, 100), key)


@dataclass(frozen=True)
class OperatorCase:
    case_id: str
    family: str
    source_kind: str
    source_project: str
    source_model: str
    source_file: str
    source_op: str
    payload: dict[str, object]
    payload_key_order: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.case_id.strip():
            raise ValueError("case_id must not be empty")
        if not self.family.strip():
            raise ValueError("family must not be empty")
        if not self.source_kind.strip():
            raise ValueError("source_kind must not be empty")
        if not self.source_project.strip():
            raise ValueError("source_project must not be empty")
        if not self.source_model.strip():
            raise ValueError("source_model must not be empty")
        if not self.source_file.strip():
            raise ValueError("source_file must not be empty")
        if not self.source_op.strip():
            raise ValueError("source_op must not be empty")
        if not self.payload:
            raise ValueError("payload must not be empty")
        normalized = {
            str(key): _normalize_payload_value(value)
            for key, value in self.payload.items()
        }
        object.__setattr__(self, "payload", normalized)
        object.__setattr__(
            self,
            "payload_key_order",
            tuple(str(key) for key in self.payload_key_order),
        )

    @property
    def payload_summary(self) -> str:
        items = (
            (key, _payload_summary_value(value))
            for key, value in sorted(
                self.payload.items(),
                key=lambda item: _payload_key_order(item[0], self.payload_key_order),
            )
        )
        return ", ".join(f"{key}={value}" for key, value in items)

    def to_json_dict(self) -> dict[str, object]:
        return {
            "case_id": self.case_id,
            "family": self.family,
            "source_kind": self.source_kind,
            "source_project": self.source_project,
            "source_model": self.source_model,
            "source_file": self.source_file,
            "source_op": self.source_op,
            "payload": _json_payload_value(self.payload),
        }


@dataclass(frozen=True)
class OperatorBenchmarkResult:
    backend: str
    device_name: str
    op: str
    dtype: str
    case: OperatorCase

    def to_json_dict(self) -> dict[str, object]:
        return {
            "backend": self.backend,
            "device_name": self.device_name,
            "op": self.op,
            "dtype": self.dtype,
            "case": self.case.to_json_dict(),
        }


@dataclass(frozen=True)
class WorkflowBenchmarkResult:
    backend: str
    device_name: str
    workflow: str
    phase: str
    dataset: str
    case_id: str
    steps: tuple[OperatorBenchmarkResult, ...]

    def to_json_dict(self) -> dict[str, object]:
        return {
            "backend": self.backend,
            "device_name": self.device_name,
            "workflow": self.workflow,
            "phase": self.phase,
            "dataset": self.dataset,
            "case_id": self.case_id,
            "steps": [step.to_json_dict() for step in self.steps],
        }
