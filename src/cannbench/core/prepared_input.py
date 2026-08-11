from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from cannbench.core.result import OperatorCase
from cannbench.datasets import get_operator_case

SCHEMA_VERSION = 1
WORKFLOW_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class OperatorInputBinding:
    op: str
    dtype: str
    dataset: str
    case_id: str
    seed: int
    output_index: int = 0

    def to_json_dict(self) -> dict[str, object]:
        return {
            "op": self.op,
            "dtype": self.dtype,
            "dataset": self.dataset,
            "case_id": self.case_id,
            "seed": self.seed,
            "output_index": self.output_index,
        }


@dataclass(frozen=True)
class PreparedOperatorInput:
    op: str
    dtype: str
    dataset: str
    seed: int
    case: OperatorCase
    input_bindings: dict[str, OperatorInputBinding] = field(default_factory=dict)

    def to_json_dict(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "op": self.op,
            "dtype": self.dtype,
            "dataset": self.dataset,
            "seed": self.seed,
            "case": self.case.to_json_dict(),
            "input_bindings": {
                name: binding.to_json_dict()
                for name, binding in self.input_bindings.items()
            },
        }


@dataclass(frozen=True)
class PreparedWorkflowStep:
    contract: str
    consumes: tuple[str, ...]
    produces: tuple[str, ...]
    prepared: PreparedOperatorInput

    def __post_init__(self) -> None:
        contract = self.contract.strip()
        if not contract:
            raise ValueError("workflow step contract must not be empty")
        consumes = tuple(str(name).strip() for name in self.consumes)
        produces = tuple(str(name).strip() for name in self.produces)
        if not produces or any(not name for name in produces):
            raise ValueError("workflow step must produce at least one named output")
        if any(not name for name in consumes):
            raise ValueError("workflow step consumed output names must not be empty")
        if len(set(produces)) != len(produces):
            raise ValueError("workflow step produced output names must be unique")
        object.__setattr__(self, "contract", contract)
        object.__setattr__(self, "consumes", consumes)
        object.__setattr__(self, "produces", produces)

    def to_json_dict(self) -> dict[str, object]:
        return {
            "contract": self.contract,
            "consumes": list(self.consumes),
            "produces": list(self.produces),
            "prepared": self.prepared.to_json_dict(),
        }


@dataclass(frozen=True)
class PreparedWorkflowInput:
    workflow: str
    phase: str
    dataset: str
    case_id: str
    steps: tuple[PreparedWorkflowStep, ...]

    def __post_init__(self) -> None:
        for field_name in ("workflow", "phase", "dataset", "case_id"):
            value = str(getattr(self, field_name)).strip()
            if not value:
                raise ValueError(f"workflow {field_name} must not be empty")
            object.__setattr__(self, field_name, value)
        steps = tuple(self.steps)
        if not steps:
            raise ValueError("prepared workflow requires at least one step")
        available: set[str] = set()
        for step in steps:
            if step.prepared.case.case_id != self.case_id:
                raise ValueError(
                    "workflow component case_id mismatch: "
                    f"expected {self.case_id}, got {step.prepared.case.case_id}"
                )
            for name in step.consumes:
                if name not in available:
                    raise ValueError(
                        f"workflow step {step.contract} consumes unknown output {name!r}"
                    )
            for name in step.produces:
                if name in available:
                    raise ValueError(
                        f"workflow step {step.contract} produces duplicate output {name!r}"
                    )
                available.add(name)
        object.__setattr__(self, "steps", steps)

    def to_json_dict(self) -> dict[str, object]:
        return {
            "schema_version": WORKFLOW_SCHEMA_VERSION,
            "workflow": self.workflow,
            "phase": self.phase,
            "dataset": self.dataset,
            "case_id": self.case_id,
            "steps": [step.to_json_dict() for step in self.steps],
        }


def build_prepared_operator_input(
    *,
    op: str,
    dtype: str,
    dataset: str,
    case_id: str,
    seed: int,
    input_bindings: dict[str, OperatorInputBinding] | None = None,
) -> PreparedOperatorInput:
    case = get_operator_case(op, dataset, case_id)
    return PreparedOperatorInput(
        op=op,
        dtype=dtype,
        dataset=dataset,
        seed=seed,
        case=OperatorCase(
            case_id=case.case_id,
            family=case.family,
            source_kind=case.source_kind,
            source_project=case.source_project,
            source_model=case.source_model,
            source_file=case.source_file,
            source_op=case.source_op,
            payload=case.payload,
        ),
        input_bindings=dict(input_bindings or {}),
    )


def write_prepared_operator_input(path: Path, prepared: PreparedOperatorInput) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(prepared.to_json_dict(), indent=2) + "\n")
    return path


def _prepared_operator_input_from_payload(
    payload: dict[str, object],
) -> PreparedOperatorInput:
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"unsupported prepared input schema_version: {payload.get('schema_version')}"
        )
    case = payload["case"]
    input_bindings = {
        str(name): OperatorInputBinding(
            op=binding["op"],
            dtype=binding["dtype"],
            dataset=binding["dataset"],
            case_id=binding["case_id"],
            seed=binding["seed"],
            output_index=binding.get("output_index", 0),
        )
        for name, binding in payload.get("input_bindings", {}).items()
    }
    return PreparedOperatorInput(
        op=payload["op"],
        dtype=payload["dtype"],
        dataset=payload["dataset"],
        seed=payload["seed"],
        case=OperatorCase(
            case_id=case["case_id"],
            family=case["family"],
            source_kind=case["source_kind"],
            source_project=case["source_project"],
            source_model=case["source_model"],
            source_file=case["source_file"],
            source_op=case["source_op"],
            payload={str(key): value for key, value in case["payload"].items()},
        ),
        input_bindings=input_bindings,
    )


def read_prepared_operator_input(path: Path) -> PreparedOperatorInput:
    return _prepared_operator_input_from_payload(json.loads(path.read_text()))


def prepare_workflow_input(workflow) -> PreparedWorkflowInput:
    return PreparedWorkflowInput(
        workflow=workflow.workflow,
        phase=workflow.phase,
        dataset=workflow.dataset,
        case_id=workflow.case_id,
        steps=tuple(
            PreparedWorkflowStep(
                contract=step.contract,
                consumes=step.consumes,
                produces=step.produces,
                prepared=PreparedOperatorInput(
                    op=step.prepared.op,
                    dtype=step.prepared.dtype,
                    dataset=step.prepared.dataset,
                    seed=step.prepared.seed,
                    case=step.prepared.case,
                ),
            )
            for step in workflow.steps
        ),
    )


def write_prepared_workflow_input(
    path: Path, prepared: PreparedWorkflowInput
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(prepared.to_json_dict(), indent=2) + "\n")
    return path


def read_prepared_workflow_input(path: Path) -> PreparedWorkflowInput:
    payload = json.loads(path.read_text())
    if payload.get("schema_version") != WORKFLOW_SCHEMA_VERSION:
        raise ValueError(
            "unsupported prepared workflow schema_version: "
            f"{payload.get('schema_version')}"
        )
    return PreparedWorkflowInput(
        workflow=payload["workflow"],
        phase=payload["phase"],
        dataset=payload["dataset"],
        case_id=payload["case_id"],
        steps=tuple(
            PreparedWorkflowStep(
                contract=step["contract"],
                consumes=tuple(step["consumes"]),
                produces=tuple(step["produces"]),
                prepared=_prepared_operator_input_from_payload(step["prepared"]),
            )
            for step in payload["steps"]
        ),
    )
