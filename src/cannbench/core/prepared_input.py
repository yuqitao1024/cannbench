from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from cannbench.core.result import OperatorCase
from cannbench.datasets import get_operator_case

SCHEMA_VERSION = 1


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


def read_prepared_operator_input(path: Path) -> PreparedOperatorInput:
    payload = json.loads(path.read_text())
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
