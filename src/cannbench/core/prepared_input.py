from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from cannbench.core.result import OperatorCase
from cannbench.datasets import get_operator_case

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class PreparedOperatorInput:
    op: str
    dtype: str
    dataset: str
    seed: int
    case: OperatorCase

    def to_json_dict(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "op": self.op,
            "dtype": self.dtype,
            "dataset": self.dataset,
            "seed": self.seed,
            "case": self.case.to_json_dict(),
        }


def build_prepared_operator_input(
    *,
    op: str,
    dtype: str,
    dataset: str,
    case_id: str,
    seed: int,
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
    )
