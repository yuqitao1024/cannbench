from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from cannbench.core.prepared_input import (
    PreparedOperatorInput,
    build_prepared_operator_input,
    read_prepared_operator_input,
)
from cannbench.datasets import get_operator_dataset

DEFAULT_DATASET_SPLITS = ("smoke", "realistic", "stress")
DEFAULT_DTYPE = "float16"


@dataclass(frozen=True)
class PreparedInputPlan:
    op: str
    dataset: str
    case_id: str
    dtype: str
    seed: int
    prepared: PreparedOperatorInput
    source_path: Path | None = None


@dataclass(frozen=True)
class BatchResultRecord:
    op: str
    dataset: str
    case_id: str
    dtype: str
    seed: int
    status: str
    prepared_input: str
    result_path: str | None = None


@dataclass(frozen=True)
class BatchFailureRecord:
    op: str
    dataset: str
    case_id: str
    dtype: str
    seed: int
    prepared_input: str
    error: str


def expand_prepared_input_plans(
    *,
    op: str | None = None,
    dtype: str | None = None,
    dataset: str | None = None,
    case_id: str | None = None,
    seed: int = 0,
    prepared_input: Path | None = None,
    prepared_dir: Path | None = None,
) -> list[PreparedInputPlan]:
    if prepared_input is not None and prepared_dir is not None:
        raise ValueError("--prepared-input and --prepared-dir are mutually exclusive")
    if prepared_input is not None:
        return [_plan_from_prepared_path(prepared_input, expected_op=op)]
    if prepared_dir is not None:
        return _plans_from_prepared_dir(prepared_dir, expected_op=op)
    if not op:
        raise ValueError("--op is required unless --prepared-input or --prepared-dir is set")
    return _plans_from_operator_selection(
        op=op,
        dtype=dtype or DEFAULT_DTYPE,
        dataset=dataset,
        case_id=case_id,
        seed=seed,
    )


def write_batch_summary_json(path: Path, rows: list[BatchResultRecord]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "result_count": len(rows),
        "results": [asdict(row) for row in rows],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


def write_batch_summary_csv(path: Path, rows: list[BatchResultRecord]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [field.name for field in fields(BatchResultRecord)]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))
    return path


def write_batch_failures_json(path: Path, rows: list[BatchFailureRecord]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "failure_count": len(rows),
        "failures": [asdict(row) for row in rows],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


def _plan_from_prepared_path(
    path: Path,
    *,
    expected_op: str | None = None,
) -> PreparedInputPlan:
    prepared = read_prepared_operator_input(path)
    if expected_op is not None and prepared.op != expected_op:
        raise ValueError(
            f"prepared input operator mismatch: expected {expected_op}, got {prepared.op} ({path})"
        )
    return PreparedInputPlan(
        op=prepared.op,
        dataset=prepared.dataset,
        case_id=prepared.case.case_id,
        dtype=prepared.dtype,
        seed=prepared.seed,
        prepared=prepared,
        source_path=path,
    )


def _plans_from_prepared_dir(
    prepared_dir: Path,
    *,
    expected_op: str | None = None,
) -> list[PreparedInputPlan]:
    if not prepared_dir.is_dir():
        raise ValueError(f"prepared input directory does not exist: {prepared_dir}")
    paths = sorted(prepared_dir.glob("*.json"))
    if not paths:
        raise ValueError(f"prepared input directory does not contain any json manifests: {prepared_dir}")
    return [
        _plan_from_prepared_path(path, expected_op=expected_op)
        for path in paths
    ]


def _plans_from_operator_selection(
    *,
    op: str,
    dtype: str,
    dataset: str | None,
    case_id: str | None,
    seed: int,
) -> list[PreparedInputPlan]:
    operator_dataset = get_operator_dataset(op)
    selected_datasets = (dataset,) if dataset is not None else DEFAULT_DATASET_SPLITS
    plans: list[PreparedInputPlan] = []

    for split in selected_datasets:
        cases = operator_dataset.get(split).cases
        if case_id is not None:
            cases = [case for case in cases if case.case_id == case_id]
        for case in cases:
            prepared = build_prepared_operator_input(
                op=op,
                dtype=dtype,
                dataset=split,
                case_id=case.case_id,
                seed=seed,
            )
            plans.append(
                PreparedInputPlan(
                    op=op,
                    dataset=split,
                    case_id=case.case_id,
                    dtype=dtype,
                    seed=seed,
                    prepared=prepared,
                )
            )

    if case_id is not None and not plans:
        raise ValueError(f"Unknown case_id for operator selection: {case_id}")
    return plans
