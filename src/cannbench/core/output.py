import csv
import json
from pathlib import Path

from cannbench.core.result import OperatorBenchmarkResult, WorkflowBenchmarkResult


def build_benchmark_artifact_stem(
    *,
    op: str,
    dataset: str,
    case_id: str,
    dtype: str,
    seed: int,
) -> str:
    return f"{op}-{dataset}-{case_id}-{dtype}-seed{seed}"


def write_benchmark_outputs(
    output_dir: Path,
    run_name: str,
    result: OperatorBenchmarkResult,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    created: dict[str, Path] = {}

    json_path = output_dir / f"{run_name}.json"
    json_path.write_text(json.dumps(result.to_json_dict(), indent=2) + "\n")
    created["json"] = json_path

    csv_path = output_dir / f"{run_name}.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "backend",
                "device_name",
                "op",
                "dtype",
                "case_id",
                "family",
                "payload",
                "source_model",
            ]
        )
        writer.writerow(
            [
                result.backend,
                result.device_name,
                result.op,
                result.dtype,
                result.case.case_id,
                result.case.family,
                result.case.payload_summary,
                result.case.source_model,
            ]
        )
    created["csv"] = csv_path

    return created


def write_workflow_benchmark_outputs(
    output_dir: Path,
    run_name: str,
    result: WorkflowBenchmarkResult,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{run_name}.json"
    json_path.write_text(json.dumps(result.to_json_dict(), indent=2) + "\n")
    return {"json": json_path}
