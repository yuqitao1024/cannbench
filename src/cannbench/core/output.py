import csv
import json
from pathlib import Path

from cannbench.core.result import OperatorBenchmarkResult

SUPPORTED_OUTPUT_FORMATS = frozenset({"json", "csv"})


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
    formats: tuple[str, ...],
) -> dict[str, Path]:
    unsupported_formats = sorted(set(formats) - SUPPORTED_OUTPUT_FORMATS)
    if unsupported_formats:
        raise ValueError(f"unsupported output format: {', '.join(unsupported_formats)}")

    output_dir.mkdir(parents=True, exist_ok=True)
    created: dict[str, Path] = {}

    if "json" in formats:
        json_path = output_dir / f"{run_name}.json"
        json_path.write_text(json.dumps(result.to_json_dict(), indent=2) + "\n")
        created["json"] = json_path

    if "csv" in formats:
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
                    "warmup",
                    "iterations",
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
                    result.warmup,
                    result.iterations,
                ]
            )
        created["csv"] = csv_path

    return created
