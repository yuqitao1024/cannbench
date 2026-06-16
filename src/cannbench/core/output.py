import csv
import json
from pathlib import Path

from cannbench.core.result import OperatorBenchmarkResult

SUPPORTED_OUTPUT_FORMATS = frozenset({"json", "csv", "md"})


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
                    "latency_ms_avg",
                    "latency_ms_p50",
                    "latency_ms_p95",
                    "latency_ms_p99",
                    "throughput_ops_per_sec",
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
                    result.metrics.latency_ms_avg,
                    result.metrics.latency_ms_p50,
                    result.metrics.latency_ms_p95,
                    result.metrics.latency_ms_p99,
                    result.metrics.throughput_ops_per_sec,
                ]
            )
        created["csv"] = csv_path

    if "md" in formats:
        md_path = output_dir / f"{run_name}.md"
        md_path.write_text(
            "\n".join(
                [
                    "# CannBench Operator Benchmark",
                    "",
                    "| field | value |",
                    "| --- | --- |",
                    f"| backend | {result.backend} |",
                    f"| device_name | {result.device_name} |",
                    f"| op | {result.op} |",
                    f"| dtype | {result.dtype} |",
                    f"| case_id | {result.case.case_id} |",
                    f"| family | {result.case.family} |",
                    f"| payload | {result.case.payload_summary} |",
                    f"| source_model | {result.case.source_model} |",
                    f"| latency_ms_avg | {result.metrics.latency_ms_avg} |",
                    f"| latency_ms_p50 | {result.metrics.latency_ms_p50} |",
                    f"| latency_ms_p95 | {result.metrics.latency_ms_p95} |",
                    f"| latency_ms_p99 | {result.metrics.latency_ms_p99} |",
                    f"| throughput_ops_per_sec | {result.metrics.throughput_ops_per_sec} |",
                    "",
                ]
            )
        )
        created["md"] = md_path

    return created
