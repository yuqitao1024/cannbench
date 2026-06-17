from __future__ import annotations

import json
from pathlib import Path


def _read_perf_result(run_dir: Path) -> dict[str, object]:
    perf_dir = run_dir / "perf"
    json_files = sorted(perf_dir.glob("*.json"))
    if not json_files:
        raise ValueError(f"no perf json files found in {perf_dir}")
    return json.loads(json_files[0].read_text())


def _perf_row(result: dict[str, object]) -> str:
    case = result["case"]
    metrics = result["metrics"]
    return (
        f"| {result['backend']} | {result['device_name']} | {result['op']} | "
        f"{case['case_id']} | {result['dtype']} | {metrics['latency_ms_avg']} |"
    )


def write_local_report(
    *,
    output_path: Path,
    nvidia_dir: Path,
    ascend_dir: Path,
    accuracy_path: Path,
) -> Path:
    nvidia = _read_perf_result(nvidia_dir)
    ascend = _read_perf_result(ascend_dir)
    accuracy = json.loads(accuracy_path.read_text())

    lines = [
        "# CannBench Local Comparison Report",
        "",
        "## Performance",
        "",
        "| backend | device_name | op | case_id | dtype | latency_ms_avg |",
        "| --- | --- | --- | --- | --- | --- |",
        _perf_row(nvidia),
        _perf_row(ascend),
        "",
        "## Accuracy",
        "",
        "| field | value |",
        "| --- | --- |",
        f"| passed | {accuracy['passed']} |",
        f"| op | {accuracy['op']} |",
        f"| case_id | {accuracy['case_id']} |",
        f"| numel | {accuracy['numel']} |",
        f"| mismatch_count | {accuracy['mismatch_count']} |",
        f"| max_abs_error | {accuracy['max_abs_error']} |",
        f"| max_rel_error | {accuracy['max_rel_error']} |",
        f"| mean_abs_error | {accuracy['mean_abs_error']} |",
        f"| rmse | {accuracy['rmse']} |",
        f"| rtol | {accuracy['rtol']} |",
        f"| atol | {accuracy['atol']} |",
        "",
        "## Artifacts",
        "",
        "| field | path |",
        "| --- | --- |",
        f"| nvidia_profile | {nvidia_dir / 'profile'} |",
        f"| ascend_profile | {ascend_dir / 'profile'} |",
        f"| nvidia_perf | {nvidia_dir / 'perf'} |",
        f"| ascend_perf | {ascend_dir / 'perf'} |",
        f"| accuracy | {accuracy_path} |",
        "",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines))
    return output_path
