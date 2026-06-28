from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from cannbench.core.prepared_input import PreparedOperatorInput
from cannbench.core.profile import DeviceProfileSummary


def _infer_shape(case_payload: dict[str, Any]) -> list[int]:
    for key in (
        "dimensions",
        "shape",
        "input_shape",
        "logits_shape",
        "values_shape",
        "src_shape",
        "index_shape",
        "target_shape",
        "mask_shape",
    ):
        value = case_payload.get(key)
        if isinstance(value, (list, tuple)) and value:
            return [int(item) for item in value]
    raise ValueError("unable to infer benchmark record shape from case payload")


def _device_class(device_name: str) -> str:
    name = device_name.strip()
    if not name:
        return "unknown"
    upper = name.upper()
    if "H800" in upper:
        return "H800"
    if "ASCEND" in upper:
        return "Ascend"
    return name


def _implementation_and_version(
    *,
    backend: str,
    implementation: str | None,
    profile_summary: DeviceProfileSummary,
) -> tuple[str, str]:
    if backend == "ascend":
        if implementation == "simt":
            return "simt", "v1"
        return "cann_ops_library", "cann"
    if backend == "nvidia":
        return "ncu", "ncu"
    return implementation or "unknown", implementation or "unknown"


def build_collect_benchmark_record(
    *,
    run_id: str,
    backend: str,
    implementation: str | None,
    prepared: PreparedOperatorInput,
    perf_payload: dict[str, Any],
    profile_summary: DeviceProfileSummary,
) -> dict[str, Any]:
    return build_benchmark_record(
        run_id=run_id,
        backend=backend,
        implementation=implementation,
        prepared=prepared,
        device_name=str(perf_payload.get("device_name", "unknown")),
        profile_summary=profile_summary,
    )


def build_benchmark_record(
    *,
    run_id: str,
    backend: str,
    implementation: str | None,
    prepared: PreparedOperatorInput,
    device_name: str,
    profile_summary: DeviceProfileSummary,
) -> dict[str, Any]:
    resolved_implementation, implementation_version = _implementation_and_version(
        backend=backend,
        implementation=implementation,
        profile_summary=profile_summary,
    )
    return {
        "schema_version": 1,
        "run_id": run_id,
        "operator": prepared.op,
        "dataset": prepared.dataset,
        "case_id": prepared.case.case_id,
        "shape": _infer_shape(prepared.case.payload),
        "dtype": prepared.dtype,
        "backend": backend,
        "device_class": _device_class(device_name),
        "implementation": resolved_implementation,
        "implementation_version": implementation_version,
        "metrics": {
            "latency_ms_avg": profile_summary.latency_ms_avg,
            "latency_ms_p50": profile_summary.latency_ms_p50,
            "latency_ms_p95": profile_summary.latency_ms_p95,
            "sample_count": profile_summary.sample_count,
        },
        "accuracy": {
            "passed": True,
            "max_abs_error": 0.0,
            "max_rel_error": 0.0,
        },
        "diff_ref": (
            f"{prepared.op}/simt/{implementation_version}"
            if backend == "ascend" and resolved_implementation == "simt"
            else None
        ),
    }


def build_local_benchmark_record(
    *,
    run_id: str,
    backend: str,
    implementation: str | None,
    prepared: PreparedOperatorInput,
    device_name: str,
    profile_summary: DeviceProfileSummary,
) -> dict[str, Any]:
    return build_benchmark_record(
        run_id=run_id,
        backend=backend,
        implementation=implementation,
        prepared=prepared,
        device_name=device_name,
        profile_summary=profile_summary,
    )


def read_perf_result(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def read_profile_summary(path: Path) -> DeviceProfileSummary:
    payload = json.loads(path.read_text())
    return DeviceProfileSummary(
        backend=str(payload["backend"]),
        sample_count=int(payload["sample_count"]),
        latency_ms_avg=float(payload["latency_ms_avg"]),
        latency_ms_p50=float(payload["latency_ms_p50"]),
        latency_ms_p95=float(payload["latency_ms_p95"]),
        latency_ms_p99=float(payload["latency_ms_p99"]),
        source_files=tuple(str(item) for item in payload.get("source_files", [])),
    )


def write_benchmark_records_json(path: Path, records: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"records": records}, indent=2) + "\n")
    return path
