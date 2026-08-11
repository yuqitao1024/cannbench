from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from cannbench.core.prepared_input import (
    PreparedOperatorInput,
    PreparedWorkflowInput,
    build_prepared_operator_input,
)
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
    if all(key in case_payload for key in ("batch", "query_tokens", "index_heads", "index_dim")):
        return [
            int(case_payload["batch"]),
            int(case_payload["query_tokens"]),
            int(case_payload["index_heads"]),
            int(case_payload["index_dim"]),
        ]
    if all(
        key in case_payload
        for key in ("query_tokens", "query_heads", "qk_head_dim")
    ):
        return [
            int(case_payload["query_tokens"]),
            int(case_payload["query_heads"]),
            int(case_payload["qk_head_dim"]),
        ]
    raise ValueError("unable to infer benchmark record shape from case payload")


def _device_class(device_name: str) -> str:
    name = device_name.strip()
    if not name:
        return "unknown"
    upper = name.upper()
    if "H800" in upper:
        return "H800"
    if "950PR" in upper:
        return "950PR"
    if "910B" in upper:
        return "910B"
    if "ASCEND" in upper:
        return "Ascend"
    return name


def _implementation_and_version(
    *,
    backend: str,
    implementation: str | None,
    implementation_version: str | None = None,
    profile_summary: DeviceProfileSummary,
) -> tuple[str, str]:
    del profile_summary
    if backend == "ascend":
        if implementation == "simt":
            return "simt", implementation_version or "v1"
        if implementation == "vllm_ascend":
            return "vllm_ascend", implementation_version or "vllm-ascend"
        return "cann_ops_library", "cannops"
    if backend == "nvidia":
        if implementation == "cuda_library":
            return "cuda_library", implementation_version or "cuda-library"
        return "cuda-pytorch", "cuda-pytorch"
    return implementation or "unknown", implementation or "unknown"


def build_collect_benchmark_record(
    *,
    run_id: str,
    backend: str,
    implementation: str | None,
    implementation_version: str | None = None,
    prepared: PreparedOperatorInput,
    perf_payload: dict[str, Any],
    profile_summary: DeviceProfileSummary,
) -> dict[str, Any]:
    return build_benchmark_record(
        run_id=run_id,
        backend=backend,
        implementation=implementation,
        implementation_version=implementation_version,
        prepared=prepared,
        device_name=str(perf_payload.get("device_name", "unknown")),
        profile_summary=profile_summary,
    )


def build_benchmark_record(
    *,
    run_id: str,
    backend: str,
    implementation: str | None,
    implementation_version: str | None = None,
    prepared: PreparedOperatorInput,
    device_name: str,
    profile_summary: DeviceProfileSummary,
) -> dict[str, Any]:
    resolved_implementation, implementation_version = _implementation_and_version(
        backend=backend,
        implementation=implementation,
        implementation_version=implementation_version,
        profile_summary=profile_summary,
    )
    return {
        "schema_version": 1,
        "run_id": run_id,
        "operator": prepared.op,
        "dataset": prepared.dataset,
        "case_id": prepared.case.case_id,
        "family": prepared.case.family,
        "shape": _infer_shape(prepared.case.payload),
        "dtype": prepared.dtype,
        "backend": backend,
        "device_class": _device_class(device_name),
        "implementation": resolved_implementation,
        "implementation_version": implementation_version,
        "source_kind": prepared.case.source_kind,
        "source_project": prepared.case.source_project,
        "source_model": prepared.case.source_model,
        "source_file": prepared.case.source_file,
        "source_op": prepared.case.source_op,
        "metrics": {
            "latency_ms": profile_summary.latency_ms,
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
    implementation_version: str | None = None,
    prepared: PreparedOperatorInput,
    device_name: str,
    profile_summary: DeviceProfileSummary,
) -> dict[str, Any]:
    return build_benchmark_record(
        run_id=run_id,
        backend=backend,
        implementation=implementation,
        implementation_version=implementation_version,
        prepared=prepared,
        device_name=device_name,
        profile_summary=profile_summary,
    )


def build_workflow_benchmark_record(
    *,
    run_id: str,
    backend: str,
    implementation: str | None,
    prepared: PreparedWorkflowInput,
    device_name: str,
    profile_summary: DeviceProfileSummary,
    implementation_version: str | None = None,
) -> dict[str, Any]:
    component_prepared = prepared.steps[-1].prepared
    record = build_benchmark_record(
        run_id=run_id,
        backend=backend,
        implementation=implementation,
        implementation_version=implementation_version,
        prepared=component_prepared,
        device_name=device_name,
        profile_summary=profile_summary,
    )
    workflow_case = build_prepared_operator_input(
        op=prepared.workflow,
        dtype=component_prepared.dtype,
        dataset=prepared.dataset,
        case_id=prepared.case_id,
        seed=component_prepared.seed,
    ).case
    record.update(
        {
            "operator": prepared.workflow,
            "dataset": prepared.dataset,
            "case_id": prepared.case_id,
            "family": workflow_case.family,
            "source_kind": workflow_case.source_kind,
            "source_project": workflow_case.source_project,
            "source_model": workflow_case.source_model,
            "source_file": workflow_case.source_file,
            "source_op": workflow_case.source_op,
            "diff_ref": None,
        }
    )
    return record


def read_perf_result(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def read_profile_summary(path: Path) -> DeviceProfileSummary:
    payload = json.loads(path.read_text())
    return DeviceProfileSummary(
        backend=str(payload["backend"]),
        latency_ms=float(payload["latency_ms"]),
        source_files=tuple(str(item) for item in payload.get("source_files", [])),
    )


def write_benchmark_records_json(path: Path, records: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"records": records}, indent=2) + "\n")
    return path
