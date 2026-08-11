from dataclasses import replace

import pytest

from cannbench.operators.shape_trace import (
    DeviceExecutionTrace,
    ShapeAxis,
    ShapeStage,
    ShapeTensor,
    ShapeTrace,
    ShapeTraceKey,
    shape_trace_to_payload,
)


def _valid_trace() -> ShapeTrace:
    q = ShapeTensor(
        id="q",
        label="Q",
        axes=(
            ShapeAxis("H", 128, "query heads", "preserved"),
            ShapeAxis("Dqk", 576, "QK feature", "contracted"),
        ),
    )
    k = ShapeTensor(
        id="k",
        label="K^T",
        axes=(
            ShapeAxis("Dqk", 576, "QK feature", "contracted"),
            ShapeAxis("S", 2048, "selected tokens", "produced"),
        ),
    )
    scores = ShapeTensor(
        id="scores",
        label="scores",
        axes=(
            ShapeAxis("H", 128, "query heads", "preserved"),
            ShapeAxis("S", 2048, "selected tokens", "produced"),
        ),
    )
    return ShapeTrace(
        schema_version=1,
        operator="dsa_decode",
        dataset="realistic",
        case_id="case",
        phase="decode",
        group="deepseek-v32",
        symbols=(ShapeAxis("H", 128, "query heads", "preserved"),),
        stages=(
            ShapeStage(
                id="qk",
                component="sparse_attention",
                title="QK",
                operation="matmul",
                formula="[H,Dqk] x [Dqk,S] -> [H,S]",
                scope="one query row",
                tensors=(q, k, scores),
                input_ids=("q", "k"),
                output_ids=("scores",),
                contracted_axes=("Dqk",),
                insight="Dqk contracts.",
            ),
        ),
        device_execution=DeviceExecutionTrace(
            status="unavailable",
            implementation="simt",
            version=None,
            message="No device trace.",
            kernels=(),
        ),
    )


def test_shape_trace_serializes_nested_tuples_to_json_payload():
    payload = shape_trace_to_payload(_valid_trace())
    assert payload["schema_version"] == 1
    assert payload["stages"][0]["tensors"][0]["axes"][1]["value"] == 576


def test_shape_trace_rejects_non_positive_axis_values():
    with pytest.raises(ValueError, match="axis value must be positive"):
        ShapeAxis("S", 0, "selected tokens", "produced")


def test_shape_trace_rejects_missing_stage_tensor_references():
    trace = _valid_trace()
    with pytest.raises(ValueError, match="unknown tensor id"):
        replace(trace.stages[0], input_ids=("missing",))


def test_shape_stage_rejects_duplicate_tensor_ids():
    trace = _valid_trace()
    duplicate_q = replace(trace.stages[0].tensors[0], label="Duplicate Q")
    with pytest.raises(ValueError, match="duplicate tensor id.*q"):
        replace(trace.stages[0], tensors=(*trace.stages[0].tensors, duplicate_q))


def test_shape_trace_key_keeps_phase_group_and_identity():
    key = ShapeTraceKey("dsa_prefill", "realistic", "case", "prefill", "deepseek-v32")
    assert shape_trace_to_payload(key) == {
        "operator": "dsa_prefill",
        "dataset": "realistic",
        "case_id": "case",
        "phase": "prefill",
        "group": "deepseek-v32",
    }
