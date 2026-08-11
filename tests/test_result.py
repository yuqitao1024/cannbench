import json
from csv import reader

import pytest

from cannbench.core.output import write_benchmark_outputs
from cannbench.core.result import (
    OperatorBenchmarkResult,
    OperatorCase,
    WorkflowBenchmarkResult,
)


def _operator_case(
    *,
    case_id: str,
    family: str,
    dimensions: tuple[int, ...],
    dim: int,
    source_kind: str,
    source_project: str,
    source_model: str,
    source_file: str,
    source_op: str,
) -> OperatorCase:
    return OperatorCase(
        case_id=case_id,
        family=family,
        source_kind=source_kind,
        source_project=source_project,
        source_model=source_model,
        source_file=source_file,
        source_op=source_op,
        payload={"dimensions": dimensions, "dim": dim},
    )


def _sample_result() -> OperatorBenchmarkResult:
    return OperatorBenchmarkResult(
        backend="nvidia",
        device_name="Fake GPU",
        op="softmax",
        dtype="float16",
        case=_operator_case(
            case_id="tiny_logits",
            family="lm_logits",
            dimensions=(128, 128),
            dim=-1,
            source_kind="synthetic_smoke",
            source_project="cannbench",
            source_model="smoke_fixture",
            source_file="tests/fixtures",
            source_op="softmax",
        ),
    )


def test_result_to_json_dict_contains_core_fields():
    result = OperatorBenchmarkResult(
        backend="nvidia",
        device_name="Fake GPU",
        op="softmax",
        dtype="float16",
        case=_operator_case(
            case_id="t5_attention",
            family="attention",
            dimensions=(4, 8, 1024, 1024),
            dim=-1,
            source_kind="real_model",
            source_project="TritonBench",
            source_model="T5Small",
            source_file="tritonbench/models/t5.py",
            source_op="softmax",
        ),
    )

    payload = result.to_json_dict()

    assert payload["backend"] == "nvidia"
    assert "metrics" not in payload
    assert payload["case"] == {
        "case_id": "t5_attention",
        "family": "attention",
        "source_kind": "real_model",
        "source_project": "TritonBench",
        "source_model": "T5Small",
        "source_file": "tritonbench/models/t5.py",
        "source_op": "softmax",
        "payload": {
            "dimensions": [4, 8, 1024, 1024],
            "dim": -1,
        },
    }


def test_write_benchmark_outputs_creates_json_and_csv(tmp_path):
    result = _sample_result()

    paths = write_benchmark_outputs(tmp_path, "sample-run", result)

    assert sorted(paths.keys()) == ["csv", "json"]
    assert json.loads(paths["json"].read_text())["backend"] == "nvidia"
    with paths["csv"].open(newline="") as handle:
        rows = list(reader(handle))
    assert rows[0] == [
        "backend",
        "device_name",
        "op",
        "dtype",
        "case_id",
        "family",
        "payload",
        "source_model",
    ]
    assert rows[1] == [
        "nvidia",
        "Fake GPU",
        "softmax",
        "float16",
        "tiny_logits",
        "lm_logits",
        "dimensions=128x128, dim=-1",
        "smoke_fixture",
    ]


def test_workflow_result_serializes_ordered_component_results():
    component = _sample_result()
    result = WorkflowBenchmarkResult(
        backend="nvidia",
        device_name="Fake GPU",
        workflow="test_workflow",
        phase="decode",
        dataset="smoke",
        case_id="tiny_logits",
        steps=(component,),
    )

    payload = result.to_json_dict()

    assert payload["workflow"] == "test_workflow"
    assert payload["phase"] == "decode"
    assert payload["case_id"] == "tiny_logits"
    assert payload["steps"] == [component.to_json_dict()]

def test_operator_case_payload_summary_is_stable():
    case = OperatorCase(
        case_id="tiny_token_lookup",
        family="token_lookup",
        source_kind="synthetic_smoke",
        source_project="cannbench",
        source_model="fixture",
        source_file="built-in",
        source_op="torch.nn.Embedding",
        payload={"num_embeddings": 128, "embedding_dim": 64, "index_shape": (32,)},
    )

    assert case.payload_summary == "index_shape=32, embedding_dim=64, num_embeddings=128"
