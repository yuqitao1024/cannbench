import json
from csv import reader

import pytest

from cannbench.core.output import write_benchmark_outputs
from cannbench.core.result import (
    BenchmarkMetrics,
    OperatorBenchmarkResult,
    OperatorCase,
    build_softmax_case,
)


def _sample_result() -> OperatorBenchmarkResult:
    return OperatorBenchmarkResult(
        backend="nvidia",
        device_name="Fake GPU",
        op="softmax",
        dtype="float16",
        case=build_softmax_case(
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
        metrics=BenchmarkMetrics(
            iterations=10,
            warmup=5,
            latency_ms_avg=1.0,
            latency_ms_p50=1.0,
            latency_ms_p95=1.1,
            latency_ms_p99=1.2,
            throughput_ops_per_sec=1000.0,
        ),
    )


def test_result_to_json_dict_contains_core_fields():
    result = OperatorBenchmarkResult(
        backend="nvidia",
        device_name="Fake GPU",
        op="softmax",
        dtype="float16",
        case=build_softmax_case(
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
        metrics=BenchmarkMetrics(
            iterations=10,
            warmup=5,
            latency_ms_avg=1.2,
            latency_ms_p50=1.1,
            latency_ms_p95=1.5,
            latency_ms_p99=1.6,
            throughput_ops_per_sec=833.33,
        ),
    )

    payload = result.to_json_dict()

    assert payload["backend"] == "nvidia"
    assert payload["metrics"]["latency_ms_avg"] == 1.2
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


def test_write_benchmark_outputs_creates_json_csv_and_markdown(tmp_path):
    result = _sample_result()

    paths = write_benchmark_outputs(tmp_path, "sample-run", result, ("json", "csv", "md"))

    assert sorted(paths.keys()) == ["csv", "json", "md"]
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
        "latency_ms_avg",
        "latency_ms_p50",
        "latency_ms_p95",
        "latency_ms_p99",
        "throughput_ops_per_sec",
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
        "1.0",
        "1.0",
        "1.1",
        "1.2",
        "1000.0",
    ]
    markdown = paths["md"].read_text()
    assert "| backend | nvidia |" in markdown
    assert "| case_id | tiny_logits |" in markdown
    assert "| source_model | smoke_fixture |" in markdown


def test_write_benchmark_outputs_creates_only_requested_formats(tmp_path):
    paths = write_benchmark_outputs(tmp_path, "json-only", _sample_result(), ("json",))

    assert sorted(paths.keys()) == ["json"]
    assert paths["json"].name == "json-only.json"
    assert not (tmp_path / "json-only.csv").exists()
    assert not (tmp_path / "json-only.md").exists()


def test_write_benchmark_outputs_rejects_unsupported_formats(tmp_path):
    with pytest.raises(ValueError, match="unsupported output format"):
        write_benchmark_outputs(tmp_path, "bad-run", _sample_result(), ("json", "yaml"))


@pytest.mark.parametrize("dimensions", [(), (0, 128), (128, -1)])
def test_build_softmax_case_rejects_invalid_dimensions(dimensions: tuple[int, ...]):
    with pytest.raises(ValueError, match="dimensions must be"):
        build_softmax_case(
            case_id="case",
            family="attention",
            dimensions=dimensions,
            dim=-1,
            source_kind="synthetic",
            source_project="cannbench",
            source_model="fixture",
            source_file="tests/fixtures",
            source_op="softmax",
        )


@pytest.mark.parametrize("dim", [-4, 3])
def test_build_softmax_case_rejects_invalid_dim(dim: int):
    with pytest.raises(ValueError, match="dim must address an axis"):
        build_softmax_case(
            case_id="case",
            family="attention",
            dimensions=(4, 8, 16),
            dim=dim,
            source_kind="synthetic",
            source_project="cannbench",
            source_model="fixture",
            source_file="tests/fixtures",
            source_op="softmax",
        )


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

    assert case.payload_summary == "embedding_dim=64, index_shape=32, num_embeddings=128"
