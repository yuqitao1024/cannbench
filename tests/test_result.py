import json
from csv import reader

import pytest

from cannbench.core.output import write_benchmark_outputs
from cannbench.core.result import BenchmarkMetrics, OperatorBenchmarkResult, SoftmaxShape


def _sample_result() -> OperatorBenchmarkResult:
    return OperatorBenchmarkResult(
        backend="nvidia",
        device_name="Fake GPU",
        op="softmax",
        dtype="float16",
        shape=SoftmaxShape(rows=128, cols=128, dim=-1),
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
        shape=SoftmaxShape(rows=128, cols=128, dim=-1),
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
    assert payload["shape"] == {"rows": 128, "cols": 128, "dim": -1}


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
        "rows",
        "cols",
        "dim",
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
        "128",
        "128",
        "-1",
        "1.0",
        "1.0",
        "1.1",
        "1.2",
        "1000.0",
    ]
    assert "| backend | nvidia |" in paths["md"].read_text()


def test_write_benchmark_outputs_creates_only_requested_formats(tmp_path):
    paths = write_benchmark_outputs(tmp_path, "json-only", _sample_result(), ("json",))

    assert sorted(paths.keys()) == ["json"]
    assert paths["json"].name == "json-only.json"
    assert not (tmp_path / "json-only.csv").exists()
    assert not (tmp_path / "json-only.md").exists()


def test_write_benchmark_outputs_rejects_unsupported_formats(tmp_path):
    with pytest.raises(ValueError, match="unsupported output format"):
        write_benchmark_outputs(tmp_path, "bad-run", _sample_result(), ("json", "yaml"))


@pytest.mark.parametrize("rows", [0, -1])
def test_softmax_shape_rejects_non_positive_rows(rows: int):
    with pytest.raises(ValueError, match="rows must be > 0"):
        SoftmaxShape(rows=rows, cols=128, dim=-1)


@pytest.mark.parametrize("cols", [0, -1])
def test_softmax_shape_rejects_non_positive_cols(cols: int):
    with pytest.raises(ValueError, match="cols must be > 0"):
        SoftmaxShape(rows=128, cols=cols, dim=-1)


@pytest.mark.parametrize("dim", [-3, 2, 3])
def test_softmax_shape_rejects_invalid_dim(dim: int):
    with pytest.raises(ValueError, match="dim must be one of"):
        SoftmaxShape(rows=128, cols=128, dim=dim)
