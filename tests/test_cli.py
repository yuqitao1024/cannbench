import runpy

import pytest

from cannbench.cli import build_parser, main
from cannbench.core.result import BenchmarkMetrics, OperatorBenchmarkResult, SoftmaxShape


def test_build_parser_exposes_operator_subcommand():
    parser = build_parser()
    args = parser.parse_args(
        [
            "operator",
            "--backend",
            "nvidia",
            "--op",
            "softmax",
            "--rows",
            "256",
            "--cols",
            "256",
            "--dim",
            "-1",
        ]
    )

    assert args.command == "operator"
    assert args.backend == "nvidia"
    assert args.op == "softmax"
    assert args.rows == 256
    assert args.cols == 256
    assert args.dim == -1


def test_build_parser_rejects_ascend_backend():
    parser = build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "operator",
                "--backend",
                "ascend",
                "--op",
                "softmax",
                "--rows",
                "256",
                "--cols",
                "256",
            ]
        )


def test_main_runs_operator_benchmark_and_writes_outputs(tmp_path, monkeypatch):
    captured: dict[str, object] = {}
    result = OperatorBenchmarkResult(
        backend="nvidia",
        device_name="Fake GPU",
        op="softmax",
        dtype="float16",
        shape=SoftmaxShape(rows=16, cols=32, dim=-1),
        metrics=BenchmarkMetrics(
            iterations=3,
            warmup=2,
            latency_ms_avg=1.0,
            latency_ms_p50=1.0,
            latency_ms_p95=1.0,
            latency_ms_p99=1.0,
            throughput_ops_per_sec=1000.0,
        ),
    )

    class FakeBackend:
        def run_softmax(self, request):
            captured["request"] = request
            return result

    monkeypatch.setattr("cannbench.cli.get_backend", lambda name: FakeBackend())

    def fake_write_benchmark_outputs(output_dir, run_name, actual_result, formats):
        captured["output_dir"] = output_dir
        captured["run_name"] = run_name
        captured["result"] = actual_result
        captured["formats"] = formats
        return {}

    monkeypatch.setattr("cannbench.cli.write_benchmark_outputs", fake_write_benchmark_outputs)

    exit_code = main(
        [
            "operator",
            "--backend",
            "nvidia",
            "--op",
            "softmax",
            "--dtype",
            "float16",
            "--rows",
            "16",
            "--cols",
            "32",
            "--dim",
            "-1",
            "--warmup",
            "2",
            "--iterations",
            "3",
            "--output-dir",
            str(tmp_path),
            "--run-name",
            "softmax-run",
        ]
    )

    request = captured["request"]
    assert exit_code == 0
    assert request.backend == "nvidia"
    assert request.op == "softmax"
    assert request.dtype == "float16"
    assert request.rows == 16
    assert request.cols == 32
    assert request.dim == -1
    assert request.warmup == 2
    assert request.iterations == 3
    assert captured["output_dir"] == tmp_path
    assert captured["run_name"] == "softmax-run"
    assert captured["result"] is result
    assert captured["formats"] == ("json", "csv", "md")


def test_main_rejects_zero_iterations():
    with pytest.raises(SystemExit):
        main(
            [
                "operator",
                "--backend",
                "nvidia",
                "--op",
                "softmax",
                "--rows",
                "16",
                "--cols",
                "32",
                "--iterations",
                "0",
            ]
        )


def test_main_rejects_negative_warmup():
    with pytest.raises(SystemExit):
        main(
            [
                "operator",
                "--backend",
                "nvidia",
                "--op",
                "softmax",
                "--rows",
                "16",
                "--cols",
                "32",
                "--warmup",
                "-1",
            ]
        )


def test_main_converts_backend_runtime_failure_to_cli_error(monkeypatch, capsys):
    class FailingBackend:
        def run_softmax(self, request):
            raise RuntimeError("CUDA is required for the nvidia backend")

    monkeypatch.setattr("cannbench.cli.get_backend", lambda name: FailingBackend())

    with pytest.raises(SystemExit) as excinfo:
        main(
            [
                "operator",
                "--backend",
                "nvidia",
                "--op",
                "softmax",
                "--rows",
                "16",
                "--cols",
                "32",
            ]
        )

    captured = capsys.readouterr()
    assert excinfo.value.code == 2
    assert "CUDA is required for the nvidia backend" in captured.err


def test_python_m_cannbench_exits_with_main_return_code(monkeypatch):
    monkeypatch.setattr("cannbench.cli.main", lambda: 7)

    with pytest.raises(SystemExit) as excinfo:
        runpy.run_module("cannbench", run_name="__main__")

    assert excinfo.value.code == 7
