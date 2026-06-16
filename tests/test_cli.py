import runpy

import pytest

from cannbench.cli import build_parser, main
from cannbench.core.result import (
    BenchmarkMetrics,
    OperatorBenchmarkResult,
    build_softmax_case,
)


def sample_result() -> OperatorBenchmarkResult:
    return OperatorBenchmarkResult(
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
            source_file="tritonbench/tritonbench/data/input_configs/hf_train/T5Small_train.json",
            source_op="softmax",
        ),
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


def test_build_parser_exposes_operator_subcommand():
    parser = build_parser()
    args = parser.parse_args(
        [
            "operator",
            "--backend",
            "nvidia",
            "--op",
            "softmax",
            "--dataset",
            "realistic",
            "--case-id",
            "t5_attention",
        ]
    )

    assert args.command == "operator"
    assert args.backend == "nvidia"
    assert args.op == "softmax"
    assert args.dataset == "realistic"
    assert args.case_id == "t5_attention"


def test_build_parser_accepts_embedding_operator():
    parser = build_parser()
    args = parser.parse_args(
        [
            "operator",
            "--backend",
            "nvidia",
            "--op",
            "embedding",
            "--dataset",
            "smoke",
            "--case-id",
            "tiny_token_lookup",
        ]
    )

    assert args.op == "embedding"


def test_build_parser_exposes_prepare_subcommand():
    parser = build_parser()
    args = parser.parse_args(
        [
            "prepare",
            "--op",
            "softmax",
            "--dtype",
            "float16",
            "--dataset",
            "smoke",
            "--case-id",
            "tiny_logits",
            "--seed",
            "7",
            "--output",
            "prepared.json",
        ]
    )

    assert args.command == "prepare"
    assert args.seed == 7


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
                "--dataset",
                "smoke",
                "--case-id",
                "tiny_logits",
            ]
        )


def test_main_runs_operator_benchmark_and_writes_outputs(tmp_path, monkeypatch):
    captured: dict[str, object] = {}
    result = sample_result()

    class FakeBackend:
        def run_operator(self, request):
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
            "--dataset",
            "realistic",
            "--case-id",
            "t5_attention",
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
    assert request.dataset == "realistic"
    assert request.case_id == "t5_attention"
    assert request.dimensions == (4, 8, 1024, 1024)
    assert request.dim == -1
    assert request.warmup == 2
    assert request.iterations == 3
    assert captured["output_dir"] == tmp_path
    assert captured["run_name"] == "softmax-run"
    assert captured["result"] is result
    assert captured["formats"] == ("json", "csv", "md")


def test_main_prepare_writes_prepared_input_manifest(tmp_path):
    output_path = tmp_path / "prepared-softmax.json"

    exit_code = main(
        [
            "prepare",
            "--op",
            "softmax",
            "--dtype",
            "float16",
            "--dataset",
            "smoke",
            "--case-id",
            "tiny_logits",
            "--seed",
            "7",
            "--output",
            str(output_path),
        ]
    )

    payload = output_path.read_text()
    assert exit_code == 0
    assert "\"schema_version\": 1" in payload
    assert "\"case_id\": \"tiny_logits\"" in payload


def test_main_runs_operator_benchmark_from_prepared_input(tmp_path, monkeypatch):
    captured: dict[str, object] = {}
    prepared_path = tmp_path / "prepared-softmax.json"
    prepared_path.write_text(
        """{
  "schema_version": 1,
  "op": "softmax",
  "dtype": "float16",
  "dataset": "smoke",
  "seed": 7,
  "case": {
    "case_id": "tiny_logits",
    "family": "lm_logits",
    "source_kind": "synthetic_smoke",
    "source_project": "cannbench",
    "source_model": "smoke_fixture",
    "source_file": "built-in",
    "source_op": "softmax",
    "payload": {
      "dimensions": [32, 128],
      "dim": -1
    }
  }
}
"""
    )
    result = sample_result()

    class FakeBackend:
        def run_operator(self, request):
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
            "--prepared-input",
            str(prepared_path),
            "--warmup",
            "2",
            "--iterations",
            "3",
            "--output-dir",
            str(tmp_path),
            "--run-name",
            "prepared-run",
        ]
    )

    request = captured["request"]
    assert exit_code == 0
    assert request.dataset == "smoke"
    assert request.case_id == "tiny_logits"
    assert request.seed == 7
    assert request.dimensions == (32, 128)
    assert captured["run_name"] == "prepared-run"


def test_main_rejects_zero_iterations():
    with pytest.raises(SystemExit):
        main(
            [
                "operator",
                "--backend",
                "nvidia",
                "--op",
                "softmax",
                "--dataset",
                "smoke",
                "--case-id",
                "tiny_logits",
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
                "--dataset",
                "smoke",
                "--case-id",
                "tiny_logits",
                "--warmup",
                "-1",
            ]
        )


def test_main_converts_backend_runtime_failure_to_cli_error(monkeypatch, capsys):
    class FailingBackend:
        def run_operator(self, request):
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
                "--dataset",
                "smoke",
                "--case-id",
                "tiny_logits",
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
