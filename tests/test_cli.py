import runpy
import tomllib

import pytest

from cannbench.cli import build_parser, main
from cannbench.core.operator_output import CapturedOperatorOutput, OutputComparisonResult
from cannbench.core.result import (
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
        warmup=2,
        iterations=3,
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


def test_build_parser_exposes_capture_output_subcommand():
    parser = build_parser()
    args = parser.parse_args(
        [
            "capture-output",
            "--backend",
            "nvidia",
            "--op",
            "softmax",
            "--dataset",
            "smoke",
            "--case-id",
            "tiny_logits",
            "--output",
            "nvidia-output",
        ]
    )

    assert args.command == "capture-output"
    assert args.backend == "nvidia"
    assert args.output.name == "nvidia-output"


def test_build_parser_exposes_compare_output_subcommand():
    parser = build_parser()
    args = parser.parse_args(
        [
            "compare-output",
            "--left",
            "nvidia-output",
            "--right",
            "ascend-output",
            "--output",
            "accuracy.json",
        ]
    )

    assert args.command == "compare-output"
    assert args.rtol == 0.001
    assert args.atol == 0.001


def test_build_parser_exposes_collect_subcommand():
    parser = build_parser()
    args = parser.parse_args(
        [
            "collect",
            "--endpoint",
            "configs/ascend.json",
            "--prepared-input",
            "prepared-softmax.json",
            "--output-dir",
            "results/ascend-softmax",
            "--run-id",
            "softmax-run",
            "--capture-output",
            "--profile-device-time",
            "--summarize-profile",
            "--warmup",
            "3",
            "--iterations",
            "5",
            "--deploy-custom-op",
        ]
    )

    assert args.command == "collect"
    assert args.capture_output is True
    assert args.profile_device_time is True
    assert args.summarize_profile is True
    assert args.warmup == 3
    assert args.iterations == 5
    assert args.deploy_custom_op is True


def test_build_parser_defaults_operator_iterations_to_one():
    parser = build_parser()
    args = parser.parse_args(
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

    assert args.iterations == 1


def test_build_parser_defaults_collect_iterations_to_one():
    parser = build_parser()
    args = parser.parse_args(
        [
            "collect",
            "--endpoint",
            "configs/ascend.json",
            "--prepared-input",
            "prepared-softmax.json",
            "--output-dir",
            "results/ascend-softmax",
            "--profile-device-time",
        ]
    )

    assert args.iterations == 1


def test_build_parser_exposes_report_subcommand():
    parser = build_parser()
    args = parser.parse_args(
        [
            "report",
            "--nvidia",
            "results/nvidia-softmax",
            "--ascend",
            "results/ascend-softmax",
            "--accuracy",
            "results/accuracy.json",
            "--output",
            "results/report.md",
        ]
    )

    assert args.command == "report"
    assert args.output.name == "report.md"


def test_build_parser_exposes_summarize_profile_subcommand():
    parser = build_parser()
    args = parser.parse_args(
        [
            "summarize-profile",
            "--backend",
            "ascend",
            "--profile-dir",
            "results/ascend/profile",
            "--output",
            "results/ascend/profile-summary.json",
        ]
    )

    assert args.command == "summarize-profile"
    assert args.backend == "ascend"


def test_build_parser_accepts_ascend_backend():
    parser = build_parser()
    args = parser.parse_args(
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

    assert args.backend == "ascend"


def test_build_parser_exposes_boolean_custom_op_deployment_flag():
    parser = build_parser()
    args = parser.parse_args(
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
            "--deploy-custom-op",
        ]
    )

    assert args.deploy_custom_op is True


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
    assert request.deploy_custom_op is False
    assert captured["output_dir"] == tmp_path
    assert captured["run_name"] == "softmax-run"
    assert captured["result"] is result
    assert captured["formats"] == ("json", "csv", "md")


def test_main_passes_custom_op_deployment_flag_to_request(tmp_path, monkeypatch):
    captured: dict[str, object] = {}
    result = sample_result()

    class FakeBackend:
        def run_operator(self, request):
            captured["request"] = request
            return result

    monkeypatch.setattr("cannbench.cli.get_backend", lambda name: FakeBackend())
    monkeypatch.setattr(
        "cannbench.cli.write_benchmark_outputs",
        lambda output_dir, run_name, actual_result, formats: {},
    )

    exit_code = main(
        [
            "operator",
            "--backend",
            "ascend",
            "--op",
            "softmax",
            "--dtype",
            "float16",
            "--dataset",
            "smoke",
            "--case-id",
            "tiny_logits",
            "--deploy-custom-op",
            "--output-dir",
            str(tmp_path),
        ]
    )

    assert exit_code == 0
    assert captured["request"].backend == "ascend"
    assert captured["request"].deploy_custom_op is True


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


def test_main_capture_output_writes_artifact(tmp_path, monkeypatch):
    captured: dict[str, object] = {}
    output = CapturedOperatorOutput(
        backend="nvidia",
        device_name="Fake GPU",
        op="softmax",
        dtype="float16",
        dataset="smoke",
        case_id="tiny_logits",
        seed=7,
        shape=(1,),
        values=(1.0,),
    )

    class FakeBackend:
        def capture_operator_output(self, request):
            captured["request"] = request
            return output

    monkeypatch.setattr("cannbench.cli.get_backend", lambda name: FakeBackend())

    def fake_write_operator_output(path, actual_output):
        captured["output_path"] = path
        captured["output"] = actual_output
        return {}

    monkeypatch.setattr("cannbench.cli.write_operator_output", fake_write_operator_output)

    exit_code = main(
        [
            "capture-output",
            "--backend",
            "nvidia",
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
            str(tmp_path / "nvidia-output"),
        ]
    )

    assert exit_code == 0
    assert captured["request"].seed == 7
    assert captured["request"].warmup == 0
    assert captured["request"].iterations == 1
    assert captured["output_path"] == tmp_path / "nvidia-output"
    assert captured["output"] is output


def test_main_compare_output_writes_report(tmp_path, monkeypatch):
    captured: dict[str, object] = {}
    comparison = OutputComparisonResult(
        passed=True,
        shape_match=True,
        left_backend="nvidia",
        right_backend="ascend",
        op="softmax",
        dtype_left="float16",
        dtype_right="float16",
        case_id="tiny_logits",
        seed_left=7,
        seed_right=7,
        shape=(1,),
        numel=1,
        mismatch_count=0,
        max_abs_error=0.0,
        max_rel_error=0.0,
        mean_abs_error=0.0,
        rmse=0.0,
        rtol=0.001,
        atol=0.001,
    )

    monkeypatch.setattr("cannbench.cli.read_operator_output", lambda path: path)
    monkeypatch.setattr(
        "cannbench.cli.compare_operator_outputs",
        lambda left, right, rtol, atol: comparison,
    )

    def fake_write_output_comparison(path, actual_comparison):
        captured["report_path"] = path
        captured["comparison"] = actual_comparison
        return path

    monkeypatch.setattr(
        "cannbench.cli.write_output_comparison", fake_write_output_comparison
    )

    exit_code = main(
        [
            "compare-output",
            "--left",
            str(tmp_path / "nvidia-output"),
            "--right",
            str(tmp_path / "ascend-output"),
            "--rtol",
            "0.001",
            "--atol",
            "0.001",
            "--output",
            str(tmp_path / "accuracy.json"),
        ]
    )

    assert exit_code == 0
    assert captured["report_path"] == tmp_path / "accuracy.json"
    assert captured["comparison"] is comparison


def test_main_collect_invokes_remote_collection(tmp_path, monkeypatch):
    captured: dict[str, object] = {}
    endpoint_path = tmp_path / "ascend.json"
    prepared_path = tmp_path / "prepared.json"
    output_dir = tmp_path / "results"

    def fake_collect_remote_artifacts(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(
        "cannbench.cli.collect_remote_artifacts", fake_collect_remote_artifacts
    )

    exit_code = main(
        [
            "collect",
            "--endpoint",
            str(endpoint_path),
            "--prepared-input",
            str(prepared_path),
            "--output-dir",
            str(output_dir),
            "--run-id",
            "softmax-run",
            "--capture-output",
            "--profile-device-time",
            "--summarize-profile",
            "--warmup",
            "3",
            "--iterations",
            "5",
        ]
    )

    assert exit_code == 0
    assert captured["endpoint_path"] == endpoint_path
    assert captured["prepared_input"] == prepared_path
    assert captured["output_dir"] == output_dir
    assert captured["run_id"] == "softmax-run"
    assert captured["capture_output"] is True
    assert captured["profile_device_time"] is True
    assert captured["summarize_profile"] is True
    assert captured["warmup"] == 3
    assert captured["iterations"] == 5
    assert captured["deploy_custom_op"] is False


def test_main_report_writes_local_report(tmp_path, monkeypatch):
    captured: dict[str, object] = {}

    def fake_write_local_report(**kwargs):
        captured.update(kwargs)
        return kwargs["output_path"]

    monkeypatch.setattr("cannbench.cli.write_local_report", fake_write_local_report)

    exit_code = main(
        [
            "report",
            "--nvidia",
            str(tmp_path / "nvidia"),
            "--ascend",
            str(tmp_path / "ascend"),
            "--accuracy",
            str(tmp_path / "accuracy.json"),
            "--output",
            str(tmp_path / "report.md"),
        ]
    )

    assert exit_code == 0
    assert captured["nvidia_dir"] == tmp_path / "nvidia"
    assert captured["ascend_dir"] == tmp_path / "ascend"
    assert captured["accuracy_path"] == tmp_path / "accuracy.json"
    assert captured["output_path"] == tmp_path / "report.md"


def test_main_summarize_profile_writes_summary(tmp_path, monkeypatch):
    captured: dict[str, object] = {}
    summary = object()

    monkeypatch.setattr(
        "cannbench.cli.read_device_profile",
        lambda profile_dir, backend: captured.setdefault("summary", summary),
    )

    def fake_write_device_profile_summary(path, actual_summary):
        captured["output_path"] = path
        captured["actual_summary"] = actual_summary
        return path

    monkeypatch.setattr(
        "cannbench.cli.write_device_profile_summary",
        fake_write_device_profile_summary,
    )

    exit_code = main(
        [
            "summarize-profile",
            "--backend",
            "ascend",
            "--profile-dir",
            str(tmp_path / "profile"),
            "--output",
            str(tmp_path / "profile-summary.json"),
        ]
    )

    assert exit_code == 0
    assert captured["output_path"] == tmp_path / "profile-summary.json"
    assert captured["actual_summary"] is summary


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
            "--deploy-custom-op",
        ]
    )

    request = captured["request"]
    assert exit_code == 0
    assert request.dataset == "smoke"
    assert request.case_id == "tiny_logits"
    assert request.seed == 7
    assert request.dimensions == (32, 128)
    assert request.deploy_custom_op is True
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


def test_package_data_includes_ascend_custom_op_defaults():
    with open("pyproject.toml", "rb") as config:
        payload = tomllib.load(config)

    package_data = payload["tool"]["setuptools"]["package-data"]

    assert "cannbench.datasets.data.*.custom_ops.ascend.default" in package_data
