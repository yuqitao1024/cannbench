import runpy
import tomllib
import json
from pathlib import Path

import pytest

from cannbench.cli import build_parser, main
from cannbench.cli import _build_canonical_run_name
from cannbench.core.execution import RemoteExecutionArtifacts, RemoteProfileArtifacts
from cannbench.core.layout import build_run_layout
from cannbench.core.operator_output import CapturedOperatorOutput, OutputComparisonResult
from cannbench.core.profile import DeviceProfileSummary, LocalDeviceProfileResult, ProfileArtifacts
from cannbench.core.remote import RemoteCollectionResult, RemoteEndpoint
from cannbench.core.result import (
    OperatorBenchmarkResult,
    OperatorCase,
    build_softmax_case,
)
from cannbench.core.prepared_input import (
    build_prepared_operator_input,
    read_prepared_operator_input,
    write_prepared_operator_input,
)
from cannbench.datasets import get_operator_dataset


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


def result_for_request(request) -> OperatorBenchmarkResult:
    return OperatorBenchmarkResult(
        backend=request.backend,
        device_name="Fake GPU",
        op=request.op,
        dtype=request.dtype,
        case=OperatorCase(
            case_id=request.case_id,
            family=request.family,
            source_kind=request.source_kind,
            source_project=request.source_project,
            source_model=request.source_model,
            source_file=request.source_file,
            source_op=request.source_op,
            payload=request.case_payload,
        ),
        warmup=request.warmup,
        iterations=request.iterations,
    )


def remote_collect_result(
    *,
    endpoint: RemoteEndpoint,
    run_id: str,
    output_dir: Path,
    prepared,
    capture_output: bool = False,
    profile_device_time: bool = False,
    device_name: str = "Ascend 910B",
    warmup: int = 10,
    iterations: int = 1,
    extra_perf_artifacts: tuple[tuple[str, bytes], ...] = (),
) -> RemoteCollectionResult:
    profile_summary = DeviceProfileSummary(
        backend=endpoint.backend,
        sample_count=1,
        latency_ms_avg=1.0,
        latency_ms_p50=1.0,
        latency_ms_p95=1.0,
        latency_ms_p99=1.0,
        source_files=("op_summary.csv",),
    )
    perf_payload = {
        "backend": endpoint.backend,
        "device_name": device_name,
        "op": prepared.op,
        "dtype": prepared.dtype,
        "case": prepared.case.to_json_dict(),
        "warmup": warmup,
        "iterations": iterations,
    }
    profile = None
    if profile_device_time:
        profile = RemoteProfileArtifacts(
            backend=endpoint.backend,
            device_name=device_name,
            profile_summary=profile_summary,
            profile_artifacts=(("op_summary.csv", b"Op Name,Task Duration(us)\nsoftmax,1000\n"),),
            perf_artifacts=(
                ("benchmark.json", (json.dumps(perf_payload) + "\n").encode("utf-8")),
                *extra_perf_artifacts,
            ),
        )
    return RemoteCollectionResult(
        endpoint=endpoint,
        run_id=run_id,
        remote_run_dir=f"{endpoint.workdir}/.cannbench-runs/{run_id}",
        local_output_dir=output_dir,
        artifacts=RemoteExecutionArtifacts(
            output_artifacts=(("tensor.json", b"{}"),) if capture_output else (),
            profile=profile,
        ),
    )


def test_build_parser_exposes_internal_run_subcommand():
    parser = build_parser()
    args = parser.parse_args(
        [
            "internal-run",
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

    assert args.command == "internal-run"
    assert args.backend == "nvidia"
    assert args.op == "softmax"
    assert args.dataset == "realistic"
    assert args.case_id == "t5_attention"


def test_build_parser_exposes_bench_subcommand():
    parser = build_parser()
    args = parser.parse_args(
        [
            "bench",
            "--backend",
            "ascend",
            "--implementation",
            "simt",
            "--implementation-version",
            "v2",
            "--endpoint",
            "configs/ascend.json",
            "--op",
            "softmax",
            "--dataset",
            "realistic",
            "--case-id",
            "t5_attention",
        ]
    )

    assert args.command == "bench"
    assert args.backend == "ascend"
    assert args.implementation == "simt"
    assert args.implementation_version == "v2"
    assert args.endpoint == Path("configs/ascend.json")
    assert args.op == "softmax"


def test_build_canonical_run_name_uses_simt_version():
    run_name = _build_canonical_run_name(
        backend="ascend",
        implementation="simt",
        implementation_version="v2",
        op="softmax",
        dataset="realistic",
        dtype="float16",
    )

    assert run_name == "opbench-ascend-950pr-simt-v2-softmax-realistic-float16"


def test_build_parser_accepts_embedding_internal_run():
    parser = build_parser()
    args = parser.parse_args(
        [
            "internal-run",
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


def test_build_parser_exposes_compare_subcommand():
    parser = build_parser()
    args = parser.parse_args(
        [
            "compare",
            "--left-backend",
            "nvidia",
            "--right-backend",
            "ascend",
            "--op",
            "softmax",
            "--dataset",
            "smoke",
            "--case-id",
            "tiny_logits",
            "--output",
            "accuracy.json",
        ]
    )

    assert args.command == "compare"
    assert args.left_backend == "nvidia"
    assert args.right_backend == "ascend"
    assert args.rtol == 0.001
    assert args.atol == 0.001


def test_build_parser_rejects_collect_subcommand():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["collect"])


def test_build_parser_accepts_prepared_dir_for_bench():
    parser = build_parser()
    args = parser.parse_args(
        [
            "bench",
            "--backend",
            "nvidia",
            "--op",
            "softmax",
            "--prepared-dir",
            "prepared/softmax",
            "--run-name",
            "softmax-batch",
        ]
    )

    assert args.prepared_dir == Path("prepared/softmax")
    assert args.run_name == "softmax-batch"


def test_build_parser_accepts_prepared_dir_for_remote_bench():
    parser = build_parser()
    args = parser.parse_args(
        [
            "bench",
            "--backend",
            "ascend",
            "--endpoint",
            "configs/ascend.json",
            "--op",
            "softmax",
            "--prepared-dir",
            "prepared/softmax",
            "--run-name",
            "softmax-batch",
        ]
    )

    assert args.prepared_dir == Path("prepared/softmax")
    assert args.run_name == "softmax-batch"


def test_build_parser_exposes_publish_subcommand():
    parser = build_parser()
    args = parser.parse_args(
        [
            "publish",
            "--source",
            "runs/softmax-run",
            "--dest",
            "published/softmax-run",
        ]
    )

    assert args.command == "publish"
    assert args.source == Path("runs/softmax-run")
    assert args.dest == Path("published/softmax-run")


def test_build_parser_exposes_serve_subcommand():
    parser = build_parser()
    args = parser.parse_args(
        [
            "serve",
            "--frontend-dir",
            "web/dist",
            "--published-dir",
            "published",
            "--host",
            "0.0.0.0",
            "--port",
            "9000",
            "--enable-gpu-upload",
        ]
    )

    assert args.command == "serve"
    assert args.frontend_dir == Path("web/dist")
    assert args.published_dir == Path("published")
    assert args.host == "0.0.0.0"
    assert args.port == 9000
    assert args.enable_gpu_upload is True


def test_build_parser_defaults_internal_run_iterations_to_one():
    parser = build_parser()
    args = parser.parse_args(
        [
            "internal-run",
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


def test_build_parser_defaults_bench_iterations_to_one():
    parser = build_parser()
    args = parser.parse_args(
        [
            "bench",
            "--backend",
            "ascend",
            "--endpoint",
            "configs/ascend.json",
            "--op",
            "softmax",
            "--dataset",
            "smoke",
            "--case-id",
            "tiny_logits",
        ]
    )

    assert args.iterations == 1


def test_build_parser_defaults_bench_dataset_to_realistic():
    parser = build_parser()
    args = parser.parse_args(
        [
            "bench",
            "--backend",
            "nvidia",
            "--op",
            "softmax",
        ]
    )

    assert args.dataset == "realistic"


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


def test_build_parser_accepts_ascend_backend():
    parser = build_parser()
    args = parser.parse_args(
        [
            "internal-run",
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


def test_build_parser_exposes_boolean_simt_op_deployment_flag():
    parser = build_parser()
    args = parser.parse_args(
        [
            "internal-run",
            "--backend",
            "ascend",
            "--op",
            "softmax",
            "--dataset",
            "smoke",
            "--case-id",
            "tiny_logits",
            "--deploy-simt-op",
        ]
    )

    assert args.deploy_simt_op is True


def test_main_runs_internal_run_and_writes_outputs(tmp_path, monkeypatch):
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
            "internal-run",
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
    assert request.deploy_simt_op is False
    assert captured["output_dir"] == tmp_path
    assert captured["run_name"] == "softmax-run"
    assert captured["result"] is result
    assert captured["formats"] == ("json", "csv", "md")


def test_main_runs_bench_and_maps_simt_to_simt_op_deployment(tmp_path, monkeypatch):
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
            "bench",
            "--backend",
            "ascend",
            "--implementation",
            "simt",
            "--implementation-version",
            "v2",
            "--op",
            "softmax",
            "--dtype",
            "float16",
            "--dataset",
            "smoke",
            "--case-id",
            "tiny_logits",
            "--output-dir",
            str(tmp_path),
        ]
    )

    assert exit_code == 0
    assert captured["request"].backend == "ascend"
    assert captured["request"].deploy_simt_op is True
    assert captured["request"].implementation_version == "v2"


def test_main_bench_single_dispatches_through_single_bench_helper(tmp_path, monkeypatch):
    captured: dict[str, object] = {}

    monkeypatch.setattr("cannbench.cli._run_single_bench", lambda args: captured.setdefault("single", args))
    monkeypatch.setattr("cannbench.cli._is_batch_mode", lambda args: False)

    exit_code = main(
        [
            "bench",
            "--backend",
            "nvidia",
            "--op",
            "softmax",
            "--dataset",
            "smoke",
            "--case-id",
            "tiny_logits",
            "--output-dir",
            str(tmp_path),
        ]
    )

    assert exit_code == 0
    assert captured["single"].command == "bench"


def test_main_local_bench_uses_local_executor(tmp_path, monkeypatch):
    captured: dict[str, object] = {}

    class FakeExecutor:
        def __init__(self, backend, write_outputs):
            captured["backend"] = backend
            captured["write_outputs"] = write_outputs

        def execute_case(self, request, *, output_dir, run_name):
            captured["request"] = request
            captured["output_dir"] = output_dir
            captured["run_name"] = run_name
            return type(
                "ExecResult",
                (),
                {
                    "artifacts": type("Artifacts", (), {"profile": None})(),
                    "result_path": output_dir / f"{run_name}.json",
                },
            )()

    class FakeBackend:
        pass

    monkeypatch.setattr("cannbench.cli.get_backend", lambda name: FakeBackend())
    monkeypatch.setattr("cannbench.cli.LocalBenchExecutor", FakeExecutor)

    exit_code = main(
        [
            "bench",
            "--backend",
            "nvidia",
            "--op",
            "softmax",
            "--dataset",
            "smoke",
            "--case-id",
            "tiny_logits",
            "--output-dir",
            str(tmp_path),
            "--run-name",
            "executor-local",
        ]
    )

    assert exit_code == 0
    assert captured["run_name"] == "softmax-smoke-tiny_logits-float16-seed0"


def test_main_remote_bench_uses_remote_executor(tmp_path, monkeypatch):
    captured: dict[str, object] = {}
    endpoint = RemoteEndpoint(
        name="ascend-a2",
        backend="ascend",
        host="user@ascend-host",
        workdir="/opt/cannbench",
        python="python3",
        env={},
    )

    class FakeExecutor:
        def __init__(self, collect_remote_artifacts, actual_endpoint, endpoint_path=None):
            captured["endpoint"] = actual_endpoint
            captured["endpoint_path"] = endpoint_path

        def execute_case(
            self,
            *,
            prepared_input,
            layout_root,
            artifact_stem,
            run_id,
            capture_output,
                warmup,
                iterations,
                deploy_simt_op,
                implementation_version=None,
            ):
            captured["prepared_input"] = prepared_input
            captured["layout_root"] = layout_root
            captured["artifact_stem"] = artifact_stem
            captured["run_id"] = run_id
            captured["capture_output"] = capture_output
            captured["warmup"] = warmup
            captured["iterations"] = iterations
            captured["deploy_simt_op"] = deploy_simt_op
            captured["implementation_version"] = implementation_version
            return type(
                "ExecResult",
                (),
                {
                    "artifacts": type(
                        "Artifacts",
                        (),
                        {
                            "profile": type(
                                "Profile",
                                (),
                                {
                                    "device_name": "Ascend 910B",
                                    "profile_summary": DeviceProfileSummary(
                                        backend="ascend",
                                        sample_count=1,
                                        latency_ms_avg=1.0,
                                        latency_ms_p50=1.0,
                                        latency_ms_p95=1.0,
                                        latency_ms_p99=1.0,
                                        source_files=("op_summary.csv",),
                                    ),
                                    "profile_artifacts": (("op_summary.csv", b"Op Name,Task Duration(us)\nsoftmax,1000\n"),),
                                    "perf_artifacts": (("benchmark.json", b"{}\n"),),
                                },
                            )()
                        },
                    )(),
                    "result_path": None,
                },
            )()

    monkeypatch.setattr("cannbench.cli.read_remote_endpoint", lambda path: endpoint)
    monkeypatch.setattr("cannbench.cli.RemoteBenchExecutor", FakeExecutor)

    exit_code = main(
        [
            "bench",
            "--backend",
            "ascend",
            "--endpoint",
            str(tmp_path / "ascend.json"),
            "--op",
            "softmax",
            "--dataset",
            "smoke",
            "--case-id",
            "tiny_logits",
            "--output-dir",
            str(tmp_path),
            "--run-id",
            "executor-remote",
        ]
    )

    assert exit_code == 0
    assert captured["endpoint"] == endpoint
    assert captured["run_id"] == "executor-remote"
    assert captured["implementation_version"] is None


def test_main_runs_single_bench_with_profile_layout_and_meta(tmp_path, monkeypatch):
    captured: dict[str, object] = {}
    result = sample_result()

    class FakeBackend:
        def run_operator(self, request):
            captured["request"] = request
            return result

        def profile_operator_device_time(self, request):
            return LocalDeviceProfileResult(
                benchmark_result=result_for_request(request),
                profile=ProfileArtifacts(
                    device_name="Fake GPU",
                    profile_summary=DeviceProfileSummary(
                        backend="nvidia",
                        sample_count=1,
                        latency_ms_avg=0.1,
                        latency_ms_p50=0.1,
                        latency_ms_p95=0.1,
                        latency_ms_p99=0.1,
                        source_files=("ncu.csv",),
                    ),
                    profile_artifacts=(
                        (
                            "ncu.csv",
                            b"Kernel Name,Metric Name,Metric Unit,Metric Value\n"
                            b"softmax,gpu__time_duration.sum,nsecond,100000\n",
                        ),
                    ),
                    perf_artifacts=(
                        (
                            "benchmark.json",
                            (
                                json.dumps(result_for_request(request).to_json_dict()) + "\n"
                            ).encode("utf-8"),
                        ),
                    ),
                ),
            )

    monkeypatch.setattr("cannbench.cli.get_backend", lambda name: FakeBackend())

    exit_code = main(
        [
            "bench",
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
            "--output-dir",
            str(tmp_path),
            "--run-name",
            "single-bench-profiled",
        ]
    )

    layout = build_run_layout(tmp_path, "single-bench-profiled")
    summary = json.loads((layout.meta_dir / "summary.json").read_text())
    benchmark_records = json.loads((layout.meta_dir / "benchmark-records.json").read_text())
    failures = json.loads((layout.meta_dir / "failures.json").read_text())

    assert exit_code == 0
    assert (layout.prepared_dir / "softmax" / "smoke" / "tiny_logits-float16-seed0.json").exists()
    assert (layout.perf_dir / "softmax-smoke-tiny_logits-float16-seed0.json").exists()
    assert (layout.profile_dir / "softmax-smoke-tiny_logits-float16-seed0" / "ncu.csv").exists()
    assert summary["metadata"]["run_name"] == "single-bench-profiled"
    assert summary["metadata"]["backend"] == "nvidia"
    assert summary["result_count"] == 1
    assert benchmark_records["records"][0]["implementation"] == "cuda-pytorch"
    assert failures["failure_count"] == 0


def test_main_runs_single_remote_bench_with_profile_layout_and_meta(tmp_path, monkeypatch):
    endpoint = RemoteEndpoint(
        name="ascend-a2",
        backend="ascend",
        host="user@ascend-host",
        workdir="/opt/cannbench",
        python="python3",
        env={},
    )
    monkeypatch.setattr("cannbench.cli.read_remote_endpoint", lambda path: endpoint)

    def fake_collect_remote_artifacts(**kwargs):
        prepared = read_prepared_operator_input(kwargs["prepared_input"])
        return remote_collect_result(
            endpoint=endpoint,
            run_id=kwargs["run_id"],
            output_dir=kwargs["output_dir"],
            prepared=prepared,
            capture_output=True,
            profile_device_time=True,
            warmup=kwargs["warmup"],
            iterations=kwargs["iterations"],
        )

    monkeypatch.setattr(
        "cannbench.cli.collect_remote_artifacts", fake_collect_remote_artifacts
    )

    exit_code = main(
        [
            "bench",
            "--backend",
            "ascend",
            "--endpoint",
            str(tmp_path / "ascend.json"),
            "--output-dir",
            str(tmp_path),
            "--run-id",
            "single-collect-profiled",
            "--op",
            "softmax",
            "--dtype",
            "float16",
            "--dataset",
            "smoke",
            "--case-id",
            "tiny_logits",
            "--capture-output",
        ]
    )

    canonical = "opbench-ascend-950pr-cannops-softmax-smoke-float16"
    layout = build_run_layout(tmp_path, canonical)
    summary = json.loads((layout.meta_dir / "summary.json").read_text())
    benchmark_records = json.loads((layout.meta_dir / "benchmark-records.json").read_text())
    failures = json.loads((layout.meta_dir / "failures.json").read_text())

    assert exit_code == 0
    assert (layout.prepared_dir / "softmax" / "smoke" / "tiny_logits-float16-seed0.json").exists()
    assert (layout.output_dir / "softmax-smoke-tiny_logits-float16-seed0" / "tensor.json").exists()
    assert (layout.profile_dir / "softmax-smoke-tiny_logits-float16-seed0" / "op_summary.csv").exists()
    assert (layout.perf_dir / "softmax-smoke-tiny_logits-float16-seed0.json").exists()
    assert summary["metadata"]["backend"] == "ascend"
    assert summary["result_count"] == 1
    assert benchmark_records["records"][0]["implementation"] == "cann_ops_library"
    assert failures["failure_count"] == 0


def test_main_single_local_bench_builds_prepared_plan_from_args(tmp_path, monkeypatch):
    captured: dict[str, object] = {}

    class FakeBackend:
        def run_operator(self, request):
            return result_for_request(request)

    monkeypatch.setattr("cannbench.cli.get_backend", lambda name: FakeBackend())

    def fake_build_prepared_operator_input(**kwargs):
        captured["built_kwargs"] = kwargs
        return build_prepared_operator_input(**kwargs)

    monkeypatch.setattr(
        "cannbench.cli.build_prepared_operator_input",
        fake_build_prepared_operator_input,
    )

    exit_code = main(
        [
            "bench",
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
            "--output-dir",
            str(tmp_path),
            "--run-name",
            "single-local-plan",
        ]
    )

    assert exit_code == 0
    assert captured["built_kwargs"] == {
        "op": "softmax",
        "dtype": "float16",
        "dataset": "smoke",
        "case_id": "tiny_logits",
        "seed": 7,
    }


def test_main_single_local_bench_defaults_to_canonical_run_name(tmp_path, monkeypatch):
    result = sample_result()

    class FakeBackend:
        def run_operator(self, request):
            return result_for_request(request)

        def profile_operator_device_time(self, request):
            return LocalDeviceProfileResult(
                benchmark_result=result_for_request(request),
                profile=ProfileArtifacts(
                    device_name="Fake GPU",
                    profile_summary=DeviceProfileSummary(
                        backend="nvidia",
                        sample_count=1,
                        latency_ms_avg=0.1,
                        latency_ms_p50=0.1,
                        latency_ms_p95=0.1,
                        latency_ms_p99=0.1,
                        source_files=("ncu.csv",),
                    ),
                    profile_artifacts=(("ncu.csv", b"metric\n"),),
                    perf_artifacts=(
                        ("benchmark.json", (json.dumps(result.to_json_dict()) + "\n").encode("utf-8")),
                    ),
                ),
            )

    monkeypatch.setattr("cannbench.cli.get_backend", lambda name: FakeBackend())

    exit_code = main(
        [
            "bench",
            "--backend",
            "nvidia",
            "--op",
            "softmax",
            "--case-id",
            "t5_attention",
            "--output-dir",
            str(tmp_path),
        ]
    )

    layout = build_run_layout(
        tmp_path, "opbench-nvidia-h800-cuda-pytorch-softmax-realistic-float16"
    )
    summary = json.loads((layout.meta_dir / "summary.json").read_text())

    assert exit_code == 0
    assert summary["metadata"]["run_name"] == "opbench-nvidia-h800-cuda-pytorch-softmax-realistic-float16"
    assert summary["records"][0]["dataset"] == "realistic"


def test_main_single_bench_uses_single_result_metadata_shape(tmp_path, monkeypatch):
    result = sample_result()

    class FakeBackend:
        def run_operator(self, request):
            return result_for_request(request)

        def profile_operator_device_time(self, request):
            return LocalDeviceProfileResult(
                benchmark_result=result_for_request(request),
                profile=ProfileArtifacts(
                    device_name="Fake GPU",
                    profile_summary=DeviceProfileSummary(
                        backend="nvidia",
                        sample_count=1,
                        latency_ms_avg=0.1,
                        latency_ms_p50=0.1,
                        latency_ms_p95=0.1,
                        latency_ms_p99=0.1,
                        source_files=("ncu.csv",),
                    ),
                    profile_artifacts=(("ncu.csv", b"metric\n"),),
                    perf_artifacts=(
                        ("benchmark.json", (json.dumps(result.to_json_dict()) + "\n").encode("utf-8")),
                    ),
                ),
            )

    monkeypatch.setattr("cannbench.cli.get_backend", lambda name: FakeBackend())

    exit_code = main(
        [
            "bench",
            "--backend",
            "nvidia",
            "--op",
            "softmax",
            "--dataset",
            "smoke",
            "--case-id",
            "tiny_logits",
            "--output-dir",
            str(tmp_path),
            "--run-name",
            "single-shape",
        ]
    )

    layout = build_run_layout(tmp_path, "single-shape")
    summary = json.loads((layout.meta_dir / "summary.json").read_text())
    failures = json.loads((layout.meta_dir / "failures.json").read_text())

    assert exit_code == 0
    assert summary["metadata"] == {
        "backend": "nvidia",
        "run_name": "single-shape",
        "implementation": None,
        "total_cases": 1,
        "success_count": 1,
        "failure_count": 0,
    }
    assert len(summary["records"]) == 1
    assert summary["records"][0]["status"] == "ok"
    assert failures["records"] == []


def test_main_passes_simt_op_deployment_flag_to_internal_run_request(tmp_path, monkeypatch):
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
            "internal-run",
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
            "--deploy-simt-op",
            "--output-dir",
            str(tmp_path),
        ]
    )

    assert exit_code == 0
    assert captured["request"].backend == "ascend"
    assert captured["request"].deploy_simt_op is True


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


def test_main_compare_captures_outputs_and_writes_report(tmp_path, monkeypatch):
    captured: dict[str, object] = {}
    left_output = CapturedOperatorOutput(
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
    right_output = CapturedOperatorOutput(
        backend="ascend",
        device_name="Fake NPU",
        op="softmax",
        dtype="float16",
        dataset="smoke",
        case_id="tiny_logits",
        seed=7,
        shape=(1,),
        values=(1.0,),
    )
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

    class FakeBackend:
        def capture_operator_output(self, request):
            requests = captured.setdefault("requests", [])
            requests.append(request)
            return left_output if request.backend == "nvidia" else right_output

    monkeypatch.setattr("cannbench.cli.get_backend", lambda name: FakeBackend())
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
            "compare",
            "--left-backend",
            "nvidia",
            "--right-backend",
            "ascend",
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
            str(tmp_path / "accuracy.json"),
        ]
    )

    assert exit_code == 0
    assert len(captured["requests"]) == 2
    assert captured["requests"][0].backend == "nvidia"
    assert captured["requests"][1].backend == "ascend"
    assert captured["requests"][0].seed == 7
    assert captured["requests"][0].warmup == 0
    assert captured["requests"][0].iterations == 1
    assert captured["report_path"] == tmp_path / "accuracy.json"
    assert captured["comparison"] is comparison


def test_main_bench_invokes_remote_collection(tmp_path, monkeypatch):
    captured: dict[str, object] = {}
    endpoint_path = tmp_path / "ascend.json"
    prepared_path = tmp_path / "prepared.json"
    output_dir = tmp_path / "results"
    endpoint = RemoteEndpoint(
        name="ascend-a2",
        backend="ascend",
        host="user@ascend-host",
        workdir="/opt/cannbench",
        python="python3",
        env={},
    )
    write_prepared_operator_input(
        prepared_path,
        build_prepared_operator_input(
            op="softmax",
            dtype="float16",
            dataset="smoke",
            case_id="tiny_logits",
            seed=0,
        ),
    )
    monkeypatch.setattr("cannbench.cli.read_remote_endpoint", lambda path: endpoint)

    def fake_collect_remote_artifacts(**kwargs):
        captured.update(kwargs)
        return remote_collect_result(
            endpoint=endpoint,
            run_id=kwargs["run_id"],
            output_dir=kwargs["output_dir"],
            prepared=read_prepared_operator_input(kwargs["prepared_input"]),
            capture_output=True,
            profile_device_time=True,
            warmup=kwargs["warmup"],
            iterations=kwargs["iterations"],
        )

    monkeypatch.setattr(
        "cannbench.cli.collect_remote_artifacts", fake_collect_remote_artifacts
    )

    exit_code = main(
        [
            "bench",
            "--backend",
            "ascend",
            "--endpoint",
            str(endpoint_path),
            "--prepared-input",
            str(prepared_path),
            "--output-dir",
            str(output_dir),
            "--run-id",
            "softmax-run",
            "--capture-output",
            "--warmup",
            "3",
            "--iterations",
            "5",
        ]
    )

    canonical = "opbench-ascend-950pr-cannops-softmax-smoke-float16"
    assert exit_code == 0
    assert captured["endpoint"] == endpoint
    assert captured["endpoint_path"] == endpoint_path
    assert captured["endpoint_path"] == endpoint_path
    assert Path(captured["prepared_input"]).is_relative_to(build_run_layout(output_dir, canonical).prepared_dir)
    assert captured["output_dir"].parent == build_run_layout(output_dir, canonical).root
    assert captured["run_id"] == "softmax-run"
    assert captured["capture_output"] is True
    assert captured["profile_device_time"] is True
    assert captured["warmup"] == 3
    assert captured["iterations"] == 5
    assert captured["deploy_simt_op"] is False


def test_main_remote_bench_builds_prepared_input_when_case_is_provided(tmp_path, monkeypatch):
    captured: dict[str, object] = {}
    endpoint_path = tmp_path / "ascend.json"
    output_dir = tmp_path / "results"
    output_dir.mkdir()
    built = object()
    endpoint = RemoteEndpoint(
        name="ascend-a2",
        backend="ascend",
        host="user@ascend-host",
        workdir="/opt/cannbench",
        python="python3",
        env={},
    )
    monkeypatch.setattr("cannbench.cli.read_remote_endpoint", lambda path: endpoint)

    def fake_build_prepared_operator_input(**kwargs):
        captured["built_kwargs"] = kwargs
        return build_prepared_operator_input(**kwargs)

    def fake_write_prepared_operator_input(path, prepared):
        captured["written"] = (path, prepared)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(prepared.to_json_dict(), indent=2) + "\n")
        return path

    monkeypatch.setattr(
        "cannbench.cli.build_prepared_operator_input", fake_build_prepared_operator_input
    )
    monkeypatch.setattr(
        "cannbench.cli.write_prepared_operator_input", fake_write_prepared_operator_input
    )

    def fake_collect_remote_artifacts(**kwargs):
        captured.update(kwargs)
        return remote_collect_result(
            endpoint=endpoint,
            run_id=kwargs["run_id"],
            output_dir=kwargs["output_dir"],
            prepared=read_prepared_operator_input(kwargs["prepared_input"]),
            profile_device_time=True,
            warmup=kwargs["warmup"],
            iterations=kwargs["iterations"],
        )

    monkeypatch.setattr(
        "cannbench.cli.collect_remote_artifacts", fake_collect_remote_artifacts
    )

    exit_code = main(
        [
            "bench",
            "--backend",
            "ascend",
            "--endpoint",
            str(endpoint_path),
            "--output-dir",
            str(output_dir),
            "--run-id",
            "softmax-run",
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
        ]
    )

    assert exit_code == 0
    assert captured["built_kwargs"] == {
        "op": "softmax",
        "dtype": "float16",
        "dataset": "smoke",
        "case_id": "tiny_logits",
        "seed": 7,
    }
    written_path, written_prepared = captured["written"]
    canonical = "opbench-ascend-950pr-cannops-softmax-smoke-float16"
    assert written_path.parent == build_run_layout(output_dir, canonical).prepared_dir / "softmax" / "smoke"
    assert written_path.suffix == ".json"
    assert captured["prepared_input"] == written_path


def test_main_remote_bench_prepared_input_defaults_to_canonical_run_name(tmp_path, monkeypatch):
    captured: dict[str, object] = {}
    endpoint_path = tmp_path / "ascend.json"
    output_dir = tmp_path / "results"
    prepared_path = tmp_path / "prepared.json"
    endpoint = RemoteEndpoint(
        name="ascend-a2",
        backend="ascend",
        host="user@ascend-host",
        workdir="/opt/cannbench",
        python="python3",
        env={},
    )
    write_prepared_operator_input(
        prepared_path,
        build_prepared_operator_input(
            op="softmax",
            dtype="float16",
            dataset="realistic",
            case_id="t5_attention",
            seed=0,
        ),
    )
    monkeypatch.setattr("cannbench.cli.read_remote_endpoint", lambda path: endpoint)

    def fake_collect_remote_artifacts(**kwargs):
        captured.update(kwargs)
        return remote_collect_result(
            endpoint=endpoint,
            run_id=kwargs["run_id"],
            output_dir=kwargs["output_dir"],
            prepared=read_prepared_operator_input(kwargs["prepared_input"]),
            profile_device_time=True,
            warmup=kwargs["warmup"],
            iterations=kwargs["iterations"],
        )

    monkeypatch.setattr(
        "cannbench.cli.collect_remote_artifacts", fake_collect_remote_artifacts
    )

    exit_code = main(
        [
            "bench",
            "--backend",
            "ascend",
            "--endpoint",
            str(endpoint_path),
            "--prepared-input",
            str(prepared_path),
            "--output-dir",
            str(output_dir),
        ]
    )

    canonical = "opbench-ascend-950pr-cannops-softmax-realistic-float16"
    layout = build_run_layout(output_dir, canonical)

    assert exit_code == 0
    assert captured["run_id"] == canonical
    assert Path(captured["prepared_input"]).is_relative_to(layout.prepared_dir)


def test_main_rejects_auto_run_name_for_mixed_prepared_dir(tmp_path, capsys):
    prepared_dir = tmp_path / "prepared"
    prepared_dir.mkdir()
    write_prepared_operator_input(
        prepared_dir / "a.json",
        build_prepared_operator_input(
            op="softmax",
            dtype="float16",
            dataset="smoke",
            case_id="tiny_logits",
            seed=1,
        ),
    )
    write_prepared_operator_input(
        prepared_dir / "b.json",
        build_prepared_operator_input(
            op="softmax",
            dtype="float16",
            dataset="stress",
            case_id="wide_vocab_lm_logits",
            seed=2,
        ),
    )

    with pytest.raises(SystemExit) as excinfo:
        main(
            [
                "bench",
                "--backend",
                "nvidia",
                "--op",
                "softmax",
                "--prepared-dir",
                str(prepared_dir),
                "--output-dir",
                str(tmp_path),
            ]
        )

    captured = capsys.readouterr()
    assert excinfo.value.code == 2
    assert "automatic run-name requires a single op/dataset/dtype combination" in captured.err


def test_main_runs_batch_remote_bench_and_writes_aggregated_artifacts(tmp_path, monkeypatch):
    endpoint = RemoteEndpoint(
        name="ascend-a2",
        backend="ascend",
        host="user@ascend-host",
        workdir="/opt/cannbench",
        python="python3",
        env={},
    )
    prepared_dir = tmp_path / "prepared"
    prepared_dir.mkdir()
    output_dir = tmp_path / "results"
    alpha = build_prepared_operator_input(
        op="softmax",
        dtype="float16",
        dataset="smoke",
        case_id="tiny_logits",
        seed=1,
    )
    beta = build_prepared_operator_input(
        op="softmax",
        dtype="float16",
        dataset="stress",
        case_id="wide_vocab_lm_logits",
        seed=2,
    )
    write_prepared_operator_input(prepared_dir / "a.json", alpha)
    write_prepared_operator_input(prepared_dir / "b.json", beta)

    captured_calls: list[dict[str, object]] = []

    monkeypatch.setattr("cannbench.cli.read_remote_endpoint", lambda path: endpoint)

    def fake_collect_remote_artifacts(**kwargs):
        captured_calls.append(kwargs)
        prepared = read_prepared_operator_input(kwargs["prepared_input"])
        return remote_collect_result(
            endpoint=endpoint,
            run_id=kwargs["run_id"],
            output_dir=kwargs["output_dir"],
            prepared=prepared,
            capture_output=True,
            profile_device_time=True,
            warmup=kwargs["warmup"],
            iterations=kwargs["iterations"],
            extra_perf_artifacts=(("benchmark.csv", b"header\nvalue\n"),),
        )

    monkeypatch.setattr(
        "cannbench.cli.collect_remote_artifacts", fake_collect_remote_artifacts
    )

    exit_code = main(
        [
            "bench",
            "--backend",
            "ascend",
            "--endpoint",
            str(tmp_path / "ascend.json"),
            "--output-dir",
            str(output_dir),
            "--run-id",
            "remote-root",
            "--run-name",
            "softmax-remote-batch",
            "--op",
            "softmax",
            "--prepared-dir",
            str(prepared_dir),
            "--capture-output",
            "--implementation",
            "simt",
        ]
    )

    layout = build_run_layout(output_dir, "softmax-remote-batch")
    summary = json.loads((layout.meta_dir / "summary.json").read_text())
    benchmark_records = json.loads((layout.meta_dir / "benchmark-records.json").read_text())
    failures = json.loads((layout.meta_dir / "failures.json").read_text())

    assert exit_code == 0
    assert [call["run_id"] for call in captured_calls] == [
        "remote-root/softmax-smoke-tiny_logits-float16-seed1",
        "remote-root/softmax-stress-wide_vocab_lm_logits-float16-seed2",
    ]
    assert all(call["endpoint"] == endpoint for call in captured_calls)
    assert all(call["deploy_simt_op"] is True for call in captured_calls)
    assert all(Path(call["prepared_input"]).is_relative_to(layout.prepared_dir) for call in captured_calls)
    assert (layout.prepared_dir / "softmax" / "smoke" / "tiny_logits-float16-seed1.json").exists()
    assert (layout.prepared_dir / "softmax" / "stress" / "wide_vocab_lm_logits-float16-seed2.json").exists()
    assert (layout.perf_dir / "softmax-smoke-tiny_logits-float16-seed1.json").exists()
    assert (layout.perf_dir / "softmax-stress-wide_vocab_lm_logits-float16-seed2.csv").exists()
    assert (layout.output_dir / "softmax-smoke-tiny_logits-float16-seed1" / "tensor.json").exists()
    assert (layout.profile_dir / "softmax-smoke-tiny_logits-float16-seed1" / "op_summary.csv").exists()
    assert (layout.profile_dir / "softmax-smoke-tiny_logits-float16-seed1" / "profile-summary.json").exists()
    assert summary["metadata"] == {
        "backend": "ascend",
        "run_name": "softmax-remote-batch",
        "implementation": "simt",
        "total_cases": 2,
        "success_count": 2,
        "failure_count": 0,
    }
    assert [row["status"] for row in summary["records"]] == ["ok", "ok"]
    assert summary["records"][0]["prepared_input"] == "prepared/softmax/smoke/tiny_logits-float16-seed1.json"
    assert summary["records"][0]["result_path"] == "perf/softmax-smoke-tiny_logits-float16-seed1.json"
    assert len(benchmark_records["records"]) == 2
    assert benchmark_records["records"][0]["backend"] == "ascend"
    assert benchmark_records["records"][0]["implementation"] == "simt"
    assert benchmark_records["records"][0]["implementation_version"] == "v1"
    assert benchmark_records["records"][0]["metrics"]["latency_ms_avg"] == 1.0
    assert failures["failure_count"] == 0
    assert failures["records"] == []


def test_main_batch_remote_bench_records_failures_and_continues(tmp_path, monkeypatch, capsys):
    endpoint = RemoteEndpoint(
        name="ascend-a2",
        backend="ascend",
        host="user@ascend-host",
        workdir="/opt/cannbench",
        python="python3",
        env={},
    )
    prepared_dir = tmp_path / "prepared"
    prepared_dir.mkdir()
    output_dir = tmp_path / "results"
    alpha = build_prepared_operator_input(
        op="softmax",
        dtype="float16",
        dataset="smoke",
        case_id="tiny_logits",
        seed=1,
    )
    beta = build_prepared_operator_input(
        op="softmax",
        dtype="float16",
        dataset="stress",
        case_id="wide_vocab_lm_logits",
        seed=2,
    )
    write_prepared_operator_input(prepared_dir / "a.json", alpha)
    write_prepared_operator_input(prepared_dir / "b.json", beta)

    run_ids: list[str] = []

    monkeypatch.setattr("cannbench.cli.read_remote_endpoint", lambda path: endpoint)

    def fake_collect_remote_artifacts(**kwargs):
        run_ids.append(kwargs["run_id"])
        prepared_input = kwargs["prepared_input"]
        if prepared_input.name == "tiny_logits-float16-seed1.json":
            raise TimeoutError("ssh timeout")
        return remote_collect_result(
            endpoint=endpoint,
            run_id=kwargs["run_id"],
            output_dir=kwargs["output_dir"],
            prepared=read_prepared_operator_input(prepared_input),
            profile_device_time=True,
            warmup=kwargs["warmup"],
            iterations=kwargs["iterations"],
        )

    monkeypatch.setattr(
        "cannbench.cli.collect_remote_artifacts", fake_collect_remote_artifacts
    )

    with pytest.raises(SystemExit) as excinfo:
        main(
            [
                "bench",
                "--backend",
                "ascend",
                "--endpoint",
                str(tmp_path / "ascend.json"),
                "--output-dir",
                str(output_dir),
                "--run-name",
                "softmax-remote-batch",
                "--op",
                "softmax",
                "--prepared-dir",
                str(prepared_dir),
            ]
        )

    layout = build_run_layout(output_dir, "softmax-remote-batch")
    summary = json.loads((layout.meta_dir / "summary.json").read_text())
    failures = json.loads((layout.meta_dir / "failures.json").read_text())
    captured = capsys.readouterr()

    assert excinfo.value.code == 2
    assert run_ids == [
        "softmax-remote-batch/softmax-smoke-tiny_logits-float16-seed1",
        "softmax-remote-batch/softmax-stress-wide_vocab_lm_logits-float16-seed2",
    ]
    assert [row["status"] for row in summary["records"]] == ["failed", "ok"]
    assert failures["failure_count"] == 1
    assert failures["records"][0]["case_id"] == "tiny_logits"
    assert failures["records"][0]["error"] == "ssh timeout"
    assert "batch bench completed with 1 failures" in captured.err


def test_main_remote_bench_requires_endpoint(tmp_path, capsys):
    prepared_dir = tmp_path / "prepared"
    prepared_dir.mkdir()

    with pytest.raises(FileNotFoundError):
        main(
            [
                "bench",
                "--backend",
                "ascend",
                "--op",
                "softmax",
                "--dataset",
                "smoke",
                "--case-id",
                "tiny_logits",
                "--endpoint",
                str(tmp_path / "missing.json"),
            ]
        )


def test_main_runs_batch_remote_bench_from_selection_expansion(tmp_path, monkeypatch):
    endpoint = RemoteEndpoint(
        name="ascend-a2",
        backend="ascend",
        host="user@ascend-host",
        workdir="/opt/cannbench",
        python="python3",
        env={},
    )
    output_dir = tmp_path / "results"
    collected_prepared: list[Path] = []

    monkeypatch.setattr("cannbench.cli.read_remote_endpoint", lambda path: endpoint)

    def fake_collect_remote_artifacts(**kwargs):
        collected_prepared.append(kwargs["prepared_input"])
        return remote_collect_result(
            endpoint=endpoint,
            run_id=kwargs["run_id"],
            output_dir=kwargs["output_dir"],
            prepared=read_prepared_operator_input(kwargs["prepared_input"]),
            profile_device_time=True,
            warmup=kwargs["warmup"],
            iterations=kwargs["iterations"],
        )

    monkeypatch.setattr(
        "cannbench.cli.collect_remote_artifacts", fake_collect_remote_artifacts
    )

    exit_code = main(
        [
            "bench",
            "--backend",
            "ascend",
            "--endpoint",
            str(tmp_path / "ascend.json"),
            "--output-dir",
            str(output_dir),
            "--run-name",
            "softmax-smoke-batch",
            "--op",
            "softmax",
            "--dataset",
            "smoke",
        ]
    )

    layout = build_run_layout(output_dir, "softmax-smoke-batch")

    assert exit_code == 0
    assert collected_prepared
    assert all(path.is_relative_to(layout.prepared_dir) for path in collected_prepared)
    assert (layout.prepared_dir / "softmax" / "smoke" / "tiny_logits-float16-seed0.json").exists()


def test_main_rejects_batch_remote_bench_when_run_directory_exists(tmp_path, monkeypatch, capsys):
    endpoint = RemoteEndpoint(
        name="ascend-a2",
        backend="ascend",
        host="user@ascend-host",
        workdir="/opt/cannbench",
        python="python3",
        env={},
    )
    prepared_dir = tmp_path / "prepared"
    prepared_dir.mkdir()
    prepared = build_prepared_operator_input(
        op="softmax",
        dtype="float16",
        dataset="smoke",
        case_id="tiny_logits",
        seed=1,
    )
    write_prepared_operator_input(prepared_dir / "a.json", prepared)
    layout = build_run_layout(tmp_path / "results", "softmax-remote-batch")
    layout.root.mkdir(parents=True)
    (layout.root / "stale.txt").write_text("stale")

    monkeypatch.setattr("cannbench.cli.read_remote_endpoint", lambda path: endpoint)

    with pytest.raises(SystemExit) as excinfo:
        main(
            [
                "bench",
                "--backend",
                "ascend",
                "--endpoint",
                str(tmp_path / "ascend.json"),
                "--output-dir",
                str(tmp_path / "results"),
                "--run-name",
                "softmax-remote-batch",
                "--op",
                "softmax",
                "--prepared-dir",
                str(prepared_dir),
            ]
        )

    captured = capsys.readouterr()
    assert excinfo.value.code == 2
    assert "batch run directory already exists and is not empty" in captured.err


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


def test_main_publish_copies_run_artifacts(tmp_path, monkeypatch):
    captured: dict[str, object] = {}

    def fake_publish_run_artifacts(source_dir, dest_dir):
        captured["source_dir"] = source_dir
        captured["dest_dir"] = dest_dir
        return object()

    monkeypatch.setattr("cannbench.cli.publish_run_artifacts", fake_publish_run_artifacts)

    exit_code = main(
        [
            "publish",
            "--source",
            str(tmp_path / "runs" / "softmax-run"),
            "--dest",
            str(tmp_path / "published" / "softmax-run"),
        ]
    )

    assert exit_code == 0
    assert captured["source_dir"] == tmp_path / "runs" / "softmax-run"
    assert captured["dest_dir"] == tmp_path / "published" / "softmax-run"


def test_main_serve_invokes_static_service(tmp_path, monkeypatch):
    captured: dict[str, object] = {}

    def fake_serve_cannbench(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr("cannbench.cli.serve_cannbench", fake_serve_cannbench)

    exit_code = main(
        [
            "serve",
            "--frontend-dir",
            str(tmp_path / "web" / "dist"),
            "--published-dir",
            str(tmp_path / "published"),
            "--host",
            "127.0.0.1",
            "--port",
            "9000",
            "--enable-gpu-upload",
        ]
    )

    assert exit_code == 0
    assert captured == {
        "frontend_dir": tmp_path / "web" / "dist",
        "published_dir": tmp_path / "published",
        "host": "127.0.0.1",
        "port": 9000,
        "enable_gpu_upload": True,
    }


def test_main_runs_internal_run_from_prepared_input(tmp_path, monkeypatch):
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
            "internal-run",
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
            "--deploy-simt-op",
        ]
    )

    request = captured["request"]
    assert exit_code == 0
    assert request.dataset == "smoke"
    assert request.case_id == "tiny_logits"
    assert request.seed == 7
    assert request.dimensions == (32, 128)
    assert request.deploy_simt_op is True
    assert captured["run_name"] == "prepared-run"


def test_main_rejects_zero_iterations():
    with pytest.raises(SystemExit):
        main(
            [
                "internal-run",
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
                "internal-run",
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


def test_main_rejects_prepared_input_and_prepared_dir_together(tmp_path, capsys):
    prepared_path = tmp_path / "prepared.json"
    prepared_path.write_text("{}")
    prepared_dir = tmp_path / "prepared"
    prepared_dir.mkdir()

    with pytest.raises(SystemExit) as excinfo:
        main(
            [
                "bench",
                "--backend",
                "nvidia",
                "--op",
                "softmax",
                "--prepared-input",
                str(prepared_path),
                "--prepared-dir",
                str(prepared_dir),
                "--run-name",
                "softmax-batch",
            ]
        )

    captured = capsys.readouterr()
    assert excinfo.value.code == 2
    assert "--prepared-input and --prepared-dir are mutually exclusive" in captured.err


def test_main_rejects_direct_selection_without_op(capsys):
    with pytest.raises(SystemExit) as excinfo:
        main(
            [
                "bench",
                "--backend",
                "nvidia",
                "--case-id",
                "tiny_logits",
            ]
        )

    captured = capsys.readouterr()
    assert excinfo.value.code == 2
    assert "--op is required unless --prepared-input is set" in captured.err


def test_main_rejects_batch_bench_without_op(tmp_path, capsys):
    prepared_dir = tmp_path / "prepared"
    prepared_dir.mkdir()
    prepared_path = prepared_dir / "tiny.json"
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

    with pytest.raises(SystemExit) as excinfo:
        main(
            [
                "bench",
                "--backend",
                "nvidia",
                "--prepared-dir",
                str(prepared_dir),
                "--run-name",
                "softmax-batch",
            ]
        )

    captured = capsys.readouterr()
    assert excinfo.value.code == 2
    assert "--op is required with --prepared-dir" in captured.err


def test_main_rejects_internal_run_prepared_dir(tmp_path, capsys):
    prepared_dir = tmp_path / "prepared"
    prepared_dir.mkdir()

    with pytest.raises(SystemExit) as excinfo:
        main(
            [
                "internal-run",
                "--backend",
                "nvidia",
                "--op",
                "softmax",
                "--prepared-dir",
                str(prepared_dir),
                "--run-name",
                "softmax-batch",
            ]
        )

    captured = capsys.readouterr()
    assert excinfo.value.code == 2
    assert "--prepared-dir is only supported for bench" in captured.err


def test_main_rejects_dataset_with_prepared_dir(tmp_path, capsys):
    prepared_dir = tmp_path / "prepared"
    prepared_dir.mkdir()

    with pytest.raises(SystemExit) as excinfo:
        main(
            [
                "bench",
                "--backend",
                "nvidia",
                "--op",
                "softmax",
                "--prepared-dir",
                str(prepared_dir),
                "--dataset",
                "smoke",
                "--run-name",
                "softmax-batch",
            ]
        )

    captured = capsys.readouterr()
    assert excinfo.value.code == 2
    assert "--dataset cannot be used with --prepared-input or --prepared-dir" in captured.err


def test_main_rejects_prepared_input_internal_run_mismatch(tmp_path, capsys):
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

    with pytest.raises(SystemExit) as excinfo:
        main(
            [
                "internal-run",
                "--backend",
                "nvidia",
                "--op",
                "embedding",
                "--prepared-input",
                str(prepared_path),
                "--run-name",
                "prepared-run",
            ]
        )

    captured = capsys.readouterr()
    assert excinfo.value.code == 2
    assert "prepared input operator mismatch: expected embedding, got softmax" in captured.err


def test_main_runs_batch_bench_from_selection_and_writes_summary(tmp_path, monkeypatch, capsys):
    captured_requests: list[object] = []
    smoke_cases = get_operator_dataset("softmax").get("smoke").cases

    class FakeBackend:
        def run_operator(self, request):
            captured_requests.append(request)
            return result_for_request(request)

        def profile_operator_device_time(self, request):
            captured_requests.append(request)
            return LocalDeviceProfileResult(
                benchmark_result=result_for_request(request),
                profile=ProfileArtifacts(
                    device_name="Fake GPU",
                    profile_summary=DeviceProfileSummary(
                        backend="nvidia",
                        sample_count=1,
                        latency_ms_avg=0.1,
                        latency_ms_p50=0.1,
                        latency_ms_p95=0.1,
                        latency_ms_p99=0.1,
                        source_files=("ncu.csv",),
                    ),
                    profile_artifacts=(
                        (
                            "ncu.csv",
                            b"Kernel Name,Metric Name,Metric Unit,Metric Value\n"
                            b"softmax,gpu__time_duration.sum,nsecond,100000\n",
                        ),
                    ),
                    perf_artifacts=(
                        (
                            "benchmark.json",
                            (
                                json.dumps(result_for_request(request).to_json_dict()) + "\n"
                            ).encode("utf-8"),
                        ),
                    ),
                ),
            )

    monkeypatch.setattr("cannbench.cli.get_backend", lambda name: FakeBackend())

    exit_code = main(
        [
            "bench",
            "--backend",
            "nvidia",
            "--op",
            "softmax",
            "--dataset",
            "smoke",
            "--output-dir",
            str(tmp_path),
            "--run-name",
            "softmax-smoke-batch",
        ]
    )

    layout = build_run_layout(tmp_path, "softmax-smoke-batch")
    summary = json.loads((layout.meta_dir / "summary.json").read_text())
    benchmark_records = json.loads((layout.meta_dir / "benchmark-records.json").read_text())
    failures = json.loads((layout.meta_dir / "failures.json").read_text())
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "[run] bench started run_name=softmax-smoke-batch backend=nvidia mode=local-batch cases=3" in captured.out
    assert "[case] start 1/3 case_id=tiny_logits dataset=smoke dtype=float16 backend=nvidia" in captured.out
    assert "[case] profiling completed case_id=tiny_logits backend=nvidia profiler=ncu" in captured.out
    assert "[run] bench completed run_name=softmax-smoke-batch backend=nvidia successes=3 failures=0" in captured.out
    assert len(captured_requests) == len(smoke_cases) * 2
    assert [request.case_id for request in captured_requests[::2]] == [case.case_id for case in smoke_cases]
    assert [request.case_id for request in captured_requests[1::2]] == [case.case_id for case in smoke_cases]
    assert (layout.prepared_dir / "softmax" / "smoke" / "tiny_logits-float16-seed0.json").exists()
    assert (layout.perf_dir / "softmax-smoke-tiny_logits-float16-seed0.json").exists()
    assert summary["metadata"]["run_name"] == "softmax-smoke-batch"
    assert summary["metadata"]["backend"] == "nvidia"
    assert summary["result_count"] == len(smoke_cases)
    assert all(row["status"] == "ok" for row in summary["records"])
    assert summary["records"][0]["result_path"].startswith("perf/softmax-smoke-")
    assert (layout.profile_dir / "softmax-smoke-tiny_logits-float16-seed0" / "ncu.csv").exists()
    assert len(benchmark_records["records"]) == len(smoke_cases)
    assert benchmark_records["records"][0]["backend"] == "nvidia"
    assert benchmark_records["records"][0]["implementation"] == "cuda-pytorch"
    assert benchmark_records["records"][0]["implementation_version"] == "cuda-pytorch"
    assert benchmark_records["records"][0]["metrics"]["latency_ms_avg"] == 0.1
    assert failures["failure_count"] == 0
    assert failures["records"] == []


def test_main_runs_batch_bench_once_per_prepared_case(tmp_path, monkeypatch):
    prepared_dir = tmp_path / "prepared"
    prepared_dir.mkdir()
    alpha = build_prepared_operator_input(
        op="softmax",
        dtype="float16",
        dataset="smoke",
        case_id="tiny_logits",
        seed=1,
    )
    beta = build_prepared_operator_input(
        op="softmax",
        dtype="float16",
        dataset="stress",
        case_id="wide_vocab_lm_logits",
        seed=2,
    )
    write_prepared_operator_input(prepared_dir / "b.json", beta)
    write_prepared_operator_input(prepared_dir / "a.json", alpha)

    seen_case_ids: list[str] = []

    class FakeBackend:
        def run_operator(self, request):
            seen_case_ids.append(request.case_id)
            return result_for_request(request)

        def profile_operator_device_time(self, request):
            raise NotImplementedError

    monkeypatch.setattr("cannbench.cli.get_backend", lambda name: FakeBackend())

    exit_code = main(
        [
            "bench",
            "--backend",
            "nvidia",
            "--op",
            "softmax",
            "--prepared-dir",
            str(prepared_dir),
            "--output-dir",
            str(tmp_path),
            "--run-name",
            "prepared-batch",
        ]
    )

    layout = build_run_layout(tmp_path, "prepared-batch")
    summary = json.loads((layout.meta_dir / "summary.json").read_text())

    assert exit_code == 0
    assert seen_case_ids == ["tiny_logits", "wide_vocab_lm_logits"]
    assert [row["case_id"] for row in summary["records"]] == ["tiny_logits", "wide_vocab_lm_logits"]


def test_main_batch_bench_records_failures_and_continues(tmp_path, monkeypatch, capsys):
    prepared_dir = tmp_path / "prepared"
    prepared_dir.mkdir()
    alpha = build_prepared_operator_input(
        op="softmax",
        dtype="float16",
        dataset="smoke",
        case_id="tiny_logits",
        seed=1,
    )
    beta = build_prepared_operator_input(
        op="softmax",
        dtype="float16",
        dataset="stress",
        case_id="wide_vocab_lm_logits",
        seed=2,
    )
    write_prepared_operator_input(prepared_dir / "a.json", alpha)
    write_prepared_operator_input(prepared_dir / "b.json", beta)

    seen_case_ids: list[str] = []

    class FakeBackend:
        def run_operator(self, request):
            seen_case_ids.append(request.case_id)
            if request.case_id == "tiny_logits":
                raise OSError("kernel launch failed")
            return result_for_request(request)

        def profile_operator_device_time(self, request):
            raise NotImplementedError

    monkeypatch.setattr("cannbench.cli.get_backend", lambda name: FakeBackend())

    with pytest.raises(SystemExit) as excinfo:
        main(
            [
                "bench",
                "--backend",
                "nvidia",
                "--op",
                "softmax",
                "--prepared-dir",
                str(prepared_dir),
                "--output-dir",
                str(tmp_path),
                "--run-name",
                "prepared-batch",
            ]
        )

    layout = build_run_layout(tmp_path, "prepared-batch")
    summary = json.loads((layout.meta_dir / "summary.json").read_text())
    failures = json.loads((layout.meta_dir / "failures.json").read_text())
    captured = capsys.readouterr()

    assert excinfo.value.code == 2
    assert "[run] bench started run_name=prepared-batch backend=nvidia mode=local-batch cases=2" in captured.out
    assert "[case] failed case_id=tiny_logits backend=nvidia error=kernel launch failed" in captured.out
    assert "[case] success case_id=wide_vocab_lm_logits backend=nvidia" in captured.out
    assert "[run] bench completed run_name=prepared-batch backend=nvidia successes=1 failures=1" in captured.out
    assert seen_case_ids == ["tiny_logits", "wide_vocab_lm_logits"]
    assert [row["status"] for row in summary["records"]] == ["failed", "ok"]
    assert failures["failure_count"] == 1
    assert failures["records"][0]["case_id"] == "tiny_logits"
    assert "completed with 1 failures" in captured.err


def test_main_batch_bench_rejects_duplicate_artifact_stems(tmp_path, capsys):
    prepared_dir = tmp_path / "prepared"
    prepared_dir.mkdir()
    prepared = build_prepared_operator_input(
        op="softmax",
        dtype="float16",
        dataset="smoke",
        case_id="tiny_logits",
        seed=7,
    )
    write_prepared_operator_input(prepared_dir / "a.json", prepared)
    write_prepared_operator_input(prepared_dir / "b.json", prepared)

    with pytest.raises(SystemExit) as excinfo:
        main(
            [
                "bench",
                "--backend",
                "nvidia",
                "--op",
                "softmax",
                "--prepared-dir",
                str(prepared_dir),
                "--output-dir",
                str(tmp_path),
                "--run-name",
                "prepared-batch",
            ]
        )

    captured = capsys.readouterr()
    assert excinfo.value.code == 2
    assert "duplicate batch artifact stem" in captured.err


def test_main_converts_backend_runtime_failure_to_cli_error(monkeypatch, capsys):
    class FailingBackend:
        def run_operator(self, request):
            raise RuntimeError("CUDA is required for the nvidia backend")

    monkeypatch.setattr("cannbench.cli.get_backend", lambda name: FailingBackend())

    with pytest.raises(SystemExit) as excinfo:
        main(
            [
                "internal-run",
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


def test_package_data_includes_ascend_simt_op_versions():
    with open("pyproject.toml", "rb") as config:
        payload = tomllib.load(config)

    package_data = payload["tool"]["setuptools"]["package-data"]

    assert "cannbench.datasets.data.*.simt.*" in package_data
