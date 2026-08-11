from pathlib import Path

import pytest

from cannbench.core.execution import (
    BenchCaseExecutor,
    BenchCaseExecutionResult,
    BenchExecutionArtifacts,
    BenchProfileArtifacts,
    LocalBenchExecutor,
    read_artifact_tree,
)
from cannbench.core.profile import DeviceProfileSummary


def test_read_artifact_tree_returns_sorted_relative_files(tmp_path: Path):
    root = tmp_path / "artifacts"
    (root / "b").mkdir(parents=True)
    (root / "a").mkdir(parents=True)
    (root / "b" / "two.txt").write_text("two", encoding="utf-8")
    (root / "a" / "one.txt").write_text("one", encoding="utf-8")

    artifacts = read_artifact_tree(root)

    assert artifacts == (
        ("a/one.txt", b"one"),
        ("b/two.txt", b"two"),
    )


def test_bench_execution_artifacts_can_represent_profiled_case():
    profile = BenchProfileArtifacts(
        backend="nvidia",
        device_name="Fake GPU",
        profile_summary=DeviceProfileSummary(
            backend="nvidia",
            latency_ms=0.1,
            source_files=("ncu.csv",),
        ),
        profile_artifacts=(("ncu.csv", b"csv"),),
        perf_artifacts=(("benchmark.json", b"{}"),),
    )

    artifacts = BenchExecutionArtifacts(
        output_artifacts=(("tensor.json", b"{}"),),
        profile=profile,
    )

    assert artifacts.profile is profile
    assert artifacts.output_artifacts[0][0] == "tensor.json"


def test_bench_case_execution_result_keeps_result_path():
    result = BenchCaseExecutionResult(
        artifacts=BenchExecutionArtifacts(),
        result_path=Path("perf/softmax.json"),
    )

    assert result.result_path == Path("perf/softmax.json")


def test_bench_case_execution_result_can_store_output_artifacts_without_profile():
    result = BenchCaseExecutionResult(
        artifacts=BenchExecutionArtifacts(output_artifacts=(("tensor.json", b"{}"),)),
        result_path=Path("output/softmax.json"),
    )

    assert result.artifacts.profile is None
    assert result.artifacts.output_artifacts == (("tensor.json", b"{}"),)


def test_bench_case_executor_requires_execute_case():
    executor = BenchCaseExecutor()

    with pytest.raises(NotImplementedError):
        executor.execute_case()


def test_bench_execution_artifacts_can_represent_sparse_attention_tuple_outputs():
    artifacts = BenchExecutionArtifacts(
        output_artifacts=(
            ("output.json", b"{}"),
            ("lse.json", b"{}"),
        )
    )

    assert len(artifacts.output_artifacts) == 2


def test_local_executor_runs_and_profiles_workflow_once(tmp_path: Path):
    calls: list[tuple[str, object]] = []
    workflow_result = object()
    profile = BenchProfileArtifacts(
        backend="nvidia",
        device_name="Fake GPU",
        profile_summary=DeviceProfileSummary(
            backend="nvidia",
            latency_ms=0.01,
            source_files=("ncu.csv",),
        ),
        profile_artifacts=(("ncu.csv", b"csv"),),
        perf_artifacts=(("benchmark.json", b"{}"),),
        component_summaries=(
            DeviceProfileSummary("nvidia", ("ncu.csv",), 0.003),
            DeviceProfileSummary("nvidia", ("ncu.csv",), 0.007),
        ),
    )

    class FakeBackend:
        def run_workflow(self, request):
            calls.append(("run", request))
            return workflow_result

        def profile_workflow_device_time(self, request):
            calls.append(("profile", request))
            return profile

        def run_operator(self, request):
            raise AssertionError("workflow execution must not run a component case")

    def write_workflow_outputs(output_dir, run_name, result):
        assert result is workflow_result
        path = output_dir / f"{run_name}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n")
        return {"json": path}

    executor = LocalBenchExecutor(
        FakeBackend(),
        lambda *args: {},
        write_workflow_outputs=write_workflow_outputs,
    )
    request = object()

    result = executor.execute_workflow(
        request,
        output_dir=tmp_path,
        run_name="workflow",
    )

    assert calls == [("run", request), ("profile", request)]
    assert result.artifacts.profile is profile
    assert result.result_path == tmp_path / "workflow.json"
