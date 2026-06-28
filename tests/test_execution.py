from pathlib import Path

import pytest

from cannbench.core.execution import (
    BenchCaseExecutor,
    BenchCaseExecutionResult,
    BenchExecutionArtifacts,
    BenchProfileArtifacts,
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
            sample_count=1,
            latency_ms_avg=0.1,
            latency_ms_p50=0.1,
            latency_ms_p95=0.1,
            latency_ms_p99=0.1,
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
