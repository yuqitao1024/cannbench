from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

from cannbench.core.profile import DeviceProfileSummary


@dataclass(frozen=True)
class BenchProfileArtifacts:
    backend: str
    device_name: str
    profile_summary: DeviceProfileSummary
    profile_artifacts: tuple[tuple[str, bytes], ...]
    perf_artifacts: tuple[tuple[str, bytes], ...]


@dataclass(frozen=True)
class BenchExecutionArtifacts:
    output_artifacts: tuple[tuple[str, bytes], ...] = ()
    profile: BenchProfileArtifacts | None = None


@dataclass(frozen=True)
class BenchCaseExecutionResult:
    artifacts: BenchExecutionArtifacts
    result_path: Path | None = None


class BenchCaseExecutor:
    def execute_case(self, *args, **kwargs) -> BenchCaseExecutionResult:
        raise NotImplementedError


class LocalBenchExecutor(BenchCaseExecutor):
    def __init__(self, backend, write_outputs) -> None:
        self._backend = backend
        self._write_outputs = write_outputs

    def execute_case(
        self,
        request,
        *,
        output_dir: Path,
        run_name: str,
    ) -> BenchCaseExecutionResult:
        result = self._backend.run_operator(request)
        outputs = self._write_outputs(output_dir, run_name, result, request.output_formats)
        result_path = outputs.get("json")
        if result_path is None and outputs:
            result_path = next(iter(outputs.values()))
        try:
            profile_result = self._backend.profile_operator_device_time(request)
        except (NotImplementedError, AttributeError):
            profile = None
        else:
            profile = profile_result.profile
        return BenchCaseExecutionResult(
            artifacts=BenchExecutionArtifacts(profile=profile),
            result_path=result_path,
        )


class RemoteBenchExecutor(BenchCaseExecutor):
    def __init__(self, collect_remote_artifacts, endpoint, endpoint_path: Path | None = None) -> None:
        self._collect_remote_artifacts = collect_remote_artifacts
        self._endpoint = endpoint
        self._endpoint_path = endpoint_path

    def execute_case(
        self,
        *,
        prepared_input: Path,
        layout_root: Path,
        artifact_stem: str,
        run_id: str,
        capture_output: bool,
        warmup: int,
        iterations: int,
        deploy_simt_op: bool,
        use_simt_op: bool = False,
        implementation: str | None = None,
        implementation_version: str | None = None,
    ) -> BenchCaseExecutionResult:
        with TemporaryDirectory(prefix=f"{artifact_stem}-", dir=layout_root) as temp_dir_name:
            temp_dir = Path(temp_dir_name)
            remote_result = self._collect_remote_artifacts(
                endpoint=self._endpoint,
                endpoint_path=self._endpoint_path,
                prepared_input=prepared_input,
                output_dir=temp_dir,
                run_id=run_id,
                capture_output=capture_output,
                profile_device_time=True,
                warmup=warmup,
                iterations=iterations,
                deploy_simt_op=deploy_simt_op,
                use_simt_op=use_simt_op,
                implementation=implementation,
                implementation_version=implementation_version,
            )
            return BenchCaseExecutionResult(
                artifacts=remote_result.artifacts,
                result_path=None,
            )


RemoteProfileArtifacts = BenchProfileArtifacts
RemoteExecutionArtifacts = BenchExecutionArtifacts


def read_artifact_tree(root: Path) -> tuple[tuple[str, bytes], ...]:
    if not root.is_dir():
        return ()
    return tuple(
        (
            str(path.relative_to(root)),
            path.read_bytes(),
        )
        for path in sorted(root.rglob("*"))
        if path.is_file()
    )
