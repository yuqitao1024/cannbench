from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

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
