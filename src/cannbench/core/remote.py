from __future__ import annotations

import json
import shlex
import subprocess
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from cannbench.core.execution import RemoteExecutionArtifacts, RemoteProfileArtifacts, read_artifact_tree
from cannbench.core.prepared_input import read_prepared_operator_input
from cannbench.core.profile import (
    expected_kernel_name_patterns,
    read_device_profile,
    write_device_profile_summary,
)


CommandRunner = Callable[[list[str]], None]


@dataclass(frozen=True)
class RemoteEndpoint:
    name: str
    backend: str
    host: str
    workdir: str
    port: int | None = None
    python: str = "python3"
    setup: str | None = None
    env: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class RemoteCollectionResult:
    endpoint: RemoteEndpoint
    run_id: str
    remote_run_dir: str
    local_output_dir: Path
    artifacts: RemoteExecutionArtifacts


def read_remote_endpoint(path: Path) -> RemoteEndpoint:
    payload = json.loads(path.read_text())
    required = ("name", "backend", "host", "workdir")
    missing = [key for key in required if not str(payload.get(key, "")).strip()]
    if missing:
        raise ValueError(f"remote endpoint missing required fields: {', '.join(missing)}")
    env = {str(key): str(value) for key, value in payload.get("env", {}).items()}
    host = str(payload["host"])
    port = payload.get("port")
    if port is None and host.rsplit(":", 1)[-1].isdigit():
        host, port_text = host.rsplit(":", 1)
        port = int(port_text)
    elif port is not None:
        port = int(port)
    return RemoteEndpoint(
        name=str(payload["name"]),
        backend=str(payload["backend"]),
        host=host,
        port=port,
        workdir=str(payload["workdir"]).rstrip("/"),
        python=str(payload.get("python", "python3")),
        setup=str(payload["setup"]) if payload.get("setup") else None,
        env=env,
    )


def _default_runner(command: list[str]) -> None:
    subprocess.run(command, check=True)


def _remote_command_env(env: dict[str, str]) -> str:
    if not env:
        return ""
    return " ".join(
        f"{shlex.quote(key)}={shlex.quote(value)}"
        for key, value in sorted(env.items())
    ) + " "


def _remote_command_prefix(endpoint: RemoteEndpoint) -> str:
    prefix = f"cd {shlex.quote(endpoint.workdir)} && "
    if endpoint.setup:
        prefix = f"{prefix}{endpoint.setup} && "
    return prefix


def _ssh_command(endpoint: RemoteEndpoint, remote_command: str) -> list[str]:
    command = ["ssh"]
    if endpoint.port is not None:
        command.extend(["-p", str(endpoint.port)])
    command.extend([endpoint.host, remote_command])
    return command


def _scp_upload_command(
    endpoint: RemoteEndpoint, local_path: Path, remote_path: str
) -> list[str]:
    command = ["scp"]
    if endpoint.port is not None:
        command.extend(["-P", str(endpoint.port)])
    command.extend([str(local_path), f"{endpoint.host}:{remote_path}"])
    return command


def _scp_download_command(
    endpoint: RemoteEndpoint, remote_path: str, local_path: Path
) -> list[str]:
    command = ["scp"]
    if endpoint.port is not None:
        command.extend(["-P", str(endpoint.port)])
    command.extend(["-r", f"{endpoint.host}:{remote_path}", str(local_path)])
    return command


def collect_remote_artifacts(
    *,
    prepared_input: Path,
    output_dir: Path,
    capture_output: bool,
    profile_device_time: bool = False,
    summarize_profile: bool = False,
    warmup: int = 10,
    iterations: int = 1,
    implementation: str | None = None,
    implementation_version: str | None = None,
    run_id: str | None = None,
    endpoint: RemoteEndpoint | None = None,
    endpoint_path: Path | None = None,
    runner: CommandRunner = _default_runner,
) -> RemoteCollectionResult:
    if endpoint is None:
        if endpoint_path is None:
            raise ValueError("endpoint or endpoint_path is required")
        endpoint = read_remote_endpoint(endpoint_path)
    if not capture_output and not profile_device_time:
        raise ValueError("remote bench requires --capture-output or device profiling")

    actual_run_id = run_id or uuid.uuid4().hex
    remote_run_dir = f"{endpoint.workdir}/.cannbench-runs/{actual_run_id}"
    remote_prepared = f"{remote_run_dir}/prepared.json"
    remote_output = f"{remote_run_dir}/output"
    remote_profile = f"{remote_run_dir}/profile"
    remote_perf = f"{remote_run_dir}/perf"
    relative_prepared = f".cannbench-runs/{actual_run_id}/prepared.json"
    relative_output = f".cannbench-runs/{actual_run_id}/output"
    relative_perf = f".cannbench-runs/{actual_run_id}/perf"

    output_dir.mkdir(parents=True, exist_ok=True)
    mkdir_targets = [remote_run_dir]
    if profile_device_time:
        mkdir_targets.append(remote_profile)
    runner(
        _ssh_command(
            endpoint,
            "mkdir -p " + " ".join(shlex.quote(target) for target in mkdir_targets),
        )
    )
    runner(_scp_upload_command(endpoint, prepared_input, remote_prepared))

    output_artifacts: tuple[tuple[str, bytes], ...] = ()
    profile_artifacts_result: RemoteProfileArtifacts | None = None
    implementation_version_arg = (
        f" --implementation-version {shlex.quote(implementation_version)}"
        if implementation_version
        else ""
    )
    implementation_arg = (
        f" --implementation {shlex.quote(implementation)}"
        if implementation
        else ""
    )

    if capture_output:
        command = (
            f"{_remote_command_prefix(endpoint)}"
            f"{_remote_command_env(endpoint.env)}"
            f"{shlex.quote(endpoint.python)} -m cannbench internal-run "
            f"--backend {shlex.quote(endpoint.backend)} "
            f"--prepared-input {shlex.quote(relative_prepared)} "
            f"--output-dir {shlex.quote(relative_output)} "
            f"--run-name captured-output{implementation_arg}{implementation_version_arg}"
        )
        runner(_ssh_command(endpoint, command))
        runner(
            _scp_download_command(endpoint, remote_output, output_dir / "output")
        )
        output_artifacts = read_artifact_tree(output_dir / "output")

    if profile_device_time:
        base_operator = (
            f"{shlex.quote(endpoint.python)} -m cannbench internal-run "
            f"--backend {shlex.quote(endpoint.backend)} "
            f"--prepared-input {shlex.quote(relative_prepared)} "
            f"--warmup {warmup} "
            f"--iterations {iterations} "
            f"--output-dir {shlex.quote(relative_perf)} "
            f"--run-name benchmark{implementation_arg}{implementation_version_arg}"
        )
        if endpoint.backend == "ascend":
            profiled_operator = (
                f"msprof op "
                f"--output={shlex.quote(remote_profile)} "
                f"--warm-up {warmup} "
                f"--launch-count {iterations} "
                f"{base_operator}"
            )
            command = (
                f"{_remote_command_prefix(endpoint)}"
                f"{_remote_command_env(endpoint.env)}"
                f"{profiled_operator}"
            )
        elif endpoint.backend == "nvidia":
            env_prefix = _remote_command_env(endpoint.env)
            ncu_operator = (
                f"{env_prefix}"
                "ncu --target-processes all --force-overwrite "
                f"--launch-skip {warmup} "
                f"--launch-count {iterations} "
                "--csv "
                f"--log-file {shlex.quote(remote_profile + '/ncu.csv')} "
                f"--export {shlex.quote(remote_profile + '/ncu-report')} "
                f"{base_operator}"
            )
            profiled_operator = ncu_operator
            command = f"{_remote_command_prefix(endpoint)}{profiled_operator}"
        else:
            raise ValueError(f"unsupported profiler backend: {endpoint.backend}")
        runner(_ssh_command(endpoint, command))
        runner(
            _scp_download_command(endpoint, remote_profile, output_dir / "profile")
        )
        runner(_scp_download_command(endpoint, remote_perf, output_dir / "perf"))
        prepared = read_prepared_operator_input(prepared_input)
        summary = read_device_profile(
            output_dir / "profile",
            backend=endpoint.backend,
            expected_kernel_name_patterns=expected_kernel_name_patterns(
                backend=endpoint.backend,
                op=prepared.op,
                implementation=implementation,
            ),
        )
        if summarize_profile:
            write_device_profile_summary(output_dir / "profile-summary.json", summary)
        perf_payload = json.loads((output_dir / "perf" / "benchmark.json").read_text())
        profile_artifacts_result = RemoteProfileArtifacts(
            backend=endpoint.backend,
            device_name=str(perf_payload.get("device_name", "unknown")),
            profile_summary=summary,
            profile_artifacts=read_artifact_tree(output_dir / "profile"),
            perf_artifacts=read_artifact_tree(output_dir / "perf"),
        )

    return RemoteCollectionResult(
        endpoint=endpoint,
        run_id=actual_run_id,
        remote_run_dir=remote_run_dir,
        local_output_dir=output_dir,
        artifacts=RemoteExecutionArtifacts(
            output_artifacts=output_artifacts,
            profile=profile_artifacts_result,
        ),
    )
