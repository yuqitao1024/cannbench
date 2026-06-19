from __future__ import annotations

import json
import shlex
import subprocess
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from cannbench.core.profile import read_device_profile, write_device_profile_summary


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
    deploy_custom_op: bool = False,
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
        raise ValueError("collect requires --capture-output or --profile-device-time")

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

    if capture_output:
        command = (
            f"{_remote_command_prefix(endpoint)}"
            f"{_remote_command_env(endpoint.env)}"
            f"{shlex.quote(endpoint.python)} -m cannbench capture-output "
            f"--backend {shlex.quote(endpoint.backend)} "
            f"--prepared-input {shlex.quote(relative_prepared)} "
            f"--output {shlex.quote(relative_output)}"
        )
        if deploy_custom_op:
            command = f"{command} --deploy-custom-op"
        runner(_ssh_command(endpoint, command))
        runner(
            _scp_download_command(endpoint, remote_output, output_dir / "output")
        )

    if profile_device_time:
        base_operator = (
            f"{shlex.quote(endpoint.python)} -m cannbench operator "
            f"--backend {shlex.quote(endpoint.backend)} "
            f"--prepared-input {shlex.quote(relative_prepared)} "
            f"--warmup {warmup} "
            f"--iterations {iterations} "
            f"--output-dir {shlex.quote(relative_perf)} "
            "--run-name benchmark"
        )
        if deploy_custom_op:
            base_operator = f"{base_operator} --deploy-custom-op"
        if endpoint.backend == "ascend":
            profiled_operator = (
                f"msprof op --output={shlex.quote(remote_profile)} {base_operator}"
            )
            command = (
                f"{_remote_command_prefix(endpoint)}"
                f"{_remote_command_env(endpoint.env)}"
                f"{profiled_operator}"
            )
        elif endpoint.backend == "nvidia":
            env_prefix = _remote_command_env(endpoint.env)
            capability_probe = (
                f"{env_prefix}{shlex.quote(endpoint.python)} -c "
                f"{shlex.quote('import torch; print(torch.cuda.get_device_capability(0)[0])')}"
            )
            ncu_operator = (
                f"{env_prefix}"
                "ncu --target-processes all --force-overwrite "
                "--csv "
                f"--log-file {shlex.quote(remote_profile + '/ncu.csv')} "
                f"--export {shlex.quote(remote_profile + '/ncu-report')} "
                f"{base_operator}"
            )
            cuda_event_operator = (
                f"{env_prefix}"
                f"{shlex.quote(endpoint.python)} -m cannbench cuda-event-profile "
                "--backend nvidia "
                f"--prepared-input {shlex.quote(relative_prepared)} "
                f"--warmup {warmup} "
                f"--iterations {iterations} "
                f"--profile-dir {shlex.quote(remote_profile)} "
                f"--output-dir {shlex.quote(relative_perf)} "
                "--run-name benchmark"
            )
            profiled_operator = (
                f"major=$({capability_probe}); "
                'if [ "$major" -ge 7 ] && command -v ncu >/dev/null 2>&1; '
                f"then {ncu_operator}; "
                f"else {cuda_event_operator}; "
                "fi"
            )
            command = f"{_remote_command_prefix(endpoint)}{profiled_operator}"
        else:
            raise ValueError(f"unsupported profiler backend: {endpoint.backend}")
        runner(_ssh_command(endpoint, command))
        runner(
            _scp_download_command(endpoint, remote_profile, output_dir / "profile")
        )
        runner(_scp_download_command(endpoint, remote_perf, output_dir / "perf"))
        if summarize_profile:
            summary = read_device_profile(output_dir / "profile", backend=endpoint.backend)
            write_device_profile_summary(output_dir / "profile-summary.json", summary)

    return RemoteCollectionResult(
        endpoint=endpoint,
        run_id=actual_run_id,
        remote_run_dir=remote_run_dir,
        local_output_dir=output_dir,
    )
