from __future__ import annotations

import json
import shlex
import subprocess
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable


CommandRunner = Callable[[list[str]], None]


@dataclass(frozen=True)
class RemoteEndpoint:
    name: str
    backend: str
    host: str
    workdir: str
    python: str = "python3"
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
    return RemoteEndpoint(
        name=str(payload["name"]),
        backend=str(payload["backend"]),
        host=str(payload["host"]),
        workdir=str(payload["workdir"]).rstrip("/"),
        python=str(payload.get("python", "python3")),
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


def collect_remote_artifacts(
    *,
    prepared_input: Path,
    output_dir: Path,
    capture_output: bool,
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
    if not capture_output:
        raise ValueError("collect currently requires --capture-output")

    actual_run_id = run_id or uuid.uuid4().hex
    remote_run_dir = f"{endpoint.workdir}/.cannbench-runs/{actual_run_id}"
    remote_prepared = f"{remote_run_dir}/prepared.json"
    remote_output = f"{remote_run_dir}/output"
    relative_prepared = f".cannbench-runs/{actual_run_id}/prepared.json"
    relative_output = f".cannbench-runs/{actual_run_id}/output"

    output_dir.mkdir(parents=True, exist_ok=True)
    runner(["ssh", endpoint.host, f"mkdir -p {shlex.quote(remote_run_dir)}"])
    runner(["scp", str(prepared_input), f"{endpoint.host}:{remote_prepared}"])

    command = (
        f"cd {shlex.quote(endpoint.workdir)} && "
        f"{_remote_command_env(endpoint.env)}"
        f"{shlex.quote(endpoint.python)} -m cannbench capture-output "
        f"--backend {shlex.quote(endpoint.backend)} "
        f"--prepared-input {shlex.quote(relative_prepared)} "
        f"--output {shlex.quote(relative_output)}"
    )
    if deploy_custom_op:
        command = f"{command} --deploy-custom-op"
    runner(["ssh", endpoint.host, command])
    runner(["scp", "-r", f"{endpoint.host}:{remote_output}", str(output_dir / "output")])

    return RemoteCollectionResult(
        endpoint=endpoint,
        run_id=actual_run_id,
        remote_run_dir=remote_run_dir,
        local_output_dir=output_dir,
    )
