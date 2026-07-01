from __future__ import annotations

import importlib
import os
import sys
import tempfile
import subprocess
from importlib.resources import as_file, files
from pathlib import Path

from cannbench.backends.torch_backend_base import TorchOperatorBackend
from cannbench.core.config import OperatorBenchmarkRequest
from cannbench.core.execution import read_artifact_tree
from cannbench.core.prepared_input import build_prepared_operator_input, write_prepared_operator_input
from cannbench.core.profile import (
    ProfileArtifacts,
    LocalDeviceProfileResult,
    expected_kernel_name_patterns,
    read_device_profile,
)
from cannbench.core.result import OperatorBenchmarkResult, OperatorCase

_ASCEND_CUSTOM_OP_MODULES = {
    "softmax": "aten_softmax",
}


def _subprocess_pythonpath() -> str:
    src_root = str(Path(__file__).resolve().parents[2])
    existing = os.environ.get("PYTHONPATH", "")
    if not existing:
        return src_root
    parts = [entry for entry in existing.split(os.pathsep) if entry]
    if src_root in parts:
        return existing
    return os.pathsep.join((src_root, existing))


class NvidiaBackend(TorchOperatorBackend):
    def __init__(self) -> None:
        super().__init__(name="nvidia", device_type="cuda")

    def _availability_error(self) -> str:
        return "CUDA is required for the nvidia backend"

    def profile_operator_device_time(
        self, request: OperatorBenchmarkRequest
    ) -> LocalDeviceProfileResult:
        self.validate_request(request)
        self._before_run_operator(request)
        torch = self._torch_module()
        if not self._is_available(torch):
            raise RuntimeError(self._availability_error())

        prepared = build_prepared_operator_input(
            op=request.op,
            dtype=request.dtype,
            dataset=request.dataset,
            case_id=request.case_id,
            seed=request.seed,
        )
        with tempfile.TemporaryDirectory(prefix="cannbench-ncu-") as temp_dir_name:
            temp_dir = Path(temp_dir_name)
            prepared_path = temp_dir / "prepared.json"
            profile_dir = temp_dir / "profile"
            perf_dir = temp_dir / "perf"
            profile_dir.mkdir(parents=True, exist_ok=True)
            perf_dir.mkdir(parents=True, exist_ok=True)
            write_prepared_operator_input(prepared_path, prepared)
            command = [
                "ncu",
                "--target-processes",
                "all",
                "--force-overwrite",
                "--launch-skip",
                str(request.warmup),
                "--launch-count",
                str(request.iterations),
                "--export",
                str(profile_dir / "ncu-report"),
                sys.executable,
                "-m",
                "cannbench",
                "internal-run",
                "--backend",
                "nvidia",
                "--prepared-input",
                str(prepared_path),
                "--warmup",
                str(request.warmup),
                "--iterations",
                str(request.iterations),
                "--output-dir",
                str(perf_dir),
                "--run-name",
                "benchmark",
            ]
            result = subprocess.run(
                command,
                cwd=temp_dir,
                env={**os.environ, "PYTHONPATH": _subprocess_pythonpath()},
                text=True,
                capture_output=True,
                check=False,
            )
            if result.returncode != 0:
                if result.stdout:
                    print(result.stdout, end="", flush=True)
                if result.stderr:
                    print(result.stderr, end="", file=sys.stderr, flush=True)
                raise RuntimeError(
                    f"ncu profiling failed (exit {result.returncode}): {result.stderr.strip()}"
                )
            if result.stdout:
                print(result.stdout, end="", flush=True)
            if result.stderr:
                print(result.stderr, end="", file=sys.stderr, flush=True)
            render_command = [
                "ncu",
                "--import",
                str(profile_dir / "ncu-report.ncu-rep"),
                "--page",
                "raw",
                "--csv",
            ]
            render_result = subprocess.run(
                render_command,
                cwd=temp_dir,
                env={**os.environ, "PYTHONPATH": _subprocess_pythonpath()},
                text=True,
                capture_output=True,
                check=False,
            )
            if render_result.returncode != 0:
                if render_result.stdout:
                    print(render_result.stdout, end="", flush=True)
                if render_result.stderr:
                    print(render_result.stderr, end="", file=sys.stderr, flush=True)
                raise RuntimeError(
                    "ncu report render failed "
                    f"(exit {render_result.returncode}): {render_result.stderr.strip()}"
                )
            if render_result.stdout:
                print(render_result.stdout, end="", flush=True)
            if render_result.stderr:
                print(render_result.stderr, end="", file=sys.stderr, flush=True)
            profile_dir.mkdir(parents=True, exist_ok=True)
            (profile_dir / "ncu.csv").write_text(render_result.stdout)
            summary = read_device_profile(
                profile_dir,
                backend="nvidia",
                expected_kernel_name_patterns=expected_kernel_name_patterns(
                    backend="nvidia",
                    op=request.op,
                ),
            )
            profile = ProfileArtifacts(
                device_name=self._device_name(torch, self._device(torch)),
                profile_summary=summary,
                profile_artifacts=read_artifact_tree(profile_dir),
                perf_artifacts=read_artifact_tree(perf_dir),
            )
        return LocalDeviceProfileResult(
            benchmark_result=OperatorBenchmarkResult(
                backend=self.name,
                device_name=profile.device_name,
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
            ),
            profile=profile,
        )


class AscendBackend(TorchOperatorBackend):
    def __init__(self) -> None:
        super().__init__(name="ascend", device_type="npu")

    def _torch_module(self):
        try:
            import torch_npu  # noqa: F401
        except ModuleNotFoundError as exc:
            raise RuntimeError("torch_npu is required for the ascend backend") from exc
        return super()._torch_module()

    def _availability_error(self) -> str:
        return "Ascend NPU is required for the ascend backend"

    def _before_run_operator(self, request: OperatorBenchmarkRequest) -> None:
        if request.deploy_custom_op:
            self._deploy_custom_op(request, request.op)

    def _custom_op_ascend_root(self, op_name: str):
        return files("cannbench.datasets.data").joinpath(
            op_name, "custom_ops", "ascend"
        )

    def _custom_op_base_dir(self, request: OperatorBenchmarkRequest, op_name: str):
        version = request.implementation_version or "v1"
        return self._custom_op_ascend_root(op_name).joinpath(version)

    def _deploy_custom_op(self, request: OperatorBenchmarkRequest, op_name: str) -> None:
        custom_op_dir = self._custom_op_base_dir(request, op_name)
        if not custom_op_dir.is_dir():
            raise RuntimeError(
                "Ascend custom op deployment requested but default custom op "
                f"directory was not found for {op_name}: {custom_op_dir}"
            )
        install_script = custom_op_dir.joinpath("install.sh")
        if not install_script.is_file():
            raise RuntimeError(
                "Ascend custom op deployment requested but install.sh was not "
                f"found for {op_name}: {install_script}"
            )
        with as_file(install_script) as script_path:
            self._run_custom_op_install(script_path)
        self._load_custom_op_module(op_name)

    def _load_custom_op_module(self, op_name: str) -> None:
        module_name = _ASCEND_CUSTOM_OP_MODULES.get(op_name)
        if module_name is None:
            return
        importlib.invalidate_caches()
        try:
            importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Ascend custom op deployment completed but Python module "
                f"{module_name!r} could not be imported"
            ) from exc

    def _run_custom_op_install(self, script_path: Path) -> None:
        result = subprocess.run(
            [str(script_path)],
            cwd=script_path.parent,
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(
                "Ascend custom op deployment failed "
                f"({script_path}, exit {result.returncode}): {result.stderr.strip()}"
            )

    def _softmax(self, torch, tensor, dim: int | None, request: OperatorBenchmarkRequest):
        if request.deploy_custom_op:
            try:
                from aten_softmax import ops as aten_softmax_ops
            except ModuleNotFoundError as exc:
                raise RuntimeError(
                    "Ascend SIMT softmax requested but aten_softmax is not importable"
                ) from exc
            return aten_softmax_ops.spatial_softmax_forward(tensor, int(dim))
        return torch.softmax(tensor, dim=dim)
