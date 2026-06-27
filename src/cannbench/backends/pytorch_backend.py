from __future__ import annotations

import importlib
import tempfile
import subprocess
from importlib.resources import as_file, files
from pathlib import Path

from cannbench.backends.torch_backend_base import TorchOperatorBackend
from cannbench.core.config import OperatorBenchmarkRequest
from cannbench.core.prepared_input import build_prepared_operator_input, write_prepared_operator_input
from cannbench.core.profile import LocalDeviceProfileResult, read_device_profile
from cannbench.core.result import OperatorBenchmarkResult, OperatorCase

_ASCEND_CUSTOM_OP_MODULES = {
    "softmax": "aten_softmax",
}


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
            write_prepared_operator_input(prepared_path, prepared)
            command = [
                "ncu",
                "--target-processes",
                "all",
                "--force-overwrite",
                "--csv",
                "--log-file",
                str(profile_dir / "ncu.csv"),
                "--export",
                str(profile_dir / "ncu-report"),
                "python3",
                "-m",
                "cannbench",
                "operator",
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
                text=True,
                capture_output=True,
                check=False,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"ncu profiling failed (exit {result.returncode}): {result.stderr.strip()}"
                )
            summary = read_device_profile(profile_dir, backend="nvidia")
            artifacts = tuple(
                (
                    str(path.relative_to(profile_dir)),
                    path.read_bytes(),
                )
                for path in sorted(profile_dir.rglob("*"))
                if path.is_file()
            )
        return LocalDeviceProfileResult(
            benchmark_result=OperatorBenchmarkResult(
                backend=self.name,
                device_name=self._device_name(torch, self._device(torch)),
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
            profile_summary=summary,
            profile_artifacts=artifacts,
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

    def _custom_op_base_dir(self, op_name: str):
        return files("cannbench.datasets.data").joinpath(
            op_name, "custom_ops", "ascend", "v1"
        )

    def _deploy_custom_op(self, request: OperatorBenchmarkRequest, op_name: str) -> None:
        del request
        custom_op_dir = self._custom_op_base_dir(op_name)
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
