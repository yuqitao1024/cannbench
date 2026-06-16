from __future__ import annotations

import subprocess
from importlib.resources import as_file, files
from pathlib import Path

from cannbench.backends.torch_backend_base import TorchOperatorBackend
from cannbench.core.config import OperatorBenchmarkRequest


class NvidiaBackend(TorchOperatorBackend):
    def __init__(self) -> None:
        super().__init__(name="nvidia", device_type="cuda")

    def _availability_error(self) -> str:
        return "CUDA is required for the nvidia backend"


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
            op_name, "custom_ops", "ascend", "default"
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
