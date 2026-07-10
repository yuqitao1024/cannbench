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
    read_device_profile,
)
from cannbench.core.result import OperatorBenchmarkResult, OperatorCase
from cannbench.operators import TorchOperatorContext, get_operator_plugin
from cannbench.operators.materialize import materialized_values_to_buffer

_SKIP_SIMT_INSTALL_ENV = "CANNBENCH_SKIP_SIMT_INSTALL"


def _subprocess_pythonpath() -> str:
    src_root = str(Path(__file__).resolve().parents[2])
    existing = os.environ.get("PYTHONPATH", "")
    if not existing:
        return src_root
    parts = [entry for entry in existing.split(os.pathsep) if entry]
    if src_root in parts:
        return existing
    return os.pathsep.join((src_root, existing))


def _ascend_msprof_op_options(
    profile_dir: Path,
    request: OperatorBenchmarkRequest,
) -> list[str]:
    del request
    return [
        f"--output={profile_dir}",
        f"--launch-count=10",
    ]


class NvidiaBackend(TorchOperatorBackend):
    def __init__(self) -> None:
        super().__init__(name="nvidia", device_type="cuda")

    def _availability_error(self) -> str:
        return "CUDA is required for the nvidia backend"

    def _operator_callable(self, torch, request, case, *, device, dtype):
        if request.implementation == "cuda_library":
            plugin = get_operator_plugin(request.op)
            if plugin.build_cuda_library_callable is None:
                raise RuntimeError(
                    f"{request.op} does not provide a cuda_library implementation"
                )
            return plugin.build_cuda_library_callable(
                TorchOperatorContext(
                    backend=self,
                    torch=torch,
                    request=request,
                    case=case,
                    device=device,
                    dtype=dtype,
                )
            )
        return super()._operator_callable(
            torch,
            request,
            case,
            device=device,
            dtype=dtype,
        )

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
                "--launch-count",
                "1",
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
                "--output-dir",
                str(perf_dir),
                "--run-name",
                "benchmark",
            ]
            if request.implementation is not None:
                command.extend(("--implementation", request.implementation))
            if request.implementation_version is not None:
                command.extend(("--implementation-version", request.implementation_version))
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
                kernel_selection=get_operator_plugin(request.op).profile_kernel_selection(
                    backend="nvidia",
                    implementation=request.implementation,
                    implementation_version=request.implementation_version,
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
        with tempfile.TemporaryDirectory(prefix="cannbench-msprof-") as temp_dir_name:
            temp_dir = Path(temp_dir_name)
            prepared_path = temp_dir / "prepared.json"
            profile_dir = temp_dir / "profile"
            perf_dir = temp_dir / "perf"
            profile_dir.mkdir(parents=True, exist_ok=True)
            perf_dir.mkdir(parents=True, exist_ok=True)
            write_prepared_operator_input(prepared_path, prepared)
            command = [
                "msprof",
                "op",
                *_ascend_msprof_op_options(profile_dir, request),
                sys.executable,
                "-m",
                "cannbench",
                "internal-run",
                "--backend",
                "ascend",
                "--prepared-input",
                str(prepared_path),
                "--output-dir",
                str(perf_dir),
                "--run-name",
                "benchmark",
            ]
            if request.implementation is not None:
                command.extend(("--implementation", request.implementation))
            if request.implementation_version is not None:
                command.extend(("--implementation-version", request.implementation_version))
            result = subprocess.run(
                command,
                cwd=temp_dir,
                env={
                    **os.environ,
                    "PYTHONPATH": _subprocess_pythonpath(),
                    _SKIP_SIMT_INSTALL_ENV: "1",
                },
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
                    f"msprof profiling failed (exit {result.returncode}): {result.stderr.strip()}"
                )
            if result.stdout:
                print(result.stdout, end="", flush=True)
            if result.stderr:
                print(result.stderr, end="", file=sys.stderr, flush=True)
            summary = read_device_profile(
                profile_dir,
                backend="ascend",
                kernel_selection=get_operator_plugin(request.op).profile_kernel_selection(
                    backend="ascend",
                    implementation=request.implementation,
                    implementation_version=request.implementation_version,
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
            ),
            profile=profile,
        )

    def _ensure_vllm_ascend_custom_ops_loaded(self) -> None:
        try:
            utils = importlib.import_module("vllm_ascend.utils")
            bootstrap_custom_op_env = getattr(
                utils, "bootstrap_custom_op_env", None
            )
            if callable(bootstrap_custom_op_env):
                bootstrap_custom_op_env(include_vendor_lib=True)
            importlib.import_module("vllm_ascend.vllm_ascend_C")
        except ImportError:
            return

    def _ascend_custom_ops(self, torch):
        self._ensure_vllm_ascend_custom_ops_loaded()
        return getattr(getattr(torch, "ops", None), "_C_ascend", None)

    def _custom_op_pair(self, torch, metadata_name: str, op_name: str):
        ops = self._ascend_custom_ops(torch)
        return getattr(ops, metadata_name, None), getattr(ops, op_name, None)

    def _materialized_values_with_padded_last_dim(
        self,
        values: tuple[float, ...],
        *,
        logical_dim: int,
        physical_dim: int,
    ):
        if logical_dim == physical_dim:
            return materialized_values_to_buffer(values)
        pad = physical_dim - logical_dim
        if pad < 0:
            raise ValueError("physical_dim must be greater than or equal to logical_dim")
        padded: list[float] = []
        zero_pad = [0.0] * pad
        for offset in range(0, len(values), logical_dim):
            padded.extend(values[offset : offset + logical_dim])
            padded.extend(zero_pad)
        return materialized_values_to_buffer(padded)

    def _materialized_kv_values(
        self,
        values: tuple[float, ...],
        *,
        batch: int,
        kv_heads: int,
        context_tokens: int,
        kept_context_tokens: int,
        logical_dim: int,
        physical_dim: int,
    ):
        if kept_context_tokens == context_tokens:
            return self._materialized_values_with_padded_last_dim(
                values,
                logical_dim=logical_dim,
                physical_dim=physical_dim,
            )
        pad = physical_dim - logical_dim
        if pad < 0:
            raise ValueError("physical_dim must be greater than or equal to logical_dim")
        padded: list[float] = []
        zero_pad = [0.0] * pad
        for batch_index in range(batch):
            for head_index in range(kv_heads):
                head_base = (
                    (batch_index * kv_heads + head_index)
                    * context_tokens
                    * logical_dim
                )
                for token_index in range(kept_context_tokens):
                    offset = head_base + token_index * logical_dim
                    padded.extend(values[offset : offset + logical_dim])
                    padded.extend(zero_pad)
        return materialized_values_to_buffer(padded)

    def _materialized_bhtd_values_as_bthd(
        self,
        values: tuple[float, ...],
        *,
        batch: int,
        heads: int,
        tokens: int,
        dim: int,
    ):
        reordered: list[float] = []
        for batch_index in range(batch):
            for token_index in range(tokens):
                for head_index in range(heads):
                    offset = (
                        ((batch_index * heads + head_index) * tokens + token_index)
                        * dim
                    )
                    reordered.extend(values[offset : offset + dim])
        return materialized_values_to_buffer(reordered)

    def _materialized_kv_values_as_bthd(
        self,
        values: tuple[float, ...],
        *,
        batch: int,
        kv_heads: int,
        context_tokens: int,
        kept_context_tokens: int,
        logical_dim: int,
        physical_dim: int,
    ):
        pad = physical_dim - logical_dim
        if pad < 0:
            raise ValueError("physical_dim must be greater than or equal to logical_dim")
        reordered: list[float] = []
        zero_pad = [0.0] * pad
        for batch_index in range(batch):
            for token_index in range(kept_context_tokens):
                for head_index in range(kv_heads):
                    offset = (
                        ((batch_index * kv_heads + head_index) * context_tokens + token_index)
                        * logical_dim
                    )
                    reordered.extend(values[offset : offset + logical_dim])
                    reordered.extend(zero_pad)
        return materialized_values_to_buffer(reordered)

    def _operator_callable(self, torch, request, case, *, device, dtype):
        if request.implementation == "vllm_ascend":
            plugin = get_operator_plugin(request.op)
            if plugin.build_vllm_ascend_callable is None:
                raise RuntimeError(
                    f"{request.op} does not provide a vllm_ascend implementation"
                )
            return plugin.build_vllm_ascend_callable(
                TorchOperatorContext(
                    backend=self,
                    torch=torch,
                    request=request,
                    case=case,
                    device=device,
                    dtype=dtype,
                )
            )
        if request.implementation == "simt":
            plugin = get_operator_plugin(request.op)
            if plugin.build_simt_callable is None:
                raise RuntimeError(f"{request.op} does not provide a SIMT implementation")
            module = self._load_simt_op_module(request, request.op)
            return plugin.build_simt_callable(
                TorchOperatorContext(
                    backend=self,
                    torch=torch,
                    request=request,
                    case=case,
                    device=device,
                    dtype=dtype,
                    implementation_module=module,
                )
            )
        return super()._operator_callable(
            torch,
            request,
            case,
            device=device,
            dtype=dtype,
        )

    def _before_run_operator(self, request: OperatorBenchmarkRequest) -> None:
        if (
            request.implementation == "simt"
            and os.environ.get(_SKIP_SIMT_INSTALL_ENV) != "1"
        ):
            self._install_simt_op(request, request.op)

    def _simt_op_root(self, op_name: str):
        return files(f"cannbench.operators.builtin.{op_name}").joinpath("simt")

    def _simt_op_base_dir(self, request: OperatorBenchmarkRequest, op_name: str):
        version = request.implementation_version or "v1"
        return self._simt_op_root(op_name).joinpath(version)

    def _install_simt_op(self, request: OperatorBenchmarkRequest, op_name: str) -> None:
        simt_op_dir = self._simt_op_base_dir(request, op_name)
        if not simt_op_dir.is_dir():
            raise RuntimeError(
                "Ascend SIMT operator deployment requested but SIMT operator "
                f"directory was not found for {op_name}: {simt_op_dir}"
            )
        install_script = simt_op_dir.joinpath("install.sh")
        if not install_script.is_file():
            raise RuntimeError(
                "Ascend SIMT operator deployment requested but install.sh was not "
                f"found for {op_name}: {install_script}"
            )
        with as_file(install_script) as script_path:
            self._run_simt_op_install(script_path)
        self._load_simt_op_module(request, op_name)

    def _simt_op_module_name(self, op_name: str, version: str | None) -> str | None:
        plugin = get_operator_plugin(op_name)
        if plugin.simt_module_name is None:
            return None
        return plugin.simt_module_name(version)

    def _load_simt_op_module(self, request: OperatorBenchmarkRequest, op_name: str):
        module_name = self._simt_op_module_name(op_name, request.implementation_version)
        if module_name is None:
            raise RuntimeError(
                "Ascend SIMT operator requested but no Python module is registered "
                f"for {op_name} version {request.implementation_version or 'v1'}"
            )
        simt_op_dir = self._simt_op_base_dir(request, op_name)
        with as_file(simt_op_dir) as simt_op_dir_path:
            simt_op_path = str(simt_op_dir_path)
            if simt_op_path not in sys.path:
                sys.path.insert(0, simt_op_path)
        importlib.invalidate_caches()
        try:
            return importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Ascend SIMT operator deployment completed but Python module "
                f"{module_name!r} could not be imported"
            ) from exc

    def _run_simt_op_install(self, script_path: Path) -> None:
        result = subprocess.run(
            [str(script_path)],
            cwd=script_path.parent,
            env={**os.environ, "PYTHON": sys.executable},
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(
                "Ascend SIMT operator deployment failed "
                f"({script_path}, exit {result.returncode}): {result.stderr.strip()}"
            )
