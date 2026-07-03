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
from cannbench.datasets.materialize import (
    materialize_lightning_indexer_inputs,
    materialized_values_to_buffer,
)

_ASCEND_SIMT_OP_MODULES = {
    ("softmax", "v1"): "aten_softmax",
    ("softmax", "v2"): "aten_softmax_v2",
    ("softmax", "v3"): "aten_softmax_v3",
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

    def _operator_callable(self, torch, request, case, *, device, dtype):
        if request.implementation == "cuda_library" and request.op in {
            "lightning_indexer",
            "sparse_attention",
        }:
            raise RuntimeError(
                "cuda_library DSA benchmarking requires a FlashMLA/DeepGEMM "
                "adapter before it can be benchmarked"
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

    def _operator_callable(self, torch, request, case, *, device, dtype):
        if request.implementation == "vllm_ascend":
            if request.op == "lightning_indexer":
                return self._vllm_ascend_lightning_indexer_callable(
                    torch,
                    request,
                    case,
                    device=device,
                    dtype=dtype,
                )
            if request.op == "sparse_attention":
                raise RuntimeError(
                    "vllm_ascend sparse_attention requires a paged-KV metadata "
                    "adapter before it can be benchmarked"
                )
        return super()._operator_callable(
            torch,
            request,
            case,
            device=device,
            dtype=dtype,
        )

    def _vllm_ascend_lightning_indexer_callable(
        self,
        torch,
        request: OperatorBenchmarkRequest,
        case,
        *,
        device,
        dtype,
    ):
        try:
            import torch_npu
        except ModuleNotFoundError as exc:
            raise RuntimeError("torch_npu is required for vllm_ascend lightning_indexer") from exc
        if not hasattr(torch_npu, "npu_lightning_indexer"):
            raise RuntimeError(
                "vllm_ascend lightning_indexer requires torch_npu.npu_lightning_indexer"
            )

        payload = materialize_lightning_indexer_inputs(
            case, dtype=request.dtype, seed=request.seed
        )
        query_shape = payload["query_shape"]
        key_shape = payload["key_shape"]
        batch, query_tokens, index_heads, index_dim = query_shape
        context_tokens = key_shape[1]

        query = self._tensor(
            torch,
            materialized_values_to_buffer(payload["query"]),
            device=device,
            dtype=dtype,
        ).reshape(batch * query_tokens, index_heads, index_dim)
        keys = self._tensor(
            torch,
            materialized_values_to_buffer(payload["keys"]),
            device=device,
            dtype=dtype,
        ).reshape(batch, context_tokens, 1, index_dim)
        weights = self._tensor(
            torch,
            materialized_values_to_buffer(payload["weights"]),
            device=device,
            dtype=dtype,
        ).reshape(payload["weight_shape"])
        actual_seq_lengths_query = self._tensor(
            torch,
            tuple((index + 1) * query_tokens for index in range(batch)),
            device=device,
            dtype=torch.int32,
        )
        actual_seq_lengths_key = self._tensor(
            torch,
            tuple((index + 1) * context_tokens for index in range(batch)),
            device=device,
            dtype=torch.int32,
        )

        def operator():
            result = torch_npu.npu_lightning_indexer(
                query=query,
                key=keys,
                weights=weights,
                actual_seq_lengths_query=actual_seq_lengths_query,
                actual_seq_lengths_key=actual_seq_lengths_key,
                block_table=None,
                layout_query="TND",
                layout_key="BSND",
                sparse_count=payload["top_k"],
                sparse_mode=3,
            )
            return result[0] if isinstance(result, tuple) else result

        return operator

    def _before_run_operator(self, request: OperatorBenchmarkRequest) -> None:
        if request.deploy_simt_op:
            self._deploy_simt_op(request, request.op)
        elif request.use_simt_op:
            self._load_simt_op_module(request, request.op)

    def _simt_op_root(self, op_name: str):
        return files("cannbench.datasets.data").joinpath(op_name, "simt")

    def _simt_op_base_dir(self, request: OperatorBenchmarkRequest, op_name: str):
        version = request.implementation_version or "v1"
        return self._simt_op_root(op_name).joinpath(version)

    def _deploy_simt_op(self, request: OperatorBenchmarkRequest, op_name: str) -> None:
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
        return _ASCEND_SIMT_OP_MODULES.get((op_name, version or "v1"))

    def _load_simt_op_module(self, request: OperatorBenchmarkRequest, op_name: str) -> None:
        module_name = self._simt_op_module_name(op_name, request.implementation_version)
        if module_name is None:
            return
        importlib.invalidate_caches()
        try:
            importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Ascend SIMT operator deployment completed but Python module "
                f"{module_name!r} could not be imported"
            ) from exc

    def _run_simt_op_install(self, script_path: Path) -> None:
        result = subprocess.run(
            [str(script_path)],
            cwd=script_path.parent,
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(
                "Ascend SIMT operator deployment failed "
                f"({script_path}, exit {result.returncode}): {result.stderr.strip()}"
            )

    def _softmax(self, torch, tensor, dim: int | None, request: OperatorBenchmarkRequest):
        if request.use_simt_op or request.deploy_simt_op:
            module_name = self._simt_op_module_name(request.op, request.implementation_version)
            if module_name is None:
                raise RuntimeError(
                    "Ascend SIMT softmax requested but no SIMT operator module is "
                    f"registered for version {request.implementation_version or 'v1'}"
                )
            try:
                aten_softmax_module = importlib.import_module(module_name)
            except ModuleNotFoundError as exc:
                raise RuntimeError(
                    f"Ascend SIMT softmax requested but {module_name} is not importable"
                ) from exc
            return aten_softmax_module.ops.spatial_softmax_forward(tensor, int(dim))
        return torch.softmax(tensor, dim=dim)
