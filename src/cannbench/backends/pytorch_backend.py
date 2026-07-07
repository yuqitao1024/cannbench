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
from cannbench.operators.builtin.lightning_indexer.materialize import (
    materialize_lightning_indexer_inputs,
)
from cannbench.operators.builtin.sparse_attention.materialize import (
    materialize_sparse_attention_inputs,
)
from cannbench.operators.materialize import materialized_values_to_buffer

_ASCEND_SIMT_OP_MODULES = {
    ("softmax", "v1"): "aten_softmax",
    ("softmax", "v2"): "aten_softmax_v2",
    ("softmax", "v3"): "aten_softmax_v3",
}
_CUDA_DSA_ADAPTER_ENV = "CANNBENCH_CUDA_DSA_ADAPTER"
_DEFAULT_CUDA_DSA_ADAPTER_MODULE = "cannbench_cuda_dsa"


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
        if request.implementation == "cuda_library":
            if request.op == "lightning_indexer":
                return self._cuda_library_lightning_indexer_callable(
                    torch,
                    request,
                    case,
                    device=device,
                    dtype=dtype,
                )
            if request.op == "sparse_attention":
                return self._cuda_library_sparse_attention_callable(
                    torch,
                    request,
                    case,
                    device=device,
                    dtype=dtype,
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

    def _resolve_cuda_dsa_adapter(self, op_name: str):
        module_name = os.environ.get(_CUDA_DSA_ADAPTER_ENV) or _DEFAULT_CUDA_DSA_ADAPTER_MODULE
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            if exc.name != module_name:
                raise
            raise RuntimeError(
                "cuda_library DSA benchmarking requires an external FlashMLA/DeepGEMM "
                f"adapter. Install {_DEFAULT_CUDA_DSA_ADAPTER_MODULE} or set "
                f"{_CUDA_DSA_ADAPTER_ENV}=<module> with callable {op_name}."
            ) from exc
        op_callable = getattr(module, op_name, None)
        if not callable(op_callable):
            raise RuntimeError(
                f"CUDA DSA adapter {module_name} must expose callable {op_name}"
            )
        return op_callable

    def _cuda_library_lightning_indexer_callable(
        self,
        torch,
        request: OperatorBenchmarkRequest,
        case,
        *,
        device,
        dtype,
    ):
        adapter_op = self._resolve_cuda_dsa_adapter("lightning_indexer")
        payload = materialize_lightning_indexer_inputs(
            case, dtype=request.dtype, seed=request.seed
        )
        query = self._tensor(
            torch,
            materialized_values_to_buffer(payload["query"]),
            device=device,
            dtype=dtype,
        ).reshape(payload["query_shape"])
        keys = self._tensor(
            torch,
            materialized_values_to_buffer(payload["keys"]),
            device=device,
            dtype=dtype,
        ).reshape(payload["key_shape"])
        weights = self._tensor(
            torch,
            materialized_values_to_buffer(payload["weights"]),
            device=device,
            dtype=dtype,
        ).reshape(payload["weight_shape"])

        def operator():
            return adapter_op(
                torch=torch,
                request=request,
                case=case,
                payload=payload,
                device=device,
                dtype=dtype,
                query=query,
                keys=keys,
                weights=weights,
                top_k=payload["top_k"],
            )

        return operator

    def _cuda_library_sparse_attention_callable(
        self,
        torch,
        request: OperatorBenchmarkRequest,
        case,
        *,
        device,
        dtype,
    ):
        adapter_op = self._resolve_cuda_dsa_adapter("sparse_attention")
        payload = materialize_sparse_attention_inputs(
            case, dtype=request.dtype, seed=request.seed
        )
        query = self._tensor(
            torch,
            materialized_values_to_buffer(payload["query"]),
            device=device,
            dtype=dtype,
        ).reshape(payload["query_shape"])
        keys = self._tensor(
            torch,
            materialized_values_to_buffer(payload["keys"]),
            device=device,
            dtype=dtype,
        ).reshape(payload["key_shape"])
        values = self._tensor(
            torch,
            materialized_values_to_buffer(payload["values"]),
            device=device,
            dtype=dtype,
        ).reshape(payload["value_shape"])
        indices = self._tensor(
            torch,
            payload["indices"],
            device=device,
            dtype=torch.int32,
        ).reshape(payload["indices_shape"])
        softmax_scale = payload["query_shape"][-1] ** -0.5

        def operator():
            return adapter_op(
                torch=torch,
                request=request,
                case=case,
                payload=payload,
                device=device,
                dtype=dtype,
                query=query,
                keys=keys,
                values=values,
                indices=indices,
                causal=payload["causal"],
                phase=payload["phase"],
                softmax_scale=softmax_scale,
            )

        return operator


class AscendBackend(TorchOperatorBackend):
    def __init__(self) -> None:
        super().__init__(name="ascend", device_type="npu")

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
                f"--output={profile_dir}",
                "--warm-up",
                str(request.warmup),
                "--launch-count",
                str(request.iterations),
                sys.executable,
                "-m",
                "cannbench",
                "internal-run",
                "--backend",
                "ascend",
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
                    "msprof op profiling failed "
                    f"(exit {result.returncode}): {result.stderr.strip()}"
                )
            if result.stdout:
                print(result.stdout, end="", flush=True)
            if result.stderr:
                print(result.stderr, end="", file=sys.stderr, flush=True)
            summary = read_device_profile(
                profile_dir,
                backend="ascend",
                expected_kernel_name_patterns=expected_kernel_name_patterns(
                    backend="ascend",
                    op=request.op,
                    implementation=request.implementation,
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

    def _torch_module(self):
        try:
            import torch_npu  # noqa: F401
        except ModuleNotFoundError as exc:
            raise RuntimeError("torch_npu is required for the ascend backend") from exc
        return super()._torch_module()

    def _availability_error(self) -> str:
        return "Ascend NPU is required for the ascend backend"

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
            if request.op == "lightning_indexer":
                return self._vllm_ascend_lightning_indexer_callable(
                    torch,
                    request,
                    case,
                    device=device,
                    dtype=dtype,
                )
            if request.op == "sparse_attention":
                return self._vllm_ascend_sparse_attention_callable(
                    torch,
                    request,
                    case,
                    device=device,
                    dtype=dtype,
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
        metadata_op, indexer_op = self._custom_op_pair(
            torch,
            "npu_vllm_quant_lightning_indexer_metadata",
            "npu_vllm_quant_lightning_indexer",
        )
        if metadata_op is not None and indexer_op is not None:
            return self._vllm_ascend_quant_lightning_indexer_callable(
                torch,
                request,
                case,
                device=device,
                dtype=dtype,
                metadata_op=metadata_op,
                indexer_op=indexer_op,
            )

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

    def _vllm_ascend_quant_lightning_indexer_callable(
        self,
        torch,
        request: OperatorBenchmarkRequest,
        case,
        *,
        device,
        dtype,
        metadata_op,
        indexer_op,
    ):
        payload = materialize_lightning_indexer_inputs(
            case, dtype=request.dtype, seed=request.seed
        )
        query_shape = payload["query_shape"]
        key_shape = payload["key_shape"]
        batch, query_tokens, index_heads, index_dim = query_shape
        context_tokens = key_shape[1]
        block_size = 128 if context_tokens % 128 == 0 else context_tokens
        blocks_per_batch = context_tokens // block_size
        quant_dtype = getattr(torch, "float8_e4m3fn", getattr(torch, "int8", dtype))
        scale_dtype = getattr(torch, "float32", dtype)

        query = self._tensor(
            torch,
            materialized_values_to_buffer(payload["query"]),
            device=device,
            dtype=quant_dtype,
        ).reshape(batch * query_tokens, index_heads, index_dim)
        keys = self._tensor(
            torch,
            materialized_values_to_buffer(payload["keys"]),
            device=device,
            dtype=quant_dtype,
        ).reshape(batch * blocks_per_batch, block_size, 1, index_dim)
        weights = self._tensor(
            torch,
            materialized_values_to_buffer(payload["weights"]),
            device=device,
            dtype=scale_dtype,
        ).reshape(batch * query_tokens, index_heads)
        query_dequant_scale = self._tensor(
            torch,
            tuple(1.0 for _ in range(batch * query_tokens * index_heads)),
            device=device,
            dtype=scale_dtype,
        ).reshape(batch * query_tokens, index_heads)
        key_dequant_scale = self._tensor(
            torch,
            tuple(1.0 for _ in range(batch * blocks_per_batch * block_size)),
            device=device,
            dtype=scale_dtype,
        ).reshape(batch * blocks_per_batch, block_size, 1)
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
        block_table = self._tensor(
            torch,
            tuple(range(batch * blocks_per_batch)),
            device=device,
            dtype=torch.int32,
        ).reshape(batch, blocks_per_batch)
        common_kwargs = {
            "actual_seq_lengths_query": actual_seq_lengths_query,
            "actual_seq_lengths_key": actual_seq_lengths_key,
            "num_heads_q": index_heads,
            "num_heads_k": 1,
            "head_dim": index_dim,
            "query_quant_mode": 0,
            "key_quant_mode": 0,
            "batch_size": batch,
            "max_seqlen_q": query_tokens,
            "max_seqlen_k": context_tokens,
            "layout_query": "TND",
            "layout_key": "PA_BSND",
            "sparse_count": payload["top_k"],
            "sparse_mode": 3,
            "pre_tokens": (1 << 63) - 1,
            "next_tokens": (1 << 63) - 1,
            "cmp_ratio": 4,
        }
        metadata = metadata_op(**common_kwargs, device=str(device))
        metadata_only_kwargs = {
            "num_heads_q",
            "num_heads_k",
            "head_dim",
            "batch_size",
            "max_seqlen_q",
            "max_seqlen_k",
        }
        indexer_kwargs = {
            key: value
            for key, value in common_kwargs.items()
            if key not in metadata_only_kwargs
        }

        def operator():
            result = indexer_op(
                query=query,
                key=keys,
                weights=weights,
                query_dequant_scale=query_dequant_scale,
                key_dequant_scale=key_dequant_scale,
                block_table=block_table,
                metadata=metadata,
                return_value=False,
                **indexer_kwargs,
            )
            return result[0] if isinstance(result, tuple) else result

        return operator

    def _vllm_ascend_sparse_attention_callable(
        self,
        torch,
        request: OperatorBenchmarkRequest,
        case,
        *,
        device,
        dtype,
    ):
        metadata_op, attention_op = self._custom_op_pair(
            torch,
            "npu_kv_quant_sparse_attn_sharedkv_metadata",
            "npu_kv_quant_sparse_attn_sharedkv",
        )
        if metadata_op is not None and attention_op is not None:
            return self._vllm_ascend_quant_sparse_attention_callable(
                torch,
                request,
                case,
                device=device,
                dtype=dtype,
                metadata_op=metadata_op,
                attention_op=attention_op,
            )

        metadata_op, attention_op = self._custom_op_pair(
            torch,
            "npu_sparse_attn_sharedkv_metadata",
            "npu_sparse_attn_sharedkv",
        )
        if metadata_op is None or attention_op is None:
            raise RuntimeError(
                "vllm_ascend sparse_attention requires "
                "torch.ops._C_ascend.npu_kv_quant_sparse_attn_sharedkv_metadata and "
                "torch.ops._C_ascend.npu_kv_quant_sparse_attn_sharedkv, or "
                "torch.ops._C_ascend.npu_sparse_attn_sharedkv_metadata and "
                "torch.ops._C_ascend.npu_sparse_attn_sharedkv"
            )

        payload = materialize_sparse_attention_inputs(
            case, dtype=request.dtype, seed=request.seed
        )
        batch, query_heads, query_tokens, head_dim = payload["query_shape"]
        _, kv_heads, context_tokens, _ = payload["key_shape"]
        selected_tokens = payload["indices_shape"][2]
        block_size = 128 if context_tokens % 128 == 0 else context_tokens
        blocks_per_batch = context_tokens // block_size

        query = self._tensor(
            torch,
            self._materialized_bhtd_values_as_bthd(
                payload["query"],
                batch=batch,
                heads=query_heads,
                tokens=query_tokens,
                dim=head_dim,
            ),
            device=device,
            dtype=dtype,
        )
        query = query.reshape(batch * query_tokens, query_heads, head_dim)
        cmp_kv = self._tensor(
            torch,
            self._materialized_kv_values_as_bthd(
                payload["keys"],
                batch=batch,
                kv_heads=kv_heads,
                context_tokens=context_tokens,
                kept_context_tokens=context_tokens,
                logical_dim=head_dim,
                physical_dim=head_dim,
            ),
            device=device,
            dtype=dtype,
        )
        cmp_kv = cmp_kv.reshape(batch * blocks_per_batch, block_size, kv_heads, head_dim)
        cmp_sparse_indices = self._tensor(
            torch,
            payload["indices"],
            device=device,
            dtype=torch.int32,
        ).reshape(batch * query_tokens, kv_heads, selected_tokens)
        cmp_block_table = self._tensor(
            torch,
            tuple(range(batch * blocks_per_batch)),
            device=device,
            dtype=torch.int32,
        ).reshape(batch, blocks_per_batch)
        cu_seqlens_q = self._tensor(
            torch,
            tuple(index * query_tokens for index in range(batch + 1)),
            device=device,
            dtype=torch.int32,
        )
        seqused_kv = self._tensor(
            torch,
            tuple(context_tokens for _ in range(batch)),
            device=device,
            dtype=torch.int32,
        )
        softmax_scale = head_dim ** -0.5

        metadata = metadata_op(
            num_heads_q=query_heads,
            num_heads_kv=kv_heads,
            head_dim=head_dim,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_ori_kv=None,
            cu_seqlens_cmp_kv=None,
            seqused_q=None,
            seqused_kv=seqused_kv,
            batch_size=batch,
            max_seqlen_q=query_tokens,
            max_seqlen_kv=context_tokens,
            ori_topk=0,
            cmp_topk=selected_tokens,
            cmp_ratio=1,
            ori_mask_mode=4,
            cmp_mask_mode=3,
            ori_win_left=0,
            ori_win_right=0,
            layout_q="TND",
            layout_kv="PA_ND",
            has_ori_kv=False,
            has_cmp_kv=True,
            device=str(device),
        )

        def operator():
            return attention_op(
                query,
                ori_kv=None,
                cmp_kv=cmp_kv,
                ori_sparse_indices=None,
                cmp_sparse_indices=cmp_sparse_indices,
                ori_block_table=None,
                cmp_block_table=cmp_block_table,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_ori_kv=None,
                cu_seqlens_cmp_kv=None,
                seqused_q=None,
                seqused_kv=seqused_kv,
                sinks=None,
                metadata=metadata,
                softmax_scale=softmax_scale,
                cmp_ratio=1,
                ori_mask_mode=4,
                cmp_mask_mode=3,
                ori_win_left=0,
                ori_win_right=0,
                layout_q="TND",
                layout_kv="PA_ND",
                return_softmax_lse=True,
            )[0]

        return operator

    def _vllm_ascend_quant_sparse_attention_callable(
        self,
        torch,
        request: OperatorBenchmarkRequest,
        case,
        *,
        device,
        dtype,
        metadata_op,
        attention_op,
    ):
        payload = materialize_sparse_attention_inputs(
            case, dtype=request.dtype, seed=request.seed
        )
        batch, query_heads, query_tokens, head_dim = payload["query_shape"]
        _, kv_heads, context_tokens, _ = payload["key_shape"]
        selected_tokens = payload["indices_shape"][2]
        a5_physical_layout = (
            head_dim == 512
            and kv_heads == 1
            and query_heads in {64, 128}
            and selected_tokens in {512, 1024}
        )
        cmp_ratio = 4 if a5_physical_layout else 1
        ori_context_tokens = context_tokens
        cmp_context_tokens = context_tokens // cmp_ratio
        ori_block_size = 128 if ori_context_tokens % 128 == 0 else ori_context_tokens
        cmp_block_size = 128 if cmp_context_tokens % 128 == 0 else cmp_context_tokens
        ori_blocks_per_batch = ori_context_tokens // ori_block_size
        cmp_blocks_per_batch = cmp_context_tokens // cmp_block_size
        kv_head_dim = 640 if a5_physical_layout else head_dim
        query_dtype = getattr(torch, "bfloat16", dtype)
        kv_dtype = getattr(torch, "float8_e4m3fn", dtype)
        scale_dtype = getattr(torch, "float32", dtype)

        query = self._tensor(
            torch,
            self._materialized_bhtd_values_as_bthd(
                payload["query"],
                batch=batch,
                heads=query_heads,
                tokens=query_tokens,
                dim=head_dim,
            ),
            device=device,
            dtype=query_dtype,
        )
        query = query.reshape(batch * query_tokens, query_heads, head_dim)
        ori_kv = None
        ori_block_table = None
        cu_seqlens_ori_kv = None
        if a5_physical_layout:
            ori_kv = self._tensor(
                torch,
                self._materialized_kv_values_as_bthd(
                    payload["values"],
                    batch=batch,
                    kv_heads=kv_heads,
                    context_tokens=context_tokens,
                    kept_context_tokens=ori_context_tokens,
                    logical_dim=head_dim,
                    physical_dim=kv_head_dim,
                ),
                device=device,
                dtype=kv_dtype,
            )
            ori_kv = ori_kv.reshape(
                batch * ori_blocks_per_batch,
                ori_block_size,
                kv_heads,
                kv_head_dim,
            )
            ori_block_table = self._tensor(
                torch,
                tuple(range(batch * ori_blocks_per_batch)),
                device=device,
                dtype=torch.int32,
            ).reshape(batch, ori_blocks_per_batch)
            cu_seqlens_ori_kv = self._tensor(
                torch,
                tuple(index * ori_context_tokens for index in range(batch + 1)),
                device=device,
                dtype=torch.int32,
            )
        cmp_kv = self._tensor(
            torch,
            self._materialized_kv_values_as_bthd(
                payload["keys"],
                batch=batch,
                kv_heads=kv_heads,
                context_tokens=context_tokens,
                kept_context_tokens=cmp_context_tokens,
                logical_dim=head_dim,
                physical_dim=kv_head_dim,
            ),
            device=device,
            dtype=kv_dtype,
        )
        cmp_kv = cmp_kv.reshape(
            batch * cmp_blocks_per_batch, cmp_block_size, kv_heads, kv_head_dim
        )
        cmp_indices = payload["indices"]
        if a5_physical_layout:
            cmp_indices = tuple(index % cmp_context_tokens for index in cmp_indices)
        cmp_sparse_indices = self._tensor(
            torch,
            cmp_indices,
            device=device,
            dtype=torch.int32,
        ).reshape(batch * query_tokens, kv_heads, selected_tokens)
        cmp_block_table = self._tensor(
            torch,
            tuple(range(batch * cmp_blocks_per_batch)),
            device=device,
            dtype=torch.int32,
        ).reshape(batch, cmp_blocks_per_batch)
        cu_seqlens_q = self._tensor(
            torch,
            tuple(index * query_tokens for index in range(batch + 1)),
            device=device,
            dtype=torch.int32,
        )
        cu_seqlens_cmp_kv = self._tensor(
            torch,
            tuple(index * cmp_context_tokens for index in range(batch + 1)),
            device=device,
            dtype=torch.int32,
        )
        seqused_kv = self._tensor(
            torch,
            tuple(context_tokens for _ in range(batch)),
            device=device,
            dtype=torch.int32,
        )
        sinks = None
        if a5_physical_layout:
            sinks = self._tensor(
                torch,
                tuple(0.0 for _ in range(query_heads)),
                device=device,
                dtype=scale_dtype,
            ).reshape((query_heads,))
        softmax_scale = head_dim ** -0.5

        metadata = metadata_op(
            num_heads_q=query_heads,
            num_heads_kv=kv_heads,
            head_dim=head_dim,
            kv_quant_mode=1,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_ori_kv=cu_seqlens_ori_kv,
            cu_seqlens_cmp_kv=cu_seqlens_cmp_kv,
            seqused_q=None,
            seqused_kv=seqused_kv,
            batch_size=batch,
            max_seqlen_q=query_tokens,
            max_seqlen_kv=context_tokens,
            ori_topk=0,
            cmp_topk=selected_tokens,
            tile_size=64,
            rope_head_dim=64,
            cmp_ratio=cmp_ratio,
            ori_mask_mode=4,
            cmp_mask_mode=3,
            ori_win_left=127,
            ori_win_right=0,
            layout_q="TND",
            layout_kv="PA_ND",
            has_ori_kv=a5_physical_layout,
            has_cmp_kv=True,
            device=str(device),
        )

        def operator():
            return attention_op(
                query,
                kv_quant_mode=1,
                ori_kv=ori_kv,
                cmp_kv=cmp_kv,
                ori_sparse_indices=None,
                cmp_sparse_indices=cmp_sparse_indices,
                ori_block_table=ori_block_table,
                cmp_block_table=cmp_block_table,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_ori_kv=None,
                cu_seqlens_cmp_kv=None,
                seqused_q=None,
                seqused_kv=seqused_kv,
                sinks=sinks,
                metadata=metadata,
                tile_size=64,
                rope_head_dim=64,
                softmax_scale=softmax_scale,
                cmp_ratio=cmp_ratio,
                ori_mask_mode=4,
                cmp_mask_mode=3,
                ori_win_left=127,
                ori_win_right=0,
                layout_q="TND",
                layout_kv="PA_ND",
                return_softmax_lse=True,
            )[0]

        return operator

    def _before_run_operator(self, request: OperatorBenchmarkRequest) -> None:
        if request.implementation == "simt":
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
        if request.implementation == "simt":
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
