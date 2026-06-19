from __future__ import annotations

from cannbench.backends.base import OperatorBackend
from cannbench.core.config import OperatorBenchmarkRequest
from cannbench.core.cuda_events import CudaEventProfileResult
from cannbench.core.operator_output import CapturedOperatorOutput
from cannbench.core.result import (
    OperatorBenchmarkResult,
    build_softmax_case,
    OperatorCase,
)
from cannbench.datasets import get_operator_case
from cannbench.datasets.materialize import (
    materialize_embedding_inputs,
    materialize_gather_inputs,
    materialize_index_select_inputs,
    materialize_index_add_inputs,
    materialize_index_put_inputs,
    materialize_masked_select_inputs,
    materialize_cross_entropy_inputs,
    materialize_scatter_add_inputs,
    materialize_scatter_inputs,
    materialize_softmax_inputs,
    materialize_take_along_dim_inputs,
    materialized_values_to_buffer,
)
from cannbench.operators import get_operator_spec


class TorchOperatorBackend(OperatorBackend):
    def __init__(self, *, name: str, device_type: str) -> None:
        super().__init__(name=name, device_type=device_type)

    def _build_result_case(self, request: OperatorBenchmarkRequest, case) -> OperatorCase:
        if request.op == "softmax":
            return build_softmax_case(
                case_id=request.case_id,
                family=request.family,
                dimensions=request.dimensions,
                dim=request.dim,
                source_kind=request.source_kind,
                source_project=request.source_project,
                source_model=request.source_model,
                source_file=request.source_file,
                source_op=request.source_op,
            )
        return OperatorCase(
            case_id=case.case_id,
            family=case.family,
            source_kind=case.source_kind,
            source_project=case.source_project,
            source_model=case.source_model,
            source_file=case.source_file,
            source_op=case.source_op,
            payload=case.payload,
        )

    def _torch_module(self):
        try:
            import torch
        except ModuleNotFoundError as exc:
            raise RuntimeError(f"PyTorch is required for the {self.name} backend") from exc
        return torch

    def _device_module(self, torch):
        return getattr(torch, self.device_type)

    def _is_available(self, torch) -> bool:
        return self._device_module(torch).is_available()

    def _synchronize(self, torch) -> None:
        self._device_module(torch).synchronize()

    def _device_name(self, torch, device) -> str:
        return self._device_module(torch).get_device_name(device)

    def _device(self, torch):
        return torch.device(self.device_type)

    def _tensor(self, torch, values, *, device, dtype):
        return torch.tensor(values, device=device, dtype=dtype)

    def _captured_output_from_tensor(
        self,
        *,
        torch,
        request: OperatorBenchmarkRequest,
        device,
        output,
    ) -> CapturedOperatorOutput:
        if hasattr(output, "detach"):
            output = output.detach()
        if hasattr(output, "cpu"):
            output = output.cpu()
        if hasattr(output, "to"):
            output = output.to(dtype=torch.float32)
        shape = tuple(int(value) for value in getattr(output, "shape", ()))
        if hasattr(output, "flatten"):
            flat = output.flatten()
        else:
            flat = output.reshape(-1)
        values = tuple(float(value) for value in flat.tolist())
        return CapturedOperatorOutput(
            backend=self.name,
            device_name=self._device_name(torch, device),
            op=request.op,
            dtype=request.dtype,
            dataset=request.dataset,
            case_id=request.case_id,
            seed=request.seed,
            shape=shape,
            values=values,
        )

    def _capture_operator_tensor(self, torch, request, case, *, device, dtype):
        if request.op == "softmax":
            payload = materialize_softmax_inputs(case, dtype=request.dtype, seed=request.seed)
            tensor = self._tensor(
                torch,
                materialized_values_to_buffer(payload["values"]),
                device=device,
                dtype=dtype,
            ).reshape(payload["shape"])
            return torch.softmax(tensor, dim=request.dim)

        if request.op == "embedding":
            payload = materialize_embedding_inputs(case, dtype=request.dtype, seed=request.seed)
            weights = self._tensor(
                torch,
                materialized_values_to_buffer(payload["weights"]),
                device=device,
                dtype=dtype,
            ).reshape(payload["num_embeddings"], payload["embedding_dim"])
            indices = self._tensor(
                torch,
                payload["indices"],
                device=device,
                dtype=torch.long,
            ).reshape(payload["index_shape"])
            module = torch.nn.Embedding(
                payload["num_embeddings"],
                payload["embedding_dim"],
                device=device,
                dtype=dtype,
            )
            module.weight = weights
            return module(indices)

        if request.op == "gather":
            payload = materialize_gather_inputs(case, dtype=request.dtype, seed=request.seed)
            input_tensor = self._tensor(
                torch,
                materialized_values_to_buffer(payload["values"]),
                device=device,
                dtype=dtype,
            ).reshape(payload["input_shape"])
            index_tensor = self._tensor(
                torch,
                payload["indices"],
                device=device,
                dtype=torch.long,
            ).reshape(payload["index_shape"])
            return torch.gather(input_tensor, payload["dim"], index_tensor)

        if request.op == "index_select":
            payload = materialize_index_select_inputs(
                case, dtype=request.dtype, seed=request.seed
            )
            input_tensor = self._tensor(
                torch,
                materialized_values_to_buffer(payload["values"]),
                device=device,
                dtype=dtype,
            ).reshape(payload["input_shape"])
            index_tensor = self._tensor(
                torch,
                payload["indices"],
                device=device,
                dtype=torch.long,
            ).reshape(payload["index_shape"])
            return torch.index_select(input_tensor, payload["dim"], index_tensor)

        if request.op == "index_add":
            payload = materialize_index_add_inputs(case, dtype=request.dtype, seed=request.seed)
            input_tensor = self._tensor(
                torch,
                materialized_values_to_buffer(payload["values"]),
                device=device,
                dtype=dtype,
            ).reshape(payload["input_shape"])
            index_tensor = self._tensor(
                torch,
                payload["indices"],
                device=device,
                dtype=torch.long,
            ).reshape(payload["index_shape"])
            src_tensor = self._tensor(
                torch,
                materialized_values_to_buffer(payload["src"]),
                device=device,
                dtype=dtype,
            ).reshape(payload["src_shape"])
            return torch.index_add(input_tensor, payload["dim"], index_tensor, src_tensor)

        if request.op == "index_put":
            payload = materialize_index_put_inputs(case, dtype=request.dtype, seed=request.seed)
            input_tensor = self._tensor(
                torch,
                materialized_values_to_buffer(payload["values"]),
                device=device,
                dtype=dtype,
            ).reshape(payload["input_shape"])
            index_tensors = tuple(
                self._tensor(
                    torch,
                    index_values,
                    device=device,
                    dtype=torch.long,
                ).reshape(index_shape)
                for index_values, index_shape in zip(
                    payload["indices"], payload["index_shapes"]
                )
            )
            values_tensor = self._tensor(
                torch,
                materialized_values_to_buffer(payload["put_values"]),
                device=device,
                dtype=dtype,
            ).reshape(payload["values_shape"])
            return torch.index_put(
                input_tensor,
                index_tensors,
                values_tensor,
                accumulate=payload["accumulate"],
            )

        if request.op == "take_along_dim":
            payload = materialize_take_along_dim_inputs(
                case, dtype=request.dtype, seed=request.seed
            )
            input_tensor = self._tensor(
                torch,
                materialized_values_to_buffer(payload["values"]),
                device=device,
                dtype=dtype,
            ).reshape(payload["input_shape"])
            index_tensor = self._tensor(
                torch,
                payload["indices"],
                device=device,
                dtype=torch.long,
            ).reshape(payload["index_shape"])
            return torch.take_along_dim(input_tensor, index_tensor, payload["dim"])

        if request.op == "masked_select":
            payload = materialize_masked_select_inputs(
                case, dtype=request.dtype, seed=request.seed
            )
            input_tensor = self._tensor(
                torch,
                materialized_values_to_buffer(payload["values"]),
                device=device,
                dtype=dtype,
            ).reshape(payload["input_shape"])
            mask_tensor = self._tensor(
                torch,
                payload["mask"],
                device=device,
                dtype=torch.bool,
            ).reshape(payload["mask_shape"])
            return torch.masked_select(input_tensor, mask_tensor)

        if request.op == "cross_entropy":
            payload = materialize_cross_entropy_inputs(
                case, dtype=request.dtype, seed=request.seed
            )
            logits = self._tensor(
                torch,
                materialized_values_to_buffer(payload["logits"]),
                device=device,
                dtype=dtype,
            ).reshape(payload["logits_shape"])
            targets = self._tensor(
                torch,
                payload["targets"],
                device=device,
                dtype=torch.long,
            ).reshape(payload["target_shape"])
            return torch.nn.functional.cross_entropy(logits, targets)

        if request.op == "scatter_add":
            payload = materialize_scatter_add_inputs(
                case, dtype=request.dtype, seed=request.seed
            )
            input_tensor = self._tensor(
                torch,
                materialized_values_to_buffer(payload["values"]),
                device=device,
                dtype=dtype,
            ).reshape(payload["input_shape"])
            index_tensor = self._tensor(
                torch,
                payload["indices"],
                device=device,
                dtype=torch.long,
            ).reshape(payload["index_shape"])
            src_tensor = self._tensor(
                torch,
                materialized_values_to_buffer(payload["src"]),
                device=device,
                dtype=dtype,
            ).reshape(payload["src_shape"])
            return torch.scatter_add(input_tensor, payload["dim"], index_tensor, src_tensor)

        if request.op == "scatter":
            payload = materialize_scatter_inputs(case, dtype=request.dtype, seed=request.seed)
            input_tensor = self._tensor(
                torch,
                materialized_values_to_buffer(payload["values"]),
                device=device,
                dtype=dtype,
            ).reshape(payload["input_shape"])
            index_tensor = self._tensor(
                torch,
                payload["indices"],
                device=device,
                dtype=torch.long,
            ).reshape(payload["index_shape"])
            src_tensor = self._tensor(
                torch,
                materialized_values_to_buffer(payload["src"]),
                device=device,
                dtype=dtype,
            ).reshape(payload["src_shape"])
            return torch.scatter(input_tensor, payload["dim"], index_tensor, src_tensor)

        raise RuntimeError(f"Unsupported operator for {self.name} output capture: {request.op}")

    def capture_operator_output(
        self, request: OperatorBenchmarkRequest
    ) -> CapturedOperatorOutput:
        self.validate_request(request)
        self._before_run_operator(request)
        spec = get_operator_spec(request.op)
        torch = self._torch_module()
        if not self._is_available(torch):
            raise RuntimeError(self._availability_error())
        if request.dtype not in spec.supported_dtypes:
            raise RuntimeError(f"Unsupported dtype for {request.op}: {request.dtype}")

        device = self._device(torch)
        dtype = getattr(torch, request.dtype)
        case = get_operator_case(request.op, request.dataset, request.case_id)
        output = self._capture_operator_tensor(
            torch,
            request,
            case,
            device=device,
            dtype=dtype,
        )
        self._synchronize(torch)
        return self._captured_output_from_tensor(
            torch=torch,
            request=request,
            device=device,
            output=output,
        )

    def _availability_error(self) -> str:
        return f"{self.device_type.upper()} is required for the {self.name} backend"

    def _before_run_operator(self, request: OperatorBenchmarkRequest) -> None:
        del request

    def run_operator(self, request: OperatorBenchmarkRequest) -> OperatorBenchmarkResult:
        self.validate_request(request)
        self._before_run_operator(request)
        spec = get_operator_spec(request.op)
        torch = self._torch_module()
        if not self._is_available(torch):
            raise RuntimeError(self._availability_error())

        device = self._device(torch)
        dtype = getattr(torch, request.dtype)
        if request.dtype not in spec.supported_dtypes:
            raise RuntimeError(f"Unsupported dtype for {request.op}: {request.dtype}")
        case = get_operator_case(request.op, request.dataset, request.case_id)

        if request.op == "softmax":
            payload = materialize_softmax_inputs(case, dtype=request.dtype, seed=request.seed)
            tensor = self._tensor(
                torch,
                materialized_values_to_buffer(payload["values"]),
                device=device,
                dtype=dtype,
            ).reshape(payload["shape"])

            for _ in range(request.warmup):
                torch.softmax(tensor, dim=request.dim)
            self._synchronize(torch)

            for _ in range(request.iterations):
                torch.softmax(tensor, dim=request.dim)
                self._synchronize(torch)
            return OperatorBenchmarkResult(
                backend=self.name,
                device_name=self._device_name(torch, device),
                op=request.op,
                dtype=request.dtype,
                case=self._build_result_case(request, case),
                warmup=request.warmup,
                iterations=request.iterations,
            )

        if request.op == "embedding":
            payload = materialize_embedding_inputs(case, dtype=request.dtype, seed=request.seed)
            weights = self._tensor(
                torch,
                materialized_values_to_buffer(payload["weights"]),
                device=device,
                dtype=dtype,
            ).reshape(payload["num_embeddings"], payload["embedding_dim"])
            indices = self._tensor(
                torch,
                payload["indices"],
                device=device,
                dtype=torch.long,
            ).reshape(payload["index_shape"])
            module = torch.nn.Embedding(
                payload["num_embeddings"],
                payload["embedding_dim"],
                device=device,
                dtype=dtype,
            )
            module.weight = weights

            for _ in range(request.warmup):
                module(indices)
            self._synchronize(torch)

            for _ in range(request.iterations):
                module(indices)
                self._synchronize(torch)
            return OperatorBenchmarkResult(
                backend=self.name,
                device_name=self._device_name(torch, device),
                op=request.op,
                dtype=request.dtype,
                case=self._build_result_case(request, case),
                warmup=request.warmup,
                iterations=request.iterations,
            )

        if request.op == "gather":
            payload = materialize_gather_inputs(case, dtype=request.dtype, seed=request.seed)
            input_tensor = self._tensor(
                torch,
                materialized_values_to_buffer(payload["values"]),
                device=device,
                dtype=dtype,
            ).reshape(payload["input_shape"])
            index_tensor = self._tensor(
                torch,
                payload["indices"],
                device=device,
                dtype=torch.long,
            ).reshape(payload["index_shape"])

            for _ in range(request.warmup):
                torch.gather(input_tensor, payload["dim"], index_tensor)
            self._synchronize(torch)

            for _ in range(request.iterations):
                torch.gather(input_tensor, payload["dim"], index_tensor)
                self._synchronize(torch)
            return OperatorBenchmarkResult(
                backend=self.name,
                device_name=self._device_name(torch, device),
                op=request.op,
                dtype=request.dtype,
                case=self._build_result_case(request, case),
                warmup=request.warmup,
                iterations=request.iterations,
            )

        if request.op == "index_select":
            payload = materialize_index_select_inputs(
                case, dtype=request.dtype, seed=request.seed
            )
            input_tensor = self._tensor(
                torch,
                materialized_values_to_buffer(payload["values"]),
                device=device,
                dtype=dtype,
            ).reshape(payload["input_shape"])
            index_tensor = self._tensor(
                torch,
                payload["indices"],
                device=device,
                dtype=torch.long,
            ).reshape(payload["index_shape"])

            for _ in range(request.warmup):
                torch.index_select(input_tensor, payload["dim"], index_tensor)
            self._synchronize(torch)

            for _ in range(request.iterations):
                torch.index_select(input_tensor, payload["dim"], index_tensor)
                self._synchronize(torch)
            return OperatorBenchmarkResult(
                backend=self.name,
                device_name=self._device_name(torch, device),
                op=request.op,
                dtype=request.dtype,
                case=self._build_result_case(request, case),
                warmup=request.warmup,
                iterations=request.iterations,
            )

        if request.op == "index_add":
            payload = materialize_index_add_inputs(case, dtype=request.dtype, seed=request.seed)
            input_tensor = self._tensor(
                torch,
                materialized_values_to_buffer(payload["values"]),
                device=device,
                dtype=dtype,
            ).reshape(payload["input_shape"])
            index_tensor = self._tensor(
                torch,
                payload["indices"],
                device=device,
                dtype=torch.long,
            ).reshape(payload["index_shape"])
            src_tensor = self._tensor(
                torch,
                materialized_values_to_buffer(payload["src"]),
                device=device,
                dtype=dtype,
            ).reshape(payload["src_shape"])

            for _ in range(request.warmup):
                torch.index_add(input_tensor, payload["dim"], index_tensor, src_tensor)
            self._synchronize(torch)

            for _ in range(request.iterations):
                torch.index_add(input_tensor, payload["dim"], index_tensor, src_tensor)
                self._synchronize(torch)
            return OperatorBenchmarkResult(
                backend=self.name,
                device_name=self._device_name(torch, device),
                op=request.op,
                dtype=request.dtype,
                case=self._build_result_case(request, case),
                warmup=request.warmup,
                iterations=request.iterations,
            )

        if request.op == "index_put":
            payload = materialize_index_put_inputs(case, dtype=request.dtype, seed=request.seed)
            input_tensor = self._tensor(
                torch,
                materialized_values_to_buffer(payload["values"]),
                device=device,
                dtype=dtype,
            ).reshape(payload["input_shape"])
            index_tensors = tuple(
                self._tensor(
                    torch,
                    index_values,
                    device=device,
                    dtype=torch.long,
                ).reshape(index_shape)
                for index_values, index_shape in zip(
                    payload["indices"], payload["index_shapes"]
                )
            )
            values_tensor = self._tensor(
                torch,
                materialized_values_to_buffer(payload["put_values"]),
                device=device,
                dtype=dtype,
            ).reshape(payload["values_shape"])

            for _ in range(request.warmup):
                torch.index_put(
                    input_tensor,
                    index_tensors,
                    values_tensor,
                    accumulate=payload["accumulate"],
                )
            self._synchronize(torch)

            for _ in range(request.iterations):
                torch.index_put(
                    input_tensor,
                    index_tensors,
                    values_tensor,
                    accumulate=payload["accumulate"],
                )
                self._synchronize(torch)
            return OperatorBenchmarkResult(
                backend=self.name,
                device_name=self._device_name(torch, device),
                op=request.op,
                dtype=request.dtype,
                case=self._build_result_case(request, case),
                warmup=request.warmup,
                iterations=request.iterations,
            )

        if request.op == "take_along_dim":
            payload = materialize_take_along_dim_inputs(
                case, dtype=request.dtype, seed=request.seed
            )
            input_tensor = self._tensor(
                torch,
                materialized_values_to_buffer(payload["values"]),
                device=device,
                dtype=dtype,
            ).reshape(payload["input_shape"])
            index_tensor = self._tensor(
                torch,
                payload["indices"],
                device=device,
                dtype=torch.long,
            ).reshape(payload["index_shape"])

            for _ in range(request.warmup):
                torch.take_along_dim(input_tensor, index_tensor, payload["dim"])
            self._synchronize(torch)

            for _ in range(request.iterations):
                torch.take_along_dim(input_tensor, index_tensor, payload["dim"])
                self._synchronize(torch)
            return OperatorBenchmarkResult(
                backend=self.name,
                device_name=self._device_name(torch, device),
                op=request.op,
                dtype=request.dtype,
                case=self._build_result_case(request, case),
                warmup=request.warmup,
                iterations=request.iterations,
            )

        if request.op == "masked_select":
            payload = materialize_masked_select_inputs(
                case, dtype=request.dtype, seed=request.seed
            )
            input_tensor = self._tensor(
                torch,
                materialized_values_to_buffer(payload["values"]),
                device=device,
                dtype=dtype,
            ).reshape(payload["input_shape"])
            mask_tensor = self._tensor(
                torch,
                payload["mask"],
                device=device,
                dtype=torch.bool,
            ).reshape(payload["mask_shape"])

            for _ in range(request.warmup):
                torch.masked_select(input_tensor, mask_tensor)
            self._synchronize(torch)

            for _ in range(request.iterations):
                torch.masked_select(input_tensor, mask_tensor)
                self._synchronize(torch)
            return OperatorBenchmarkResult(
                backend=self.name,
                device_name=self._device_name(torch, device),
                op=request.op,
                dtype=request.dtype,
                case=self._build_result_case(request, case),
                warmup=request.warmup,
                iterations=request.iterations,
            )

        if request.op == "cross_entropy":
            payload = materialize_cross_entropy_inputs(
                case, dtype=request.dtype, seed=request.seed
            )
            logits = self._tensor(
                torch,
                materialized_values_to_buffer(payload["logits"]),
                device=device,
                dtype=dtype,
            ).reshape(payload["logits_shape"])
            targets = self._tensor(
                torch,
                payload["targets"],
                device=device,
                dtype=torch.long,
            ).reshape(payload["target_shape"])

            for _ in range(request.warmup):
                torch.nn.functional.cross_entropy(logits, targets)
            self._synchronize(torch)

            for _ in range(request.iterations):
                torch.nn.functional.cross_entropy(logits, targets)
                self._synchronize(torch)
            return OperatorBenchmarkResult(
                backend=self.name,
                device_name=self._device_name(torch, device),
                op=request.op,
                dtype=request.dtype,
                case=self._build_result_case(request, case),
                warmup=request.warmup,
                iterations=request.iterations,
            )

        if request.op == "scatter_add":
            payload = materialize_scatter_add_inputs(
                case, dtype=request.dtype, seed=request.seed
            )
            input_tensor = self._tensor(
                torch,
                materialized_values_to_buffer(payload["values"]),
                device=device,
                dtype=dtype,
            ).reshape(payload["input_shape"])
            index_tensor = self._tensor(
                torch,
                payload["indices"],
                device=device,
                dtype=torch.long,
            ).reshape(payload["index_shape"])
            src_tensor = self._tensor(
                torch,
                materialized_values_to_buffer(payload["src"]),
                device=device,
                dtype=dtype,
            ).reshape(payload["src_shape"])

            for _ in range(request.warmup):
                torch.scatter_add(input_tensor, payload["dim"], index_tensor, src_tensor)
            self._synchronize(torch)

            for _ in range(request.iterations):
                torch.scatter_add(input_tensor, payload["dim"], index_tensor, src_tensor)
                self._synchronize(torch)
            return OperatorBenchmarkResult(
                backend=self.name,
                device_name=self._device_name(torch, device),
                op=request.op,
                dtype=request.dtype,
                case=self._build_result_case(request, case),
                warmup=request.warmup,
                iterations=request.iterations,
            )

        if request.op == "scatter":
            payload = materialize_scatter_inputs(case, dtype=request.dtype, seed=request.seed)
            input_tensor = self._tensor(
                torch,
                materialized_values_to_buffer(payload["values"]),
                device=device,
                dtype=dtype,
            ).reshape(payload["input_shape"])
            index_tensor = self._tensor(
                torch,
                payload["indices"],
                device=device,
                dtype=torch.long,
            ).reshape(payload["index_shape"])
            src_tensor = self._tensor(
                torch,
                materialized_values_to_buffer(payload["src"]),
                device=device,
                dtype=dtype,
            ).reshape(payload["src_shape"])

            for _ in range(request.warmup):
                torch.scatter(input_tensor, payload["dim"], index_tensor, src_tensor)
            self._synchronize(torch)

            for _ in range(request.iterations):
                torch.scatter(input_tensor, payload["dim"], index_tensor, src_tensor)
                self._synchronize(torch)
            return OperatorBenchmarkResult(
                backend=self.name,
                device_name=self._device_name(torch, device),
                op=request.op,
                dtype=request.dtype,
                case=self._build_result_case(request, case),
                warmup=request.warmup,
                iterations=request.iterations,
            )

        raise RuntimeError(f"Unsupported operator for {self.name} backend: {request.op}")

    def profile_operator_with_cuda_events(
        self, request: OperatorBenchmarkRequest
    ) -> CudaEventProfileResult:
        if self.device_type != "cuda":
            raise RuntimeError("CUDA event profiling is only supported for CUDA backends")
        self.validate_request(request)
        self._before_run_operator(request)
        spec = get_operator_spec(request.op)
        torch = self._torch_module()
        if not self._is_available(torch):
            raise RuntimeError(self._availability_error())

        device = self._device(torch)
        dtype = getattr(torch, request.dtype)
        if request.dtype not in spec.supported_dtypes:
            raise RuntimeError(f"Unsupported dtype for {request.op}: {request.dtype}")
        case = get_operator_case(request.op, request.dataset, request.case_id)
        if request.op != "softmax":
            raise RuntimeError(
                f"CUDA event profiling is not implemented for {request.op}"
            )

        payload = materialize_softmax_inputs(case, dtype=request.dtype, seed=request.seed)
        tensor = self._tensor(
            torch,
            materialized_values_to_buffer(payload["values"]),
            device=device,
            dtype=dtype,
        ).reshape(payload["shape"])

        output = None
        for _ in range(request.warmup):
            output = torch.softmax(tensor, dim=request.dim)
        self._synchronize(torch)

        durations: list[float] = []
        for _ in range(request.iterations):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            output = torch.softmax(tensor, dim=request.dim)
            end.record()
            self._synchronize(torch)
            durations.append(float(start.elapsed_time(end)))

        # Keep the final output live until after synchronization.
        del output
        benchmark_result = OperatorBenchmarkResult(
            backend=self.name,
            device_name=self._device_name(torch, device),
            op=request.op,
            dtype=request.dtype,
            case=self._build_result_case(request, case),
            warmup=request.warmup,
            iterations=request.iterations,
        )
        return CudaEventProfileResult(
            benchmark_result=benchmark_result,
            durations_ms=tuple(durations),
        )
