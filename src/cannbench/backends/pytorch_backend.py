import time

from cannbench.backends.base import OperatorBackend
from cannbench.core.config import OperatorBenchmarkRequest
from cannbench.core.result import (
    BenchmarkMetrics,
    OperatorBenchmarkResult,
    build_softmax_case,
    OperatorCase,
)
from cannbench.core.timing import summarize_timings_ms
from cannbench.datasets import get_operator_case
from cannbench.datasets.materialize import (
    materialize_embedding_inputs,
    materialize_softmax_inputs,
    materialized_values_to_buffer,
)
from cannbench.operators import get_operator_spec


class NvidiaBackend(OperatorBackend):
    def __init__(self) -> None:
        super().__init__(name="nvidia", device_type="cuda")

    def run_operator(self, request: OperatorBenchmarkRequest) -> OperatorBenchmarkResult:
        self.validate_request(request)
        spec = get_operator_spec(request.op)

        try:
            import torch
        except ModuleNotFoundError as exc:
            raise RuntimeError("PyTorch is required for the nvidia backend") from exc

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for the nvidia backend")

        device = torch.device(self.device_type)
        dtype = getattr(torch, request.dtype)
        if request.dtype not in spec.supported_dtypes:
            raise RuntimeError(f"Unsupported dtype for {request.op}: {request.dtype}")
        case = get_operator_case(request.op, request.dataset, request.case_id)

        if request.op == "softmax":
            payload = materialize_softmax_inputs(
                case,
                dtype=request.dtype,
                seed=request.seed,
            )
            tensor = torch.tensor(
                materialized_values_to_buffer(payload["values"]),
                device=device,
                dtype=dtype,
            )
            tensor = tensor.reshape(payload["shape"])

            for _ in range(request.warmup):
                torch.softmax(tensor, dim=request.dim)
            torch.cuda.synchronize()

            samples: list[float] = []
            for _ in range(request.iterations):
                started = time.perf_counter()
                torch.softmax(tensor, dim=request.dim)
                torch.cuda.synchronize()
                samples.append((time.perf_counter() - started) * 1000.0)

            summary = summarize_timings_ms(samples)
            metrics = BenchmarkMetrics(
                iterations=request.iterations,
                warmup=request.warmup,
                latency_ms_avg=summary["latency_ms_avg"],
                latency_ms_p50=summary["latency_ms_p50"],
                latency_ms_p95=summary["latency_ms_p95"],
                latency_ms_p99=summary["latency_ms_p99"],
                throughput_ops_per_sec=request.iterations / (sum(samples) / 1000.0),
            )
            return OperatorBenchmarkResult(
                backend=self.name,
                device_name=torch.cuda.get_device_name(device),
                op=request.op,
                dtype=request.dtype,
                case=build_softmax_case(
                    case_id=request.case_id,
                    family=request.family,
                    dimensions=request.dimensions,
                    dim=request.dim,
                    source_kind=request.source_kind,
                    source_project=request.source_project,
                    source_model=request.source_model,
                    source_file=request.source_file,
                    source_op=request.source_op,
                ),
                metrics=metrics,
            )

        if request.op == "embedding":
            payload = materialize_embedding_inputs(
                case,
                dtype=request.dtype,
                seed=request.seed,
            )
            weights = torch.tensor(
                materialized_values_to_buffer(payload["weights"]),
                device=device,
                dtype=dtype,
            ).reshape(payload["num_embeddings"], payload["embedding_dim"])
            indices = torch.tensor(
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
            torch.cuda.synchronize()

            samples: list[float] = []
            for _ in range(request.iterations):
                started = time.perf_counter()
                module(indices)
                torch.cuda.synchronize()
                samples.append((time.perf_counter() - started) * 1000.0)

            summary = summarize_timings_ms(samples)
            metrics = BenchmarkMetrics(
                iterations=request.iterations,
                warmup=request.warmup,
                latency_ms_avg=summary["latency_ms_avg"],
                latency_ms_p50=summary["latency_ms_p50"],
                latency_ms_p95=summary["latency_ms_p95"],
                latency_ms_p99=summary["latency_ms_p99"],
                throughput_ops_per_sec=request.iterations / (sum(samples) / 1000.0),
            )
            return OperatorBenchmarkResult(
                backend=self.name,
                device_name=torch.cuda.get_device_name(device),
                op=request.op,
                dtype=request.dtype,
                case=OperatorCase(
                    case_id=case.case_id,
                    family=case.family,
                    source_kind=case.source_kind,
                    source_project=case.source_project,
                    source_model=case.source_model,
                    source_file=case.source_file,
                    source_op=case.source_op,
                    payload=case.payload,
                ),
                metrics=metrics,
            )

        raise RuntimeError(f"Unsupported operator for nvidia backend: {request.op}")


class AscendBackend(OperatorBackend):
    def __init__(self) -> None:
        super().__init__(name="ascend", device_type="npu")

    def run_operator(self, request: OperatorBenchmarkRequest) -> OperatorBenchmarkResult:
        self.validate_request(request)
        raise RuntimeError(
            "Ascend backend implementation comes from the same PyTorch path and should be "
            "added by adapting synchronization and device discovery for torch_npu"
        )
