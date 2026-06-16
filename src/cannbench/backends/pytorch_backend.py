import time

from cannbench.backends.base import OperatorBackend
from cannbench.core.config import OperatorBenchmarkRequest
from cannbench.core.result import (
    BenchmarkMetrics,
    OperatorBenchmarkResult,
    SoftmaxShape,
)
from cannbench.core.timing import summarize_timings_ms


class NvidiaBackend(OperatorBackend):
    def __init__(self) -> None:
        super().__init__(name="nvidia", device_type="cuda")

    def run_softmax(self, request: OperatorBenchmarkRequest) -> OperatorBenchmarkResult:
        self.validate_request(request)

        try:
            import torch
        except ModuleNotFoundError as exc:
            raise RuntimeError("PyTorch is required for the nvidia backend") from exc

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for the nvidia backend")

        device = torch.device(self.device_type)
        dtype = getattr(torch, request.dtype)
        tensor = torch.randn(request.dimensions, device=device, dtype=dtype)

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
            shape=SoftmaxShape(
                dimensions=request.dimensions,
                dim=request.dim,
                case_id=request.case_id,
                family=request.family,
                source_kind=request.source_kind,
                source_project=request.source_project,
                source_model=request.source_model,
                source_file=request.source_file,
                source_op=request.source_op,
            ),
            metrics=metrics,
        )


class AscendBackend(OperatorBackend):
    def __init__(self) -> None:
        super().__init__(name="ascend", device_type="npu")

    def run_softmax(self, request: OperatorBenchmarkRequest) -> OperatorBenchmarkResult:
        self.validate_request(request)
        raise RuntimeError(
            "Ascend backend implementation comes from the same PyTorch path and should be "
            "added by adapting synchronization and device discovery for torch_npu"
        )
