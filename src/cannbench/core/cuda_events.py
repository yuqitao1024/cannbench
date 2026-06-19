from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from cannbench.core.result import OperatorBenchmarkResult


@dataclass(frozen=True)
class CudaEventProfileResult:
    benchmark_result: OperatorBenchmarkResult
    durations_ms: tuple[float, ...]


def write_cuda_event_profile_csv(
    profile_dir: Path,
    result: CudaEventProfileResult,
) -> Path:
    profile_dir.mkdir(parents=True, exist_ok=True)
    path = profile_dir / "cuda-events.csv"
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Name", "Duration(ms)"])
        for duration in result.durations_ms:
            writer.writerow([result.benchmark_result.op, duration])
    return path
