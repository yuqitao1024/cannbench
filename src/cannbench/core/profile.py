from __future__ import annotations

import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

from cannbench.core.result import OperatorBenchmarkResult
from cannbench.core.timing import summarize_timings_ms

NVIDIA_TIME_DURATION_AVG = "gpu__time_duration.avg"


@dataclass(frozen=True)
class DeviceProfileSummary:
    backend: str
    sample_count: int
    latency_ms_avg: float
    latency_ms_p50: float
    latency_ms_p95: float
    latency_ms_p99: float
    source_files: tuple[str, ...]

    def to_json_dict(self) -> dict[str, object]:
        return {
            "backend": self.backend,
            "sample_count": self.sample_count,
            "latency_ms_avg": self.latency_ms_avg,
            "latency_ms_p50": self.latency_ms_p50,
            "latency_ms_p95": self.latency_ms_p95,
            "latency_ms_p99": self.latency_ms_p99,
            "source_files": list(self.source_files),
        }


@dataclass(frozen=True)
class ProfileArtifacts:
    device_name: str
    profile_summary: DeviceProfileSummary
    profile_artifacts: tuple[tuple[str, bytes], ...]
    perf_artifacts: tuple[tuple[str, bytes], ...]


@dataclass(frozen=True)
class LocalDeviceProfileResult:
    benchmark_result: OperatorBenchmarkResult
    profile: ProfileArtifacts


def write_profile_artifacts(
    profile_dir: Path,
    artifacts: tuple[tuple[str, bytes], ...],
) -> tuple[Path, ...]:
    profile_dir.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []
    for relative_name, content in artifacts:
        path = profile_dir / relative_name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        created.append(path)
    return tuple(created)


def _unit_from_text(text: str) -> str:
    lowered = text.lower()
    match = re.search(r"\(([^)]+)\)", lowered)
    if match:
        return match.group(1)
    if "nsecond" in lowered or lowered in {"ns", "nanosecond", "nanoseconds"}:
        return "ns"
    if "usecond" in lowered or lowered in {"us", "microsecond", "microseconds"}:
        return "us"
    if "msecond" in lowered or lowered in {"ms", "millisecond", "milliseconds"}:
        return "ms"
    if "second" in lowered or lowered in {"s", "sec"}:
        return "s"
    return "ms"


def _to_ms(value: float, unit: str) -> float:
    normalized = unit.strip().lower()
    if normalized in {"ns", "nsecond", "nanosecond", "nanoseconds"}:
        return value / 1_000_000.0
    if normalized in {"us", "usecond", "microsecond", "microseconds"}:
        return value / 1_000.0
    if normalized in {"ms", "msecond", "millisecond", "milliseconds"}:
        return value
    if normalized in {"s", "sec", "second", "seconds"}:
        return value * 1000.0
    return value


def _parse_float(value: object) -> float | None:
    try:
        parsed = float(str(value).replace(",", "").strip())
    except ValueError:
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def _duration_from_metric_row(row: dict[str, str], *, backend: str) -> float | None:
    lower = {key.strip().lower(): value for key, value in row.items()}
    metric_name = lower.get("metric name") or lower.get("metric")
    metric_value = lower.get("metric value") or lower.get("value")
    if not metric_name or metric_value is None:
        return None
    normalized_metric = metric_name.strip().lower()
    if backend == "nvidia":
        if normalized_metric != NVIDIA_TIME_DURATION_AVG:
            return None
    elif "duration" not in normalized_metric:
        return None
    parsed = _parse_float(metric_value)
    if parsed is None:
        return None
    unit = lower.get("metric unit") or lower.get("unit") or _unit_from_text(metric_name)
    return _to_ms(parsed, unit)


def _duration_from_wide_row(row: dict[str, str], *, backend: str, units: dict[str, str] | None = None) -> float | None:
    if backend == "nvidia":
        value = row.get(NVIDIA_TIME_DURATION_AVG)
        if value is None:
            return None
        parsed = _parse_float(value)
        if parsed is None:
            return None
        unit = (units or {}).get(NVIDIA_TIME_DURATION_AVG, _unit_from_text(NVIDIA_TIME_DURATION_AVG))
        return _to_ms(parsed, unit)

    for key, value in row.items():
        lowered = key.strip().lower()
        if "duration" not in lowered and "elapsed" not in lowered:
            continue
        parsed = _parse_float(value)
        if parsed is None:
            continue
        return _to_ms(parsed, _unit_from_text(key))
    return None


def _looks_like_ncu_unit_row(row: dict[str, str]) -> bool:
    value = row.get(NVIDIA_TIME_DURATION_AVG)
    return value is not None and _parse_float(value) is None and _unit_from_text(value) != "ms"


def _read_csv_durations(path: Path, *, backend: str) -> list[float]:
    durations: list[float] = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        units: dict[str, str] = {}
        for row in reader:
            normalized_row = {key.strip().lower(): value for key, value in row.items() if key is not None}
            if backend == "nvidia" and _looks_like_ncu_unit_row(normalized_row):
                units = {key: value for key, value in normalized_row.items() if value}
                continue

            duration = _duration_from_metric_row(row, backend=backend)
            if duration is None:
                duration = _duration_from_wide_row(normalized_row, backend=backend, units=units)
            if duration is not None:
                durations.append(duration)
    return durations


def read_device_profile(profile_dir: Path, *, backend: str) -> DeviceProfileSummary:
    csv_files = sorted(profile_dir.rglob("*.csv"))
    samples: list[float] = []
    source_files: list[str] = []
    for csv_file in csv_files:
        durations = _read_csv_durations(csv_file, backend=backend)
        if durations:
            samples.extend(durations)
            source_files.append(str(csv_file.relative_to(profile_dir)))
    if not samples:
        raise ValueError(f"no duration samples found in profiler CSV files under {profile_dir}")
    summary = summarize_timings_ms(samples)
    return DeviceProfileSummary(
        backend=backend,
        sample_count=len(samples),
        latency_ms_avg=summary["latency_ms_avg"],
        latency_ms_p50=summary["latency_ms_p50"],
        latency_ms_p95=summary["latency_ms_p95"],
        latency_ms_p99=summary["latency_ms_p99"],
        source_files=tuple(source_files),
    )


def write_device_profile_summary(path: Path, summary: DeviceProfileSummary) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary.to_json_dict(), indent=2) + "\n")
    return path
