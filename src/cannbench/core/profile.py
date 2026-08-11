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
WORKFLOW_DEFAULT_LAUNCH_BUDGET = 64


@dataclass(frozen=True)
class ProfileKernelSelection:
    kernel_name_patterns: tuple[str, ...] = ()
    terminal_kernel_name_patterns: tuple[str, ...] = ()
    launch_count: int | None = None
    aggregate_across_files: bool = False
    nvtx_range: str | None = None


def workflow_profile_launch_count(
    selections: tuple[ProfileKernelSelection, ...],
) -> int:
    return sum(
        selection.launch_count or WORKFLOW_DEFAULT_LAUNCH_BUDGET
        for selection in selections
    )


def ncu_profile_options(
    kernel_selection: ProfileKernelSelection,
) -> tuple[str, ...]:
    if kernel_selection.nvtx_range:
        return (
            "--nvtx",
            "--nvtx-include",
            f"{kernel_selection.nvtx_range}/",
        )
    return ("--launch-count", str(kernel_selection.launch_count or 1))


@dataclass(frozen=True)
class DeviceProfileSummary:
    backend: str
    source_files: tuple[str, ...]
    latency_ms: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "latency_ms", float(self.latency_ms))

    def to_json_dict(self) -> dict[str, object]:
        return {
            "backend": self.backend,
            "latency_ms": self.latency_ms,
            "source_files": list(self.source_files),
        }


@dataclass(frozen=True)
class WorkflowProfileSummary:
    backend: str
    source_files: tuple[str, ...]
    latency_ms: float
    component_summaries: tuple[DeviceProfileSummary, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "latency_ms", float(self.latency_ms))

    @property
    def profile_summary(self) -> DeviceProfileSummary:
        return DeviceProfileSummary(
            backend=self.backend,
            source_files=self.source_files,
            latency_ms=self.latency_ms,
        )

    def to_json_dict(self) -> dict[str, object]:
        return {
            **self.profile_summary.to_json_dict(),
            "components": [
                component.to_json_dict() for component in self.component_summaries
            ],
        }


@dataclass(frozen=True)
class ProfileArtifacts:
    device_name: str
    profile_summary: DeviceProfileSummary
    profile_artifacts: tuple[tuple[str, bytes], ...]
    perf_artifacts: tuple[tuple[str, bytes], ...]
    component_summaries: tuple[DeviceProfileSummary, ...] = ()


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


def _kernel_name_from_row(row: dict[str, str]) -> str | None:
    lower = {key.strip().lower(): value for key, value in row.items()}
    for key in ("kernel name", "op name", "name"):
        value = lower.get(key)
        if value:
            return value.strip()
    return None


def _matches_kernel_name(kernel_name: str | None, patterns: tuple[str, ...]) -> bool:
    if not patterns:
        return True
    if not kernel_name:
        return False
    lowered = kernel_name.lower()
    return any(pattern.lower() in lowered for pattern in patterns)


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


@dataclass(frozen=True)
class _ProfileDurationRow:
    duration_ms: float
    kernel_name: str | None
    source_file: str


def _read_csv_duration_rows(path: Path, *, backend: str) -> list[tuple[float, str | None]]:
    durations: list[tuple[float, str | None]] = []
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
                durations.append((duration, _kernel_name_from_row(row)))
    return durations


def _read_profile_duration_rows(
    profile_dir: Path, *, backend: str
) -> list[_ProfileDurationRow]:
    rows: list[_ProfileDurationRow] = []
    for csv_file in sorted(profile_dir.rglob("*.csv")):
        source_file = str(csv_file.relative_to(profile_dir))
        rows.extend(
            _ProfileDurationRow(
                duration_ms=duration,
                kernel_name=kernel_name,
                source_file=source_file,
            )
            for duration, kernel_name in _read_csv_duration_rows(
                csv_file, backend=backend
            )
        )
    return rows


def _ordered_unique(values: list[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))


def read_workflow_profile(
    profile_dir: Path,
    *,
    backend: str,
    step_selections: tuple[ProfileKernelSelection, ...],
) -> WorkflowProfileSummary:
    if not step_selections:
        raise ValueError("workflow profile requires at least one step selection")

    rows = _read_profile_duration_rows(profile_dir, backend=backend)
    if not rows:
        raise ValueError(
            f"no duration samples found in profiler CSV files under {profile_dir}"
        )

    boundaries: list[int] = []
    for step_index, selection in enumerate(step_selections[:-1]):
        terminal_patterns = selection.terminal_kernel_name_patterns
        if not terminal_patterns:
            raise ValueError(
                f"non-final workflow step {step_index} requires terminal kernel "
                "name patterns"
            )
        matching_indices = [
            row_index
            for row_index, row in enumerate(rows)
            if _matches_kernel_name(row.kernel_name, terminal_patterns)
        ]
        if not matching_indices:
            expected = ", ".join(terminal_patterns)
            raise ValueError(
                f"non-final workflow step {step_index} has no terminal kernel "
                f"matching {expected!r}"
            )
        boundaries.append(matching_indices[-1])

    if any(current <= previous for previous, current in zip(boundaries, boundaries[1:])):
        raise ValueError("workflow terminal boundaries are out of workflow order")

    span_ends = (*boundaries, len(rows) - 1)
    span_start = 0
    component_summaries: list[DeviceProfileSummary] = []
    for step_index, (selection, span_end) in enumerate(
        zip(step_selections, span_ends, strict=True)
    ):
        if span_end < span_start:
            raise ValueError(f"workflow step {step_index} has an empty profile span")
        span = rows[span_start : span_end + 1]
        selected_rows = [
            row
            for row in span
            if _matches_kernel_name(row.kernel_name, selection.kernel_name_patterns)
        ]
        if not selected_rows:
            expected = ", ".join(selection.kernel_name_patterns)
            raise ValueError(
                f"workflow step {step_index} has no selected kernel rows "
                f"matching {expected!r}"
            )
        component_summaries.append(
            DeviceProfileSummary(
                backend=backend,
                latency_ms=sum(row.duration_ms for row in selected_rows),
                source_files=_ordered_unique(
                    [row.source_file for row in selected_rows]
                ),
            )
        )
        span_start = span_end + 1

    source_files = _ordered_unique(
        [
            source_file
            for component in component_summaries
            for source_file in component.source_files
        ]
    )
    return WorkflowProfileSummary(
        backend=backend,
        source_files=source_files,
        latency_ms=sum(
            component.latency_ms for component in component_summaries
        ),
        component_summaries=tuple(component_summaries),
    )


def read_device_profile(
    profile_dir: Path,
    *,
    backend: str,
    expected_kernel_name_patterns: tuple[str, ...] = (),
    kernel_selection: ProfileKernelSelection | None = None,
) -> DeviceProfileSummary:
    sum_matching_rows = kernel_selection is not None
    if kernel_selection is None:
        kernel_selection = ProfileKernelSelection(
            kernel_name_patterns=expected_kernel_name_patterns
        )
    csv_files = sorted(profile_dir.rglob("*.csv"))
    samples: list[float] = []
    source_files: list[str] = []
    observed_kernel_names: set[str] = set()
    for csv_file in csv_files:
        duration_rows = _read_csv_duration_rows(csv_file, backend=backend)
        observed_kernel_names.update(
            kernel_name for _, kernel_name in duration_rows if kernel_name
        )
        durations = [
            duration
            for duration, kernel_name in duration_rows
            if _matches_kernel_name(kernel_name, kernel_selection.kernel_name_patterns)
        ]
        if durations:
            if sum_matching_rows:
                samples.append(sum(durations))
            else:
                samples.extend(durations)
            source_files.append(str(csv_file.relative_to(profile_dir)))
    if not samples:
        if kernel_selection.kernel_name_patterns and observed_kernel_names:
            observed = ", ".join(sorted(observed_kernel_names))
            expected = ", ".join(kernel_selection.kernel_name_patterns)
            raise ValueError(
                "expected profiler kernel matching "
                f"{expected!r}, but observed: {observed}"
            )
        raise ValueError(f"no duration samples found in profiler CSV files under {profile_dir}")
    if kernel_selection.aggregate_across_files:
        samples = [sum(samples)]
    summary = summarize_timings_ms(samples)
    return DeviceProfileSummary(
        backend=backend,
        latency_ms=summary["latency_ms"],
        source_files=tuple(source_files),
    )


def write_device_profile_summary(path: Path, summary: DeviceProfileSummary) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary.to_json_dict(), indent=2) + "\n")
    return path
