#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SAMPLE_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)
BUILD_ROOT="${SAMPLE_DIR}/build"
RUN_ID=${RUN_ID:-$(date +%Y%m%d-%H%M%S)}

aggregate_profile() {
    local raw_dir=$1
    local parsed_dir=$2
    local kernels=$3
    local expected_launches_per_call=$4

    mkdir -p "${parsed_dir}"
    python3 - "${raw_dir}" "${parsed_dir}" "${kernels}" "${expected_launches_per_call}" <<'PY'
import csv
import math
import sys
from pathlib import Path

raw_dir = Path(sys.argv[1])
parsed_dir = Path(sys.argv[2])
kernels = tuple(sys.argv[3].split(","))
expected_launches_per_call = int(sys.argv[4])
rows = []


def duration_to_us(header, value):
    parsed = float(value.replace(",", "").strip())
    lowered = header.lower()
    if "(ns)" in lowered or "nanosecond" in lowered:
        return parsed / 1000.0
    if "(ms)" in lowered or "millisecond" in lowered:
        return parsed * 1000.0
    if "(s)" in lowered and "(us)" not in lowered:
        return parsed * 1_000_000.0
    return parsed


for csv_path in sorted(raw_dir.rglob("*.csv")):
    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            continue
        duration_key = next(
            (key for key in reader.fieldnames if "duration" in key.lower() or "elapsed" in key.lower()),
            None,
        )
        if duration_key is None:
            continue
        for row in reader:
            row_text = " ".join(str(value) for value in row.values())
            matched = [kernel for kernel in kernels if kernel in row_text]
            if len(matched) != 1:
                continue
            try:
                duration_us = duration_to_us(duration_key, row[duration_key])
            except (TypeError, ValueError):
                continue
            if math.isfinite(duration_us):
                rows.append((str(csv_path.relative_to(raw_dir)), matched[0], duration_us))

counts = {kernel: sum(row[1] == kernel for row in rows) for kernel in kernels}
if any(count == 0 for count in counts.values()):
    raise SystemExit(f"missing selected kernel rows: {counts}")
if len(set(counts.values())) != 1:
    raise SystemExit(f"selected kernel counts do not form complete calls: {counts}")
captured_calls = next(iter(counts.values()))
if expected_launches_per_call != len(kernels):
    raise SystemExit("expected launch count disagrees with selected kernel family count")

with (parsed_dir / "kernel_rows.csv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.writer(handle)
    writer.writerow(("source_file", "kernel_name", "task_duration_us"))
    writer.writerows(rows)

total_us = sum(row[2] for row in rows)
with (parsed_dir / "aggregate.csv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.writer(handle)
    writer.writerow((
        "expected_launches_per_call",
        "observed_selected_rows",
        "captured_application_calls",
        "selected_task_duration_sum_us",
        "mean_complete_boundary_us",
    ))
    writer.writerow((
        expected_launches_per_call,
        len(rows),
        captured_calls,
        f"{total_us:.9f}",
        f"{total_us / captured_calls:.9f}",
    ))
PY
}

run_scenario() {
    local scenario=$1
    local name=$2
    local kernels
    local expected_launches_per_call
    local build_dir="${BUILD_ROOT}/scenario_${scenario}"
    local raw_dir="${SAMPLE_DIR}/profiles/scenario_${scenario}/raw/${RUN_ID}"
    local parsed_dir="${SAMPLE_DIR}/profiles/scenario_${scenario}/parsed/${RUN_ID}"
    local verify_log="${raw_dir}/verification.log"
    local profile_log="${raw_dir}/profiler.log"

    if [[ ${scenario} == 0 ]]; then
        kernels="row_major_producer_kernel,explicit_blocked_pack_kernel"
        expected_launches_per_call=2
    else
        kernels="consumer_native_layout_kernel"
        expected_launches_per_call=1
    fi
    if [[ -e "${raw_dir}" || -e "${parsed_dir}" ]]; then
        echo "RUN_ID already exists for scenario ${scenario}: ${RUN_ID}" >&2
        return 1
    fi

    cmake -S "${SAMPLE_DIR}" -B "${build_dir}" \
        -DSCENARIO_NUM="${scenario}" -DCMAKE_ASC_ARCHITECTURES=dav-3510
    cmake --build "${build_dir}" -j

    mkdir -p "${raw_dir}"
    "${build_dir}/consumer_native_layout" 2>&1 | tee "${verify_log}"
    grep -q "Verification PASSED" "${verify_log}"

    msopprof --output="${raw_dir}/profile" \
        --launch-count="${expected_launches_per_call}" \
        "${build_dir}/consumer_native_layout" 2>&1 | tee "${profile_log}"
    aggregate_profile "${raw_dir}" "${parsed_dir}" "${kernels}" "${expected_launches_per_call}"
    echo "Scenario ${scenario} (${name}) raw profile: ${raw_dir}"
}

run_scenario 0 "explicit_pack"
run_scenario 1 "native_layout"
