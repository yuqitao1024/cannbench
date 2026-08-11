#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SAMPLE_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)
BUILD_ROOT="${SAMPLE_DIR}/build"
PROFILE_ROOT="${SAMPLE_DIR}/profiles"
RUN_ID=${RUN_ID:-$(date +%Y%m%d-%H%M%S)}
RUN_ROOT="${PROFILE_ROOT}/${RUN_ID}"
SUMMARY_CSV="${RUN_ROOT}/operator_calls.csv"
SAMPLES=${SAMPLES:-5}

if [[ -e "${RUN_ROOT}" ]]; then
    echo "RUN_ID already exists: ${RUN_ID}" >&2
    exit 1
fi
rm -rf "${BUILD_ROOT}"
mkdir -p "${BUILD_ROOT}" "${RUN_ROOT}"
printf 'scenario,sample,expected_launches,observed_launches,stage_durations_us,stage_sum_us\n' > "${SUMMARY_CSV}"

run_case() {
    local scenario=$1
    local expected_launches=$2
    local build_dir="${BUILD_ROOT}/scenario_${scenario}"

    cmake -S "${SAMPLE_DIR}" -B "${build_dir}" \
        -DCMAKE_ASC_ARCHITECTURES=dav-3510 \
        -DSCENARIO_NUM="${scenario}"
    cmake --build "${build_dir}" --parallel
    "${build_dir}/sharded_histogram_topk" | tee "${build_dir}/direct.log"
    grep -q "Verification PASSED" "${build_dir}/direct.log"

    for ((sample = 1; sample <= SAMPLES; ++sample)); do
        local sample_dir="${RUN_ROOT}/scenario_${scenario}/sample_${sample}"
        mkdir -p "${sample_dir}"
        msopprof \
            --output="${sample_dir}" \
            --aic-metrics=BasicInfo \
            --launch-count="${expected_launches}" \
            "${build_dir}/sharded_histogram_topk" 2>&1 | tee "${sample_dir}/msopprof.log"

        # Case 1 rows are stages of one operator call: sum them before cross-call statistics.
        python3 - "${scenario}" "${sample}" "${expected_launches}" "${sample_dir}" "${SUMMARY_CSV}" <<'PY'
import csv
from pathlib import Path
import sys

scenario, sample, expected_text, root_text, summary_text = sys.argv[1:]
expected_launches = int(expected_text)
root = Path(root_text)
expected_names = (
    ("baseline_histogram_threshold_kernel",)
    if scenario == "0"
    else ("shard_histogram_kernel", "reduce_histogram_threshold_kernel")
)
csv_paths = list(root.rglob("OpBasicInfo.csv"))
if not csv_paths:
    csv_paths = list(root.rglob("*.csv"))

selected = []
for path in csv_paths:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "Op Name" not in reader.fieldnames:
            continue
        duration_key = next((name for name in reader.fieldnames if name.startswith("Task Duration(")), None)
        if duration_key is None:
            continue
        for row in reader:
            op_name = row["Op Name"]
            for expected_name in expected_names:
                if expected_name in op_name:
                    selected.append((expected_name, float(row[duration_key])))
                    break

observed_names = [name for name, _ in selected]
if len(selected) != expected_launches or sorted(observed_names) != sorted(expected_names):
    raise SystemExit(
        f"launch audit failed: expected={expected_names}, observed={selected}, csv_paths={csv_paths}"
    )
durations = [duration for _, duration in selected]
with Path(summary_text).open("a", newline="", encoding="utf-8") as handle:
    writer = csv.writer(handle)
    writer.writerow((scenario, sample, expected_launches, len(selected),
                     ";".join(f"{value:.6f}" for value in durations), f"{sum(durations):.6f}"))
print(f"scenario={scenario} sample={sample} stage_sum_us={sum(durations):.6f}")
PY
    done
}

run_case 0 1
run_case 1 2

python3 - "${SUMMARY_CSV}" <<'PY' | tee "${RUN_ROOT}/performance_summary.txt"
import csv
from pathlib import Path
import statistics
import sys

rows = list(csv.DictReader(Path(sys.argv[1]).open(encoding="utf-8")))
for scenario in ("0", "1"):
    values = [float(row["stage_sum_us"]) for row in rows if row["scenario"] == scenario]
    print(
        f"scenario={scenario} calls={len(values)} median_stage_sum_us={statistics.median(values):.6f} "
        f"min_us={min(values):.6f} max_us={max(values):.6f}"
    )
PY

printf 'Raw profiles: %s\n' "${RUN_ROOT}"
