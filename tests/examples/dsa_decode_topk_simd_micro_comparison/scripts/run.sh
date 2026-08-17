#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
EXAMPLE_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)
RESULT_ROOT=${RESULT_ROOT:-"/tmp/dsa-topk-simd-micro-exp-$(date +%Y%m%d-%H%M%S)"}
BUILD_DIR=${BUILD_DIR:-"${RESULT_ROOT}/build"}

if [[ -e ${RESULT_ROOT} ]]; then
    echo "RESULT_ROOT already exists: ${RESULT_ROOT}" >&2
    exit 1
fi
mkdir -p "${RESULT_ROOT}"

{
    echo "source_revision=$(git -C "${EXAMPLE_DIR}" rev-parse HEAD 2>/dev/null || echo unavailable)"
    echo "target=dav-3510"
    echo "hostname=$(hostname)"
    bisheng --version 2>/dev/null | head -n 1 || true
    msopprof --version 2>/dev/null | head -n 2 || true
} > "${RESULT_ROOT}/environment.txt"
sha256sum "${EXAMPLE_DIR}"/*.asc "${EXAMPLE_DIR}/host_common.h" \
    > "${RESULT_ROOT}/source_sha256.txt"

cmake -S "${EXAMPLE_DIR}" -B "${BUILD_DIR}" \
    -DCMAKE_ASC_ARCHITECTURES=dav-3510 2>&1 | tee "${RESULT_ROOT}/configure.log"
cmake --build "${BUILD_DIR}" -j 2 2>&1 | tee "${RESULT_ROOT}/build.log"
sha256sum "${BUILD_DIR}/dsa_decode_topk_simt_v2_baseline" \
    "${BUILD_DIR}/dsa_decode_topk_simd_micro" \
    > "${RESULT_ROOT}/executable_sha256.txt"

for implementation in simt_v2_baseline simd_micro; do
    executable="${BUILD_DIR}/dsa_decode_topk_${implementation}"
    "${executable}" 2>&1 | tee "${RESULT_ROOT}/${implementation}_correctness.log"
    grep -q "Verification PASSED" "${RESULT_ROOT}/${implementation}_correctness.log"
done

for round in $(seq 1 10); do
    if (( round % 2 == 1 )); then
        order=(simt_v2_baseline simd_micro)
    else
        order=(simd_micro simt_v2_baseline)
    fi
    for implementation in "${order[@]}"; do
        executable="${BUILD_DIR}/dsa_decode_topk_${implementation}"
        if [[ ${implementation} == simt_v2_baseline ]]; then
            kernel=dsa_decode_topk_simt_v2_kernel
        else
            kernel=dsa_decode_topk_simd_micro_kernel
        fi
        sample_dir="${RESULT_ROOT}/profiles/round_${round}/${implementation}"
        raw_dir="${sample_dir}/raw"
        parsed="${sample_dir}/parsed.json"
        mkdir -p "${raw_dir}"
        printf 'msopprof --output=%q --aic-metrics=Default --launch-count=1 %q\n' \
            "${raw_dir}" "${executable}" > "${sample_dir}/command.txt"
        msopprof --output="${raw_dir}" --aic-metrics=Default --launch-count=1 \
            "${executable}" 2>&1 | tee "${sample_dir}/msopprof.log"
        grep -q "Verification PASSED" "${sample_dir}/msopprof.log"
        python3 "${SCRIPT_DIR}/parse_profile.py" --raw "${raw_dir}" \
            --kernel "${kernel}" --expected-launches 1 \
            --expected-block-dim 64 --expected-frequency 1650 \
            --output "${parsed}"
    done
done

python3 - "${RESULT_ROOT}" <<'PY'
import json
from pathlib import Path
import statistics
import sys

root = Path(sys.argv[1])
names = ("simt_v2_baseline", "simd_micro")
summary = {"rounds": []}
samples = {name: [] for name in names}
for round_index in range(1, 11):
    row = {"round": round_index}
    for name in names:
        parsed = json.loads(
            (root / "profiles" / f"round_{round_index}" / name / "parsed.json").read_text()
        )
        current = parsed["frequency_mhz"]
        rated = parsed["rated_frequency_mhz"]
        if current != 1650.0 or rated != 1650.0:
            raise SystemExit(
                f"frequency parity rejected: round={round_index} implementation={name} "
                f"current={current} rated={rated}"
            )
        duration = parsed["task_duration_us"]
        samples[name].append(duration)
        row[name] = duration
    row["candidate_minus_baseline_us"] = row["simd_micro"] - row["simt_v2_baseline"]
    summary["rounds"].append(row)

for name in names:
    values = samples[name]
    summary[name] = {
        "samples_us": values,
        "median_us": statistics.median(values),
        "min_us": min(values),
        "max_us": max(values),
    }
deltas = [row["candidate_minus_baseline_us"] for row in summary["rounds"]]
baseline_median = summary["simt_v2_baseline"]["median_us"]
candidate_median = summary["simd_micro"]["median_us"]
summary["paired_delta_median_us"] = statistics.median(deltas)
summary["candidate_wins"] = sum(delta < 0 for delta in deltas)
summary["ties"] = sum(delta == 0 for delta in deltas)
summary["candidate_improvement_percent"] = (
    (baseline_median - candidate_median) / baseline_median * 100.0
)
(root / "summary.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
print(json.dumps(summary, indent=2, sort_keys=True))
PY

ARCHIVE_PATH="${RESULT_ROOT}.tar.gz"
CHECKSUM_PATH="${ARCHIVE_PATH}.sha256"
tar -czf "${ARCHIVE_PATH}" -C "$(dirname "${RESULT_ROOT}")" "$(basename "${RESULT_ROOT}")"
sha256sum "${ARCHIVE_PATH}" > "${CHECKSUM_PATH}"
printf 'evidence_archive=%s\n' "${ARCHIVE_PATH}"
printf 'evidence_checksum=%s\n' "${CHECKSUM_PATH}"
