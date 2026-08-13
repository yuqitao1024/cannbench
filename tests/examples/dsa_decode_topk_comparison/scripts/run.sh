#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
EXAMPLE_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)
RESULT_ROOT=${RESULT_ROOT:-"${EXAMPLE_DIR}/evidence/$(date +%Y%m%d-%H%M%S)"}
BUILD_DIR=${BUILD_DIR:-"${RESULT_ROOT}/build"}

if [[ -e ${RESULT_ROOT} ]]; then
    echo "RESULT_ROOT already exists: ${RESULT_ROOT}" >&2
    exit 1
fi
mkdir -p "${RESULT_ROOT}"
cmake -S "${EXAMPLE_DIR}" -B "${BUILD_DIR}" -DCMAKE_ASC_ARCHITECTURES=dav-3510 2>&1 |
    tee "${RESULT_ROOT}/configure.log"
cmake --build "${BUILD_DIR}" -j 2>&1 | tee "${RESULT_ROOT}/build.log"

for implementation in vllm_ascend simt_v2; do
    executable="${BUILD_DIR}/dsa_decode_topk_${implementation}"
    "${executable}" 2>&1 | tee "${RESULT_ROOT}/${implementation}_correctness.log"
    grep -q "Verification PASSED" "${RESULT_ROOT}/${implementation}_correctness.log"
done

for implementation in vllm_ascend simt_v2; do
    executable="${BUILD_DIR}/dsa_decode_topk_${implementation}"
    kernel="dsa_decode_topk_${implementation}_kernel"
    case "${implementation}" in
        vllm_ascend) expected_block_dim=4 ;;
        simt_v2) expected_block_dim=64 ;;
    esac
    for sample in 1 2 3 4 5; do
        raw_dir="${RESULT_ROOT}/profiles/${implementation}/sample_${sample}/raw"
        parsed="${RESULT_ROOT}/profiles/${implementation}/sample_${sample}/parsed.json"
        mkdir -p "${raw_dir}"
        msopprof --output="${raw_dir}" --aic-metrics="Default" --launch-count=1 \
            "${executable}" 2>&1 | tee "${raw_dir}/msopprof.log"
        python3 "${SCRIPT_DIR}/parse_profile.py" --raw "${raw_dir}" \
            --kernel "${kernel}" --expected-launches 1 --expected-block-dim "${expected_block_dim}" \
            --output "${parsed}"
    done
done

python3 - "${RESULT_ROOT}" <<'PY'
import json
from pathlib import Path
import statistics
import sys

root = Path(sys.argv[1])
summary = {}
for implementation in ("vllm_ascend", "simt_v2"):
    rows = [json.loads((root / "profiles" / implementation / f"sample_{i}" / "parsed.json").read_text())
            for i in range(1, 6)]
    durations = [row["task_duration_us"] for row in rows]
    frequencies = [row["frequency_mhz"] for row in rows]
    rated = [row["rated_frequency_mhz"] for row in rows]
    if any(current != target for current, target in zip(frequencies, rated)):
        raise SystemExit(f"frequency parity rejected: {implementation} current={frequencies} rated={rated}")
    summary[implementation] = {
        "samples_us": durations,
        "median_us": statistics.median(durations),
        "min_us": min(durations),
        "max_us": max(durations),
        "frequency_mhz": frequencies,
    }
reference_frequency = summary["vllm_ascend"]["frequency_mhz"]
if summary["simt_v2"]["frequency_mhz"] != reference_frequency:
    raise SystemExit(
        "frequency parity rejected across implementations: "
        f"vllm_ascend={reference_frequency} "
        f"simt_v2={summary['simt_v2']['frequency_mhz']}")
baseline = summary["vllm_ascend"]["median_us"]
candidate = summary["simt_v2"]["median_us"]
summary["ratio_vllm_over_simt"] = baseline / candidate
(root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print(json.dumps(summary, indent=2, sort_keys=True))
PY
