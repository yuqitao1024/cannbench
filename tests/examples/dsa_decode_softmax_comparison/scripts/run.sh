#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
EXAMPLE_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)
RUN_ID=${RUN_ID:-$(date +%Y%m%d-%H%M%S)}
RESULT_ROOT=${RESULT_ROOT:-"${EXAMPLE_DIR}/results/${RUN_ID}"}
BUILD_DIR="${RESULT_ROOT}/build"

if [[ -e "${RESULT_ROOT}" ]]; then
    echo "result directory already exists: ${RESULT_ROOT}" >&2
    exit 1
fi
mkdir -p "${RESULT_ROOT}"

cmake -S "${EXAMPLE_DIR}" -B "${BUILD_DIR}" -DCMAKE_ASC_ARCHITECTURES=dav-3510
cmake --build "${BUILD_DIR}" -j

for implementation in vllm_ascend simt_v2; do
    executable="dsa_decode_softmax_${implementation}"
    implementation_root="${RESULT_ROOT}/${implementation}"
    mkdir -p "${implementation_root}/correctness" "${implementation_root}/parsed"
    "${BUILD_DIR}/${executable}" 2>&1 | tee "${implementation_root}/correctness/verification.log"
    grep -q "Verification PASSED" "${implementation_root}/correctness/verification.log"
done

for implementation in vllm_ascend simt_v2; do
    executable="dsa_decode_softmax_${implementation}"
    kernel="${executable}_kernel"
    implementation_root="${RESULT_ROOT}/${implementation}"
    for sample in 1 2 3 4 5; do
        raw_dir="${implementation_root}/raw/sample_${sample}"
        mkdir -p "${raw_dir}"
        msopprof --output="${raw_dir}/profile" --aic-metrics=Default --launch-count=1 \
            "${BUILD_DIR}/${executable}" 2>&1 | tee "${raw_dir}/profiler.log"
        python3 "${SCRIPT_DIR}/parse_profile.py" \
            --raw "${raw_dir}" --kernel "${kernel}" --sample "${sample}" \
            --expected-block-dim 16 \
            --output "${implementation_root}/parsed/sample_${sample}.csv"
    done
done

python3 - "${RESULT_ROOT}" <<'PY'
import csv
from pathlib import Path
import statistics
import sys

root = Path(sys.argv[1])
records = {}
frequencies = set()
rated_frequencies = set()
for implementation in ("vllm_ascend", "simt_v2"):
    rows = []
    for path in sorted((root / implementation / "parsed").glob("sample_*.csv")):
        with path.open(encoding="utf-8") as handle:
            parsed = list(csv.DictReader(handle))
        if len(parsed) != 1:
            raise SystemExit(f"expected one structured row in {path}")
        rows.append(parsed[0])
        frequencies.add(parsed[0]["frequency_mhz"])
        rated_frequencies.add(parsed[0]["rated_frequency_mhz"])
    if len(rows) != 5:
        raise SystemExit(f"expected five samples for {implementation}")
    records[implementation] = [float(row["task_duration_us"]) for row in rows]
if len(frequencies) != 1:
    raise SystemExit(f"measured frequency parity failed: {sorted(frequencies)}")
if len(rated_frequencies) != 1 or frequencies != rated_frequencies:
    raise SystemExit(
        "current/rated frequency parity failed: "
        f"{sorted(frequencies)}/{sorted(rated_frequencies)}"
    )
with (root / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.writer(handle)
    writer.writerow((
        "implementation", "samples_us", "median_us", "minimum_us", "maximum_us",
        "frequency_mhz", "rated_frequency_mhz",
    ))
    for name, values in records.items():
        writer.writerow((name, ";".join(f"{value:.6f}" for value in values),
                         f"{statistics.median(values):.6f}", f"{min(values):.6f}",
                         f"{max(values):.6f}", next(iter(frequencies)),
                         next(iter(rated_frequencies))))
with (root / "ratio.txt").open("w", encoding="utf-8") as handle:
    baseline = statistics.median(records["vllm_ascend"])
    candidate = statistics.median(records["simt_v2"])
    handle.write(f"vllm_ascend_over_simt_v2={baseline / candidate:.6f}\n")
PY

{
    echo "soc_target=dav-3510"
    echo "device=0"
    echo "samples_per_implementation=5"
    echo "timing_field=Task Duration(us)"
    echo "expected_launches_per_collection=1"
    echo "source_hash_vllm_service=332679ebb84a571582d7f3fd3f5cd086415ff24188b3c4ef0370187ec3de02b3"
    echo "source_hash_vllm_vf=8ddafeda9671ac77f9fe1bc86bad512c1a821bd4229b19a825a43525e6e63d10"
    echo "source_hash_simt_v2=5817bea44cfb8c86837b8f5cbf561725cf6b37f5a9a05f4784cd02690b2ea4eb"
    sha256sum "${BUILD_DIR}/dsa_decode_softmax_vllm_ascend" \
        "${BUILD_DIR}/dsa_decode_softmax_simt_v2"
} > "${RESULT_ROOT}/launch_manifest.txt"

echo "Raw profiles and structured results retained at ${RESULT_ROOT}"
