#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SAMPLE_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)
BUILD_ROOT="${SAMPLE_DIR}/build"
RUN_ID=${RUN_ID:-$(date +%Y%m%d-%H%M%S)}

parse_kernel_rows() {
    local raw_dir=$1
    local parsed_dir=$2
    local kernel=$3
    local output="${parsed_dir}/kernel_rows.csv"
    local matched=0

    mkdir -p "${parsed_dir}"
    : > "${output}"
    while IFS= read -r csv_file; do
        if grep -F "${kernel}" "${csv_file}" >> "${output}"; then
            matched=1
        fi
    done < <(find "${raw_dir}" -type f -name "*.csv" -print | sort)

    if [[ ${matched} -ne 1 ]]; then
        echo "No profiler CSV row matched kernel ${kernel}" >&2
        return 1
    fi
    echo "Parsed kernel rows: ${output}"
}

run_scenario() {
    local scenario=$1
    local name=$2
    local kernel
    local build_dir="${BUILD_ROOT}/scenario_${scenario}"
    local raw_dir="${SAMPLE_DIR}/profiles/scenario_${scenario}/raw/${RUN_ID}"
    local parsed_dir="${SAMPLE_DIR}/profiles/scenario_${scenario}/parsed/${RUN_ID}"
    local verify_log="${raw_dir}/verification.log"
    local profile_log="${raw_dir}/profiler.log"

    if [[ ${scenario} == 0 ]]; then
        kernel="shared_gather_independent_kernel"
    else
        kernel="shared_gather_reuse_kernel"
    fi
    if [[ -e "${raw_dir}" || -e "${parsed_dir}" ]]; then
        echo "RUN_ID already exists for scenario ${scenario}: ${RUN_ID}" >&2
        return 1
    fi

    cmake -S "${SAMPLE_DIR}" -B "${build_dir}" \
        -DSCENARIO_NUM="${scenario}" -DCMAKE_ASC_ARCHITECTURES=dav-3510
    cmake --build "${build_dir}" -j

    mkdir -p "${raw_dir}"
    "${build_dir}/shared_gather" 2>&1 | tee "${verify_log}"
    grep -q "Verification PASSED" "${verify_log}"

    msopprof --output="${raw_dir}/profile" "${build_dir}/shared_gather" 2>&1 | tee "${profile_log}"
    parse_kernel_rows "${raw_dir}" "${parsed_dir}" "${kernel}"
    echo "Scenario ${scenario} (${name}) raw profile: ${raw_dir}"
}

run_scenario 0 "independent_consumers"
run_scenario 1 "shared_producer"
