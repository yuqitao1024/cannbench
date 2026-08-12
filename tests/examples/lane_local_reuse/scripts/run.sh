#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CASE_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
BUILD_ROOT=${BUILD_ROOT:-"${CASE_ROOT}/build"}
PROFILE_ROOT=${PROFILE_ROOT:-"${CASE_ROOT}/profiles"}
RUN_ID=${RUN_ID:-$(date +%Y%m%d-%H%M%S)}
PROFILE_ELEMENT_COUNT=$((64 * 512 * 4 - 13))

parse_kernel_rows() {
    local raw_dir=$1
    local parsed_dir=$2
    local kernel_name=$3
    local output="${parsed_dir}/kernel_rows.csv"
    local matched=0

    mkdir -p "${parsed_dir}"
    : > "${output}"
    while IFS= read -r csv_file; do
        if grep -F "${kernel_name}" "${csv_file}" >> "${output}"; then
            matched=1
        fi
    done < <(find "${raw_dir}" -type f -name '*.csv' -print | sort)
    if [[ ${matched} -ne 1 ]]; then
        echo "No profiler CSV row matched ${kernel_name}" >&2
        return 1
    fi
}

run_scenario() {
    local scenario=$1
    local kernel_name=$2
    local build_dir="${BUILD_ROOT}/scenario_${scenario}"
    local raw_dir="${PROFILE_ROOT}/scenario_${scenario}/raw/${RUN_ID}"
    local parsed_dir="${PROFILE_ROOT}/scenario_${scenario}/parsed/${RUN_ID}"
    local verify_log="${raw_dir}/verification.log"
    local profile_log="${raw_dir}/profile.log"

    if [[ -e "${raw_dir}" || -e "${parsed_dir}" ]]; then
        echo "RUN_ID already exists for scenario ${scenario}: ${RUN_ID}" >&2
        return 1
    fi

    cmake -S "${CASE_ROOT}" -B "${build_dir}" \
        -DCMAKE_ASC_ARCHITECTURES=dav-3510 \
        -DSCENARIO_NUM="${scenario}"
    cmake --build "${build_dir}" --parallel

    mkdir -p "${raw_dir}"
    "${build_dir}/lane_local_reuse" 2>&1 | tee "${verify_log}"
    grep -q "Verification PASSED" "${verify_log}"

    msopprof --output="${raw_dir}" \
        "${build_dir}/lane_local_reuse" "${PROFILE_ELEMENT_COUNT}" 2>&1 | tee "${profile_log}"
    grep -q "Verification PASSED" "${profile_log}"
    parse_kernel_rows "${raw_dir}" "${parsed_dir}" "${kernel_name}"
    find "${raw_dir}" -type d -name 'OPPROF_*' | grep -q .
    printf 'scenario=%s raw_profile=%s parsed_rows=%s\n' "${scenario}" "${raw_dir}" "${parsed_dir}/kernel_rows.csv"
}

run_scenario 0 "recompute_lane_metadata_kernel"
run_scenario 1 "shared_ub_metadata_kernel"
run_scenario 2 "lane_local_reuse_kernel"
