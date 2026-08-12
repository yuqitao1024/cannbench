#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CASE_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
BUILD_ROOT=${BUILD_ROOT:-"${CASE_ROOT}/build"}
PROFILE_ROOT=${PROFILE_ROOT:-"${CASE_ROOT}/profiles"}
RUN_ID=${RUN_ID:-$(date +%Y%m%d-%H%M%S)}

correctness_widths=(1 255 256 257 1023 1024 1025 4095 4096 4097 8192)
profile_widths=(255 256 257 1023 1024 1025 4095 4096 4097 8192)

expected_kernel() {
    local scenario=$1
    local width=$2
    if [[ ${scenario} -eq 0 ]]; then
        echo "generic_row_normalize_kernel"
    elif [[ ${scenario} -eq 1 ]]; then
        if [[ ${width} -le 256 ]]; then
            echo "small_row_normalize_kernel"
        elif [[ ${width} -le 4096 ]]; then
            echo "medium_row_normalize_kernel"
        else
            echo "wide_row_normalize_kernel"
        fi
    elif [[ ${width} -eq 1024 ]]; then
        echo "exact_1024_row_normalize_kernel"
    else
        echo "exact_overfit_fallback_kernel"
    fi
}

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
    local strategy=$2
    local build_dir="${BUILD_ROOT}/scenario_${scenario}"

    cmake -S "${CASE_ROOT}" -B "${build_dir}" \
        -DCMAKE_ASC_ARCHITECTURES=dav-3510 \
        -DSCENARIO_NUM="${scenario}"
    cmake --build "${build_dir}" --parallel

    for width in "${correctness_widths[@]}"; do
        local verify_log="${build_dir}/correctness_width_${width}.log"
        local kernel_name
        kernel_name=$(expected_kernel "${scenario}" "${width}")
        "${build_dir}/row_width_dispatch" "${width}" 2>&1 | tee "${verify_log}"
        grep -q "Verification PASSED" "${verify_log}"
        grep -q "launch_count=1 kernel=${kernel_name}" "${verify_log}"
    done

    for width in "${profile_widths[@]}"; do
        local kernel_name
        local raw_dir="${PROFILE_ROOT}/scenario_${scenario}/width_${width}/raw/${RUN_ID}"
        local parsed_dir="${PROFILE_ROOT}/scenario_${scenario}/width_${width}/parsed/${RUN_ID}"
        local profile_log="${raw_dir}/profile.log"
        kernel_name=$(expected_kernel "${scenario}" "${width}")
        if [[ -e "${raw_dir}" || -e "${parsed_dir}" ]]; then
            echo "RUN_ID already exists: scenario=${scenario} width=${width} RUN_ID=${RUN_ID}" >&2
            return 1
        fi
        mkdir -p "${raw_dir}"
        msopprof --output="${raw_dir}" \
            "${build_dir}/row_width_dispatch" "${width}" 2>&1 | tee "${profile_log}"
        grep -q "Verification PASSED" "${profile_log}"
        grep -q "launch_count=1 kernel=${kernel_name}" "${profile_log}"
        parse_kernel_rows "${raw_dir}" "${parsed_dir}" "${kernel_name}"
        find "${raw_dir}" -type d -name 'OPPROF_*' | grep -q .
        printf 'scenario=%s strategy=%s width=%s kernel=%s raw=%s\n' \
            "${scenario}" "${strategy}" "${width}" "${kernel_name}" "${raw_dir}"
    done
}

run_scenario 0 "generic"
run_scenario 1 "bucketed"
run_scenario 2 "exact_overfit"
