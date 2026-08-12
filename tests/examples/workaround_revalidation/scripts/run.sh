#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SAMPLE_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)
BUILD_ROOT="${SAMPLE_DIR}/build"
RUN_ID=${RUN_ID:-$(date +%Y%m%d-%H%M%S)}

run_scenario() {
    local scenario=$1
    local name=$2
    local kernel
    local block_dim
    local expected_launches_per_call=3
    local build_dir="${BUILD_ROOT}/scenario_${scenario}"
    local raw_dir="${SAMPLE_DIR}/profiles/scenario_${scenario}/raw/${RUN_ID}"
    local parsed_dir="${SAMPLE_DIR}/profiles/scenario_${scenario}/parsed/${RUN_ID}"
    local rows_file="${parsed_dir}/kernel_rows.csv"
    local aggregate_file="${parsed_dir}/aggregate.csv"

    if [[ ${scenario} == 0 ]]; then
        kernel="row_sum_single_block_workaround_kernel"
        block_dim=1
    else
        kernel="row_sum_unique_scratch_multiblock_kernel"
        block_dim=64
    fi
    if [[ -e "${raw_dir}" || -e "${parsed_dir}" ]]; then
        echo "RUN_ID already exists: ${RUN_ID}" >&2
        return 1
    fi

    cmake -S "${SAMPLE_DIR}" -B "${build_dir}" -DSCENARIO_NUM="${scenario}" \
        -DCMAKE_ASC_ARCHITECTURES=dav-3510
    cmake --build "${build_dir}" -j
    mkdir -p "${raw_dir}" "${parsed_dir}"
    "${build_dir}/workaround_revalidation" 2>&1 | tee "${raw_dir}/verification.log"
    grep -q "Ownership model PASSED" "${raw_dir}/verification.log"
    grep -q "Verification PASSED" "${raw_dir}/verification.log"

    msopprof --output="${raw_dir}/profile" \
        --launch-count="${expected_launches_per_call}" \
        "${build_dir}/workaround_revalidation" 2>&1 |
        tee "${raw_dir}/profiler.log"
    python3 "${SCRIPT_DIR}/parse_profile.py" \
        --raw "${raw_dir}" \
        --kernel "${kernel}" \
        --expected-launches "${expected_launches_per_call}" \
        --rows-output "${rows_file}" \
        --aggregate-output "${aggregate_file}"
    {
        echo "scenario=${scenario}"
        echo "kernel=${kernel}"
        echo "block_dim=${block_dim}"
        echo "threads_per_block=512"
        echo "expected_launches_per_call=${expected_launches_per_call}"
        echo "aggregation=sum selected Task Duration rows per application call"
    } > "${parsed_dir}/launch_manifest.txt"
    echo "Scenario ${scenario} (${name}) raw profile: ${raw_dir}"
}

run_scenario 0 "single_block_workaround"
run_scenario 1 "fixed_multiblock"
