#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CASE_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
BUILD_ROOT=${BUILD_ROOT:-"${CASE_ROOT}/build"}
PROFILE_ROOT=${PROFILE_ROOT:-"${CASE_ROOT}/profiles"}

mkdir -p "${BUILD_ROOT}" "${PROFILE_ROOT}"

run_scenario() {
    local scenario=$1
    local kernel_name=$2
    local build_dir="${BUILD_ROOT}/scenario_${scenario}"
    local verify_log="${build_dir}/verify.log"
    local profile_dir
    local raw_dir

    rm -rf "${build_dir}"
    cmake -S "${CASE_ROOT}" -B "${build_dir}" -DSCENARIO_NUM="${scenario}"
    cmake --build "${build_dir}" -j

    "${build_dir}/phase_aware_tile" 2>&1 | tee "${verify_log}"
    grep -q "Verification PASSED" "${verify_log}"

    profile_dir=$(mktemp -d "${PROFILE_ROOT}/scenario_${scenario}_XXXXXX")
    raw_dir="${profile_dir}/raw"
    mkdir -p "${raw_dir}"
    {
        date -Iseconds
        command -v msopprof
        msopprof --version || true
    } > "${profile_dir}/environment.txt" 2>&1

    msopprof \
        --output="${raw_dir}" \
        --aic-metrics=Default \
        "${build_dir}/phase_aware_tile" 2>&1 | tee "${profile_dir}/msopprof.log"

    grep -q "Verification PASSED" "${profile_dir}/msopprof.log"
    grep -q "${kernel_name}" "${profile_dir}/msopprof.log"
    find "${raw_dir}" -type d -name 'OPPROF_*' | grep -q .
    printf 'scenario=%s raw_profile=%s\n' "${scenario}" "${raw_dir}"
}

run_scenario 0 "conservative_phase_buffers_kernel"
run_scenario 1 "peak_live_set_reuse_kernel"
run_scenario 2 "oversized_outer_tile_kernel"
