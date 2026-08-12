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

    rm -rf "${build_dir}"
    cmake -S "${CASE_ROOT}" -B "${build_dir}" -DSCENARIO_NUM="${scenario}"
    cmake --build "${build_dir}" -j

    "${build_dir}/deterministic_compaction" 2>&1 | tee "${verify_log}"
    grep -q "Verification PASSED" "${verify_log}"

    profile_dir=$(mktemp -d "${PROFILE_ROOT}/scenario_${scenario}_XXXXXX")
    msopprof \
        --output="${profile_dir}" \
        --aic-metrics=Default \
        "${build_dir}/deterministic_compaction" 2>&1 | tee "${profile_dir}/msopprof.log"

    grep -q "Verification PASSED" "${profile_dir}/msopprof.log"
    grep -q "${kernel_name}" "${profile_dir}/msopprof.log"
    find "${profile_dir}" -type d -name 'OPPROF_*' | grep -q .
    printf 'scenario=%s profile=%s\n' "${scenario}" "${profile_dir}"
}

run_scenario 0 "stable_atomic_scan_compaction_kernel"
run_scenario 1 "packed_prefix_compaction_kernel"
