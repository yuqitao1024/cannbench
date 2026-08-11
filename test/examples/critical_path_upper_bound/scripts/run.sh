#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CASE_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
BUILD_ROOT=${BUILD_ROOT:-"${CASE_ROOT}/build"}
PROFILE_ROOT=${PROFILE_ROOT:-"${CASE_ROOT}/profiles"}
ANALYSIS_0=""
ANALYSIS_1=""

mkdir -p "${BUILD_ROOT}" "${PROFILE_ROOT}"

run_scenario() {
    local scenario=$1
    local lane_b_kernel=$2
    local build_dir="${BUILD_ROOT}/scenario_${scenario}"
    local verify_log="${build_dir}/verify.log"
    local profile_dir
    local raw_dir
    local analysis_file

    rm -rf "${build_dir}"
    cmake -S "${CASE_ROOT}" -B "${build_dir}" -DSCENARIO_NUM="${scenario}"
    cmake --build "${build_dir}" -j

    "${build_dir}/critical_path_upper_bound" 2>&1 | tee "${verify_log}"
    grep -q "Verification PASSED" "${verify_log}"

    profile_dir=$(mktemp -d "${PROFILE_ROOT}/scenario_${scenario}_XXXXXX")
    raw_dir="${profile_dir}/raw"
    analysis_file="${profile_dir}/analysis.json"
    mkdir -p "${raw_dir}"
    {
        date -Iseconds
        command -v msopprof
        msopprof --version || true
    } > "${profile_dir}/environment.txt" 2>&1

    msopprof \
        --output="${raw_dir}" \
        --aic-metrics=Default \
        --launch-count=4 \
        "${build_dir}/critical_path_upper_bound" 2>&1 | tee "${profile_dir}/msopprof.log"

    grep -q "Verification PASSED" "${profile_dir}/msopprof.log"
    grep -q "lane_a_stage0_kernel" "${profile_dir}/msopprof.log"
    grep -q "${lane_b_kernel}" "${profile_dir}/msopprof.log"
    find "${raw_dir}" -type d -name 'OPPROF_*' | grep -q .
    python3 "${SCRIPT_DIR}/analyze_profile.py" profile \
        --raw "${raw_dir}" \
        --log "${profile_dir}/msopprof.log" \
        --scenario "${scenario}" \
        --output "${analysis_file}"
    if [[ "${scenario}" == "0" ]]; then
        ANALYSIS_0="${analysis_file}"
    else
        ANALYSIS_1="${analysis_file}"
    fi
    printf 'scenario=%s raw_profile=%s analysis=%s\n' "${scenario}" "${raw_dir}" "${analysis_file}"
}

run_scenario 0 "lane_b_candidate_baseline_kernel"
run_scenario 1 "lane_b_candidate_counterfactual_kernel"

comparison_dir=$(mktemp -d "${PROFILE_ROOT}/comparison_XXXXXX")
python3 "${SCRIPT_DIR}/analyze_profile.py" compare \
    --baseline "${ANALYSIS_0}" \
    --counterfactual "${ANALYSIS_1}" \
    --output "${comparison_dir}/comparison.json"
printf 'comparison=%s\n' "${comparison_dir}/comparison.json"
