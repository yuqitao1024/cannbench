#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CASE_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
BUILD_ROOT=${BUILD_ROOT:-"${CASE_ROOT}/build"}

cmake -S "${CASE_ROOT}" -B "${BUILD_ROOT}" \
    -DCMAKE_ASC_ARCHITECTURES=dav-3510
cmake --build "${BUILD_ROOT}" --parallel
"${BUILD_ROOT}/vcv_per_aiv_ub_layout" | tee "${BUILD_ROOT}/verification.log"
grep -q "Verification PASSED" "${BUILD_ROOT}/verification.log"

printf 'Build artifacts retained at %s\n' "${BUILD_ROOT}"
