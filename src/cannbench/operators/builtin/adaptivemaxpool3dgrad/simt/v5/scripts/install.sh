#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

detect_npu_arch() {
  if [[ -n "${NPU_ARCH:-}" ]]; then
    printf '%s\n' "${NPU_ARCH}"
    return
  fi

  local ascend_home="${ASCEND_TOOLKIT_HOME:-${ASCEND_HOME_PATH:-}}"
  if [[ -n "${ascend_home}" && -d "${ascend_home}/x86_64-linux/asc/impl/c_api/instr_impl/npu_arch_3510" ]]; then
    printf '%s\n' "dav-3510"
    return
  fi

  echo "Unable to determine NPU_ARCH automatically; please export NPU_ARCH" >&2
  exit 1
}

export NPU_ARCH
NPU_ARCH="$(detect_npu_arch)"
export PIP_NO_BUILD_ISOLATION=1

cd "${PROJECT_ROOT}"
"${PYTHON:-python}" -m pip install -e . --no-build-isolation --no-deps "$@"

