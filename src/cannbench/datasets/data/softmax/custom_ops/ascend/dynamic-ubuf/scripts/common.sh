#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

source_ascend_env() {
  if [[ -f /usr/local/Ascend/cann/set_env.sh ]]; then
    # shellcheck disable=SC1091
    source /usr/local/Ascend/cann/set_env.sh
    return
  fi

  if [[ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]]; then
    # shellcheck disable=SC1091
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    return
  fi

  echo "Ascend environment script not found" >&2
  exit 1
}

prepare_default_env() {
  source_ascend_env
  export PIP_NO_BUILD_ISOLATION=1
}

detect_npu_arch() {
  if [[ -n "${NPU_ARCH:-}" ]]; then
    printf '%s\n' "${NPU_ARCH}"
    return
  fi

  if [[ -d /usr/local/Ascend/cann-9.0.0/x86_64-linux/asc/impl/c_api/instr_impl/npu_arch_3510 ]]; then
    printf '%s\n' "dav-3510"
    return
  fi

  echo "Unable to determine NPU_ARCH automatically; please export NPU_ARCH" >&2
  exit 1
}

prepare_default_build_env() {
  source_ascend_env
  export NPU_ARCH
  NPU_ARCH="$(detect_npu_arch)"
  export PIP_NO_BUILD_ISOLATION=1
}
