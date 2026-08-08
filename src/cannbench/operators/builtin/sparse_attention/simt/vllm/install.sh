#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON:-python}"

cd "${SCRIPT_DIR}"
"${PYTHON_BIN}" -m pip install -e . --no-build-isolation --no-deps "$@"
