#!/usr/bin/env bash
set -euo pipefail

if [[ "${EUID}" -ne 0 ]]; then
  echo "install.sh must be run as root" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_DIR="${SCRIPT_DIR}"
INSTALL_ROOT="/opt/cannbench"
INSTALL_DIR="${INSTALL_ROOT}/cannbench-release"
SERVICE_SRC="${SOURCE_DIR}/deploy/systemd/cannbench-serve.service"
SERVICE_DEST="/etc/systemd/system/cannbench-serve.service"

if [[ ! -f "${SOURCE_DIR}/pyproject.toml" || ! -d "${SOURCE_DIR}/src" ]]; then
  echo "install.sh must be executed from the release root directory" >&2
  exit 1
fi

if [[ ! -f "${SERVICE_SRC}" ]]; then
  echo "missing systemd service template: ${SERVICE_SRC}" >&2
  exit 1
fi

mkdir -p "${INSTALL_ROOT}"
rm -rf "${INSTALL_DIR}"
mkdir -p "${INSTALL_DIR}"
cp -a "${SOURCE_DIR}/." "${INSTALL_DIR}/"

cp "${SERVICE_SRC}" "${SERVICE_DEST}"

systemctl daemon-reload
systemctl enable --now cannbench-serve

echo "CANNBench installed to ${INSTALL_DIR}"
echo "Service started: cannbench-serve"
