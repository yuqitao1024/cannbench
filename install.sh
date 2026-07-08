#!/usr/bin/env bash
set -euo pipefail

fatal() {
  echo "$1" >&2
  exit 1
}

install_nginx_if_needed() {
  if command -v nginx >/dev/null 2>&1; then
    return
  fi

  if command -v apt-get >/dev/null 2>&1; then
    apt-get update
    DEBIAN_FRONTEND=noninteractive apt-get install -y nginx
    return
  fi

  if command -v dnf >/dev/null 2>&1; then
    dnf install -y nginx
    return
  fi

  if command -v yum >/dev/null 2>&1; then
    yum install -y nginx
    return
  fi

  fatal "nginx is not installed and no supported package manager was found"
}

if [[ "${EUID}" -ne 0 ]]; then
  fatal "install.sh must be run as root"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_DIR="${SCRIPT_DIR}"
INSTALL_DIR="/opt/cannbench"
SERVICE_SRC="${SOURCE_DIR}/deploy/systemd/cannbench-serve.service"
SERVICE_DEST="/etc/systemd/system/cannbench-serve.service"
NGINX_TEMPLATE_SRC="${SOURCE_DIR}/deploy/nginx/cannbench-https.conf"
NGINX_CONF_DEST="/etc/nginx/conf.d/cannbench.conf"
NGINX_SSL_DIR="/etc/nginx/ssl/cannbench"
SSL_CERT_SRC="/etc/nginx/ssl/cannbench/fullchain.pem"
SSL_KEY_SRC="/etc/nginx/ssl/cannbench/privkey.pem"
HTTPS_ENABLED=0

if [[ ! -f "${SOURCE_DIR}/pyproject.toml" || ! -d "${SOURCE_DIR}/src" ]]; then
  fatal "install.sh must be executed from the release root directory"
fi

if [[ ! -f "${SERVICE_SRC}" ]]; then
  fatal "missing systemd service template: ${SERVICE_SRC}"
fi

if [[ ! -f "${NGINX_TEMPLATE_SRC}" ]]; then
  fatal "missing nginx config template: ${NGINX_TEMPLATE_SRC}"
fi

if [[ -f "${SSL_CERT_SRC}" && -f "${SSL_KEY_SRC}" ]]; then
  HTTPS_ENABLED=1
fi

rm -rf "${INSTALL_DIR}"
mkdir -p "${INSTALL_DIR}"
cp -a "${SOURCE_DIR}/." "${INSTALL_DIR}/"

cp "${SERVICE_SRC}" "${SERVICE_DEST}"

systemctl daemon-reload
systemctl enable --now cannbench-serve

if [[ "${HTTPS_ENABLED}" -eq 1 ]]; then
  install_nginx_if_needed
  mkdir -p "${NGINX_SSL_DIR}"
  mkdir -p "$(dirname "${NGINX_CONF_DEST}")"
  install -m 644 "${NGINX_TEMPLATE_SRC}" "${NGINX_CONF_DEST}"
  nginx -t
  systemctl enable --now nginx
  systemctl reload nginx
fi

echo "CANNBench installed to ${INSTALL_DIR}"
echo "Service started: cannbench-serve"
echo "Application is bound to 127.0.0.1:8000 by default."

if [[ "${HTTPS_ENABLED}" -eq 1 ]]; then
  echo "HTTPS gateway deployed via nginx."
else
  echo "HTTPS not enabled. Place certificate files at ${SSL_CERT_SRC} and ${SSL_KEY_SRC}, then rerun ./install.sh."
fi
