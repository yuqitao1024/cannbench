# Release and Deployment

CannBench release packaging is intended for copying the same benchmark tool and prepared inputs to isolated NVIDIA, Ascend, or frontend deployment machines.

## Build

```bash
make release
```

The build creates:

```text
dist/cannbench-release/
dist/cannbench-release.tar.gz
```

The release includes:

- Python sources and project metadata.
- Built frontend assets from `web/dist`.
- Published benchmark data from `published/`.
- Default prepared inputs under `prepared/<operator>/<dataset>/`.
- Deployment helpers under `deploy/`.
- `install.sh` for service deployment.

Prepared inputs are generated during release packaging with default `dtype=float16` and `seed=7`.

## Manual Use After Unpack

```bash
tar -xzf cannbench-release.tar.gz
cd cannbench-release
export PYTHONPATH="$(pwd)/src"
python3 -m cannbench bench \
  --backend nvidia \
  --op softmax \
  --prepared-dir prepared/softmax/realistic \
  --output-dir runs
```

This mode does not require installing the Python package, but the target runtime dependencies still need to exist on the machine.

## Service Install

After unpacking the release anywhere on the target machine:

```bash
sudo ./install.sh
```

If you place the TLS certificate and private key at the fixed paths below before installation:

```text
/etc/nginx/ssl/cannbench/fullchain.pem
/etc/nginx/ssl/cannbench/privkey.pem
```

then the same `install.sh` command will also install and configure `nginx` HTTPS automatically.

The installer copies the release to:

```text
/opt/cannbench/cannbench-release
```

It also installs the systemd service and starts it.

## Systemd Unit

The release contains:

```text
deploy/systemd/cannbench-serve.service
```

The default service runs as `root` and listens only on `127.0.0.1:8000`. GPU upload is disabled by default.

External traffic should be terminated by a reverse proxy such as `nginx`, which is also where TLS should be configured.

Typical service operations:

```bash
sudo systemctl daemon-reload
sudo systemctl restart cannbench-serve
sudo systemctl status cannbench-serve
sudo journalctl -u cannbench-serve -f
```

## HTTPS Deployment with Nginx

The release includes an `nginx` template:

```text
deploy/nginx/cannbench-https.conf
```

Recommended topology:

```text
Internet
  -> nginx :443
  -> cannbench serve :127.0.0.1:8000
```

Typical setup on Alibaba Cloud ECS:

1. In Alibaba Cloud Certificate Management Service, apply for a free public certificate for your domain.
2. Download the `Nginx` certificate package.
3. Copy the certificate files to the server:

```text
/etc/nginx/ssl/cannbench/fullchain.pem
/etc/nginx/ssl/cannbench/privkey.pem
```

4. Run:

```bash
sudo ./install.sh
```

This flow installs `nginx` automatically when `apt-get`, `dnf`, or `yum` is available, reads the certificate from:

```text
/etc/nginx/ssl/cannbench/fullchain.pem
/etc/nginx/ssl/cannbench/privkey.pem
```

and writes the site config to:

```text
/etc/nginx/conf.d/cannbench.conf
```

5. Test and reload `nginx` if you later modify the config:

```bash
sudo nginx -t
sudo systemctl reload nginx
```

6. In the Alibaba Cloud security group, allow inbound TCP `80` and `443`.

The provided template redirects all `HTTP` traffic on port `80` to `HTTPS` and proxies all requests to the local CannBench service.

## Enabling GPU JSON Upload

GPU JSON upload is disabled by default. Enable it only when importing data:

```bash
python3 -m cannbench serve \
  --frontend-dir web/dist \
  --published-dir published \
  --host 127.0.0.1 \
  --port 8000 \
  --enable-gpu-upload
```

For service deployment, edit the systemd unit temporarily and add `--enable-gpu-upload`, then restart the service. Disable it again after upload.

## Release Hygiene

Do not archive transient profiler outputs in `published/`.

Expected published layout:

```text
published/
  index.json
  <run-name>/
    meta/
      benchmark-records.json
```

See [../contracts/published-data-contract.md](../contracts/published-data-contract.md) for the stable frontend contract.
