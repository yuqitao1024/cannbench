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

The default service runs as `root` and serves the frontend on port `80`. GPU upload is disabled by default.

Typical service operations:

```bash
sudo systemctl daemon-reload
sudo systemctl restart cannbench-serve
sudo systemctl status cannbench-serve
sudo journalctl -u cannbench-serve -f
```

## Enabling GPU JSON Upload

GPU JSON upload is disabled by default. Enable it only when importing data:

```bash
python3 -m cannbench serve \
  --frontend-dir web/dist \
  --published-dir published \
  --host 0.0.0.0 \
  --port 80 \
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
