# Remote Benchmarking

CannBench can run operator benchmarks on a remote backend host over SSH while keeping orchestration, artifact collection, and report generation on the controller machine.

## Endpoint Config

Example endpoint file:

```json
{
  "name": "ascend-a2",
  "backend": "ascend",
  "host": "user@ascend-host",
  "port": 22,
  "workdir": "/opt/cannbench/cannbench-release",
  "python": "python3",
  "setup": ". /usr/local/Ascend/ascend-toolkit/set_env.sh",
  "env": {
    "ASCEND_VISIBLE_DEVICES": "0"
  }
}
```

Fields:

- `name`: human-readable endpoint name.
- `backend`: `nvidia` or `ascend`.
- `host`: SSH target.
- `port`: optional SSH port.
- `workdir`: CannBench working directory on the remote host.
- `python`: Python command used on the remote host.
- `setup`: optional shell setup command sourced before execution.
- `env`: optional environment variables.

The remote host must already contain CannBench sources or an unpacked release package under `workdir`.

## Remote Single Case

```bash
cannbench bench \
  --backend nvidia \
  --endpoint configs/h800.json \
  --op softmax \
  --case-id t5_attention \
  --output-dir runs
```

If no prepared input is supplied, CannBench builds one locally from `op`, `dataset`, `case-id`, `dtype`, and `seed`, then uploads it to the remote run directory.

## Remote Batch

Use an explicit prepared directory for batch remote runs:

```bash
cannbench bench \
  --backend ascend \
  --endpoint configs/ascend.json \
  --op softmax \
  --prepared-dir prepared/softmax/realistic \
  --output-dir runs \
  --warmup 0 \
  --iterations 1
```

The controller uploads each prepared manifest, starts remote execution, and downloads generated artifacts into the local run directory.

## Profiling Behavior

`bench` is expected to collect device-side time by default.

- Ascend remote profiling wraps execution with `msprof op`.
- NVIDIA remote profiling uses `ncu`.
- Raw profiler outputs are stored under `profile/`.
- Parsed frontend-facing records are stored under `meta/benchmark-records.json`.

Raw profiler outputs are collection artifacts, not published data. Use `publish` to copy only frontend-facing data into `published/`.

## Output Capture

Add `--capture-output` when output tensors are needed for later correctness checks:

```bash
cannbench bench \
  --backend ascend \
  --endpoint configs/ascend.json \
  --prepared-input prepared-softmax.json \
  --output-dir runs/ascend-softmax \
  --capture-output
```

Output capture is separate from performance measurement and should not be interpreted as part of device-side kernel time.

## Failure Replay

Batch runs write `failures.json`. For a failed case, rerun the corresponding prepared manifest directly:

```bash
cannbench bench \
  --backend ascend \
  --endpoint configs/ascend.json \
  --prepared-input runs/<run-name>/prepared/<operator>/<dataset>/<case>-float16-seed0.json \
  --output-dir runs/retry
```
