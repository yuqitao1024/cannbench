# CannBench

CannBench is a benchmarking tool for single-card accelerator performance testing, with a primary focus on NVIDIA GPUs and Huawei Ascend NPUs.

The project is designed for two practical benchmarking scenarios:

1. Single-operator performance testing
2. Single-card model inference benchmarking with TTFS and TPS metrics

## Goals

- Provide a simple and reproducible way to measure single-card performance
- Support both NVIDIA and Ascend environments under one project
- Separate low-level operator benchmarks from end-to-end model serving benchmarks
- Make benchmark outputs easy to compare across devices, drivers, runtimes, and model configurations

## Benchmark Scope

### 1. Single-Operator Performance Testing

This mode is intended for focused kernel or operator-level performance analysis.

Typical use cases:

- Compare the latency and throughput of the same operator on different cards
- Validate runtime or compiler optimization effects
- Measure precision-specific behavior such as FP32, FP16, BF16, or INT8
- Identify bottlenecks before running full model benchmarks

Representative test dimensions:

- Operator type
- Operator dataset and case selection
- Input and output shapes
- Data type
- Batch size
- Warmup iterations
- Measured iterations
- Runtime backend

Example operator categories:

- MatMul / GEMM
- Attention-related operators
- Convolution
- Normalization
- Activation functions
- Elementwise operators

### 2. Single-Card Model TTFS / TPS Testing

This mode is intended for model-level inference benchmarking on a single accelerator card.

Key metrics:

- TTFS: Time To First Token
- TPS: Tokens Per Second

Typical use cases:

- Evaluate interactive inference latency
- Compare model serving performance across cards
- Measure the impact of precision, sequence length, batch size, and runtime settings
- Establish a standard single-card baseline before scaling out

Representative test dimensions:

- Model name
- Framework or serving backend
- Precision
- Prompt length
- Output length
- Batch size
- Sampling configuration
- Runtime and device settings

## Supported Hardware

Planned primary targets:

- NVIDIA single-GPU benchmarking
- Huawei Ascend single-card benchmarking

The project is focused on single-device testing first. Multi-card and distributed benchmarking may be added later if needed.

## Output Expectations

CannBench aims to produce benchmark results that are easy to archive and compare, including:

- Device information
- Driver and runtime information
- Benchmark configuration
- Latency metrics
- Throughput metrics
- TTFS and TPS results for model tests
- Structured output for later aggregation

Recommended output formats:

- JSON result files
- CSV exports for comparison
- Markdown reports

## Proposed Project Structure

```text
cannbench/
├── README.md
├── benchmarks/
│   ├── operators/
│   └── models/
├── backends/
│   ├── nvidia/
│   └── ascend/
├── scripts/
├── results/
└── .agents/
```

## Getting Started

### Install

CannBench itself is a small Python package, but the current runnable operator path also requires:

- PyTorch installed in the target environment
- A usable NVIDIA CUDA runtime for the `nvidia` backend
- PyTorch with `torch_npu` installed in the target environment for the `ascend` backend

Install the project package first:

```bash
python3 -m pip install -e ".[dev]"
```

Then install the matching PyTorch runtime stack for the target machine before running the benchmark.

### Build a Release Package

Build a self-contained release directory and tarball:

```bash
make release
```

The release package includes:

- Python sources and project metadata
- Built frontend assets from `web/dist`
- Default prepared inputs under `prepared/<operator>/<dataset>/`

Prepared inputs are generated with `dtype=float16` and `seed=7` by default. This allows the same release package to be unpacked on different GPU and Ascend machines while reusing the same input manifests.

### Unified CLI

Use `bench` for a single local case:

```bash
cannbench bench \
  --backend nvidia \
  --op softmax \
  --case-id t5_attention \
  --warmup 10 \
  --iterations 1
```

`bench` defaults to `--dataset realistic`. If `--run-name` is omitted, CannBench generates the canonical run name automatically:

```text
opbench-<backend>-<device>-<implementation>-<operator>-<dataset>-<dtype>
```

Example auto-generated names:

- `opbench-nvidia-h800-cuda-pytorch-softmax-realistic-float16`
- `opbench-ascend-950pr-cannops-softmax-realistic-float16`
- `opbench-ascend-950pr-simt-v1-softmax-realistic-float16`

Use `bench` without `--case-id` to expand the selected built-in dataset split for one operator. This writes one batch run directory with per-case artifacts plus `summary.json`, `summary.csv`, and `failures.json`. When the selected backend exposes a local device-time profiling path, batch `bench` also emits normalized frontend records to `meta/benchmark-records.json`:

```bash
cannbench bench \
  --backend nvidia \
  --op softmax \
  --output-dir runs \
  --warmup 10 \
  --iterations 1
```

Use `bench` with `--endpoint` for remote single-case execution. It can either consume an existing prepared input or generate one automatically from `op/dataset/case-id/dtype/seed` before uploading it to the remote host:

```bash
cannbench bench \
  --backend nvidia \
  --endpoint configs/h800.json \
  --op softmax \
  --case-id t5_attention \
  --output-dir runs \
  --run-id h800-softmax-realistic
```

Use `bench --prepared-dir --endpoint` for remote batch execution over a prepared manifest set. The command preserves the prepared manifests under the batch run, stores per-case outputs in stable local paths, emits the same batch summary artifacts as local `bench`, and also writes normalized frontend records to `meta/benchmark-records.json`. If `--run-name` is omitted, automatic naming is only allowed when the prepared manifests all share the same `operator/dataset/dtype` combination:

```bash
cannbench bench \
  --backend ascend \
  --endpoint configs/ascend.json \
  --op softmax \
  --prepared-dir prepared/softmax/realistic \
  --output-dir runs
```

`meta/benchmark-records.json` is the publish-facing artifact for frontend consumption. `meta/summary.json` remains an internal batch index for execution status, prepared-input references, and failure replay.

Use `publish` to mirror selected run artifacts into `published/` without raw profiler files:

```bash
cannbench publish \
  --source runs/h800-softmax-realistic \
  --dest published/h800-softmax-realistic
```

Because `publish` mirrors the full `meta/` directory, `published/<run>/meta/benchmark-records.json` is immediately available to the frontend or any higher-level aggregation service.

Use `serve` to host the frontend and published results. GPU JSON upload is disabled unless explicitly enabled:

```bash
cannbench serve \
  --frontend-dir web/dist \
  --published-dir published
```

```bash
cannbench serve \
  --frontend-dir web/dist \
  --published-dir published \
  --enable-gpu-upload
```

For public cloud deployment, the release package also includes a `systemd` unit template at:

```text
deploy/systemd/cannbench-serve.service
```

Update `User`, `Group`, `WorkingDirectory`, `PYTHONPATH`, and `ExecStart` for the target machine before installing it under `/etc/systemd/system/`.

The release package also includes an `install.sh` helper. After unpacking the release in any directory, run:

```bash
sudo ./install.sh
```

This installs the release under `/opt/cannbench/cannbench-release`, installs the `systemd` unit, reloads `systemd`, and starts `cannbench-serve`.

### Run a benchmark

`bench` is the user-facing execution command for both local and remote runs. It selects shapes from built-in operator datasets instead of raw ad hoc shape CLI arguments.

```bash
cannbench bench \
  --backend nvidia \
  --op softmax \
  --case-id t5_attention \
  --warmup 10 \
  --iterations 1 \
  --output-dir results
```

- `smoke`: small synthetic cases for functionality checks
- `realistic`: model-shaped cases with source metadata
- `stress`: operator-specific boundary cases

Dataset catalogs and case tables are documented under `src/cannbench/datasets/data/<operator>/README.md`.

This command writes a canonical run directory under `results/`, for example:

- `results/opbench-nvidia-h800-cuda-pytorch-softmax-realistic-float16/`
- `results/opbench-nvidia-h800-cuda-pytorch-softmax-realistic-float16/meta/summary.json`
- `results/opbench-nvidia-h800-cuda-pytorch-softmax-realistic-float16/meta/benchmark-records.json`

### Prepare Shared Inputs

Prepared inputs make it possible to run the same generated data on different backend machines, such as one NVIDIA host and one Ascend host.

```bash
cannbench prepare \
  --op softmax \
  --dtype float16 \
  --dataset smoke \
  --case-id tiny_logits \
  --seed 7 \
  --output prepared-softmax.json
```

Run the prepared input on a backend machine:

```bash
cannbench bench \
  --backend nvidia \
  --prepared-input prepared-softmax.json \
  --warmup 10 \
  --iterations 1 \
  --output-dir results
```

### Ascend Backend Status

The Ascend backend is wired into the same benchmark framework as NVIDIA:

- Same operator names
- Same dataset manifests
- Same seeded input materialization
- Same prepared-input flow
- Same JSON / CSV / Markdown output writers

Ascend execution requires a target machine with PyTorch and `torch_npu`. The repository includes a built-in Ascend SIMT `softmax` operator project under version `v1`. The SIMT deployment hook is intentionally a boolean flag:

```bash
cannbench bench \
  --backend ascend \
  --prepared-input prepared-softmax.json \
  --deploy-custom-op
```

When `--deploy-custom-op` is set, CannBench looks for:

```text
src/cannbench/datasets/data/<operator>/custom_ops/ascend/v1/install.sh
```

If that path is absent, the run fails with a clear error. If `--deploy-custom-op` is not set, CannBench skips SIMT deployment and uses the default CANN ops library behavior available in the target runtime.

### Performance Viewer

CannBench includes a static single-page frontend for inspecting normalized benchmark results, comparing GPU H800, CANN ops library, and multiple Ascend SIMT operator versions.

```bash
cd web
npm install
npm run dev
```

The first version loads sample data from:

```text
web/src/data/sample/benchmark-results.json
```

GPU result import is represented as a hidden JSON text validation flow and is disabled by policy until a server upload endpoint is intentionally enabled in a later milestone. The frontend validator accepts only normalized GPU benchmark performance records and rejects sensitive fields such as hostnames, environment data, commands, logs, paths, raw profiler data, source code, or diffs.

### Output Correctness Comparison

CannBench uses `compare` for CPU-side correctness checks across backends. The command runs the same operator case on both sides, captures normalized outputs, and writes a local comparison artifact without mixing output transfer time into device profiling.

```bash
cannbench compare \
  --left-backend nvidia \
  --right-backend ascend \
  --op softmax \
  --dtype float16 \
  --dataset smoke \
  --case-id tiny_logits \
  --seed 7 \
  --rtol 0.001 \
  --atol 0.001 \
  --output results/softmax-accuracy.json
```

If the Ascend side should use a SIMT implementation instead of the CANN ops library, add `--left-deploy-custom-op` or `--right-deploy-custom-op` on the corresponding side.

Generate a local Markdown report from collected NVIDIA and Ascend run directories:

```bash
cannbench report \
  --nvidia results/nvidia-softmax \
  --ascend results/ascend-softmax \
  --accuracy results/softmax-accuracy.json \
  --output results/softmax-report.md
```

### Remote Bench

CannBench can run output capture and device-side profiling on a remote backend host over SSH and copy the artifacts back to the local controller machine.

Example endpoint config:

```json
{
  "name": "ascend-a2",
  "backend": "ascend",
  "host": "user@ascend-host",
  "port": 22,
  "workdir": "/opt/cannbench",
  "python": "python3",
  "env": {
    "ASCEND_VISIBLE_DEVICES": "0"
  }
}
```

Collect a remote run with device-side profiling and optional output capture:

```bash
cannbench bench \
  --backend ascend \
  --endpoint configs/ascend.json \
  --prepared-input prepared-softmax.json \
  --output-dir results/ascend-softmax \
  --run-id softmax-run \
  --capture-output
```

The remote host must already have CannBench installed in `workdir`. The local controller copies the prepared input to the remote run directory, runs `cannbench internal-run` remotely, and downloads the generated `output/`, `profile/`, and `perf/` artifacts back to `output-dir` when requested.

The `port` field is optional and defaults to the SSH client default when omitted.

Collect device-side profiler artifacts from the remote host:

```bash
cannbench bench \
  --backend ascend \
  --endpoint configs/ascend.json \
  --prepared-input prepared-softmax.json \
  --output-dir results/ascend-softmax \
  --run-id softmax-run \
  --warmup 10 \
  --iterations 1
```

For Ascend endpoints, CannBench wraps an internal remote benchmark worker with `msprof op` and downloads:

```text
results/ascend-softmax/profile/
results/ascend-softmax/perf/
```

For NVIDIA endpoints, CannBench profiles operator device time with `ncu`. Output capture and profiling can be requested in the same `bench --endpoint ...` call by passing `--capture-output`; device-side profiling is always enabled for `bench`.

CannBench parses downloaded profiler artifacts on the controller and stores normalized device-time summaries under each run's `meta/` directory together with the frontend-facing benchmark records.

### Current Scope

Implemented now:

- Python CLI entrypoint
- Manifest-driven operator dataset selection
- Operator benchmark request/result schema with source metadata
- Timing summaries with p50/p95/p99
- JSON / CSV / Markdown report writers
- Prepared-input generation for cross-machine backend comparisons
- CPU-side output comparison across NVIDIA and Ascend backends
- SSH/SCP-based remote benchmark execution and artifact collection
- Remote device-side profiler artifact collection for Ascend and NVIDIA
- Local Markdown report generation across NVIDIA, Ascend, and accuracy artifacts
- Normalized benchmark record generation for publish and frontend loading
- Static frontend performance viewer with chart, case table, repository diff panel, and GPU JSON upload validation
- NVIDIA PyTorch backend for single-card operator tests
- Ascend PyTorch backend adapter with optional default custom-op deployment hook
- Built-in Ascend custom `softmax` operator source project
- Built-in operator datasets and dispatch for:
  - `softmax`
  - `embedding`
  - `gather`
  - `index_select`
  - `take_along_dim`
  - `masked_select`
  - `cross_entropy`
  - `scatter_add`
  - `scatter`
  - `index_add`
  - `index_put`

Planned next:

- Harden profiler parsers against real `msprof op` and `ncu` output variants from target machines
- Built-in Ascend custom operator projects for more operator datasets
- Real-hardware validation on NVIDIA CUDA and Ascend NPU hosts
- Model-level TTFS / TPS benchmarks

## Design Principles

- Reproducible: benchmark configuration should be explicit and versionable
- Comparable: results should be normalized and easy to compare across devices
- Extensible: new operators, models, and backends should be easy to add
- Practical: focus on metrics that are useful for real hardware evaluation

## Roadmap

- Add built-in Ascend custom operator examples beyond `softmax`
- Expand operator datasets and realistic shape coverage
- Add TTFS and TPS model benchmark pipeline
- Standardize result schema
- Add benchmark reports and comparison scripts

## References

The following projects and documents are useful references for CannBench design and implementation:

### Model-Level Benchmarking

- vLLM Benchmarks README
  https://github.com/vllm-project/vllm/blob/main/benchmarks/README.md

  Useful as a reference for serving benchmarks, latency and throughput testing, benchmark dataset handling, and benchmark CLI organization.

### Operator-Level Benchmarking

- TritonBench
  https://github.com/meta-pytorch/tritonbench

  Useful as a reference for operator-focused benchmarking, example inputs, and performance comparison workflows for PyTorch custom operators.

- DeepSpeedExamples Benchmarks
  https://github.com/deepspeedai/DeepSpeedExamples/tree/master/benchmarks

  Useful as a reference for practical benchmarking layouts and performance evaluation examples around training and inference workloads.

### NVIDIA Performance Guidance

- NVIDIA CUDA C++ Best Practices Guide
  https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html

  Useful as a reference for measurement methodology, CUDA performance analysis, optimization principles, and benchmarking discipline on NVIDIA platforms.

## Status

This repository is in the first implementation stage. The current scope is:

- Single-card benchmarking only
- First operator benchmark framework for NVIDIA and Ascend backends
- Shared schema, timing, and report output layers
- A built-in Ascend custom `softmax` operator source project is included
- Static frontend performance result viewing is included
- Model TTFS/TPS benchmarking is not implemented yet

## License

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.
