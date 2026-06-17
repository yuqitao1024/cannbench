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

### Run an operator benchmark

The current operator path supports NVIDIA CUDA execution and an Ascend NPU backend adapter using the same dataset and materialization framework.

```bash
cannbench operator \
  --backend nvidia \
  --op softmax \
  --dtype float16 \
  --dataset realistic \
  --case-id t5_attention \
  --warmup 10 \
  --iterations 50 \
  --output-dir results \
  --run-name nvidia-softmax-smoke
```

The operator path selects shapes from built-in operator datasets instead of raw ad hoc shape CLI arguments:

- `smoke`: small synthetic cases for functionality checks
- `realistic`: model-shaped cases with source metadata
- `stress`: operator-specific boundary cases

Dataset catalogs and case tables are documented under `src/cannbench/datasets/data/<operator>/README.md`.

This command writes:

- `results/nvidia-softmax-smoke.json`
- `results/nvidia-softmax-smoke.csv`
- `results/nvidia-softmax-smoke.md`

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
cannbench operator \
  --backend nvidia \
  --prepared-input prepared-softmax.json \
  --warmup 10 \
  --iterations 50 \
  --output-dir results \
  --run-name nvidia-softmax-prepared
```

### Ascend Backend Status

The Ascend backend is wired into the same operator framework as NVIDIA:

- Same operator names
- Same dataset manifests
- Same seeded input materialization
- Same prepared-input flow
- Same JSON / CSV / Markdown output writers

Ascend execution requires a target machine with PyTorch and `torch_npu`. This repository does not currently include a built-in Ascend custom operator implementation. The custom-op deployment hook is intentionally a boolean flag:

```bash
cannbench operator \
  --backend ascend \
  --prepared-input prepared-softmax.json \
  --deploy-custom-op
```

When `--deploy-custom-op` is set, CannBench looks for:

```text
src/cannbench/datasets/data/<operator>/custom_ops/ascend/default/install.sh
```

If that path is absent, the run fails with a clear error. If `--deploy-custom-op` is not set, CannBench skips custom-op deployment and uses the default Ascend operator library behavior available in the target runtime.

### Output Correctness Comparison

CannBench can capture operator outputs separately from performance measurement. This is intended for NVIDIA-vs-Ascend consistency checks where each backend runs on its own machine and the output artifacts are copied back to the controller machine for CPU-side comparison.

Capture an output artifact:

```bash
cannbench capture-output \
  --backend nvidia \
  --prepared-input prepared-softmax.json \
  --output results/nvidia-softmax-output
```

Capture the same prepared input on Ascend:

```bash
cannbench capture-output \
  --backend ascend \
  --prepared-input prepared-softmax.json \
  --output results/ascend-softmax-output
```

Compare the two artifacts locally:

```bash
cannbench compare-output \
  --left results/nvidia-softmax-output \
  --right results/ascend-softmax-output \
  --rtol 0.001 \
  --atol 0.001 \
  --output results/softmax-accuracy.json
```

Output capture is not part of the performance sampling window. It runs as a separate correctness phase so CPU transfers and comparison work do not pollute device-side profiling results.

Generate a local Markdown report from collected NVIDIA and Ascend run directories:

```bash
cannbench report \
  --nvidia results/nvidia-softmax \
  --ascend results/ascend-softmax \
  --accuracy results/softmax-accuracy.json \
  --output results/softmax-report.md
```

### Remote Collection

CannBench can also run output capture on a remote backend host over SSH and copy the artifact back to the local controller machine.

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

Collect an output artifact from the remote host:

```bash
cannbench collect \
  --endpoint configs/ascend.json \
  --prepared-input prepared-softmax.json \
  --output-dir results/ascend-softmax \
  --run-id softmax-run \
  --capture-output
```

The remote host must already have CannBench installed in `workdir`. The local controller copies the prepared input to the remote run directory, runs `cannbench capture-output` remotely, and downloads the resulting `output` artifact directory to `output-dir`.

The `port` field is optional and defaults to the SSH client default when omitted.

Collect device-side profiler artifacts from the remote host:

```bash
cannbench collect \
  --endpoint configs/ascend.json \
  --prepared-input prepared-softmax.json \
  --output-dir results/ascend-softmax \
  --run-id softmax-run \
  --profile-device-time \
  --summarize-profile \
  --warmup 10 \
  --iterations 50
```

For Ascend endpoints, CannBench wraps the remote operator command with `msprof op` and downloads:

```text
results/ascend-softmax/profile/
results/ascend-softmax/perf/
```

For NVIDIA endpoints, CannBench wraps the remote operator command with `ncu` and downloads the same local artifact directories. Output capture and profiling can be requested in the same `collect` call by passing both `--capture-output` and `--profile-device-time`; internally they still run as separate phases.

When `--summarize-profile` is provided, CannBench also parses the downloaded profiler CSV artifacts on the local controller and writes:

```text
results/<run>/profile-summary.json
```

Summarize profiler CSV artifacts locally:

```bash
cannbench summarize-profile \
  --backend ascend \
  --profile-dir results/ascend-softmax/profile \
  --output results/ascend-softmax/profile-summary.json
```

```bash
cannbench summarize-profile \
  --backend nvidia \
  --profile-dir results/nvidia-softmax/profile \
  --output results/nvidia-softmax/profile-summary.json
```

### Current Scope

Implemented now:

- Python CLI entrypoint
- Manifest-driven operator dataset selection
- Operator benchmark request/result schema with source metadata
- Timing summaries with p50/p95/p99
- JSON / CSV / Markdown report writers
- Prepared-input generation for cross-machine backend comparisons
- Output artifact capture and CPU-side output comparison
- SSH/SCP-based remote output collection from backend hosts
- Remote device-side profiler artifact collection for Ascend and NVIDIA
- Local Markdown report generation across NVIDIA, Ascend, and accuracy artifacts
- Local profiler CSV summarization into normalized device-side latency metrics
- NVIDIA PyTorch backend for single-card operator tests
- Ascend PyTorch backend adapter with optional default custom-op deployment hook
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
- Built-in Ascend custom operator projects under each operator dataset directory
- Real-hardware validation on NVIDIA CUDA and Ascend NPU hosts
- Model-level TTFS / TPS benchmarks

## Design Principles

- Reproducible: benchmark configuration should be explicit and versionable
- Comparable: results should be normalized and easy to compare across devices
- Extensible: new operators, models, and backends should be easy to add
- Practical: focus on metrics that are useful for real hardware evaluation

## Roadmap

- Add built-in Ascend custom operator examples, starting with `softmax`
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
- Ascend custom operator source projects are not included yet
- Model TTFS/TPS benchmarking is not implemented yet

## License

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.
