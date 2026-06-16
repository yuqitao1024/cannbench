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

Install the project package first:

```bash
python3 -m pip install -e ".[dev]"
```

Then install the matching PyTorch + CUDA stack for your machine before running the benchmark.

### Run the first operator benchmark

The current runnable path is the NVIDIA single-card `softmax` benchmark.

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

The operator path now selects shapes from built-in softmax datasets instead of raw `rows` / `cols` CLI arguments:

- `smoke`: small synthetic cases for functionality checks
- `realistic`: TritonBench-derived model shapes with source metadata
- `stress`: operator-specific softmax boundary cases

The detailed softmax dataset catalog, case tables, TritonBench baseline commit, and curation rules are documented in [src/cannbench/datasets/data/softmax/README.md](/root/aiagent/cannbench/src/cannbench/datasets/data/softmax/README.md).

This command writes:

- `results/nvidia-softmax-smoke.json`
- `results/nvidia-softmax-smoke.csv`
- `results/nvidia-softmax-smoke.md`

### Current Scope

Implemented now:

- Python CLI entrypoint
- Manifest-driven softmax dataset selection
- Softmax operator benchmark request/result schema with source metadata
- Timing summaries with p50/p95/p99
- JSON / CSV / Markdown report writers
- NVIDIA PyTorch backend for single-card `softmax`

Planned next:

- Ascend runtime path
- More operators
- Model-level TTFS / TPS benchmarks

## Design Principles

- Reproducible: benchmark configuration should be explicit and versionable
- Comparable: results should be normalized and easy to compare across devices
- Extensible: new operators, models, and backends should be easy to add
- Practical: focus on metrics that are useful for real hardware evaluation

## Roadmap

- Add Ascend backend for single-operator benchmarking
- Add more NVIDIA operators beyond the initial `softmax` path
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
- First runnable operator path: NVIDIA `softmax`
- Shared schema, timing, and report output layers
- Model TTFS/TPS benchmarking is not implemented yet

## License

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.
