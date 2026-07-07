# CannBench

CannBench is a benchmark framework for single-card accelerator performance analysis. It currently focuses on operator-level comparison across NVIDIA GPU baselines, Ascend CANN ops library baselines, and Ascend SIMT operator versions.

The project also includes a static frontend for publishing and inspecting benchmark results.

## Features

- Unified operator benchmark CLI for local and remote execution.
- Manifest-driven datasets with `smoke`, `realistic`, and `stress` splits.
- Reproducible prepared inputs for cross-machine NVIDIA and Ascend comparison.
- Device-side profiling integration for NVIDIA and Ascend runs.
- Normalized published data layout for frontend consumption.
- Static performance viewer with GPU, CANN ops library, and SIMT version comparison.
- Versioned Ascend SIMT operator source projects, currently centered on `softmax`.

## Current Scope

Implemented:

- Single-card operator benchmarks.
- NVIDIA PyTorch baseline.
- Ascend CANN ops library baseline.
- Ascend SIMT operator benchmark path.
- Remote SSH benchmark execution.
- CPU-side output comparison.
- Published benchmark record loading in the frontend.

Not implemented yet:

- Model-level TTFS / TPS benchmark pipeline.
- Multi-card or distributed benchmark orchestration.

## Install

Install CannBench in editable mode:

```bash
python3 -m pip install -e ".[dev]"
```

Target machines must provide the matching runtime stack:

- NVIDIA: PyTorch and a usable CUDA runtime.
- Ascend: PyTorch, `torch_npu`, CANN toolkit, and `msprof` when profiling.

## Quick Start

Run all `softmax` realistic cases locally on NVIDIA:

```bash
cannbench bench \
  --backend nvidia \
  --op softmax \
  --dataset realistic \
  --output-dir runs
```

Run one case:

```bash
cannbench bench \
  --backend nvidia \
  --op softmax \
  --case-id t5_attention \
  --warmup 10 \
  --iterations 1
```

Run Ascend SIMT `softmax` version `v2`:

```bash
cannbench bench \
  --backend ascend \
  --implementation simt \
  --implementation-version v2 \
  --op softmax \
  --dataset realistic \
  --output-dir runs
```

Publish a run for frontend loading:

```bash
cannbench publish \
  --source runs/opbench-nvidia-h800-cuda-pytorch-softmax-realistic-float16 \
  --dest published/opbench-nvidia-h800-cuda-pytorch-softmax-realistic-float16
```

Serve the frontend and published data:

```bash
cannbench serve \
  --frontend-dir web/dist \
  --published-dir published
```

Build a release package:

```bash
make release
```

## Documentation

- CLI usage: [docs/guides/cli-usage.md](docs/guides/cli-usage.md)
- Remote benchmarking: [docs/guides/remote-benchmarking.md](docs/guides/remote-benchmarking.md)
- Release and deployment: [docs/guides/release-and-deployment.md](docs/guides/release-and-deployment.md)
- Frontend and GPU upload policy: [docs/guides/frontend-and-upload.md](docs/guides/frontend-and-upload.md)
- Published data contract: [docs/contracts/published-data-contract.md](docs/contracts/published-data-contract.md)
- Adding a new operator: [docs/guides/adding-operator-benchmark.zh-CN.md](docs/guides/adding-operator-benchmark.zh-CN.md)
- CUDA optimization notes: [docs/optimization/cuda-operator-optimization-best-practices.md](docs/optimization/cuda-operator-optimization-best-practices.md)
- DSA fused operator design: [docs/designs/dsa-inference-fusion-spec.md](docs/designs/dsa-inference-fusion-spec.md)

## Design Principles

- Reproducible: benchmark configuration and generated inputs should be explicit.
- Comparable: results should use normalized schemas and stable run names.
- Extensible: new operators should be added through isolated operator plugins.
- Practical: published metrics should focus on device-side performance data.

## References

- TritonBench: https://github.com/meta-pytorch/tritonbench
- vLLM benchmarks: https://github.com/vllm-project/vllm/blob/main/benchmarks/README.md
- NVIDIA CUDA C++ Best Practices Guide: https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html

## License

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.
