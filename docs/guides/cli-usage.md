# CannBench CLI Usage

This guide covers the user-facing CannBench commands. It intentionally avoids backend installation details; target machines still need their NVIDIA or Ascend runtime stacks installed separately.

## Commands

CannBench exposes these primary commands:

- `bench`: run local or remote operator benchmarks.
- `prepare`: generate a deterministic prepared-input manifest.
- `compare`: capture two backend outputs and compare correctness on the controller side.
- `publish`: copy frontend-facing run artifacts into `published/`.
- `serve`: host the static frontend and published result files.

`internal-run` is reserved for CannBench remote execution internals and should not be used directly in normal workflows.

## Operator Benchmarking

Run one case:

```bash
cannbench bench \
  --backend nvidia \
  --op softmax \
  --case-id t5_attention \
  --warmup 10 \
  --iterations 1 \
  --output-dir runs
```

Run a dataset split. If `--case-id` is omitted, `bench` expands all cases in the selected split:

```bash
cannbench bench \
  --backend nvidia \
  --op softmax \
  --dataset realistic \
  --warmup 10 \
  --iterations 1 \
  --output-dir runs
```

Dataset split defaults:

- `--dataset` defaults to `realistic`.
- `--dtype` defaults to `float16`.
- `--warmup` defaults to `10`.
- `--iterations` defaults to `1`.

Dataset meanings:

- `smoke`: small synthetic cases for functionality checks.
- `realistic`: real-model shapes with source metadata.
- `stress`: operator-specific boundary and stress cases.

## Implementations

NVIDIA PyTorch baseline. Do not pass `--implementation`; the NVIDIA backend defaults to the PyTorch CUDA baseline:

```bash
cannbench bench \
  --backend nvidia \
  --op softmax
```

Ascend CANN ops library baseline:

```bash
cannbench bench \
  --backend ascend \
  --implementation cann_ops_library \
  --op softmax
```

Ascend SIMT operator:

```bash
cannbench bench \
  --backend ascend \
  --implementation simt \
  --implementation-version v2 \
  --op softmax
```

For SIMT runs, CannBench deploys the selected operator version from:

```text
src/cannbench/operators/builtin/<operator>/simt/<version>/install.sh
```

If the selected SIMT version does not exist, the run fails instead of silently falling back to the CANN ops library.

## Run Names

If `--run-name` is omitted, CannBench generates a canonical name:

```text
opbench-<backend>-<device>-<implementation>-<operator>-<dataset>-<dtype>
```

Examples:

```text
opbench-nvidia-h800-cuda-pytorch-softmax-realistic-float16
opbench-ascend-950pr-cannops-softmax-realistic-float16
opbench-ascend-950pr-simt-v2-softmax-realistic-float16
```

The run-name is the stable publication and frontend lookup key. See [../contracts/published-data-contract.md](../contracts/published-data-contract.md) for the full contract.

## Output Layout

A batch `bench` run writes one run directory:

```text
runs/<run-name>/
  prepared/
  perf/
  profile/
  output/
  meta/
    summary.json
    benchmark-records.json
  summary.json
  summary.csv
  failures.json
```

Important files:

- `meta/benchmark-records.json`: frontend-facing normalized performance records.
- `meta/summary.json`: internal execution status and replay metadata.
- `summary.json` and `summary.csv`: human-readable batch summaries.
- `profile/`: raw profiler artifacts; not published.

## Prepared Inputs

Prepared inputs allow different machines to consume the same generated benchmark data.

```bash
cannbench prepare \
  --op softmax \
  --dtype float16 \
  --dataset realistic \
  --case-id t5_attention \
  --seed 7 \
  --output prepared-softmax.json
```

Run the prepared input:

```bash
cannbench bench \
  --backend nvidia \
  --prepared-input prepared-softmax.json \
  --output-dir runs
```

Run a prepared directory:

```bash
cannbench bench \
  --backend ascend \
  --op softmax \
  --prepared-dir prepared/softmax/realistic \
  --output-dir runs
```

`--prepared-input` and `--prepared-dir` are mutually exclusive.

## Correctness Comparison

`compare` captures outputs from two backends and performs the comparison locally.

```bash
cannbench compare \
  --left-backend nvidia \
  --right-backend ascend \
  --op softmax \
  --dtype float16 \
  --dataset realistic \
  --case-id t5_attention \
  --seed 7 \
  --rtol 0.001 \
  --atol 0.001 \
  --output runs/softmax-accuracy.json
```

Use `--left-deploy-simt-op` or `--right-deploy-simt-op` when one side should use an Ascend SIMT implementation.

## Publishing

Publish a collected run:

```bash
cannbench publish \
  --source runs/opbench-nvidia-h800-cuda-pytorch-softmax-realistic-float16 \
  --dest published/opbench-nvidia-h800-cuda-pytorch-softmax-realistic-float16
```

`publish` mirrors the publishable metadata and avoids raw profiler files. The frontend discovers published runs through `published/index.json`.
