# Published Data Contract

This document defines the stable contract between CannBench data producers, the published artifact layout, and the frontend loader.

## Scope

This contract applies to:

- `published/index.json`
- `published/<run-name>/meta/benchmark-records.json`
- frontend loading and aggregation behavior

It does not define internal execution artifacts such as `summary.json`, raw profiler files, or temporary run directories under `runs/`.

## Published Layout

The published root must follow this structure:

```text
published/
  index.json
  <run-name>/
    meta/
      benchmark-records.json
```

Rules:

- `published/index.json` is the only discovery entrypoint for the frontend.
- Each published run must have exactly one `meta/benchmark-records.json`.
- Frontend code must not hardcode individual run directories.
- Frontend code must not rely on a synthetic aggregate directory such as `published/default/`.

## Run Index Contract

`published/index.json` must use this schema:

```json
{
  "runs": [
    "opbench-ascend-950pr-cannops-softmax-realistic-float16",
    "opbench-ascend-950pr-simt-v1-softmax-realistic-float16"
  ]
}
```

Rules:

- `runs` is an ordered array of published run names.
- The frontend loads runs in this order and merges all records into one in-memory dataset.
- A run listed in `index.json` must exist as `published/<run-name>/meta/benchmark-records.json`.
- A published run directory that is not listed in `index.json` is considered undiscoverable by the frontend.

## Run Name Contract

### Canonical Format

Run names must follow this format:

```text
opbench-<backend>-<device>-<implementation>-<operator>-<dataset>-<dtype>
```

Example:

```text
opbench-ascend-950pr-cannops-softmax-realistic-float16
opbench-ascend-950pr-simt-v1-softmax-realistic-float16
opbench-nvidia-h800-cuda-pytorch-softmax-realistic-float16
```

### Segment Definitions

- `opbench`
  - fixed prefix
- `<backend>`
  - current values: `nvidia`, `ascend`
- `<device>`
  - normalized device class
  - current values: `h800`, `950pr`
- `<implementation>`
  - execution implementation identifier
  - current values:
    - `cuda-pytorch`
    - `cannops`
    - `simt-v1`
    - future SIMT versions must use `simt-v2`, `simt-v3`, and so on
- `<operator>`
  - operator name, for example `softmax`, `embedding`
- `<dataset>`
  - current values: `smoke`, `realistic`, `stress`
- `<dtype>`
  - current values start with `float16`

### Run Name Semantics

`<implementation>` in `run-name` identifies the published benchmark lane, not the low-level record field vocabulary.

Current mapping:

- `cuda-pytorch`
  - NVIDIA baseline lane
  - benchmark records under this run currently use:
    - `implementation: "cuda-pytorch"`
    - `implementation_version: "cuda-pytorch"`
- `cannops`
  - Ascend CANN ops library baseline lane
  - benchmark records under this run currently use:
    - `implementation: "cann_ops_library"`
    - `implementation_version: "cannops"`
- `simt-v1`
  - Ascend SIMT operator lane
  - benchmark records under this run currently use:
    - `implementation: "simt"`
    - `implementation_version: "v1"`

This distinction is intentional:

- `run-name` is the publication and retrieval key
- record-level `implementation` and `implementation_version` are the normalized frontend data fields

### Naming Rules

- All segments must be lowercase ASCII.
- Segments are separated only by `-`.
- Do not embed timestamps in `run-name`.
- Do not add extra freeform tags to `run-name`.
- If historical capture time is needed, store it in JSON metadata fields, not in the run name.

## Benchmark Record Contract

Each `published/<run-name>/meta/benchmark-records.json` must use this shape:

```json
{
  "records": [
    {
      "schema_version": 1,
      "run_id": "opbench-ascend-950pr-cannops-softmax-realistic-float16",
      "operator": "softmax",
      "dataset": "realistic",
      "case_id": "gptj_attention",
      "family": "attention",
      "shape": [1, 16, 128, 128],
      "dtype": "float16",
      "backend": "ascend",
      "device_class": "950PR",
      "implementation": "cann_ops_library",
      "implementation_version": "cannops",
      "source_kind": "real_model",
      "source_project": "TritonBench",
      "source_model": "GPTJForCausalLM",
      "source_file": "hf_train/GPTJForCausalLM_train.json",
      "source_op": "aten._softmax.default",
      "metrics": {
        "latency_ms_avg": 0.016,
        "latency_ms_p50": 0.016,
        "latency_ms_p95": 0.018,
        "sample_count": 1
      },
      "accuracy": {
        "passed": true,
        "max_abs_error": 0.0,
        "max_rel_error": 0.0
      },
      "diff_ref": null
    }
  ]
}
```

Rules:

- `run_id` must exactly equal the parent `<run-name>`.
- `implementation` is the semantic execution class.
- `implementation_version` is the normalized version token used by the frontend and diff tooling.
- `device_class` is the normalized display and grouping key.
- `metrics.latency_ms_avg`, `metrics.latency_ms_p50`, `metrics.latency_ms_p95`, and `metrics.sample_count` are required frontend fields.
- `diff_ref` must be non-null only for SIMT records.

### NVIDIA Published Record Example

NVIDIA published records must follow the same schema:

```json
{
  "records": [
    {
      "schema_version": 1,
      "run_id": "opbench-nvidia-h800-cuda-pytorch-softmax-realistic-float16",
      "operator": "softmax",
      "dataset": "realistic",
      "case_id": "gptj_attention",
      "family": "attention",
      "shape": [1, 16, 128, 128],
      "dtype": "float16",
      "backend": "nvidia",
      "device_class": "H800",
      "implementation": "cuda-pytorch",
      "implementation_version": "cuda-pytorch",
      "source_kind": "real_model",
      "source_project": "TritonBench",
      "source_model": "GPTJForCausalLM",
      "source_file": "hf_train/GPTJForCausalLM_train.json",
      "source_op": "aten._softmax.default",
      "metrics": {
        "latency_ms_avg": 0.011,
        "latency_ms_p50": 0.011,
        "latency_ms_p95": 0.012,
        "sample_count": 1
      },
      "accuracy": {
        "passed": true,
        "max_abs_error": 0.0,
        "max_rel_error": 0.0
      },
      "diff_ref": null
    }
  ]
}
```

NVIDIA rules:

- published NVIDIA run names must currently use `cuda-pytorch` in the `run-name`
- published NVIDIA records must currently use `implementation: "cuda-pytorch"`
- published NVIDIA records must currently use `implementation_version: "cuda-pytorch"`
- `diff_ref` must remain `null` for NVIDIA records

## GPU Upload Publication Rules

GPU JSON upload is a special ingestion path, but it must converge to the same published contract.

Rules:

- the upload endpoint accepts GPU benchmark data only
- uploaded records must pass frontend and backend validation before publication
- accepted uploaded records must still be published as:
  - `published/<run-name>/meta/benchmark-records.json`
- uploaded GPU data must not introduce a separate directory shape
- uploaded GPU data must still be referenced through `published/index.json`
- uploaded GPU records must remain constrained to:
  - `backend: "nvidia"` or `backend: "gpu"`
  - `implementation: "cuda-pytorch"`
  - `implementation_version: "cuda-pytorch"`
  - `diff_ref: null`

Security rules:

- no source code
- no diffs
- no environment dumps
- no hostnames, usernames, paths, logs, or raw profiler payloads

Operational rule:

- upload enablement is controlled by server configuration
- publication shape is not allowed to change when upload is enabled

## Frontend Loading Contract

Frontend behavior must remain:

1. Load `published/index.json`
2. Read `runs`
3. Fetch each `published/<run-name>/meta/benchmark-records.json`
4. Merge all records into one array
5. Render whatever records are present

Rules:

- Partial publication is allowed.
- A published set does not need to cover all operators, all datasets, or all backends.
- The frontend must render available data without assuming completeness.

## Retrieval Rules

Published data is discovered by structured metadata, not by path guessing alone.

Rules:

- Use `published/index.json` to discover runs.
- Use `run_id`, `backend`, `device_class`, `implementation`, `implementation_version`, `operator`, `dataset`, and `dtype` for filtering and grouping.
- Do not reconstruct business logic from ad hoc filename parsing outside the canonical `run-name` grammar defined above.

## Current Canonical Published Runs

At the time of writing, the repository archives:

- `opbench-ascend-950pr-cannops-softmax-realistic-float16`
- `opbench-ascend-950pr-simt-v1-softmax-realistic-float16`

Any future additions must follow this same contract.
