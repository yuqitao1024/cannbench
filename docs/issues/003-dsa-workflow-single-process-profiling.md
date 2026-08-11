# Issue 003: Run DSA workflow profiling in one process

## Status

Resolved

## Resolution

Workflow-capable plugins now produce one prepared workflow manifest per case.
Local and remote `bench` paths pass that manifest to the generic workflow
runner, which executes every step in one Python process and keeps produced
device objects in an in-memory output map. The downstream step therefore
receives the exact tensor object returned by its producer; prepared workflow
steps do not use recursive `input_bindings`.

Profiling wraps one `internal-run-workflow` invocation and captures the full
launch sequence. Component plugins provide ordinary kernel-name patterns and
terminal main-kernel patterns. The common parser uses the last terminal match
for each non-final step to partition physical CSV rows into ordered,
non-overlapping spans, then applies each component's ordinary selection only
inside its own span.

Each workflow case now writes:

```text
profile/<workflow-artifact-stem>/
  <raw profiler tree>
  profile-summary.json
  components/
    0-<first-operator>.json
    1-<second-operator>.json
```

The workflow latency is the exact sum of the selected component rows from that
single raw profile. The published benchmark record remains one workflow record
with the existing schema, canonical run name, and `metrics.latency_ms` field.
Ascend SIMT, vllm-ascend, and NVIDIA CUDA library implementations all use the
same workflow path.

## Context

The DSA decode and prefill workflows have a real device-side dependency:

```text
lightning_indexer -> indices -> sparse_attention
```

The current workflow expansion turns these components into independent
`PreparedInputPlan` entries. Each component is then benchmarked and profiled
through a separate `cannbench internal-run` process. The prepared
`sparse_attention` input contains an `indices` input binding that describes how
to run `lightning_indexer`; it does not contain a device tensor produced by the
earlier workflow step.

Consequently, profiling the workflow currently behaves like this:

```text
process A: lightning_indexer                         # component profile
process B: lightning_indexer -> sparse_attention     # attention profile
```

Process B must rerun `lightning_indexer` because an NPU tensor from process A
cannot be passed directly into it. A multi-stage Indexer implementation can
therefore produce several Lightning Indexer kernel rows inside the raw Sparse
Attention profiler artifact even though there is only one logical Indexer call
in the workflow.

The Sparse Attention plugin currently excludes explicitly named Lightning
Indexer kernels from its SIMT latency summary, and the published workflow
latency is formed by adding the separately reduced component summaries. This
avoids counting the named Indexer kernels twice, but the raw execution does not
match the actual workflow and relies on kernel-name filtering to recover the
intended measurement boundary.

## Problem

- Lightning Indexer is physically executed again while profiling Sparse
  Attention.
- The raw Sparse Attention profile contains kernels from its upstream producer.
- Component attribution depends on substring-based kernel-name filters rather
  than an explicit workflow execution boundary.
- Generic vLLM Ascend lowering names such as `cat`, `cast`, `slice`, or `add`
  can be ambiguous if both components emit them.
- The reported workflow latency is a synthetic sum of independent component
  profiles, not a measurement derived from one end-to-end workflow execution.

## Expected Behavior

Profile the DSA workflow with one Python process and one live set of device
tensors:

```python
indices = lightning_indexer(...)
output, lse = sparse_attention(..., indices=indices)
```

The exact Indexer output tensor should be passed directly to Sparse Attention.
Lightning Indexer should execute once per workflow invocation.

The profiler output must still expose auditable component boundaries:

- Lightning Indexer latency includes only its selected kernels.
- Sparse Attention latency includes only its lowering, main, and postprocess
  kernels.
- Workflow latency is the sum of the selected stages from that same workflow
  invocation, or a directly measured end-to-end device interval with the
  component breakdown retained.
- Input materialization, initialization, and profiler replay are classified
  explicitly and are not silently attributed to either component.

## Implemented Direction

The workflow execution path now:

1. Materializes the component inputs once.
2. Builds both component callables in the same process and on the same device.
3. Calls Lightning Indexer once and passes its returned NPU tensor directly to
   Sparse Attention.
4. Records one launch sequence and attributes it through plugin-owned terminal
   kernel patterns without rerunning the producer.
5. Reduces the single raw profile according to the two component plugins'
   selection policies.
6. Preserves the current published benchmark-record schema and canonical run
   names.

Keep the implementation workflow-local. Do not add DSA operator-name branches
to the CLI, common backend, profile parser, or result layer. If a shared runner
extension is required, it should operate on generic `OperatorWorkflow` steps
and tensor bindings rather than concrete operator names.

## Acceptance Criteria

- One profiled DSA workflow invocation launches Lightning Indexer exactly once.
- Sparse Attention receives the exact tensor object, storage, and dtype returned
  by Lightning Indexer without host serialization or regeneration.
- Raw profile evidence shows the expected ordered sequence:
  `lightning_indexer -> sparse_attention`.
- Component summaries are derived from that single execution and contain no
  cross-component kernel attribution.
- Workflow latency and component latency accounting are documented and
  reproducible from raw profiler rows.
- SIMT and vLLM Ascend decode and prefill workflows pass accuracy checks.
- Existing standalone operator benchmarking remains supported.
- Published data contracts and run-name structure remain unchanged.

## Current Evidence

The 2026-08-05 V3.2 decode profiler artifacts show the issue directly. The
SIMT V2 Sparse Attention raw profile contains the multi-stage Lightning Indexer
kernels used to produce `indices`, followed by the Sparse Attention fused and
combine kernels. The Sparse Attention summary correctly excludes the named
Indexer kernels, but only after both components have executed in the same raw
profile collection.

Relevant implementation points:

- `src/cannbench/operators/builtin/dsa_decode/__init__.py`: declares the
  `indices` producer binding.
- `src/cannbench/operators/builtin/dsa_prefill/__init__.py`: declares the same
  dependency for prefill.
- `src/cannbench/backends/torch_backend_base.py`: resolves a binding by building
  and executing the producer callable.
- `src/cannbench/cli.py`: expands workflow steps into independent prepared
  execution plans and later aggregates component records.
- `src/cannbench/operators/builtin/sparse_attention/__init__.py`: defines the
  current kernel-name selection used to exclude unrelated raw rows.

## Non-Goals

- Fusing Lightning Indexer and Sparse Attention into one device kernel.
- Changing either operator's numerical contract or implementation dispatch.
- Removing standalone component profiles.
- Changing the published benchmark record schema.
