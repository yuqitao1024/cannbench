# Single-Process Workflow Profiling Design

Date: 2026-08-11
Status: approved

## Goal

Replace CannBench's flattened component execution for every `OperatorWorkflow`
with one generic, single-process workflow runner. The runner must pass live device
tensors between steps, profile one physical launch sequence, preserve the current
published workflow record contract, and support Ascend SIMT, vllm-ascend, and
NVIDIA CUDA implementations without concrete operator-name branches in public
layers.

The first real-device target is the DeepSeek V3.2 DSA decode workflow:

```text
lightning_indexer -> indices -> sparse_attention
```

The design also applies to DSA prefill and to future workflow plugins expressed
through `OperatorWorkflow`.

## Current Problem

The CLI currently flattens every workflow into independent
`PreparedInputPlan` entries. A remote DSA workflow therefore executes as:

```text
process A: lightning_indexer
process B: lightning_indexer -> sparse_attention
```

Process B resolves the Attention `indices` input binding by rebuilding and
executing the Indexer. SIMT, vllm-ascend, and CUDA all inherit this behavior.
Published latency avoids double-counting the second Indexer through name
filtering, but the raw trace is not a physical workflow invocation and the exact
device tensor from process A cannot reach process B.

## Design Decisions

### Unified Default

All workflow `bench` executions use the single-process path by default, both
locally and remotely. There is no separate legacy mode. Normal non-workflow
operator execution remains unchanged.

The public CLI and common backend code detect workflow capability through
`OperatorPlugin.build_workflow` and `OperatorPlugin.list_workflows`. They do not
test for `dsa_decode`, `dsa_prefill`, or component operator names.

### One Raw Profile

Each workflow case produces:

```text
profile/<workflow-artifact-stem>/
  <raw profiler tree>
  profile-summary.json
  components/
    <step-index>-<operator>.json
```

`profile-summary.json` contains the workflow device latency. Each component
summary is derived from the same raw launch sequence. Raw files are not copied
into per-component directories.

The run still publishes one workflow benchmark record. Its schema, canonical
run name, run ID structure, operator, dataset, case, dtype, device class,
implementation fields, accuracy fields, and `metrics.latency_ms` field remain
unchanged.

### Full Capture And Ordered Name Attribution

Profilers collect the complete workflow launch sequence. CannBench does not
require MSTX or NVTX to establish workflow step boundaries.

`ProfileKernelSelection` gains terminal-kernel patterns for workflow
attribution. A non-final step must identify at least one terminal main kernel.
The parser reads raw launch rows in physical order, finds the last matching
terminal kernel for each non-final step, and partitions the sequence into
contiguous, non-overlapping step spans. It then applies the existing component
kernel-name selection only inside that component's span.

This rule handles generic helper names such as `cat`, `cast`, and `copy`: a row
can match more than one component's ordinary whitelist, but its physical span
allows it to belong to only one step. No raw row can contribute to two component
latencies.

The operator plugin owns implementation-specific kernel selection and terminal
patterns. Public profile code only implements ordered span partitioning.

The parser fails explicitly when:

- a non-final step has no terminal patterns;
- none of its terminal patterns occur;
- terminal boundaries are out of workflow order;
- a computed span is empty;
- selected workflow rows cannot produce a component summary.

Unselected rows remain in the raw profile and are not silently deleted.

### Timing Boundary

Workflow device latency is the sum of the selected rows in all disjoint
component spans:

```text
workflow_device_latency = sum(component_device_latency)
```

This preserves the current published device-work contract. Host dispatch,
profiler initialization, input materialization, static cache construction,
output copies, and correctness comparison remain outside the metric.

Detailed metrics such as `PipeTimeline` and `InstrTimeline` are attribution
runs. Their reported durations are not compared directly with `BasicInfo`
latency.

## Prepared Workflow Contract

Add serializable prepared-workflow types next to the existing prepared operator
input types:

```python
@dataclass(frozen=True)
class PreparedWorkflowStep:
    contract: str
    consumes: tuple[str, ...]
    produces: tuple[str, ...]
    prepared: PreparedOperatorInput


@dataclass(frozen=True)
class PreparedWorkflowInput:
    workflow: str
    phase: str
    dataset: str
    case_id: str
    steps: tuple[PreparedWorkflowStep, ...]
```

The JSON representation has its own schema version and embeds each step's
existing prepared operator payload. Reader validation requires:

- at least one step;
- non-empty contract and produced names;
- every consumed name to have been produced by an earlier step;
- no produced name to be declared twice;
- every embedded prepared case to match the workflow case ID;
- step order to remain identical after a write/read round trip.

The workflow plugin remains responsible for dataset mapping and component
ordering. For example, DSA's private `realistic -> realistic_decode` mapping
stays in the DSA package.

## Backend Execution

Add a generic `WorkflowBenchmarkRequest` containing backend, implementation,
implementation version, AIC metric name, and `PreparedWorkflowInput`.

`OperatorBackend` exposes workflow execution methods, and
`TorchOperatorBackend` supplies the shared sequential implementation:

1. Validate the workflow request and target device.
2. Build an `OperatorBenchmarkRequest` for the current prepared step.
3. Install or load the step implementation through existing implementation
   dispatch.
4. Build the step callable with explicit `bound_inputs` from the workflow
   output map. Explicit workflow bindings bypass recursive
   `_resolve_input_bindings`.
5. Execute the callable exactly once.
6. Map a scalar return to the single declared produced name, or map tuple/list
   elements to produced names by position.
7. Retain the returned device objects in the output map for downstream steps.
8. Synchronize only at the existing workflow measurement boundary, not between
   steps.

The downstream callable receives the same tensor object returned by the
producer. There is no host serialization, device-to-host copy, rematerialized
producer call, or detached tensor replacement.

Return arity mismatches, missing consumed names, duplicate output names, and
unsupported implementations are explicit errors.

## CLI And Remote Execution

The CLI keeps operator plans and workflow plans distinct. A workflow selection
produces one prepared workflow per case instead of flattening its steps.

Add an internal-only `internal-run-workflow` command that accepts one prepared
workflow manifest and writes one workflow result bundle. It is not a public
benchmark entry point.

Remote execution performs:

1. Preinstall every unique SIMT component once when SIMT is selected.
2. Upload one prepared workflow manifest.
3. Invoke one profiler command around one `internal-run-workflow` process.
4. Download one raw profile tree and one workflow perf tree.
5. Build component and workflow summaries from the downloaded raw tree.

Ascend `msopprof` uses a workflow launch-count budget derived from all component
selections and captures the full sequence. NVIDIA NCU likewise captures the
full workflow rather than selecting only one component NVTX range. Existing
operator-local NVTX annotations may remain, but correctness does not depend on
them.

Local profiling follows the same prepared workflow and backend execution path.

## Plugin Ownership

Component operator packages continue to own:

- ordinary profile kernel-name selection;
- implementation-specific terminal kernel patterns;
- launch-count expectations;
- CUDA NVTX annotations used by standalone component profiling.

Workflow operator packages continue to own:

- workflow schema and step order;
- component dataset mapping;
- consumes/produces names;
- workflow case selection.

No concrete operator names or DSA-specific conditions are added to
`cli.py`, common config, backend base classes, common profile parsing, or result
models.

## Verification

Implementation follows test-driven development.

Unit coverage must prove:

- prepared workflow JSON round trips without changing order or bindings;
- invalid consumes/produces graphs fail clearly;
- each workflow step executes exactly once;
- a downstream step receives the producer's exact tensor object;
- scalar and tuple/list outputs map correctly;
- recursive prepared input bindings are not executed in workflow mode;
- overlapping helper kernel names are assigned to one physical span only;
- missing or out-of-order terminal kernels fail attribution;
- local workflow CLI uses one workflow backend call;
- remote workflow execution uploads one manifest, starts one internal workflow
  process, and downloads one profile tree;
- remote SIMT preinstall still installs each unique component once;
- non-workflow local and remote commands remain unchanged;
- SIMT, vllm-ascend, and CUDA library requests all use the generic runner.

Repository verification includes `pytest -q`, `git diff --check`, and targeted
searches that reject new DSA/operator-name branches in public layers.

Real-device verification on Ascend 950PR includes:

- V3.2 decode correctness for SIMT v2 and vllm-ascend;
- one DSA prefill correctness run for both implementations;
- raw launch audit showing one Indexer sequence followed by one Attention
  sequence;
- fresh `BasicInfo` comparison under the same stack;
- separate `PipeTimeline` and `InstrTimeline` collections for V3.2 decode;
- local retention of raw profiles, component summaries, commands, environment
  manifest, revision, and loaded module provenance.

NVIDIA CUDA receives full automated path coverage. If no usable GPU endpoint is
available, the final report states that NVIDIA real-device validation remains
open.

## Alternatives Rejected

### MSTX/NVTX As The Required Boundary

Explicit ranges are attractive but create different profiler integration and
parsing paths for Ascend and NVIDIA. They are unnecessary because workflow
steps already execute serially and terminal kernel names can partition the raw
sequence. Ranges may remain supplementary evidence.

### DSA-Specific Standalone Profiler Script

A dedicated script could pass Indexer output to Attention quickly, but it would
duplicate framework materialization, implementation loading, artifact layout,
and profile selection. It would not complete the generic workflow issue and
would violate the plugin architecture.

### Keep Flattened Components And Add A Conformance Trace

The existing conformance runner proves tensor chaining for correctness, but it
does not change production `bench` behavior and cannot make the published
profile a true workflow invocation. Keeping flattened component runs therefore
does not meet the goal.

## Completion Criteria

The issue is complete when one local or remote workflow case maps to one Python
process, every step runs once, downstream steps receive live producer tensors,
one raw trace yields non-overlapping component summaries and one compatible
workflow record, all tests pass, and the Ascend V3.2 decode trace demonstrates
the expected physical order for both SIMT v2 and vllm-ascend.
