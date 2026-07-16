# AGENTS.md

This file defines repository-specific rules for AI coding agents working in CannBench.

The goal is to keep operator integrations isolated, preserve the plugin architecture, and avoid reintroducing central branching on concrete operator names.

## Project Intent

CannBench is an operator benchmark framework for:

- NVIDIA PyTorch CUDA baselines
- Ascend CANN ops library baselines
- Ascend SIMT operator implementations
- Workflow-style fused operators built from component operators

The repository architecture is intentionally plugin-driven. New operator work should extend operator packages, not common framework layers.

## Hard Rules

### 1. Keep operator-specific logic inside operator packages

For a normal operator, new business logic should live under:

```text
src/cannbench/operators/builtin/<operator>/
```

This includes:

- dataset and case loading
- input materialization
- torch baseline callable construction
- SIMT module naming
- SIMT callable construction
- profile kernel selection
- profile launch-count rules
- payload-summary ordering
- external implementation adapters when they are operator-specific

Do not add concrete operator-name branches to common framework files just because one operator needs special handling.

### 2. Workflow operators must be independent plugins

A workflow or fused operator must be implemented as its own plugin package, for example:

```text
src/cannbench/operators/builtin/<workflow_operator>/
```

Do not centralize workflow-specific phase rules, component mappings, or dataset split mappings in shared top-level files.

Workflow-specific details must stay in the workflow operator package, including:

- workflow case schema
- workflow dataset loading
- workflow step expansion
- workflow component ordering
- workflow-only dataset mapping such as `realistic -> realistic_decode`

### 3. Do not hardcode concrete operators in public layers

Do not add concrete operator names or workflow names to:

- `src/cannbench/cli.py`
- `src/cannbench/core/config.py`
- `src/cannbench/backends/base.py`
- `src/cannbench/backends/torch_backend_base.py`
- `src/cannbench/backends/pytorch_backend.py`
- `src/cannbench/core/result.py`

If you think a change is required in one of these files, prove first that the requirement cannot be expressed through `OperatorPlugin` or an operator-local module.

### 4. Dataset validation belongs to the operator plugin

Do not maintain a global dataset allowlist in CLI or config code.

The public layer may accept `--dataset` as a plain string. The selected operator plugin is responsible for deciding whether that dataset exists.

This is required so operator-private dataset splits can be added without editing public layers.

### 5. Execution-path branching is allowed only by implementation type

Public backends may branch on implementation class, for example:

- `simt`
- `cuda_library`
- `vllm_ascend`

Public backends must not branch on specific operator names.

### 6. Implementation-level tests must live with the operator

Tests that validate implementation source layout or implementation-specific behavior must live under the operator package, for example:

```text
src/cannbench/operators/builtin/<operator>/simt/test/
```

Do not place implementation-level tests in global `tests/` if they are really checking one operator's source tree or one operator's custom kernel project.

Global `tests/` should be reserved for framework-level behavior:

- registry
- plugin discovery
- CLI
- publish
- serve
- profile parser behavior
- backend dispatch behavior

### 7. Prefer plugin extension over backend helper growth

Do not add helper methods such as:

- `_softmax`
- `_index_add`
- `_topk`
- `_lightning_indexer`
- `_sparse_attention`

to common backend base classes.

If a torch baseline needs operator logic, build that callable inside the operator plugin.

### 8. Keep published data compatible with current contracts

When updating benchmark records, run names, or published artifacts:

- preserve the current published data contract
- preserve canonical run-name structure
- keep frontend-facing record schemas stable unless the change is intentional and coordinated

If a schema change is required, update the relevant contract or guide in `docs/`.

### 9. Ascend SIMT operator API boundary

For Ascend SIMT and mixed SIMT/TensorAPI operator implementations in this repository:

- use only C API, Tensor API, and SIMT API in operator source code
- kernel-local Mutex API may be used for intra-core pipeline synchronization when the same behavior cannot be expressed with the allowed copy/compute ordering alone
- do not introduce new dependencies on C++ Basic API facilities

In practice, avoid adding new usage of:

- `basic_api/kernel_basic_intf.h`
- `basic_api/kernel_operator_block_sync_intf.h`
- `kernel_operator.h`
- `AscendC::LocalTensor`
- `SetFlag` / `WaitFlag` / `PipeBarrier` style Basic API synchronization helpers
- `CrossCoreSetFlag` / `CrossCoreWaitFlag` style inter-core synchronization helpers

Allowed exception inside the operator-local SIMT boundary:

- `AscendC::Mutex::Lock`
- `AscendC::Mutex::Unlock`

Use this exception only for kernel-local pipeline synchronization. Do not use it to reintroduce inter-core coordination or to justify broader Basic API usage.

This is the target implementation boundary for ongoing operator work.

Transitional note:

- when a task is explicitly prioritized as "function first", agents may temporarily continue debugging or validating existing code that still contains legacy Basic API usage
- however, new design work and follow-up cleanup should converge back to the `C API + Tensor API + SIMT API` boundary

## Expected Operator Package Shape

A normal operator package should usually look like:

```text
src/cannbench/operators/builtin/<operator>/
  __init__.py
  cases.py
  materialize.py
  data/
    smoke.json
    realistic.json
    stress.json
```

A SIMT-capable operator may additionally include:

```text
  simt/
    test/
    v1/
    v2/
    v3/
```

A workflow operator should usually look like:

```text
src/cannbench/operators/builtin/<workflow_operator>/
  __init__.py
  cases.py
  materialize.py
  data/
    smoke.json
    realistic.json
    stress.json
```

## Expected Plugin Responsibilities

Use `OperatorPlugin` as the primary extension surface.

An operator plugin should own as much of the following as applicable:

- `get_dataset`
- `get_case`
- `materialize_inputs`
- `build_torch_callable`
- `build_simt_callable`
- `simt_module_name`
- `build_cuda_library_callable`
- `build_vllm_ascend_callable`
- `build_profile_kernel_selection`
- `profile_launch_count`
- `payload_summary_key_order`
- `build_workflow`
- `list_workflows`
- `component_operator_names`

If a new requirement can be represented by one of these hooks, use the hook instead of changing common code.

## Before Editing Common Layers

Before changing any file under:

- `src/cannbench/backends/`
- `src/cannbench/core/`
- `src/cannbench/cli.py`

verify all of the following:

1. The behavior cannot be expressed in the operator plugin.
2. The behavior is genuinely shared across multiple operators.
3. The change does not introduce a concrete operator-name branch.
4. The change does not move workflow-specific rules into public layers.

If any answer is "no", do not make the common-layer change.

## Documentation Duties

When changing extension boundaries, also update the relevant docs if needed:

- `docs/guides/adding-operator-benchmark.zh-CN.md`
- `docs/guides/cli-usage.md`
- `docs/contracts/published-data-contract.md`

## Verification Expectations

At minimum, after architecture-affecting changes, run:

```bash
pytest -q
```

If you changed plugin boundaries, also check for accidental public-layer hardcoding with targeted searches before claiming the work is done.
