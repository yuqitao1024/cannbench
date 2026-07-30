# Ascend DSA SIMT Custom-Op Rollout Design

## Status

Approved design for moving the current DSA SIMT path from Python reference
wrappers toward real Ascend `<<<>>>` custom-ops while preserving the existing
CannBench plugin boundaries.

This spec covers the full intended scope:

- `lightning_indexer`
- `sparse_attention`
- `prefill`
- `decode`

Implementation priority is allowed to be staged later in the plan, but scope is
defined here in full.

Current implementation status:

- the first real custom-op slice exists for `lightning_indexer prefill
  family_4x64`
- unsupported `lightning_indexer` phases/families continue to use the Python
  reference fallback
- `lightning_indexer decode`, `lightning_indexer family_64x128`, and all
  `sparse_attention` custom-op slices remain pending
- runtime exactness for the real `PrivateUse1` path still requires validation
  in an Ascend environment with `torch`, `torch_npu`, `bisheng`, and an
  available NPU runtime

## Goal

Build real Ascend SIMT custom-op implementations behind the existing CannBench
SIMT plugin interfaces for:

- `lightning_indexer`
- `sparse_attention`

across both:

- `prefill`
- `decode`

while preserving:

- the current operator plugin boundaries
- the current `implementation=simt, implementation_version=v1` integration
  boundary
- safe fallback behavior when a real custom-op is unavailable or a shape is not
  yet supported

## Non-Goals

This design does not introduce:

- `aclnn`
- workflow-level fused public kernels for `dsa_prefill` or `dsa_decode`
- concrete DSA branches in `cli.py`, `core/`, or shared backend classes
- generic-shape-first implementation
- a pure-SIMT strategy for matrix-heavy work better served by cube core

## Constraints

- Ascend-only scope
- keep all operator-specific logic inside:
  - `src/cannbench/operators/builtin/lightning_indexer/`
  - `src/cannbench/operators/builtin/sparse_attention/`
- keep workflow-specific logic inside:
  - `src/cannbench/operators/builtin/dsa_prefill/`
  - `src/cannbench/operators/builtin/dsa_decode/`
- continue to use `softmax`-style custom-op engineering shape:
  - `setup.py`
  - `install.sh`
  - `scripts/common.sh`
  - `register.asc`
  - Python `_C` loading
  - `torch.ops` dispatch
- matrix-heavy work stays on cube core in the long-term target architecture
- vector, sparse, control, merge, layout, and glue work moves to SIMT

## Intended Full Scope

The final intended scope covered by this spec is:

### `lightning_indexer`

- `prefill`
  - `family_4x64`
  - `family_64x128`
- `decode`
  - `family_4x64`
  - `family_64x128`

### `sparse_attention`

- `prefill`
  - `family_hd128`
  - `family_hd512`
- `decode`
  - `family_hd128`
  - `family_hd512`

The fact that implementation starts with only one family does not reduce the
required product scope above.

## Main Decision

Use the existing `simt/v1` package locations as the long-lived public
integration points for the real custom-op rollout:

- `src/cannbench/operators/builtin/lightning_indexer/simt/v1/aten_dsa_lightning_indexer/`
- `src/cannbench/operators/builtin/sparse_attention/simt/v1/aten_dsa_sparse_attention/`

These packages evolve from:

- Python reference wrapper only

to:

- real custom-op when `_C` is available and the current shape/phase is
  supported
- Python reference fallback for unsupported shapes, unsupported phases, or
  build/runtime environments where the custom-op is not available

This keeps:

- plugin integration stable
- versioning stable
- rollout risk low

and avoids duplicating the same dispatch path again under `v2`.

## Package Structure

Each operator package should follow the `softmax` SIMT engineering shape.

### `lightning_indexer`

- `aten_dsa_lightning_indexer/__init__.py`
  - load `_C`
  - expose `ops`
- `aten_dsa_lightning_indexer/ops.py`
  - route to real `torch.ops` when available and supported
  - otherwise route to Python reference fallback
- `aten_dsa_lightning_indexer/csrc/register.asc`
  - minimal Python extension entry
- `aten_dsa_lightning_indexer/csrc/*.asc`
  - host-side registration and bridge logic
- `aten_dsa_lightning_indexer/csrc/simt/*.asc`
  - actual `<<<>>>` kernel implementation
- `setup.py`
  - `bisheng -x asc --enable-simt`
- `install.sh`, `scripts/install.sh`, `scripts/common.sh`
  - Ascend environment and install flow

### `sparse_attention`

- `aten_dsa_sparse_attention/__init__.py`
  - load `_C`
  - expose `ops`
- `aten_dsa_sparse_attention/ops.py`
  - route to real `torch.ops` when available and supported
  - otherwise route to Python reference fallback
- `aten_dsa_sparse_attention/csrc/register.asc`
  - minimal Python extension entry
- `aten_dsa_sparse_attention/csrc/*.asc`
  - host-side registration and bridge logic
- `aten_dsa_sparse_attention/csrc/simt/*.asc`
  - actual `<<<>>>` kernel implementation
- `setup.py`
  - `bisheng -x asc --enable-simt`
- `install.sh`, `scripts/install.sh`, `scripts/common.sh`
  - Ascend environment and install flow

## Public Runtime Interfaces

### `lightning_indexer`

The runtime namespace remains:

```python
torch.ops.aten_dsa_lightning_indexer.lightning_indexer_forward(
    query,
    keys,
    weights,
    top_k: int,
    phase: str,
    family: str,
) -> Tensor
```

The Python package continues to expose:

```python
lightning_indexer_forward(query, keys, weights, *, top_k: int, phase: str, family: str)
```

### `sparse_attention`

The runtime namespace remains:

```python
torch.ops.aten_dsa_sparse_attention.sparse_attention_forward(
    query,
    keys,
    values,
    indices,
    phase: str,
    family: str,
    causal: bool,
) -> tuple[Tensor, Tensor]
```

The Python package continues to expose:

```python
sparse_attention_forward(
    query,
    keys,
    values,
    indices,
    *,
    phase: str,
    family: str,
    causal: bool,
)
```

## Routing Model

The runtime routing strategy is the same for both operators.

### Preferred path

Use real `torch.ops` when all of the following are true:

- `_C` is loaded
- the namespace op is registered
- the requested `phase` is implemented
- the requested `family` is implemented

### Fallback path

Use the existing Python reference wrapper when any of the following are true:

- `_C` is not built
- `_C` is not importable
- the namespace op is not registered
- the requested `phase` is not yet implemented
- the requested `family` is not yet implemented

This keeps `v1` usable throughout the rollout.

## `lightning_indexer` Semantics

The externally visible semantics remain aligned with the current reference:

```python
scores = einsum("bqhd,bcd->bqhc", query, keys)
scores = relu(scores)
scores = scores * weights.unsqueeze(-1)
reduced = scores.sum(dim=2)
indices = topk(reduced, top_k).indices.to(int32)
```

Output contract:

- shape: `[B, Q, top_k]`
- dtype: `int32`

The real custom-op implementation does not need to mirror PyTorch operator
composition internally. It only needs to preserve the output contract.

## `sparse_attention` Semantics

The externally visible semantics remain aligned with the current reference:

- gather selected keys and values
- compute query-key scores
- apply causal masking when required
- softmax over selected tokens
- reduce with selected values
- return:
  - attention output
  - log-sum-exp tensor

Output contract:

- output tensor shape follows the current plugin-local reference contract
- LSE tensor shape follows the current plugin-local reference contract

As with `lightning_indexer`, the real custom-op implementation may choose a
different internal decomposition as long as the public contract remains stable.

## First Deliverable

Although this spec covers the full target scope, the first real custom-op
deliverable should be:

- operator: `lightning_indexer`
- phase: `prefill`
- family: `family_4x64`

Reasons:

- smaller state space
- easier SIMT launch and buffer organization
- lower risk for proving the full custom-op toolchain
- best first step for validating:
  - build
  - install
  - registration
  - `torch.ops` dispatch
  - CannBench plugin integration

This is a priority decision, not a scope reduction.

## Programming Model

The long-term architecture remains the agreed mixed model:

- cube core for matrix-heavy work
- SIMT for vector, sparse, control, merge, and layout work

The first deliverable does not need to fully realize that decomposition. Its
job is to establish:

- a real `<<<>>>` execution path
- correct operator registration
- exact reference parity

Then later phases can move the implementation toward the full mixed cube-plus-
SIMT target.

## Rollout Phases

The design assumes the following rollout sequence.

### Phase 1

- `lightning_indexer`
- `prefill`
- `family_4x64`

### Phase 2

- `lightning_indexer`
- remaining families and `decode`

### Phase 3

- `sparse_attention`
- `prefill`
- first implemented family

### Phase 4

- `sparse_attention`
- remaining families and `decode`

### Phase 5

- performance-directed optimization toward the mixed cube-plus-SIMT target

## Testing And Acceptance

Acceptance applies to the full scope, with each phase validating the relevant
subset.

### 1. Build And Registration

Prove that:

- the package installs through the operator-local SIMT install flow
- `_C` imports successfully
- the expected `torch.ops` namespace entry exists

### 2. Wrapper Routing

Prove that:

- supported `phase + family` pairs route to the real custom-op
- unsupported `phase + family` pairs route to Python fallback

### 3. Correctness

Prove that:

- each newly implemented `phase + family` pair matches the current Python
  reference exactly
- `lightning_indexer` preserves `int32` output dtype
- `sparse_attention` preserves tuple-output contract

### 4. CannBench Integration

Prove that:

- `implementation=simt`
- `implementation_version=v1`
- operator-local plugin dispatch still works through the backend path
- DSA workflow coverage remains valid for both:
  - `dsa_prefill`
  - `dsa_decode`

### 5. Repository Verification

The repository must remain green under:

```bash
pytest -q
```

## Follow-On Work

After the first real custom-op succeeds, subsequent work should:

1. expand `lightning_indexer` family coverage
2. add `lightning_indexer decode`
3. bring `sparse_attention` onto the same real custom-op engineering shell
4. expand `sparse_attention` family and phase coverage
5. optimize toward the final mixed cube-plus-SIMT performance target
