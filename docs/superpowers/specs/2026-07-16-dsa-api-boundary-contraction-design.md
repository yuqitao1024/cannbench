# DSA SIMT API Boundary Contraction Design

Date: 2026-07-16

## Goal

Contract the Ascend DSA SIMT implementation boundary for `lightning_indexer` and
`sparse_attention` to `C API + Tensor API + SIMT API` only.

This change must preserve all of the following:

- current operator plugin boundaries
- current family and phase coverage
- current public custom-op entry points
- current main fusion paths

This change must remove new or existing dependencies on C++ Basic API
facilities from the target score-kernel implementations.

## Scope

This design applies only to operator-local SIMT implementation files under:

- `src/cannbench/operators/builtin/lightning_indexer/simt/v1/aten_dsa_lightning_indexer/csrc/simt/lightning_indexer_score_family_4x64.asc`
- `src/cannbench/operators/builtin/lightning_indexer/simt/v1/aten_dsa_lightning_indexer/csrc/simt/lightning_indexer_score_family_64x128.asc`
- `src/cannbench/operators/builtin/sparse_attention/simt/v1/aten_dsa_sparse_attention/csrc/simt/sparse_attention_score_family_hd128.asc`
- `src/cannbench/operators/builtin/sparse_attention/simt/v1/aten_dsa_sparse_attention/csrc/simt/sparse_attention_score_family_hd512.asc`

The following are explicitly out of scope:

- plugin registration changes
- backend dispatch changes
- workflow expansion changes
- dataset contract changes
- public CLI or config changes
- score-path fallback to `at::matmul`, `at::bmm`, or bridge-side full-operator rewrites

## Current State

`lightning_indexer` and `sparse_attention` already expose independent operator
plugins and custom-op entry points. The remaining boundary violations are
localized inside score-kernel implementation files.

Current violations include:

- `basic_api/kernel_basic_intf.h`
- `basic_api/kernel_operator_block_sync_intf.h`
- `kernel_operator.h`
- `AscendC::LocalTensor`
- `SetFlag`
- `WaitFlag`
- `PipeBarrier`
- `CrossCoreSetFlag`
- `CrossCoreWaitFlag`

These violations are concentrated in:

- `lightning_indexer` score kernels for `family_4x64` and `family_64x128`
- `sparse_attention` score kernels for `family_hd128` and `family_hd512`

## Design Constraints

### Repository constraints

- Keep operator-specific logic inside operator packages.
- Do not introduce new public-layer branching on operator names.
- Do not move workflow-specific rules into shared framework files.
- Do not add backend helper growth for operator-specific logic.

### API-boundary constraints

Allowed implementation boundary:

- C API
- Tensor API
- SIMT API

Disallowed implementation boundary:

- Basic API synchronization helpers
- Basic API local-tensor facilities
- `kernel_operator.h`

### Behavioral constraints

The implementation must preserve:

- `lightning_indexer`
  - `prefill` and `decode` behavior already covered by current plugin and custom-op paths
  - `family_4x64`
  - `family_64x128`
- `sparse_attention`
  - `prefill`
  - `decode`
  - `family_hd128`
  - `family_hd512`

The implementation must not regress the current main fusion intent:

- `lightning_indexer` keeps tiled score body plus existing postprocess path
- `sparse_attention` keeps fused gather-plus-score intent and avoids full
  selected-KV materialization

## Recommended Approach

Use operator-local score-kernel rewrites while keeping public and bridge shapes
unchanged.

This means:

- rewrite only the score-kernel implementation files
- preserve host bridge call sites and custom-op names
- preserve existing operator plugin interfaces
- preserve current family dispatch
- preserve current workflow/operator composition

This is the lowest-risk approach because it removes the violating API surface
without expanding shared framework abstractions or changing plugin contracts.

## Detailed Design

### 1. `lightning_indexer`

#### Public and bridge behavior

No changes:

- keep `aten_dsa_lightning_indexer::lightning_indexer_forward`
- keep current tiled bridge loops over query and context
- keep existing postprocess kernels and launch sites
- keep existing result tensor layouts and types

#### Internal score-kernel design

The two score kernels remain Tensor API score bodies:

- `MakeTensor`
- `MakeCopy`
- `MakeMmad`

Required changes:

- remove Basic API includes
- remove all explicit Basic API event synchronization
- remove `PipeBarrier`
- express the tile pipeline using only allowed Tensor API and SIMT-compatible
  ordering

Preserved properties:

- tile-local query/context score computation
- output score layout consumed by current postprocess kernels
- no fallback to bridge-side `at::matmul` or `at::bmm`

### 2. `sparse_attention`

#### Public and bridge behavior

No changes:

- keep `aten_dsa_sparse_attention::sparse_attention_forward`
- keep current family and phase dispatch
- keep existing bridge helpers:
  - query pack
  - score gather
  - postprocess
  - decode-direct
- keep output contracts `(output, lse)`

#### Internal score-gather design

The score path remains split by responsibility:

- SIMT/AIV side performs gather and layout-friendly packing from `indices`
- Tensor API/AIC side performs score-body compute
- postprocess/decode-direct continues to consume `scores`

Required changes:

- remove Basic API includes
- remove `kernel_operator.h`
- remove `AscendC::LocalTensor`
- remove `CrossCoreSetFlag` and `CrossCoreWaitFlag`
- remove all Basic API event synchronization

Replacement strategy:

- replace the current Basic-API-based cross-core L1 handoff with
  operator-local tile scratch handoff
- the scratch lifetime is limited to the active score tile
- gathered key data remains tile-local rather than becoming a full public or
  long-lived selected-keys tensor

Preserved properties:

- gather remains inside the operator score path
- full selected-KV materialization is still avoided
- bridge does not regress to `at::matmul` or `at::bmm`
- family/phase dispatch stays unchanged

## Data Flow

### `lightning_indexer`

Target data flow stays:

1. bridge tiles query and context
2. score kernel computes tile-local score body using Tensor API
3. postprocess kernel applies weighting and TopK update
4. final selected indices remain the operator result

Only the internal score-kernel synchronization strategy changes.

### `sparse_attention`

Target data flow stays:

1. optional query pack prepares query tile
2. score-gather path gathers selected keys for the active tile
3. Tensor API score body produces score tile
4. postprocess or decode-direct consumes scores and values
5. final `(output, lse)` are written directly

Only the internal tile handoff changes from Basic API coordination to allowed
operator-local scratch coordination.

## Error Handling and Failure Policy

This work must fail closed instead of silently broadening behavior.

If a target path cannot be expressed within the allowed API boundary without
changing operator behavior, do not:

- add new Basic API usage
- route the operator through public-layer special cases
- degrade to an unrelated bridge-side fallback path

Instead:

- keep the failure local to the implementation work
- surface the constraint through tests or explicit runtime guard behavior if
  needed

For this change, the expected implementation path is considered feasible, so the
design assumes no behavioral fallback is introduced.

## Testing Strategy

### New source-boundary tests

Add or update build-shell tests so the target score sources assert the absence
of disallowed API usage.

`lightning_indexer` score-source assertions:

- no `basic_api/`
- no `SetFlag`
- no `WaitFlag`
- no `PipeBarrier`

`sparse_attention` score-source assertions:

- no `basic_api/`
- no `kernel_operator.h`
- no `AscendC::LocalTensor`
- no `SetFlag`
- no `WaitFlag`
- no `PipeBarrier`
- no `CrossCoreSetFlag`
- no `CrossCoreWaitFlag`

### Existing behavior tests that must remain green

- SIMT build-shell tests for both operators
- dispatch tests
- reference-path tests
- custom-op accuracy tests
- backend integration tests already covering SIMT execution paths

### Required verification

At minimum run:

```bash
pytest -q
```

In addition, run targeted searches to confirm no disallowed Basic API usage
remains in the target operator score sources and no accidental public-layer
hardcoding was introduced.

## Acceptance Criteria

The work is complete only when all of the following are true:

1. The four target score-kernel source files use only `C API + Tensor API +
   SIMT API`.
2. No disallowed Basic API symbols remain in the target files.
3. `lightning_indexer` preserves current family coverage and bridge/postprocess
   structure.
4. `sparse_attention` preserves current family and phase coverage.
5. `sparse_attention` preserves the main gather-plus-score fusion intent and
   does not reintroduce full selected-KV materialization.
6. Existing operator/plugin/public-layer contracts remain unchanged.
7. Verification passes, including `pytest -q`.

## Non-Goals

This design does not attempt to:

- redesign operator APIs
- redesign workflow operators
- optimize numerical precision policy
- remove all GM traffic
- collapse multi-kernel score/postprocess flows into a single monolithic kernel
- generalize a shared DSA internal kernel framework

## Implementation Notes

Implementation should proceed test-first:

1. add failing source-boundary tests
2. confirm they fail for the expected Basic API usage
3. rewrite implementation locally inside operator packages
4. rerun focused tests
5. rerun full verification

This keeps the architecture change narrow and makes any regression visible at
the operator package boundary where it belongs.
