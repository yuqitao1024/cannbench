# Lightning Indexer FP16 Fused-Kernel Design

## Scope

This design covers the next `lightning_indexer` fusion milestone:

- operator: `lightning_indexer`
- dtype: `float16`
- families:
  - `family_4x64`
  - `family_64x128`
- phases:
  - `prefill`
  - `decode`

Out of scope for this milestone:

- `float32`
- `bfloat16`
- plugin-local fallback behavior changes
- workflow-level fusion of `dsa_prefill` or `dsa_decode`
- collapsing host tiling into a single full-operator launch

The public custom-op interface remains unchanged:

- `lightning_indexer_forward(Tensor query, Tensor keys, Tensor weights, int top_k, str phase, str family) -> Tensor`

## Goal

Turn the current `float16` fast paths from:

1. score kernel launch
2. postprocess/top-k kernel launch

into a fused-kernel implementation where each tile launch performs:

1. score computation
2. `relu`
3. head-weighted reduction
4. top-k merge into the running tile state

The host is still allowed to launch multiple tiles per operator invocation. The target is:

- single kernel implementation per family tile path
- not single launch for the entire operator

## Current Baseline

Today `lightning_indexer` is already a standalone custom op, but the `float16` fast paths are still split internally.

For both `family_4x64` and `family_64x128`, host code in `lightning_indexer.asc` currently:

1. allocates `best_scores_tile` / `best_indices_tile`
2. launches a score kernel
3. launches a postprocess kernel
4. repeats over context tiles

Recent fixes made this path stable, but the implementation is still not fused at kernel granularity.

Additional current-state details:

- `family_4x64` uses a correctness-first per-head score fallback
- `family_64x128` also uses a correctness-first per-head score fallback
- both families now have validated runtime and reference correctness on the current Ascend environment

## Main Decision

Implement fused tile kernels separately per family instead of building a shared generalized framework in this milestone.

Recommended structure:

- `family_4x64`
  - one fused tile kernel path for `float16`
- `family_64x128`
  - one fused tile kernel path for `float16`

Reason:

- the two families already carry different correctness constraints
- `4x64` and `64x128` previously failed for different reasons
- family-local fused kernels are easier to validate and debug than a new generalized abstraction

## Proposed Architecture

### Host-side contract

Host code continues to own:

- query tiling
- context tiling
- allocation of running tile state:
  - `best_scores_tile`
  - `best_indices_tile`
- final copy from tile outputs into the operator result tensor
- stream lifetime handling with `record_tensor_on_stream(...)`

Host code no longer allocates or uses intermediate GM score tensors for the fused `float16` paths.

### Device-side contract

Each fused tile kernel receives:

- `query_tile`
- `key_tile`
- `weights_tile`
- current `best_scores_tile`
- current `best_indices_tile`
- tiling metadata

Each fused tile kernel updates the running top-k state in-place for the current `(query_tile, context_tile)` region.

### Dispatch rules

Only these cases switch to the new fused tile implementation:

- `family_4x64 + float16 + prefill`
- `family_4x64 + float16 + decode`
- `family_64x128 + float16 + prefill`
- `family_64x128 + float16 + decode`

All other dtype and fallback paths remain unchanged.

## Kernel Data Flow

Within one fused tile launch, the kernel performs:

1. compute the score contribution for the active row/head slice
2. apply `relu`
3. multiply by the per-head weight
4. reduce across heads into a per-context scalar score
5. merge the new scores with the running `best_scores_tile`
6. update `best_indices_tile` with context offsets

The kernel therefore replaces the current GM boundary between:

- score production
- top-k postprocess

No intermediate `score_tile` should be materialized in GM for the fused `float16` fast path.

## Family-Specific Plan

### `family_4x64`

`family_4x64` should use a family-local fused tile kernel that preserves the currently validated per-head correctness behavior.

The key requirement is to avoid reintroducing the old broken multi-row score layout path. The fused implementation should therefore start from the currently stable row/head execution shape and move the reduction/top-k logic into the same kernel.

### `family_64x128`

`family_64x128` should also keep the currently validated correctness-first launch shape as the baseline for the fused kernel.

This milestone should not try to restore the old wider multi-row score path at the same time as introducing fusion. The fused kernel should be built on the stable path first, then optimized later.

## Phase Handling

For `lightning_indexer`, `prefill` and `decode` share the same mathematical work:

- `QK`
- `relu`
- weighted head reduction
- top-k selection

The difference is in the tensor shapes and the case family selection, not in the postprocess algorithm itself.

This milestone therefore uses one fused family implementation per family for both phases, instead of creating separate `prefill` and `decode` kernels with duplicated logic.

## Host-Side Refactor Plan

In `lightning_indexer.asc`:

- replace the current `score -> postprocess` helper chain for fused `float16` families
- remove GM `score_tile` allocation from the fused `float16` path
- keep tile-level `best_scores_tile` and `best_indices_tile`
- keep existing stream recording behavior
- leave non-`float16` and fallback paths unchanged

Expected end state:

- fused `float16` family path does not call a separate postprocess launcher
- fused `float16` family path does not allocate `score_tile` as a GM intermediate

## Correctness Requirements

The fused implementation must preserve current observable behavior for:

- `family_4x64`
- `family_64x128`
- `prefill`
- `decode`
- `float16`

Required correctness properties:

- output indices match the current reference path
- no runtime crash on current Atlas 350 remote environment
- top-k ordering remains descending by reduced score
- context offsets remain correct across context tiling

## Testing Strategy

### Local structure tests

Update source-structure tests to verify:

- fused `float16` family paths no longer allocate GM `score_tile`
- fused `float16` family paths no longer call separate postprocess launchers
- family-local fused kernel launch helpers exist
- non-`float16` and fallback paths remain present

### Remote correctness tests

Minimum required remote checks after each family is brought up:

- `family_4x64`
  - decode reference test
  - prefill reference test
- `family_64x128`
  - decode reference test
  - prefill reference test

The remote validation environment remains:

- Atlas 350
- `torch`
- `torch_npu`
- `bisheng`
- current `cannbench` remote worktree

## Risks

### Risk 1: reintroducing old row-layout correctness bugs

Both families previously needed correctness-first fallbacks. Fusion must not silently bring back the broken wider-row score layout.

Mitigation:

- build fused kernels from the currently validated execution shape
- treat wider-row optimization as a later step

### Risk 2: tile-state merge bugs

Moving top-k merge into the kernel introduces new in-kernel ordering and overwrite rules.

Mitigation:

- validate with exact reference tests first
- keep host tiling unchanged in this milestone

### Risk 3: stream lifetime regressions

Removing `score_tile` changes which tensors need allocator stream recording.

Mitigation:

- preserve current host-side `record_tensor_on_stream(...)` discipline
- re-run the remote correctness suite after the refactor

## Non-Goals For This Milestone

This milestone does not attempt to:

- fuse fallback/reference execution
- add `float32` or `bfloat16` fused kernels
- convert the full operator to one launch for all tiles
- optimize multicore utilization beyond correctness needs
- refactor `dsa_prefill` or `dsa_decode` workflow plugins

## Expected Result

After this milestone:

- `lightning_indexer` remains a standalone custom op
- `float16` fast paths for `family_4x64` and `family_64x128` use fused tile kernels
- `prefill` and `decode` both reuse those fused family paths
- workflow-level DSA execution remains two-stage:
  - `lightning_indexer`
  - `sparse_attention`
