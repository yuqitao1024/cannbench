# Sparse Attention HD128 Prefill Fused Design

## Scope

This design covers the first fused sparse attention kernel milestone:

- operator: `sparse_attention`
- family: `family_hd128`
- phase: `prefill`
- dtype: `float16`

Out of scope for this milestone:

- `family_hd512`
- `decode`
- `float32`
- `bfloat16`
- global performance tuning after functional bring-up

The public operator interface remains unchanged:

- `sparse_attention_forward(Tensor query, Tensor keys, Tensor values, Tensor indices, str phase, str family, bool causal) -> (Tensor, Tensor)`

Non-`float16` inputs and non-`hd128 prefill` cases continue to use the existing implementation path.

## Goal

Replace the current `hd128 prefill` two-stage flow:

1. score gather kernel writing `scores` to GM
2. postprocess kernel reading `scores` and gathered values from GM

with a single fused mixed kernel that:

- gathers keys and values on chip
- computes `QK`
- keeps score tiles on chip
- runs softmax and `lse` on AIV SIMT
- reduces weighted values on chip
- writes only final `output` and `lse` to GM

The fused kernel must eliminate both GM intermediate tensors:

- `scores`
- `selected_values`

## Current Baseline

Current `family_hd128` prefill flow in `sparse_attention.asc` is:

1. optional query pack
2. `run_sparse_attention_score_gather_family_hd128_tile(...)`
3. `run_sparse_attention_family_hd128_tile(...)`

The current score kernel already uses a mixed `AIC + AIV` structure and stable cross-core synchronization on Atlas 350.

The current postprocess kernel is an AIV SIMT kernel that consumes:

- GM `scores`
- GM `selected_values`
- GM `indices`

This means the expensive sparse attention middle state is still materialized in GM, which is the main target of this fusion.

## Proposed Architecture

Add a new internal fused launch for `family_hd128` prefill `float16`:

- new internal host entry:
  - `run_sparse_attention_family_hd128_prefill_fused_tile(...)`
- new internal device launch:
  - one fused mixed kernel dedicated to `hd128 prefill fp16`

The dispatcher in `sparse_attention_forward_privateuse1(...)` changes as follows:

- `family_hd128 + phase == "prefill" + query.dtype == float16`
  - use new fused path
- all other cases
  - keep current path unchanged

The fused path does not allocate GM tensors for:

- `scores`
- `selected_values`

`query pack` is not fused in this milestone.

Reason:

- this milestone is intentionally limited to `float16`
- `float16` already bypasses query pack today
- removing score/value GM intermediates is the dominant fusion win

## Kernel Pipeline

The fused kernel keeps the existing mixed-role split:

- `AIV` for sparse gather and SIMT postprocess work
- `AIC` for `QK`

Per row, the kernel executes an `AIV -> AIC -> AIV` pipeline.

For each logical row `(batch, head, query_token)`:

1. `AIV` gathers a tile of keys using `indices`
2. gathered key tile is staged for `AIC`
3. `AIC` computes `QK` for that tile
4. `AIC` keeps score results in `L0C`
5. `AIV` copies score tile from `L0C` to `UB` using Tensor API `CopyL0C2UB`
6. `AIV` performs:
   - score scaling
   - causal / invalid masking
   - online max tracking
   - online exp-sum tracking
7. `AIV` gathers the matching value tile using the same indices
8. `AIV` computes weighted partial output accumulation in `UB`
9. after all selected-token tiles complete:
   - `AIV` finalizes normalization
   - `AIV` writes final `output`
   - `AIV` writes final `lse`

Only final outputs are written to GM.

## On-Chip Buffering Strategy

### AIV-side buffers

`AIV` uses `UB` for:

- gathered key/value tiles needed for SIMT-side processing
- score tiles copied from `L0C`
- online softmax state
- partial output accumulation

### AIC-side buffers

`AIC` uses:

- current query tile
- gathered key tile
- `L0C` for score tile output

### Cross-role transfer

This design explicitly relies on same-AICore `L0C -> UB` transfer for the handoff from `AIC` to `AIV`.

That transfer is the core mechanism that lets postprocess consume score tiles without first materializing them in GM.

## Tiling Model

The fused kernel preserves the existing row-batch launch style that was already validated for stability:

- host launches the mixed kernel for up to `used_core_num` rows at a time
- each block handles one row in that launch batch
- host loops over `row_start += used_core_num`

This is intentionally more conservative than a single-launch full-row sweep.

Reason:

- it matches the synchronization shape already validated for `hd128`
- it reduces first-milestone debugging risk
- it still preserves multicore parallelism

Selected-token processing remains tiled inside the kernel.

The first implementation may keep the existing `hd128` selected-token tile size if it fits the new buffering plan. If buffer pressure requires change, the tile size may be reduced, but the row-level contract remains unchanged.

## Synchronization Model

Atlas 350 synchronization rules stay aligned with the validated `hd128` path:

- mixed mode: `4`
- only `subblock0` participates on AIV side
- per-launch active pair count is bounded so `flagId` remains in the legal range

The synchronization semantics change from:

- score gather kernel completion -> separate postprocess kernel launch

to:

- per-tile gather readiness
- per-tile `QK` completion
- per-tile `L0C -> UB` consumption safety

The design should continue using per-block flag assignment within the legal Atlas 350 range, following the same validated pattern as the current multicore `hd128` score kernel.

## Host-Side Changes

In `sparse_attention.asc`:

- add a dedicated fused launch helper for `family_hd128` prefill fp16
- remove GM `scores` allocation from that path
- remove GM `selected_values` preparation from that path
- keep existing old helper functions for all fallback paths

The dispatcher logic should be explicit:

- if `family == "family_hd128"` and `phase == "prefill"` and `query.scalar_type() == Half`
  - use fused helper
- else
  - use existing current implementation

This preserves compatibility while making the new path easy to isolate in testing.

## Correctness Requirements

The fused path must preserve current observable behavior for `hd128 prefill fp16`:

- output numerics remain within existing tolerance
- `lse` numerics remain within existing tolerance
- invalid sparse indices behave as zero contribution
- causal masking behavior matches current prefill behavior

Expected output dtype behavior remains unchanged:

- internal accumulation may remain `float`
- final returned `output` preserves current public contract

## Testing Strategy

### Local structural tests

Update or add build-shell/source-structure tests to verify:

- new fused `hd128 prefill` path exists
- old `hd128 prefill` path no longer allocates GM `scores` for fp16 fused case
- fused kernel uses mixed-kernel structure
- source references indicate `L0C -> UB` handoff
- fallback paths for decode, hd512, and non-fp16 remain present

### Remote correctness tests

Required first-pass remote validation:

- `hd128_prefill_smoke`
- `hd128_prefill_top2048`

At minimum, verify:

- no runtime crash
- no deadlock / hang
- outputs match current reference under existing tolerance

### Regression coverage

Existing decode and hd512 tests must continue passing because their paths are intentionally untouched.

## Error Handling and Fallback

If the fused path cannot support an input shape or dtype in this milestone, it must not silently enter a partially supported implementation.

For this milestone:

- only `family_hd128 + prefill + float16` uses the fused path
- all other cases continue using the existing path

This avoids widening unsupported behavior while the fused kernel is still being brought up.

## Future Extensions

This design is intentionally staged.

Planned later follow-ons:

1. `family_hd128` prefill `bfloat16`
2. `family_hd512` prefill fused path
3. decode fused path using the same `AIV -> AIC -> AIV` principle where applicable
4. performance tuning beyond first functional bring-up

The first implementation should keep file structure and host dispatch clean enough that those later steps can reuse the same pattern rather than requiring another architectural rewrite.

## Risks

Main technical risks:

- on-chip buffer pressure from simultaneous score tile, value tile, and accumulation state
- new mixed-kernel synchronization hazards around tile handoff
- online softmax integration introducing subtle numeric drift
- `L0C -> UB` layout handling being more restrictive than the current GM-based postprocess path

Mitigations:

- keep first milestone to `hd128 prefill fp16` only
- preserve validated multicore launch batching pattern
- validate first on smoke, then on `top2048`
- keep old path intact as fallback for all non-target cases

## Implementation Outline

1. add fused host helper and dispatch branch for `hd128 prefill fp16`
2. add fused device kernel source for `hd128 prefill`
3. integrate `AIV -> AIC -> AIV` pipeline with `L0C -> UB` handoff
4. remove GM middle tensor allocation from the fused path
5. add/update source-structure tests
6. remote validate smoke and realistic prefill cases
