# Ascend SIMT DSA Operators Design

## Status

Design accepted. Initial implementation scaffolding exists for:

- `lightning_indexer` SIMT dispatch
- `sparse_attention` SIMT dispatch
- v1 reference wrappers for `prefill` and `decode`

High-performance mixed cube-plus-SIMT kernels remain to be substituted behind
the same operator-local interfaces. Scope remains limited to Ascend-only DSA
operator acceleration in CannBench, without changing workflow boundaries or
public backend architecture.

## Context

CannBench models DSA as workflow operators built from two component operators:

- `lightning_indexer`
- `sparse_attention`

The current benchmark structure already keeps DSA workflow expansion inside
operator plugins:

- `dsa_prefill` expands to `lightning_indexer -> sparse_attention`
- `dsa_decode` expands to `lightning_indexer -> sparse_attention`

That boundary is correct and should be preserved. The new requirement is to add
high-performance Ascend SIMT implementations for the two component operators
without adding concrete DSA branches to public backend layers.

## Goal

Build Ascend-only high-performance implementations for DSA component operators
using a mixed programming model:

- matrix-heavy work stays on cube core via SIMD-style compute
- vector, sparse, layout, control, and merge work moves to SIMT

The design must cover both:

- `prefill`
- `decode`

Both execution modes are required product scope for this work. Priority affects
implementation order only; it does not reduce feature scope.

Implementation priority is:

1. `prefill`
2. `decode`

The target is not merely functional parity. The target is to approach or exceed
the current `vllm_ascend` or external CUDA-library baselines on the benchmarked
case families.

## Non-Goals

- Do not fuse `dsa_prefill` or `dsa_decode` into a single public workflow-level
  kernel in the first phase.
- Do not add DSA-specific dispatch branches to `cli.py`, `core/`, or shared
  backend classes.
- Do not optimize for fully generic shapes first.
- Do not use a pure-SIMT implementation for large matrix products that are
  better served by cube core.

## Existing Case Families

After case expansion, the current benchmark set contains at least three
important shape families.

### Family A: A5-compatible

`lightning_indexer`

- `index_heads=64`
- `index_dim=128`
- `top_k in {512, 1024}`

`sparse_attention`

- `query_heads=64`
- `kv_heads=1`
- `head_dim=512`
- `selected_tokens in {512, 1024}`

### Family B: V4-Pro-like

`lightning_indexer`

- `index_heads=64`
- `index_dim=128`
- larger `batch`, `query_tokens`, `context_tokens`
- `top_k=1024`

`sparse_attention`

- `query_heads=128`
- `kv_heads=1`
- `head_dim=512`
- `selected_tokens=1024`

### Family C: V3.2

`lightning_indexer`

- `index_heads=4`
- `index_dim=64`
- `top_k=2048`

`sparse_attention`

- `query_heads=128`
- `kv_heads=1`
- `head_dim=128`
- `selected_tokens=2048`

These families differ enough that a single heavily parameterized kernel family
is unlikely to remain competitive across all of them.

## Main Decision

Use one fast-path framework per operator, but split the underlying high
performance kernels into two kernel families rather than one monolithic fast
path.

Recommended grouping:

- `lightning_indexer`
  - `family_64x128`: covers A5-compatible and V4-Pro-like
  - `family_4x64`: covers V3.2
- `sparse_attention`
  - `family_hd512`: covers A5-compatible and V4-Pro-like
  - `family_hd128`: covers V3.2

Each operator also keeps a lower-optimization fallback path for correctness,
benchmark coverage, and future extension.

## Alternatives Considered

### 1. Single Generic Fast Path

One fast path with runtime-parameterized tiling, buffering, and merge logic.

Pros:

- less code branching
- simpler mental model

Cons:

- tile and buffer choices become compromise values
- register and local-memory budgets drift toward the largest cases
- likely to leave substantial performance on the table for at least one family

Rejected for first phase because it conflicts with the performance target.

### 2. Operator-Local Dual Path

Per operator:

- one fast-path framework with family-specific specializations
- one fallback path

Pros:

- matches CannBench plugin boundaries
- keeps workflow architecture stable
- allows aggressive specialization only where it matters
- contains operator-specific complexity inside operator packages

Cons:

- requires explicit family dispatch inside each operator plugin
- duplicates some execution-path structure between families

Chosen approach.

### 3. Workflow-Level Fused Engine

Implement `dsa_prefill` and `dsa_decode` as fully fused operators.

Pros:

- highest theoretical performance ceiling

Cons:

- conflicts with current architecture and workflow contracts
- raises validation and rollout complexity sharply
- obscures component-level benchmarking

Deferred. Not appropriate for first phase.

## Architecture

Workflow boundaries remain unchanged.

- `dsa_prefill` stays a workflow over component operators
- `dsa_decode` stays a workflow over component operators

Implementation stays inside component operator packages:

- `src/cannbench/operators/builtin/lightning_indexer/simt/`
- `src/cannbench/operators/builtin/sparse_attention/simt/`

Each operator plugin provides:

- case-family dispatch logic
- fast-path entry selection
- fallback entry selection
- profile kernel naming and launch-count behavior as needed

No public backend changes should branch on DSA operator names.

## Programming Model

Use a mixed execution model:

- cube core / SIMD for high-arithmetic-intensity matrix work
- SIMT for vector processing, sparse access, control, and reductions

The division of responsibility should stay stable across families.

### Cube Core Responsibilities

- main `Q x K^T` score computation
- selected-token attention matmul body
- `P x V`-style dense compute on already selected data

### SIMT Responsibilities

- page and block-table walk
- sparse gather and sparse scatter support
- layout transform and metadata interpretation
- dequant scale application
- head reduction where needed
- local and hierarchical TopK selection
- online softmax bookkeeping
- partial merge and split-KV merge logic

## Operator Design

### lightning_indexer

The performance-critical rule is that the implementation must not materialize
full logits to global memory before TopK.

#### Prefill

For `prefill`, the main path is:

1. cube core computes tile-local `Q x K^T`
2. SIMT applies weights and performs per-tile reduction
3. SIMT computes local TopK candidates
4. tile outputs feed a hierarchical TopK merge
5. final output is only the selected indices

Desired property:

- full logits never become a long-lived global-memory tensor

#### Decode

For `decode`, the same family split remains, but scheduling changes:

- `query_tokens` is typically very small
- metadata and page-walk overhead matters more
- launch count and synchronization become first-class concerns

The decode path should use smaller query tiling and shallower merge structure
than the prefill path.

### sparse_attention

The performance-critical rule is that the implementation must not expand a
large full selected-KV temporary tensor if the same access pattern can be
streamed.

#### Prefill

For `prefill`, the main path is:

1. SIMT consumes sparse indices and interprets page or block metadata
2. SIMT organizes selected-KV access
3. cube core performs selected-token `QK` and `PV` compute
4. SIMT performs dequant, masking, online softmax, and merge
5. final outputs are attention output and associated reduction state

Desired property:

- selected-KV access is streamed or block-buffered rather than globally
  materialized in full

#### Decode

For `decode`, the emphasis shifts to latency:

- query workload is smaller
- fixed overhead from sparse gather, metadata handling, and merge is larger
  relative to compute

The decode implementation should therefore fuse as much SIMT-side pre and post
processing around the main kernel path as is practical without destroying
maintainability.

## Fast-Path Family Dispatch

Dispatch stays operator-local.

### lightning_indexer

`family_64x128`

- match `index_heads=64 && index_dim=128`
- covers A5-compatible and V4-Pro-like

`family_4x64`

- match `index_heads=4 && index_dim=64`
- covers V3.2

Any unmatched shape uses fallback.

### sparse_attention

`family_hd512`

- match `head_dim=512 && kv_heads=1`
- sub-variants may further specialize on:
  - `query_heads in {64, 128}`
  - `selected_tokens in {512, 1024}`

`family_hd128`

- match `head_dim=128 && kv_heads=1 && query_heads=128`
- current benchmark target is V3.2 with `selected_tokens=2048`

Any unmatched shape uses fallback.

## Fallback Strategy

Fallback remains inside the operator plugin and serves three purposes:

- correctness
- benchmark coverage
- profile visibility

Fallback is not expected to match the external-library fast path.

A fallback implementation may still be SIMT-based, but can use more conservative
tiling and less aggressive fusion.

## Implementation Priority

Implementation order should be:

1. `lightning_indexer` prefill `family_64x128`
2. `sparse_attention` prefill `family_hd512`
3. `lightning_indexer` prefill `family_4x64`
4. `sparse_attention` prefill `family_hd128`
5. decode implementations for both operators and both kernel families

Rationale:

- prefill offers larger compute envelopes and better opportunity to validate the
  mixed cube-plus-SIMT design
- prefill is the right place to prove:
  - no full-logit materialization
  - no full selected-KV expansion
- decode is more sensitive to fixed overhead and should follow after the main
  dataflow is proven

This priority order does not make decode optional. A complete delivery must
include both:

- `dsa_prefill`
- `dsa_decode`

## Dataflow Contract Between Operators

The `lightning_indexer` output layout and metadata contract must be designed so
that `sparse_attention` can consume it without large reshapes or global-memory
repacking.

This implies:

- stable sparse-index layout conventions
- stable metadata semantics for page or block interpretation
- avoidance of redundant intermediate layout conversion between workflow steps

Workflow-level performance will not improve if both component operators are fast
in isolation but their interface forces large data movement.

## Performance Principles

The design should enforce the following principles.

### 1. Do Not Materialize Full Logits

`lightning_indexer` must prefer:

- tile matmul
- local reduction
- local TopK
- cross-tile merge

over:

- full global-memory logits
- separate global TopK pass

### 2. Do Not Materialize Full Selected-KV

`sparse_attention` must prefer:

- streamed page-aware gather
- block buffering
- online reduction

over:

- global temporary expansion of selected-KV

### 3. Keep Cube and SIMT Responsibilities Stable

Do not let vector-heavy control logic creep into cube-oriented kernels, and do
not move large matrix products into SIMT merely for convenience.

### 4. Specialize Family Parameters Aggressively

Within each kernel family, use static tiling and static buffer assumptions where
possible instead of broad runtime branching.

### 5. Optimize Workflow, Not Only Standalone Operators

Operator-local wins that create interface friction are not acceptable if the
workflow result regresses.

## Verification Plan

Verification should happen at three levels.

### 1. Correctness

Compare family fast paths against current baselines for:

- selected indices
- attention outputs
- reduction outputs such as `lse` where applicable

Allow numerical tolerance for floating-point outputs, but keep comparison policy
explicit per output type.

### 2. Kernel-Level Profiling

Check whether the dominant runtime remains in the intended cube-heavy kernels
rather than in SIMT gather, TopK, metadata, or merge overhead.

If the SIMT-side orchestration dominates runtime, the fusion boundary or data
movement strategy is wrong.

### 3. Workflow-Level Benchmarking

Measure:

- `dsa_prefill`
- `dsa_decode`

Do not judge success only by standalone operator benchmarks. The workflow result
must improve under realistic case families.

## Success Criteria

### Phase 1

- `prefill` for `family_64x128` and `family_hd512` approaches or matches
  `vllm_ascend` on the main realistic cases

### Phase 2

- V3.2-oriented `family_4x64` and `family_hd128` are added without breaking the
  first family

### Phase 3

- decode paths are optimized for low latency and workflow-level DSA results
  become competitive with or better than the current external baselines

Across all phases, the intended feature-complete end state still includes both
prefill and decode support for the targeted DSA operator families.

## Risks

- a single operator-local interface may still hide workflow-level layout costs
- aggressive fusion may increase maintenance cost if profiling hooks are weak
- decode may require more radical launch minimization than prefill
- V3.2 may need deeper specialization than currently assumed because its head
  shape diverges substantially from the `64x128` or `hd512` families

## Open Questions For Implementation Planning

- exact SIMT kernel source layout and versioning per family
- metadata format shared between `lightning_indexer` and `sparse_attention`
- fallback reference path definition for unsupported shapes
- profiling kernel naming conventions for new SIMT families
- correctness thresholds per output type and dtype
