# Lightning Indexer Context-Sharded Decode Design

## Status

Implemented as the first two-launch context-sharded `lightning_indexer` decode
path. The later parameterized single-kernel implementation supersedes it for
the enabled S16, S8, S4, S2, and S1 planner tiers; this document is retained as
historical design context.

This design prioritizes the production-shaped DeepSeek V3.2 decode case. It does
not yet implement the A/B/C comparison matrix from the parallel-splitting
research document.

## Context

The current `family_64x128` mixed kernel assigns one logical task to each
`(batch, query)` row. For the target case:

```text
B = 2
Q = 2
C = 32768
H = 64
D = 128
K = 2048
```

that produces only four row tasks. Each task serially traverses 256 context
tiles of 128 tokens, so long-context decode exposes little scheduling
parallelism.

The research in
`src/cannbench/operators/builtin/lightning_indexer/PARALLEL_SPLITTING_RESEARCH.zh-CN.md`
compares row-only scheduling with context sharding, Q atoms, full-score
workspaces, and local-TopK merge schemes. This design selects the first
high-priority implementation from those options.

## Preliminary Synchronization Fix

Before implementing context sharding, commit `d608c0b` corrects the existing
mixed-kernel handshake:

- cross-core synchronization mode changes from 4 to 2
- every AI Core group reuses flag 0
- both sub-AIVs execute the mode-2 set/wait handshake
- only sub-AIV 0 performs the existing row postprocess

Mode 2 is required because one AIC synchronizes with both AIVs in the same AI
Core. Flag counters are local to that synchronization group, so the flag ID is
not derived from `block_idx`.

The fix was validated on the remote Ascend 950PR environment:

- `bisheng --npu-arch=dav-3510` build succeeded
- operator-local tests passed: 68 passed, 4 skipped
- registered NPU custom-op tests passed: 12 passed
- the full V3.2 target shape returned the same selected score set as the
  PyTorch reference, including causal valid lengths 32767 and 32768

The observed 52.287 ms was a single cold validation launch and is not a
performance baseline.

## Goal

Reduce target-case latency by splitting context work across 16 logical tasks
while loading each Key shard once for both Query rows.

The new path must:

- preserve the current public custom-op and plugin interfaces
- preserve the target case's TopK score-set semantics
- use both AIVs in each `1:2` mixed-kernel group
- launch new SIMT VF and TopK blocks with 1024 threads
- avoid synchronization between different logical tasks or AI Core groups
- retain the current implementation for every non-target shape
- remain entirely inside the `lightning_indexer` operator package

## Non-Goals

This milestone does not add:

- runtime tuning across shard sizes
- scheme A (`Q=2`, shard 2048)
- scheme C (`Q=1`, shard 4096)
- local TopK plus final candidate merge
- double buffering
- arbitrary Q-atom or context-shard sizes
- prefill context sharding
- changes to CLI, backend, core, result, or published-data contracts
- removal of all transitional Basic API usage from the existing fused kernel

The API-boundary cleanup remains follow-up work. This performance-first slice
does not add global synchronization and does not expand the existing legacy
dependency outside the operator-local SIMT implementation.

## Target Fast Path

The C++ bridge selects the new path only when static metadata matches:

```text
phase = decode
family = family_64x128
dtype = bfloat16
B = 2
Q = 2
C = 32768
H = 64
D = 128
K = 2048
```

The production case supplies `valid_context_lengths` rows of `[32767, 32768]`.
Those values are handled on device and are not part of the host-side fast-path
predicate.

The bridge does not read `valid_context_lengths` on the host. Reading that NPU
tensor would introduce an unwanted host-device synchronization. Each AIV
applies its Query row's length on device instead.

All other inputs use the current implementation unchanged.

## Execution Overview

The fast path has two device launches on the current NPU stream:

```text
query + keys + weights
        |
        v
Q=2 atom, context-sharded mixed score kernel
        |
        v
[B, Q, C] BF16 reduced-score workspace
        |
        v
independent SIMT TopK kernel
        |
        v
[B, Q, K] int32 indices
```

Stream ordering is the only synchronization between the two launches. No
cross-task device barrier is required.

## Score-Kernel Task Mapping

The context shard is fixed at 4096 tokens:

```text
context_shard_count = C / 4096 = 8
task_count = B * context_shard_count = 16
blockDim = 16

batch_index = task_id / 8
shard_index = task_id % 8
context_start = shard_index * 4096
context_end = context_start + 4096
```

Each logical block owns exactly one `(batch, context_shard)` pair. The launch
does not apply the existing `kMaxUsedCoreNum = 11` cap; that cap came from the
old per-block flag allocation and is not a physical-core count. The device
scheduler maps the 16 logical blocks to available hardware.

The target context length is exactly divisible by 4096, so this milestone does
not need a shard tail. Per-Query valid lengths are still honored inside every
shard.

## Q-Atom Cube Work

The AIC loads both Query rows and arranges their heads as one M dimension:

```text
M = Q * H = 2 * 64 = 128
K = D = 128
N = context_tile = 32
```

For every 32-token Key tile, one M=128 MMAD computes both Query rows. The Key
tile is loaded once and reused by the complete Query atom. The shared score
layout produced for each AIV by dual-M Fixpipe is:

```text
[head=64, context_tile=32]
```

The first 64 score rows belong to Query 0; the second 64 rows belong to Query
1. Fixpipe `dualDstCtl=1` routes those halves to sub-AIV 0 and sub-AIV 1 at the
same UB offset. Each half is exactly 8 KiB. C310 mixed-kernel validation showed
that offsets at or above 8 KiB trap with error 341, so a 128-token tile cannot
be consumed directly even when the logical `LocalTensor` is declared larger.

## AIC/AIV Handshake

The first implementation is single-buffered and uses cross-core mode 2 with
flag 0.

For each score tile:

1. Both AIVs set flag 0 after finishing any previous use of the shared score
   buffer.
2. The AIC waits on flag 0 before overwriting that buffer.
3. The AIC performs MMAD and Fixpipe into the shared buffer.
4. The AIC sets flag 0 after the score tile is ready.
5. Both AIVs wait on flag 0 before reading the score tile.

Both sub-AIVs must execute every set/wait call. Neither may return before the
handshake.

Double buffering is deferred. A later version may use flags 0 and 1 for
ping-pong buffers after the single-buffered path is correct and profiled.

## AIV Postprocess

Sub-AIV ownership is fixed:

```text
sub-AIV 0 -> Query 0
sub-AIV 1 -> Query 1
```

Each AIV processes its Query row over the 32 context positions in the shared
tile. For each position it:

1. converts the MMAD result to BF16 as in the current kernel
2. applies ReLU
3. multiplies by the corresponding BF16 head weight
4. accumulates the 64 weighted head values
5. converts the final reduced score to BF16
6. writes the score to its non-overlapping workspace row

Positions at or beyond that Query's `valid_context_lengths` value are written
as negative infinity. This supports asymmetric Query lengths without a host
readback.

The arithmetic and conversion order must remain aligned with the current
kernel so that context sharding changes scheduling, not score semantics.

The postprocess VF uses `__launch_bounds__(1024)` and launches
`dim3(1024, 1, 1)`. Threads stride by `blockDim.x`; the configuration follows
Ascend SIMT practice rather than the current CUDA-like 256-thread constant.
It may be reduced only if the remote compiler resource report demonstrates
register or stack spilling at 1024 threads.

## Score Workspace

The workspace shape and type are:

```text
shape = [2, 2, 32768]
dtype = bfloat16
size = 2 * 2 * 32768 * 2 bytes = 256 KiB
```

Each `(batch, shard, query_in_atom)` writer owns a unique range. No atomics,
locks, or final score reduction are needed.

The bridge allocates the workspace as a temporary tensor on the Query device
and records all raw-launch tensor storage on the current NPU stream.

## TopK Kernel

The second launch is a SIMT-only TopK kernel with four logical blocks, one per
`(batch, query)` row.

Each TopK block also launches 1024 SIMT threads with
`__launch_bounds__(1024)`.

Each block keeps 2048 best candidates and merges 2048 workspace scores at a
time. The existing 4096-entry UB capacity can hold:

```text
2048 retained candidates + 2048 new scores
```

Each row therefore requires 16 merge rounds for 32768 scores. Workspace scores
are converted from BF16 to float for comparison. Candidate ordering is:

1. higher score first
2. lower global context index first when scores are equal

The output remains sorted `[B, Q, 2048]` `int32` indices. The public operation
continues to return indices only.

## Error Handling and Fallback

Existing C++ validation remains authoritative for device, rank, dtype, shape,
contiguity, and `top_k` constraints.

The fast path is selected only after those checks pass. Workspace-allocation,
kernel-launch, or asynchronous NPU failures propagate through the normal
PyTorch/torch_npu error path. The implementation does not catch a fast-path
failure and silently rerun the old kernel.

Non-target static shapes route directly to the existing fused implementation.
There is no public behavior change and no framework-level operator branch.

## Code Ownership

Implementation and tests stay under:

```text
src/cannbench/operators/builtin/lightning_indexer/
```

Expected files include:

```text
simt/v1/aten_dsa_lightning_indexer/csrc/lightning_indexer.asc
simt/v1/aten_dsa_lightning_indexer/csrc/simt/
simt/test/
```

No concrete `lightning_indexer` logic is added to `cli.py`, `core/`, shared
backends, or global framework tests.

## Test Strategy

### Source and dispatch contracts

- target metadata selects the context-sharded path
- other metadata selects the current path
- task count is 16 and mapping is `(batch, shard)`
- Query atom M is 128
- the context tile is 32 so each dual-M AIV destination is at most 8 KiB
- synchronization uses mode 2 and flag 0
- score postprocess and TopK use 1024 SIMT threads
- both AIVs execute the handshake
- sub-AIV 0 and sub-AIV 1 select different Query halves
- the score workspace is BF16 `[B, Q, C]`
- the TopK launch uses four row tasks and 2048-score merge tiles

### NPU correctness

- full target case matches the PyTorch reference's selected score set
- output shape is `[2, 2, 2048]` and dtype is `int32`
- Query 0 and Query 1 use distinct inputs and produce independently verified
  outputs
- asymmetric valid lengths mask the correct suffix for each Query
- repeated launches complete without deadlock or flag-counter imbalance

### Regression

- prefill paths remain unchanged
- `family_4x64` remains unchanged apart from the preliminary synchronization
  correction
- `family_32x128` and non-target `family_64x128` shapes remain on the old path
- the full repository test suite passes
- targeted searches show no new public-layer hardcoding

### Remote build

The source must build with:

```text
bisheng --npu-arch=dav-3510
```

on the provided Ascend 950PR endpoint before the implementation is considered
complete.

## Performance Acceptance

Performance comparison uses the corrected current path at commit `d608c0b` as
the baseline, on the same remote device and with identical input tensors.

For both paths:

- run warmup iterations before recording samples
- record at least 30 synchronized device timings
- compare median latency
- retain profiler evidence for kernel launch count and active task structure

The new path becomes the default for the target metadata only if its median is
stably lower than the corrected baseline. If it is not faster, keep the
implementation and measurements for the later A/B/C comparison, but do not
select it as the default runtime path.

## Follow-Up Work

After this priority path is measured, later iterations may add:

- scheme A: Q=2 atom with 2048-token shards
- scheme C: single-Query tasks with 4096-token shards
- local TopK plus final merge
- score-only profiler microbenchmarks
- flag 0/1 double buffering
- runtime selection across B/Q/C/K shapes
- removal of remaining transitional Basic API dependencies
