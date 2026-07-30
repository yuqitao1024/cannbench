# Lightning Indexer V3.2 Prefill Row-Parallel Design

## Status

Closed by the performance gate. The Q=2 dual-AIV candidate was implemented and
validated, but its standalone median was 11.11% slower than the corrected
baseline, so automatic dispatch remains disabled. The separate 32-mixed-task,
1024-thread common-path correction was retained. Candidate workflow timing was
intentionally skipped after the required standalone gate failed.

## Context

The target case is:

```text
case = deepseek_v32_flashmla_prefill_q4096_ctx32768_top2048
phase = prefill
family = family_64x128
dtype = bfloat16
B = 1
Q = 4096
C = 32768
H = 64
D = 128
K = 2048
query_start = 28672
```

The Query rows have right-aligned causal valid lengths from 28673 through
32768. The lengths are supplied in `valid_context_lengths`; the host must not
read them back from the NPU.

The current `family_64x128` fused kernel caps its launch at 11 mixed tasks.
That constant predates commit `d608c0b`, which changed the AIC/AIV handshake
from per-block mode-4 flags to mode 2 with flag 0. The old cap is therefore a
missed follow-up, not a hardware limit.

Core counts in this design use both relevant units:

```text
1 mixed task under KERNEL_TYPE_MIX_AIC_1_2 = 1 AIC + 2 AIV
11 mixed tasks = 11 AIC + 22 AIV
16 mixed tasks = 16 AIC + 32 AIV
```

The exact V3.2 decode fast path already launches 16 mixed tasks and therefore
uses 32 AIVs. Other prefill and decode shapes still use the capped fused
kernel.

## Goal

Improve standalone `lightning_indexer` latency for the exact V3.2 prefill case
by:

1. correcting the obsolete 11-task cap before measuring a baseline
2. packing two Query rows into one M=128 Cube atom
3. assigning one Query row to each AIV in a mixed task
4. retaining row-parallel scheduling instead of sharding context

The candidate must preserve the current output contract and must not regress
the full `dsa_prefill` workflow.

## Non-Goals

This milestone does not add:

- support for every `family_64x128` prefill shape
- context sharding for prefill
- a 256 MiB full-score prefill workspace
- local-TopK candidate merging across context shards
- runtime autotuning
- double buffering
- changes to CLI, core, shared backends, result schemas, or workflow plugins
- new Basic API dependencies

Existing transitional synchronization may remain while function and
performance are validated. Follow-up cleanup must converge on the repository's
C API, Tensor API, and SIMT API boundary.

## Approaches Considered

### 1. Corrected row-only kernel

Change the existing cap from 11 to 16 mixed tasks while leaving each task
responsible for one Query row. This is the required corrected baseline and the
smallest-risk prerequisite. It exposes 32 AIVs to the launch, although only
sub-AIV 0 performs postprocess work in the current implementation.

This option should improve scheduling but leaves half of the AIV compute
capacity unused.

### 2. Q=2 row atoms with both AIVs

Pack two adjacent Query rows into M=128. One AIC loads their shared Key tile
once, and dual-destination Fixpipe gives one M=64 half to each AIV. Each AIV
updates TopK for its own Query row.

This is the selected first optimization because prefill already has 4096 rows
of natural parallelism. It reuses the successful decode Q-atom and dual-AIV
mechanics without adding context-shard coordination or a large GM workspace.

### 3. Decode-style context sharding and separate TopK

The decode fast path could be generalized so every prefill row writes full
reduced scores and a second kernel performs TopK. For this case the BF16 score
workspace would be:

```text
1 * 4096 * 32768 * 2 bytes = 256 MiB
```

This may reduce repeated in-kernel TopK work, but adds substantial GM traffic
and a large temporary allocation. It is deferred as the first comparison
candidate if the selected Q=2 fused path fails its performance gate.

## Corrected Baseline

The existing `family_4x64` and `family_64x128` fused kernels use the same
obsolete `kMaxUsedCoreNum = 11` rule and the same corrected mode-2/flag-0
handshake. Update both limits to 16 mixed tasks so the common implementation
uses at most 16 AICs and 32 AIVs.

This correction is measured independently from the new exact-shape candidate:

```text
old published baseline: 11 mixed tasks
corrected prefill baseline: 16 mixed tasks, existing one-row algorithm
candidate: 16 mixed tasks, Q=2 dual-AIV algorithm
```

The candidate speedup is reported only against the corrected baseline.

## Exact Fast-Path Dispatch

The new prefill path is selected only when static metadata matches:

```text
phase = prefill
family = family_64x128
original dtype = bfloat16
B = 1
Q = 4096
C = 32768
H = 64
D = 128
K = 2048
```

The bridge does not inspect `valid_context_lengths` values on the host. All
other metadata continues through the corrected existing fused kernel.

The implementation stays under the `lightning_indexer` package and does not
add concrete operator branches to public framework layers.

## Task Mapping

There are 2048 Query atoms:

```text
query_atom_size = 2
query_atom_count = Q / 2 = 2048
mixed_task_count = 16

for atom_index = task_id; atom_index < 2048; atom_index += 16:
    query_0 = atom_index * 2
    query_1 = query_0 + 1
```

Each mixed task therefore owns 128 Query atoms. Different tasks never write
the same output rows, so no inter-task synchronization or atomics are needed.

## Cube Work and Key Reuse

For each Query atom and context tile, the AIC computes:

```text
M = 2 * H = 128
K = D = 128
N = context_tile = 32
```

The Key tile is loaded once for both Query rows. Fixpipe uses `dualDstCtl=1`
and sends the first 64 score rows to sub-AIV 0 and the second 64 rows to
sub-AIV 1. Each destination is:

```text
64 * 32 * sizeof(float) = 8 KiB
```

This matches the C310 per-AIV shared destination boundary already validated by
the decode implementation. A larger N is not assumed safe in this milestone.

## AIC/AIV Synchronization

The path uses mode 2 and flag 0. Both AIVs participate in every handshake:

1. both AIVs signal that the shared score buffer is free
2. the AIC waits before overwriting the buffer
3. the AIC runs MMAD and dual-destination Fixpipe
4. the AIC signals that scores are ready
5. both AIVs wait before reading their own score half

Neither AIV may return early. No flag is derived from `block_idx`, because the
counter is local to the AIC plus two-AIV synchronization group.

The first version remains single-buffered with flag 0. Flags 0 and 1 are
reserved for a measured double-buffering follow-up.

## AIV Postprocess and TopK

Sub-AIV ownership is fixed per atom:

```text
sub-AIV 0 -> query_0
sub-AIV 1 -> query_1
```

For every context position, each AIV preserves the current arithmetic order:

1. convert the MMAD value to BF16
2. apply ReLU
3. multiply by the Query row's BF16 head weight
4. accumulate 64 weighted head values in float
5. convert the reduced value through the current BF16 rounding point
6. merge it into that row's running TopK state

Each AIV reads its own `valid_context_lengths[query_index]`. Context positions
outside the causal length must not enter TopK.

The SIMT VF uses `__launch_bounds__(1024)` and launches 1024 threads, following
the established Ascend setting. A lower thread count is permitted only if the
remote compiler resource report or measured result proves 1024 is worse.

Running TopK scores and indices remain row-local. The candidate does not
materialize the full `[1,4096,32768]` score tensor.

## Correctness and Stability

Tests must cover:

- source contracts for 16 mixed tasks, Q=2, mode 2, flag 0, dual AIV, and 1024
  threads
- exact output shape `[1,4096,2048]` and int32 dtype
- causal valid lengths from 28673 through 32768
- selected score-set equivalence with the existing PyTorch reference
- repeated launches to detect unbalanced mode-2 flag counters
- non-target prefill and decode dispatch remaining on the common path
- operator-local regression tests for both fused families

Because equivalent scores may produce different valid indices, correctness is
judged by the repository's existing score-set contract unless a test explicitly
requires deterministic tie ordering.

## Performance Gate

Use isolated remote builds with the same compiler, device, seed, warmups, and
sample count.

The gate has three levels:

1. The 16-task corrected baseline must remain correct and stable.
2. The exact Q=2 candidate's standalone median must beat that corrected
   baseline.
3. Full `dsa_prefill` correctness must pass and workflow latency must not
   regress versus the corrected baseline.

The exact dispatch remains enabled only if all three levels pass. If the Q=2
candidate fails, retain the corrected common path and use the deferred
full-score/separate-TopK design as the next comparison rather than stacking
unmeasured changes.

Record median, minimum, maximum, compiler stack/register usage, and the exact
commits in the operator-local SIMT README.

## Expected File Scope

Implementation should be limited to:

- the two existing fused family kernels for the corrected task cap
- one operator-local exact prefill kernel source
- the operator-local build map and C++ bridge
- operator-local source and NPU tests
- the operator-local SIMT README

No shared framework or workflow files should need implementation changes.
