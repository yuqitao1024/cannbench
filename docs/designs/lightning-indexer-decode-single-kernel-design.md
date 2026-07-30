# Lightning Indexer Parameterized Decode Single-Kernel Design

## Status

Implemented and performance-validated design replacing the two-launch
context-sharded `family_64x128` decode path with one parameterized mixed
kernel. The enabled path covers every supported planner tier (S16, S8, S4,
S2, and S1); unsupported shapes retain the existing generic fused fallback.

This design extends the exact `B=2, Q=2` implementation described in
`lightning-indexer-context-sharded-decode-design.md`. It does not
make the complete Lightning Indexer algorithm dynamic across every head,
context, and TopK shape.

## Goal

Fuse context-sharded score generation and final TopK into one device launch
for the DeepSeek V3.2 `family_64x128` decode family while avoiding a hardcoded
`B=2, Q=2` task map.

The new path must:

- accept runtime batch size `B` and query count `Q`
- preserve the two-Query atom and Key reuse when possible
- select a safe context-shard count without launching more than 32 mixed tasks
- use all 32 AICs and 64 AIVs during score generation for the production
  `B=2, Q=2` case
- run shard-local TopK only when one shard contains more than 2048 scores
- run one final row-level TopK after all shard candidates are ready
- synchronize candidate producers and consumers inside one kernel
- keep the exact V3.2 decode case at least as fast as the current two-kernel
  device-time baseline
- retain the existing fused implementation as a correctness and performance
  fallback
- keep all implementation and dispatch logic inside the `lightning_indexer`
  operator package

## Supported Fast-Path Family

The parameterized single-kernel candidate has the following fixed family
contract:

```text
phase = decode
family = family_64x128
dtype = bfloat16
H = 64
D = 128
C = 32768
K = 2048
B = runtime
Q = runtime
```

`H`, `D`, `C`, and `K` remain fixed because they determine the MMAD shape,
score workspace, TopK candidate sizes, and dynamic UB allocation. Other values
continue to use the existing generic fused kernel.

This is family-level reuse rather than an exact-case specialization. It is not
a universal Lightning Indexer kernel.

## Runtime Task Planning

Two Query rows form one Query atom:

```text
query_atom_count = ceil(Q / 2)
base_task_count = B * query_atom_count
```

The host selects the largest context-shard count from:

```text
{16, 8, 4, 2, 1}
```

that satisfies:

```text
base_task_count * context_shard_count <= 32
```

The resulting launch size is:

```text
mixed_task_count = base_task_count * context_shard_count
```

If `base_task_count > 32`, the bridge does not launch the barrier-based
kernel. It falls back to the existing generic fused `family_64x128` kernel,
which distributes rows across at most 32 tasks and loops over additional rows.

For the production target:

```text
B = 2
Q = 2
query_atom_count = 1
base_task_count = 2
context_shard_count = 16
mixed_task_count = 32
```

This changes the current score phase from 16 AICs and 32 AIVs to the full
device complement of 32 AICs and 64 AIVs. Each task processes a 2048-token
context shard instead of the current 4096-token shard.

Representative planning examples are:

| B | Q | Base tasks | Shards | Mixed tasks |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 1 | 1 | 16 | 16 |
| 2 | 2 | 2 | 16 | 32 |
| 3 | 2 | 3 | 8 | 24 |
| 4 | 2 | 4 | 8 | 32 |
| 8 | 2 | 8 | 4 | 32 |
| 16 | 2 | 16 | 2 | 32 |
| 32 | 2 | 32 | 1 | 32 |

## Mixed-Task Mapping

Each logical mixed task owns one `(batch, query_atom, context_shard)` tuple:

```text
task_id = ((batch_index * query_atom_count + atom_index)
           * context_shard_count) + shard_index
```

The two AIVs in the mixed task map to the two Query rows in the atom:

```text
query_index = atom_index * 2 + sub_block_index
```

The context shard boundaries are derived from `context_shard_count`. Since
`C=32768` and the allowed shard counts are powers of two, all shards are
uniform and no context tail is introduced.

The AIC normally loads two Query rows and computes the existing
`M=128, K=128, N=32` MMAD. For an odd-`Q` tail atom, it loads only the valid
Query row and uses `M=64`; Fixpipe disables dual destination for that atom.
The padding AIV still performs every synchronization operation but does not
read shared scores, write a score row, or launch TopK.

## Score and Candidate Workspaces

The bridge allocates a temporary BF16 score tensor:

```text
shape = [B, Q, 32768]
size = B * Q * 32768 * 2 bytes
```

Every `(batch, query, shard)` writer owns a disjoint range. The single-kernel
design removes the device launch boundary, not the initial full-score GM
write. Each AIV reads only its own completed shard for local TopK.

When a shard contains more than `K=2048` entries, shard-local TopK reuses the
first 2048 entries of that score shard as its BF16 candidate-score slot.
Candidate values originated as BF16 scores, so storing them back as BF16 does
not introduce another rounding step. A 2048-entry shard already contains
exactly `K` entries, so local selection cannot reduce it and is skipped.

When `1 < context_shard_count < 16`, the bridge also allocates a temporary
int32 candidate-index tensor:

```text
shape = [B, Q, context_shard_count, 2048]
size = B * Q * context_shard_count * 2048 * 4 bytes
```

Each shard owns one candidate-index slot. The 16-shard production path skips
local TopK and does not allocate this tensor because raw score indices are
implicit. The one-shard path writes final indices directly to the public
output.

For the production target, temporary GM storage is therefore only the 256 KiB
BF16 score workspace.

The output remains:

```text
shape = [B, Q, 2048]
dtype = int32
```

## Local AIC/AIV Synchronization

The existing per-tile AIC-to-AIV protocol remains mode 2 with flag 0:

1. Both AIVs signal that the shared score buffer is reusable.
2. The paired AIC waits before overwriting the buffer.
3. The AIC executes MMAD and Fixpipe.
4. The AIC signals that the score tile is ready.
5. Both AIVs wait before launching the postprocess VF.

Both AIVs execute every synchronization operation, including the padding AIV
for odd `Q`.

The implementation migrates this protocol from Basic API
`CrossCoreSetFlag/WaitFlag` calls to public C synchronization calls:

```text
AIV ready: asc_sync_block_arrive(PIPE_V, 0)
AIC waits: asc_sync_block_wait(PIPE_S, 0)
AIC ready: asc_sync_block_arrive(PIPE_FIX, 0)
AIV waits: asc_sync_block_wait(PIPE_S, 0)
```

On dav-3510, `asc_sync_block_arrive/wait` maps to the current mode-2 behavior:
one AIC arrival releases both paired AIV waits, and both AIV arrivals are
required to release the paired AIC wait. The implementation must not replace
this with the mode-4/1:1 `asc_sync_intra_arrive/wait` API.

Flag 0 remains shared by the sequential local and global synchronization
phases. CANN permits reuse across modes after all set/wait pairs from the
previous mode have completed. No local mode-2 operation remains outstanding
when an AIV enters the global barrier.

## Conditional Per-Shard Local TopK

After producing its complete score shard, a valid AIV computes:

```text
shard_size = 32768 / context_shard_count
needs_local_topk = shard_size > 2048
```

Only an AIV with `needs_local_topk=true` performs shard-local TopK. It does so
without waiting for other mixed tasks, allowing an early-finishing shard to
start selection while other AIC/AIV groups are still generating scores.

| Context shards | Scores per shard | Shard-local TopK | Final candidates per row |
| ---: | ---: | :--- | ---: |
| 16 | 2048 | skipped | 32768 raw scores |
| 8 | 4096 | 4096 to 2048 | 16384 |
| 4 | 8192 | 8192 to 2048 | 8192 |
| 2 | 16384 | 16384 to 2048 | 4096 |
| 1 | 32768 | 32768 to 2048, written as final output | 2048 |

Before the local TopK VF reads GM, its AIV executes:

```text
asc_sync_vec()
asc_sync_data_barrier(mem_dsb_t::DSB_DDR)
```

The first operation waits for all earlier Vector-side postprocess VF work on
that AIV. On dav-3510, `asc_sync_pipe(PIPE_V)` is not a valid public C API
operation; `asc_sync_vec()` is the supported Vector completion wrapper. The
second operation closes the GM read-after-write dependency before the same AIV
reads its score shard.

Local TopK processes the shard in 2048-score tiles:

1. Load a tile and sort its BF16 scores with 1024 SIMT threads.
2. Use lower global context index as the tie breaker.
3. Merge the sorted tile with the retained sorted Top2048 list.
4. After the final tile, write sorted BF16 scores into the first 2048 entries
   of the score shard and write matching int32 indices into the shard's
   candidate-index slot.

For the production 16-shard plan, each shard contains exactly 2048 scores, so
all 64 AIVs skip local TopK and leave their raw scores in place. For smaller
shard-count tiers, every valid AIV processes multiple 2048-score tiles and
publishes one local Top2048 list.

An odd-`Q` padding AIV skips score and local-TopK work but follows every global
barrier executed by valid AIVs.

## Global AIV Barrier

When `context_shard_count > 1`, every launched AIV executes a mode-0 barrier
after score generation and any required local TopK:

```text
asc_sync_inter_arrive(PIPE_V, 0)
asc_sync_inter_wait(PIPE_S, 0)
```

The arrive is ordered on `PIPE_V`, so raw-score or local-candidate stores
complete before another AIV reads them. The wait blocks until all AIVs in this
launch have arrived. Exactly one mode-0 barrier is executed, independent of
the shard count. A one-shard launch has no cross-shard dependency and skips
the barrier.

For a `1:2` mixed launch with `N` logical mixed tasks, the barrier has `2*N`
AIV participants. A partial launch does not wait for inactive physical AIVs.
The production launch has `N=32`, so all 64 physical AIVs participate. AICs do
not participate in the barrier.

Flag 0 is reused only after all local mode-2 operations are fully consumed.
The kernel uses batch scheduling (`__schedmode__(1)`) to avoid
partial-residency deadlock under concurrent streams.

When `context_shard_count > 1`, no AIV may return before this barrier,
including an odd-`Q` padding AIV or an AIV that skipped local TopK.

## Final Row-Level TopK

After the required barrier, exactly one shard-0 AIV per valid `(batch, query)`
row runs the final TopK. The one-shard tier reaches this step directly because
it has no cross-shard dependency:

```text
row_index = batch_index * Q + query_index
per_shard_candidates = min(shard_size, 2048)
row_candidate_count = context_shard_count * per_shard_candidates
```

If `shard_size == 2048`, the final VF reads raw BF16 scores and derives their
global context indices from the shard and element positions. If
`shard_size > 2048`, it reads each shard's retained BF16 scores and explicit
int32 candidate indices.

The final VF keeps 2048 retained candidates in UB and consumes candidates in
2048-entry tiles. It runs once per row and writes the public int32 output. For
the production target, four AIVs each process all 32768 raw scores. Per row,
the 8-, 4-, 2-, and 1-shard tiers respectively process 16384, 8192, 4096, and
2048 locally retained candidates.

The final row-level VF needs no following global barrier. Kernel completion
provides the stream-visible output boundary. A one-shard owner writes the
result of its local TopK directly to output and does not reread candidates.

All local and final TopK work uses 1024 SIMT threads and preserves the exact
global ordering:

1. higher BF16 score first
2. lower global context index first for equal scores

## Device Library Layout

The context postprocess, conditional local-TopK, and final row-level TopK VFs
must be compiled into the same device library so the mixed kernel can call all
phases. The standalone TopK device launch and its bridge declaration are
removed only after the combined library builds, runs, and profiles
successfully.

This creates a multi-VF device library, so validation must explicitly cover
the previous msopprof failure signature. The existing minimal two-VF repro
remains independent and is not used as a substitute for profiling the real
combined kernel.

## Dispatch and Performance Gating

The parameterized kernel and conditional two-stage TopK are selected for every
supported `B/Q` combination with `base_task_count <= 32`. There is no separate
runtime gate that simply reaffirms this decision: the generic fused fallback
is used only for an unsupported family contract or a planner result of zero.

The July 29, 2026 gates were measured on Ascend 950PR with CANN 9.2.0. Every
requested tier passed its correctness score-set check and improved synchronized
wall time against the prior path, so S16, S8, S4, S2, and S1 remain enabled.

| Tier | Representative `[B,Q]` | Score-set check | New / old wall median (ms) | Reduction | Dispatch |
| --- | --- | --- | ---: | ---: | --- |
| S16 | `[2,2]` | pass | 2.034242 / 2.1796345 | 6.67% | enabled |
| S8 | `[3,2]` | pass | 1.662996 / 26.159979 | 93.64% | enabled |
| S4 | `[5,1]` | pass | 2.012016 / 26.156473 | 92.31% | enabled |
| S2 | `[9,1]` | pass | 3.363967 / 26.156201 | 87.14% | enabled |
| S1 | `[17,1]` | pass | 6.267947 / 26.170409 | 76.05% | enabled |

For the production S16 target, a same-seed (`29`) BasicInfo comparison used
five warmups and 20 samples. The old two-launch median was `578.193481 us`
for score generation plus `1566.729981 us` for standalone TopK, or
`2144.923462 us` combined. The parameterized one-launch kernel measured
`1984.271485 us`, a `7.489870%` kernel-duration reduction. Synchronized wall
time fell from `2.1796345 ms` to `2.034242 ms` (`6.67%`), and the score sets
matched exactly. The new msopprof trace reports `Block Dim = 32` and
`Mix Block Dim = 64`, confirming the target's 32 AIC + 64 AIV mixed launch.
Artifacts are retained remotely at `/tmp/msopprof-li-kernel-NDQdwo`.

## Error Handling and Fallback

Existing C++ validation remains authoritative for device placement, ranks,
contiguity, dtype, and family dimensions.

Unsupported family metadata routes to the existing generic fused kernel.
Asynchronous build, launch, or device failures propagate normally; the bridge
does not catch a failed single-kernel launch and retry after partial device
execution.

## Verification

### Source contracts

- runtime `B/Q` values are passed to the device launcher
- Query atom count and shard count follow the documented formulas
- mixed task count never exceeds 32
- `needs_local_topk` is exactly `shard_size > 2048`
- the 16-shard tier skips local TopK and the candidate-index workspace
- the 8-, 4-, and 2-shard tiers allocate candidate indices as
  `[B, Q, shards, 2048]` and run local TopK once per valid shard producer
- the one-shard tier runs local TopK once and writes its result directly to
  output
- `context_shard_count > 1` executes exactly one mode-0 barrier; the one-shard
  tier executes none
- when `context_shard_count > 1`, no AIV returns before the single required
  mode-0 barrier
- odd `Q` uses an `M=64` tail atom and does not read or write a padding Query
  row
- local and final TopK VFs receive explicit row and shard metadata
- postprocess, local-TopK, and final-TopK VFs use 1024 threads
- synchronization uses public C API calls, not new Basic API cross-core calls
- the barrier kernel is batch scheduled
- the bridge performs one device launch for an enabled fast-path shape

### NPU correctness

- the production `B=2, Q=2` result matches the selected-score reference
- representative even and odd `Q` shapes match the reference
- asymmetric valid context lengths mask the correct suffix per row
- local and final TopK preserve score order and global-index tie order
- final row candidate counts for the 16-, 8-, 4-, 2-, and 1-shard tiers are
  respectively 32768, 16384, 8192, 4096, and 2048
- repeated launches complete without deadlock or stale flag counters
- concurrent-stream stress completes without deadlock
- output shape and dtype remain `[B, Q, 2048]` and `int32`

### Build and profiler

- `bisheng --npu-arch=dav-3510` builds the combined multi-VF library
- msopprof captures the combined kernel without the prior two-VF error
- the profile reports the expected mixed task count for each shard tier
- the production score phase reports 32 AICs and 64 AIVs
- the production profile shows no shard-local TopK and four final row-level
  TopK owners
- smaller-shard-count profiles show one local TopK per valid shard followed by
  one final TopK per valid row
- the target profile contains one Lightning Indexer device kernel

### Regression

- prefill dispatch and kernels are unchanged
- non-V3.2 family shapes retain the generic fallback
- the full repository test suite passes
- targeted searches find no concrete operator branching in public layers

## Non-Goals

This milestone does not add:

- dynamic `H`, `D`, `C`, or `K` in the context-sharded kernel
- removal of the BF16 full-score GM workspace
- double buffering or new flag-0/flag-1 ping-pong scheduling
- prefill changes
- public CLI, backend, core, result, or published-schema changes

## Follow-Up

After the single-kernel path passes its gate, the next optimization may add
flag-0/flag-1 double buffering inside the score phase and measure additional
copy, MMAD, Fixpipe, and postprocess overlap. A later design may maintain local
TopK candidates incrementally during postprocess to remove the initial
full-score GM round trip. A distributed multi-level TopK tree remains a
separate option only if the single final-row pass becomes the measured
bottleneck for another shard tier.
