# Lightning Indexer Fused Distributed TopK Design

## Status

Approved for implementation on 2026-08-10 as a measured experiment. The user
selected the single-kernel fusion approach and granted a task-local exception
for `AscendC::SyncAll()`. This exception does not modify `AGENTS.md` and does
not apply to other operators or later work.

## Scope

This design changes only the canonical V2 BF16 decode specialization:

```text
B = 2
Q = 2
C = 32768
H = 64
D = 128
TopK = 2048
context shards = 16
```

The context-sharded Score kernel remains a separate mixed AIC/AIV launch. The
five pure-AIV distributed TopK launches are fused into one pure-AIV launch.
All other decode shapes, prefill paths, V1 code, and the existing one-block
decode radix fallback remain unchanged.

## Motivation

The retained canonical path launches these five pure-AIV TopK stages in stream
order:

1. high-byte histogram;
2. global high-byte selection;
3. conditional low-byte histogram;
4. global low-byte selection and per-shard output offsets; and
5. deterministic compaction.

The stage algorithms are already validated and performant. This experiment
tests only whether replacing four intervening launch boundaries with four
device-wide `SyncAll()` barriers reduces complete Indexer and workflow latency.
It does not redesign radix selection or change the TopK contract.

## Selected Architecture

The distributed TopK device source will expose one fused outer kernel and one
host launcher. The outer kernel launches 64 AIV tasks, matching:

```text
row_count * context_shard_count = 4 * 16 = 64
```

It invokes the existing stage VFs serially:

```text
high_histogram VF, 1024 threads
  -> GM visibility step
  -> SyncAll
select_high VF, 256 threads
  -> GM visibility step
  -> SyncAll
low_histogram VF, 1024 threads
  -> GM visibility step
  -> SyncAll
select_low_offsets VF, 256 threads
  -> GM visibility step
  -> SyncAll
compact VF, 1024 threads
```

Histogram and compaction stages use all 64 outer tasks. Reducer stages use only
the first four tasks, one per output row. Tasks with `blockIdx.x >= row_count`
perform no reducer work but must return from the VF call to the outer kernel and
participate in every `SyncAll()`. No outer task may exit before the final stage.

The launch allocates `kCompactUbufBytes`, currently 8 KiB, as dynamic UB. This
is the maximum scratch requirement among the five stages. Each VF reuses that
region only after the preceding VF has completed and the full-core barrier has
passed.

## Synchronization And Memory Visibility

Sequential `asc_vf_call` invocations order work on each AIV task. Four
`AscendC::SyncAll()` calls provide the cross-task phase boundaries previously
provided by kernel completion.

The producer VFs communicate with the next phase through GM workspaces:

- high histogram writes `high_histograms`;
- high selection writes `radix_state`;
- low histogram writes `low_histograms`;
- low selection writes `radix_state` and `shard_offsets`.

Before each `SyncAll()`, the producer VF must complete all per-block stores and
perform the target-stack cache-maintenance operation needed to make those GM
writes visible to other AIV tasks. The implementation will follow the included
CANN arch35 SIMT precedent: a block barrier followed by one
`__builtin_cce_dcci(nullptr, 1, 0)` call from thread zero. Correctness testing
must distinguish a missing visibility step from a barrier-ordering defect.

The experiment uses the narrowest header that exposes `AscendC::SyncAll()` on
CANN 9.2.0. Adding that Basic API dependency is covered only by this task-local
exception. It must not introduce `SetFlag`, `WaitFlag`, `PipeBarrier`,
`CrossCoreSetFlag`, `CrossCoreWaitFlag`, GM spin loops, or Mutex-based inter-core
coordination.

## Host Dispatch And Workspaces

The canonical predicate remains:

```text
batch_size == 2 && query_count == 2 && context_shard_count == 16
```

The host bridge continues to allocate the existing BF16 score workspace,
two histogram workspaces, radix state, shard offsets, and `int32` output. It
replaces the five distributed TopK launcher calls with one fused launcher call.
The preceding context-sharded Score launch and tensor stream recording remain
unchanged.

Noncanonical context-sharded decode continues to call the independent
single-block radix TopK kernel. No public backend, CLI, registry, published-data
schema, or operator-plugin hook changes are required.

## Correctness Contract

The fused path must preserve the existing canonical distributed TopK contract:

- exactly 2048 unique indices per valid row;
- every index inside the valid context range;
- unordered score-set equivalence with the trusted reference;
- deterministic completion when multiple scores equal the selected threshold;
- correct masking for shortened valid-context lengths;
- stable repeated results;
- unchanged `int32` output shape `[2, 2, 2048]`.

Device validation covers canonical random, masked-tail, tied-threshold,
near-threshold, and negative-score cases. It also exercises the noncanonical
fallback and the matching V1 regression so the new dispatch does not broaden
its shape scope.

## Test Strategy

Development follows test-first source contracts. The first failing tests will
require:

- one fused distributed TopK outer kernel and one exported launcher;
- all five existing VF calls in the required order;
- exactly four `AscendC::SyncAll()` phase boundaries;
- fixed 64-task canonical launch geometry;
- reducer task masking without early outer-kernel exit;
- dynamic UB sized to `kCompactUbufBytes`;
- producer-side block completion and cache maintenance before each barrier;
- one fused host launch only under the existing canonical predicate;
- unchanged one-block radix fallback for all other supported shapes.

After source tests pass, run the complete local `pytest -q` suite. Target-device
validation uses CANN 9.2.0, Bisheng production `-O3`, and `dav-3510`, followed
by resource metadata inspection and the correctness matrix above.

## Performance Experiment And Retention

Collect two clean-process baseline/candidate pairs on Ascend 950PR at the same
frequency, inputs, seed, warmup, repetitions, profiler mode, and selected-kernel
boundary. Resolve and record the loaded package path plus device-library hash
inside each process.

Report these boundaries separately:

- Score kernel;
- complete TopK, using five stage rows for the baseline and the fused row for
  the candidate;
- complete Lightning Indexer;
- Sparse Attention fused kernel and Combine;
- complete selected DSA decode workflow;
- excluded materialization helpers such as Cast.

Retain the fused implementation only if correctness passes and neither complete
Lightning Indexer nor complete selected DSA workflow is slower than its paired
baseline in both clean-process pairs. No positive percentage threshold is
required. If either boundary regresses in either pair, restore the retained
five-launch implementation and document the candidate as rejected evidence.

## Non-Goals

This experiment does not:

- fuse Score with TopK;
- remove or resize the GM score and radix workspaces;
- change BF16 ordered-key mapping, histogram layout, threshold selection,
  shard-offset construction, or compaction order;
- sort the returned TopK indices;
- generalize full-core synchronization as an allowed repository-wide pattern;
- change the published benchmark schema or timing boundary.
