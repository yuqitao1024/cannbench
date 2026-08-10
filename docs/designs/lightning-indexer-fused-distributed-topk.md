# Lightning Indexer Fused Distributed TopK Design

## Status

Retained after target-device validation on 2026-08-10. The user selected the
single-kernel fusion approach and granted a task-local exception for
`AscendC::SyncAll()`. This exception does not modify `AGENTS.md` and does not
apply to other operators or later work.

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

## Measured Result And Retention

The experiment was built and validated on `Ascend950PR_9589` with CANN 9.2.0,
the 2026-08-04 Bisheng build, PyTorch 2.10.0, torch_npu 2.10.0.post2,
production `-O3`, and `dav-3510`. All retained `BasicInfo` rows reported a
current and rated frequency of 1650 MHz.

The isolated source revisions and distributed TopK ELF hashes were:

| Variant | Revision | Source archive SHA-256 | Distributed TopK ELF SHA-256 |
| --- | --- | --- | --- |
| Baseline | `c7b23cf` | `a0d095d5b82aa0904afaf7431829fb810455b28262d12864159e92c8a5f5e20f` | `d7211e075f7a576e8ac148187180619afea2164903c29259c35d466281af01bf` |
| Candidate | `4a63e64` | `391c8b1558ad589eac952c8699c01dcedd49277040798befffc348663a234e7f` | `ab08e711a56c6435d24b25d184c95e6f36312595b96ea1997c8d2d31af953f0c` |

The candidate production build completed successfully. A separate
`--cce-res-usage` build used the same compiler inputs and reported zero Stack
bytes in all five VFs. Register counts in execution order were 24 for high
histogram, 16 for high selection, 24 for low histogram, 33 for low selection
and offsets, and 26 for compaction. The resource ELF SHA-256 was
`c69d593f85e34859faa3b50bc4643725706bb4f64fbd4190e8d66feef26a5d75`.

Device correctness passed the following fresh-process checks:

- canonical random, masked-tail, tied-threshold, near-threshold, and
  negative-score candidate cases at seed 7, with five repeats per case;
- a noncanonical V2 `B=1,Q=4` fallback case at seed 19, with five repeats;
- the independently built V1 canonical regression at seed 7, with five
  repeats.

Every check produced unique in-range indices, matched the reference TopK score
multiset, and returned a stable index set across repeats. The tied-threshold
case also returned the deterministic expected low-index set.

Two alternating clean-process `BasicInfo` pairs used the same seed-7 DSA decode
case. Times below are microseconds. Baseline TopK is the sum of the five stage
rows; candidate TopK is the single fused row. The selected Sparse Attention V2
path emitted one fused row and no separate Combine row, so the selected
workflow is Indexer plus Sparse fused. Materialization Cast remains excluded.

| Pair 1 boundary | Baseline | Candidate | Candidate delta |
| --- | ---: | ---: | ---: |
| Score | 56.611 | 56.340 | -0.271 (-0.48%) |
| Complete TopK | 36.238 | 28.640 | -7.598 (-20.97%) |
| Complete Indexer | 92.849 | 84.980 | -7.869 (-8.47%) |
| Sparse Attention fused | 97.392 | 97.259 | -0.133 (-0.14%) |
| Selected DSA workflow | 190.241 | 182.239 | -8.002 (-4.21%) |

| Pair 2 boundary | Baseline | Candidate | Candidate delta |
| --- | ---: | ---: | ---: |
| Score | 56.788 | 56.877 | +0.089 (+0.16%) |
| Complete TopK | 35.704 | 28.598 | -7.106 (-19.90%) |
| Complete Indexer | 92.492 | 85.475 | -7.017 (-7.59%) |
| Sparse Attention fused | 97.635 | 97.787 | +0.152 (+0.16%) |
| Selected DSA workflow | 190.127 | 183.262 | -6.865 (-3.61%) |

The candidate improves both complete Indexer and selected workflow boundaries
in both pairs, so it satisfies the no-regression retention rule and is
retained. An earlier `pair1-baseline` collection overlapped an unrelated device
job and was interrupted; its partial rows are explicitly excluded from these
tables and from the retention decision.

Controller-side raw artifacts, including all four clean profile trees and
build, resource, and accuracy logs, are under:

```text
/tmp/cannbench-lightning-topk-fused-results.EY6ail/
```

The corresponding remote source, logs, and profile trees remain under:

```text
/root/cannbench-lightning-topk-c7b23cf/
/root/cannbench-lightning-topk-4a63e64/
/root/cannbench-lightning-topk-perf/clean-pair1-baseline/
/root/cannbench-lightning-topk-perf/clean-pair1-candidate/
/root/cannbench-lightning-topk-perf/clean-pair2-baseline/
/root/cannbench-lightning-topk-perf/clean-pair2-candidate/
```

## Non-Goals

This experiment does not:

- fuse Score with TopK;
- remove or resize the GM score and radix workspaces;
- change BF16 ordered-key mapping, histogram layout, threshold selection,
  shard-offset construction, or compaction order;
- sort the returned TopK indices;
- generalize full-core synchronization as an allowed repository-wide pattern;
- change the published benchmark schema or timing boundary.
