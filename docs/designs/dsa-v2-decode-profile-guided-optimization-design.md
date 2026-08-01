# DSA V2 Decode Profile-Guided Optimization Design

## Status

Approved for staged implementation on 2026-08-01. The launch-geometry-only
experiment was rejected after device measurement. The 1024-thread,
head-parallel Lightning Indexer decode and V3.2 full-score prefill reductions
and the Sparse Attention QK=128 tile are retained after correctness and
repeated performance validation. Later stages remain gated by device results.

## Scope And Baseline

This design covers the BF16 DeepSeek V3.2 decode workflow case:

```text
B = 2
Q = 2
C = 32768
Indexer: H = 64, D = 128, K = 2048
Attention: query_heads = 128, qk_head_dim = 576, value_head_dim = 512
selected_tokens = 2048
```

The retained baseline is repository commit `f79f99c`, with V2 operator source
from `af10aba`. It was measured on Ascend 950PR (`dav-3510`) with CANN 9.2.0,
production `-O3`, and the same CannBench case, dtype, seed, stream boundary,
and selected-kernel rules for every implementation.

The `Default` profile collected on 2026-08-01 reports:

| Component | Kernel or group | Duration (us) | Workflow share |
| --- | --- | ---: | ---: |
| Indexer | context-sharded score | 292.722 | 35.3% |
| Indexer | unordered radix Top-K | 131.422 | 15.9% |
| Sparse Attention | Head64 fused | 367.710 | 44.4% |
| Sparse Attention | combine | 36.226 | 4.4% |
| Workflow | selected-kernel sum | 828.080 | 100.0% |

The matching vLLM-Ascend `Default` profile is 97.674 us for Lightning Indexer
and 74.161 us for Sparse Attention, or 171.835 us for the component sum. Both
paths use BF16 tensors. FP8 KV compression does not explain this comparison.

Raw SIMT V2 BasicInfo, Default, and InstrTimeline data is preserved in:

```text
/mnt/c/Users/yuqitao/Downloads/
  cannbench-dsa-v2-decode-profiles-f79f99c.tar.gz
```

## Profile Findings

### Indexer Score

The score launch uses 32 mixed tasks. Each task owns one Query atom and one of
16 context shards and executes 64 context tiles. Every AIV therefore invokes
the postprocess VF 64 times.

The VF is launched with 1024 threads, but its loop maps only the first 32
threads to the 32 context positions. Each active thread serially converts,
applies ReLU, multiplies, and accumulates all 64 heads. InstrTimeline records
exactly 64 `VF_SIMT` regions per AIV lane, averaging about 4.11 us each and
about 263 us in aggregate. AIC and AIV spans remain near 280-292 us because the
single-buffered score-tile handshake makes AIC progress depend on postprocess
completion.

The score kernel reads about 17.45 MiB-equivalent profiler KB, close to the
17.16 MiB-equivalent profiler KB read by vLLM-Ascend Lightning Indexer. The
observed gap is therefore more consistent with VF granularity, reduction
schedule, and producer/consumer serialization than with additional GM bytes.

### Unordered Radix Top-K

The V2 Top-K launch is one pure-Vector block per output row. Four blocks perform
two BF16 radix passes, threshold compaction, and deterministic threshold-tie
completion. InstrTimeline exposes each block as one long `VF_SIMT` region. The
stage is 31.0% of Indexer time and passes both gates from the original design:
it exceeds 100 us and 20% of Indexer time.

### Sparse Attention

The automatic decode plan uses 32 logical Head64 P4 tasks. Each task owns 512
selected tokens, split into eight 64-token tiles. Per AIV lane, the fused
kernel executes approximately 148 SIMT VF calls:

```text
3 initialization/query calls
+ 8 * (9 QK key-pack calls + 1 softmax call)
+ 8 * (4 value-pack calls + 4 output-update calls)
+ 1 output-write call
```

InstrTimeline records about 305 us of `VF_SIMT` work inside a roughly 351 us
Vector span. It also records repeated AIV/AIC device-flag waits and individual
cross-pipeline events as long as 128-244 us. The vLLM-Ascend sparse-attention
Vector trace spans about 91 us and has about 62 us of union busy time. The
formal vLLM trace is Vector-only because replaying its Cube pipeline fails on
this toolchain, so timeline structure is comparable but InstrTimeline latency
is not a full-kernel comparison.

## Contracts

All stages must preserve:

- V1 and V2 package/version isolation;
- operator-local dispatch with no concrete operator branches in public layers;
- BF16 inputs and score workspace, FP32 accumulation, and `int32` indices;
- unordered Top-K score-set semantics, uniqueness, valid-range masking, and
  deterministic repeated results;
- Sparse Attention output and LSE tolerances for the canonical and boundary
  cases;
- the production stream boundary and current selected-kernel timing contract;
- the target `C API + Tensor API + SIMT API` boundary for new code. Existing
  transitional Basic API code may be measured or minimally adjusted, but new
  optimization logic must not expand that dependency.

## Approach Decision

Three starting points were considered.

### A. Optimize Score Postprocess First (Selected)

Start with a compile-time thread-count matrix that leaves indexing, arithmetic
order, memory layout, tile size, and synchronization unchanged. It directly
tests whether launching 1024 threads for 32 useful context positions contributes
material overhead. It is the smallest reversible experiment and has no expected
numerical effect.

If geometry alone is insufficient, compare a C/SIMD vector implementation and
a head-parallel SIMT reduction. Those variants are separate experiments because
the head-parallel reduction changes FP32 addition order and requires stronger
accuracy validation.

### B. Optimize Sparse Attention First

Larger QK or Value tiles could remove many VF calls and flag handshakes, but
they alter L1/L0/UB pressure and a complex mixed-pipeline protocol. Sparse
Attention has the largest theoretical workflow upside, but it is a higher-risk
first change and needs compiler resource metadata before implementation.

### C. Continue Top-K First

A distributed context-shard histogram can parallelize the 131 us radix stage,
but it adds several launches and GM histogram traffic. Even eliminating Top-K
entirely improves this workflow by at most 15.9%. It remains a measured fallback
after the score experiment, not the first implementation.

## Optimization Backlog And Gates

Items are ordered by current evidence. An item is not automatically implemented
because it appears here; each gate prevents carrying a negative optimization
into V2.

### S0: Score VF Launch-Geometry Matrix (Rejected)

Build otherwise identical variants with 32, 64, 128, 256, and 1024 VF threads.
Keep one context position per thread and the serial 64-head loop unchanged.
Collect repeated direct-operator latency and the CannBench decode workflow.

Retain a new geometry only if:

- all V2 accuracy cases pass for at least five repeated launches;
- score median improves by at least 5%;
- Indexer and workflow medians do not regress; and
- a second clean-process run confirms the direction.

The matrix was run on the isolated `e781b64` release with production `-O3` and
CannBench BasicInfo. Every candidate passed five repeated canonical,
masked-tail, and tied-threshold device accuracy launches. Each workflow
collection exposes the score kernel twice, once in the Indexer component and
once while profiling the dependent Sparse Attention component:

| VF threads | Score observations (us) | Score median (us) | Workflow (us) |
| ---: | ---: | ---: | ---: |
| 1024 baseline | 292.353, 291.947 | 292.150 | 828.131 |
| 32 | 287.592, 287.962 | 287.777 | 825.124 |
| 64 | 288.882, 286.516 | 287.699 | 821.123 |
| 128 | 286.551, 287.938 | 287.244 | 822.570 |
| 256 | 288.314, 285.424 | 286.869 | 821.561 |

The best score result is only 1.8% below baseline and the best workflow result
is 0.8% below baseline. This misses the 5% score gate. The experiment was
stopped without changing production geometry because it only removes inactive
threads; it does not parallelize the serial 64-head reduction. Raw artifacts
are preserved under:

```text
/tmp/cannbench-dsa-v2-score-s0-baseline/
/tmp/cannbench-dsa-v2-score-s0-runs/
```

### S1: 1024-Thread Head-Parallel Score Reduction (Retained)

Keep the 1024-thread VF and map all threads to useful work:

```text
32 warps per VF * 32 lanes = 1024 threads
warp index                   = context position in the 32-position tile
lane index                   = head index 0..31
heads processed by each lane = lane, lane + 32
reduction                    = five asc_shfl_down steps within the warp
writer                       = lane 0 for that context position
```

This uses every thread, needs no UB scratch, and adds no block-wide or
inter-core synchronization. Keep the existing per-head Float-to-BF16, ReLU,
weight multiply, and BF16-to-Float conversion points. Only the FP32 addition
order changes from a serial 64-value sum to a warp tree.

The path must pass canonical, masked-tail, all-equal, near-threshold,
negative-score, and repeated-stability cases. Do not accept score tolerance
alone: the selected Top-K score multiset must still match the trusted
reference. Retain it only if the score median improves by at least 10%, the
full Indexer and workflow do not regress, and a clean-process repeat confirms
the gain. If the tree order changes threshold membership, test a two-stage
warp reduction that preserves deterministic groups before considering a C/SIMD
fallback.

The first production-`-O3` device build passed five repeats of the canonical,
masked-tail, and tied-threshold cases. It measured score at 147.664 and
148.670 us and the workflow at 681.981 us. The fixed two-head lane body was then
written explicitly because Bisheng did not honor the requested unroll pragma.
The final source compiles without that warning and additionally passes five
repeats of signed near-threshold and negative-score cases.

The clean-process final repeat reports:

| Boundary | Baseline (us) | Head-parallel (us) | Improvement |
| --- | ---: | ---: | ---: |
| Score median | 292.150 | 147.763 | 49.4% |
| Indexer | 424.768 | 279.847 | 34.1% |
| Workflow | 828.131 | 685.502 | 17.2% |

The final score observations are 148.182 and 147.343 us. Radix Top-K remains
about 131.5 us and Sparse Attention remains about 405.7 us, so the gain is
isolated to the intended score stage. Raw S1 artifacts are preserved under:

```text
/tmp/cannbench-dsa-v2-score-s1-runs/
```

The official post-S1 `Default` and `InstrTimeline` recollection used CannBench's
`--aic-metrics` option without source edits. `Default` measured score at
147.936 us, radix Top-K at 132.544 us, fused Sparse Attention at 368.502 us,
and Combine at 36.266 us. `InstrTimeline` reproduced the same boundary within
normal run variance. The complete local artifacts are preserved under:

```text
/tmp/cannbench-dsa-v2-decode-current-metrics/
```

### S2: Score Producer/Consumer Pipelining

If postprocess remains on the critical path, prototype flag-0/flag-1 ping-pong
buffers so AIC tile `n+1` can overlap AIV postprocess for tile `n`. Before
implementation, prove the physical LCM/UB allocation and flag lifetime on
`dav-3510`; the earlier 8 KiB per-sub-AIV shared-score limit remains binding.

Proceed only if Default or InstrTimeline shows at least 15% exposed non-overlap
after S0/S1. The protocol must stay kernel-local and must not introduce
inter-core coordination between logical tasks.

### P0: Propagate Head-Parallel Reduction To Prefill

Both V2 BF16 prefill score paths have the same `H=64`, 32-context tile, and
1024-thread launch followed by a serial 64-head loop in only the first 32
threads:

- `lightning_indexer_prefill_q2_family_64x128` reduces the score and then runs
  an in-VF block-wide Top-K merge;
- `lightning_indexer_prefill_full_score_family_64x128` writes BF16 scores for a
  later radix Top-K launch.

After the decode reduction is retained, apply the same 32-warp mapping to each
prefill source as separate experiments. The q2 tail path must keep every warp
alive until `asc_syncthreads`; warps beyond `current_context` skip score work
but cannot return early. The full-score path may use uniform per-warp masking.
Run the existing sampled-reference and repeated-stability prefill tests, then
measure the complete V3.2 prefill workflow. Retain each path independently only
if its selected-kernel boundary improves by at least 10% without changing the
Top-K score set.

The mapping was implemented in both sources. The q2 source builds successfully
and keeps all tail warps alive through the existing block-wide barrier, but it
is not currently called by the V2 public dispatch and therefore has no claimed
runtime gain. The canonical V3.2 q4096 case exercises the full-score source.

Real-device accuracy on Ascend 950PR passed three repeats each for canonical,
masked-tail, tied-threshold, signed near-threshold, and negative-score inputs.
Every case preserved the sampled Top-K score multiset, valid unique indices,
and a stable selected index set. Production `-O3` BasicInfo results were:

| Boundary | Baseline (ms) | Optimized 1 (ms) | Optimized 2 (ms) | Gain vs baseline |
| --- | ---: | ---: | ---: | ---: |
| Full-score kernel | 276.019 | 136.848 | 136.839 | 50.4% |
| Lightning Indexer | 299.560 | 160.543 | 160.583 | 46.4% |
| DSA workflow | 635.266 | 494.284 | 497.748 | 21.6% |

Radix Top-K stayed near 23.6 ms and Sparse Attention ranged from 333.7 to
337.2 ms, isolating the retained gain to full-score reduction. Raw artifacts
are preserved under:

```text
/tmp/cannbench-dsa-v2-prefill-baseline-runs/
/tmp/cannbench-dsa-v2-prefill-optimized-runs/
```

### A0: Sparse Attention QK Tile 64 -> 128 (Retained)

Changing `kHead64QkTile` to 128 reduces QK iterations per selected tile from
9 to 5 and removes 32 key-pack VF calls per AIV task. The source-level L1
worksheet changes from approximately 180,224 bytes to 196,608 bytes:

```text
query                  73,728
double-buffered keys   32,768
scores                 16,384
probabilities           8,192
double-buffered values 32,768
PV                     32,768
total                 196,608 bytes
```

Compiler metadata, not this worksheet, decides whether the variant is viable.
Reject it on spills, reduced useful occupancy, resource errors, or less than a
5% fused-kernel gain.

The production `-O3` `dav-3510` build completed without resource or spill
diagnostics in the changed fused kernel. Full V3.2 decode accuracy passed for
all 128 heads and both query rows in both batches, including injected negative,
out-of-range, and causal-future indices. Output and LSE both reported zero
mismatches at `atol=rtol=0.05`.

Two clean-process CannBench BasicInfo runs reported:

| Boundary | S1 baseline (us) | QK=128 run 1 (us) | QK=128 run 2 (us) | Gain vs baseline |
| --- | ---: | ---: | ---: | ---: |
| Sparse Attention fused | 368.502 | 326.919 | 327.363 | 11.2% |
| Sparse Attention + Combine | 404.768 | 363.529 | 362.868 | 10.3% |
| Full decode workflow | 685.248 | 643.484 | 642.075 | 6.3% |

The unchanged Indexer measured 279.955 and 279.207 us, confirming that the
gain is isolated to Sparse Attention. The QK=128 variant therefore passes the
5% fused-kernel gate and is retained. Raw artifacts are preserved under:

```text
/tmp/cannbench-dsa-v2-sparse-a0-runs/
/tmp/cannbench-dsa-v2-sparse-a0-f94694b/
```

### A1: Sparse Attention Value Tile 128 -> 256

Changing `kHead64ValueTile` to 256 reduces Value iterations from 4 to 2 and
removes about 32 value-pack/output-update VF calls per AIV task. With QK still
64, the source-level L1 worksheet is approximately 245,760 bytes. This is close
to the expected capacity and must be compiled and inspected before device use.

Do not combine QK=128 and Value=256 in the first experiment: their source-level
total is about 262,144 bytes before alignment, compiler temporaries, or runtime
reservation and therefore has no defensible safety margin.

### A2: Sparse Attention Partition Matrix

Measure automatic P1, P2, and P4 variants with identical semantics and include
Combine in the boundary. Fewer partitions reduce partial-output and Combine
work but reduce parallel task count and increase selected-token work per task.
Retain P4 unless a repeated full-component median proves otherwise.

### A3: Sparse Attention VF And Flag Coarsening

The post-A0 InstrTimeline shows that each AIV lane spends about 292 us of its
roughly 325 us span in `VF_SIMT`. The compiler already presents the three
initialization/query calls as one contiguous approximately 7.6 us VF region,
and the gaps between adjacent key-pack and Value-pack VFs are generally below
0.4 us. Folding those calls therefore has insufficient standalone upside.

The selected first experiment is pairwise PV coarsening at QK=128, Value=128,
and automatic P4. Add a second 32-by-128 FP32 PV slot in UB, increasing the
source-level UB worksheet from approximately 147,840 bytes to 164,224 bytes.
Keep the two existing L1 Value gather slots and the existing ready flags. AIC
computes two consecutive 128-dimension PV tiles into alternating UB slots and
publishes one ready event for the pair. AIV applies `old_scale` and accumulates
both slots into the corresponding 256 output dimensions with one VF, then
acknowledges the pair. The odd/tail case publishes and consumes a single tile.

The pipeline ownership is:

```text
L1 Value slot s   AIV produces -> AIC consumes -> AIC requests reuse
UB PV slot s      AIC produces -> AIV consumes -> pair acknowledgement
running_output    AIV owns for the complete logical task
```

This schedule keeps the existing one-tile-ahead Value gather: after AIC loads
the second tile of a pair, it requests the first L1 slot for the next pair;
AIV may refill that slot while Cube finishes the second PV. It adds no flag ID,
no cross-core primitive, and no Basic API dependency. A four-tile version is
rejected because its source-level UB requirement is approximately 196,992
bytes before compiler temporaries and runtime reservation.

Retain pairwise coarsening only if full V3.2 decode output and LSE accuracy pass,
two clean-process BasicInfo runs show at least a 5% fused-kernel improvement,
and the full workflow does not regress. If it misses the gate, revert the
kernel and source-contract changes while preserving the negative result here.

The pairwise PV experiment was rejected. Its production `-O3` `dav-3510`
build completed without a fused-kernel resource or spill diagnostic, and full
V3.2 decode accuracy passed with zero output and LSE mismatches at
`atol=rtol=0.05`. The additional UB and coarser update nevertheless regressed
the measured boundary in two clean CannBench BasicInfo processes:

| Boundary | A0 baseline (us) | PV pair run 1 (us) | PV pair run 2 (us) | Median change |
| --- | ---: | ---: | ---: | ---: |
| Indexer | 279.207 | 280.374 | 279.283 | +0.2% |
| Sparse Attention fused | 327.363 | 336.378 | 336.616 | +2.8% |
| Sparse Attention + Combine | 362.868 | 372.703 | 373.067 | +2.8% |
| Full decode workflow | 642.075 | 653.077 | 652.350 | +1.7% |

The implementation and source-contract changes were reverted, and published
data remains on A0. Raw controller artifacts are preserved under
`/tmp/cannbench-dsa-v2-sparse-a3-pvpair-runs/`; the unique remote build,
accuracy result, and profiler outputs are under
`/tmp/cannbench-dsa-v2-sparse-a3-pvpair-eeccd1f/` on the Ascend 950PR host.
This result rules out pairwise PV buffering as a useful A3 continuation on the
current schedule; future work should not retry it without a different producer
or buffer-lifetime mechanism.

Do not move cross-core waits into an opaque monolithic VF merely to reduce the
visible event count. Accept only a reduction in repeated BasicInfo/Default
latency with unchanged correctness.

### A4: Combine And Output Materialization

Profile whether partition outputs can be reduced with less GM traffic or
whether a direct-output mode is profitable at a lower partition count. Preserve
the output/LSE contract and include any replacement helper kernel in the timing
boundary. The current 36 us Combine stage caps the standalone gain.

### T0: Distributed Context-Shard Histogram Microbenchmark

Implement only the isolated launch-chain microbenchmark described in the V2
unordered-radix design. Include histogram production, digit reducers, offsets,
and compaction. Proceed to production integration only if the complete chain is
below 105 us, which is a 20% improvement over the current 131.422 us stage.

### T1: Single-Block Radix Refinements

If T0 misses its gate, profile the current block for atomic histogram pressure,
barrier count, threshold-equal scan cost, and four-block imbalance. Consider a
warp-private histogram merge and a bounded equal-threshold compaction only when
the corresponding canonical or tie-heavy microbenchmark identifies that work
as dominant.

### W0: Workflow-Level Cleanup

Keep dependency materialization and helper launches visible in raw profiles.
The Cast helper is currently outside the selected Sparse Attention boundary;
do not claim its removal as a component gain unless the published timing
contract is intentionally updated. Consider score-to-histogram fusion only
after S0-S2 and T0 establish that avoiding the BF16 score workspace is worth
the added ownership and synchronization complexity.

### C0: API Boundary Convergence

Performance changes must not add Basic API dependencies. Once a winning score
or attention schedule is stable, replace retained `LocalTensor`, Basic flag,
and `PipeBarrier` usage with C API, Tensor API, SIMT API, or the allowed
kernel-local Mutex exception. Treat this as a separately validated cleanup so
API migration cannot hide a performance regression.

## Validation Matrix

Each retained change runs:

1. operator-local source and dispatch tests;
2. V2 canonical, masked-tail, tied-threshold, and repeated-stability accuracy;
3. V1 regression for the same decode case;
4. remote production `-O3` build targeting `dav-3510`;
5. synchronized direct-operator repetitions in a clean process;
6. CannBench `dsa_decode` BasicInfo workflow collection;
7. Default or InstrTimeline recollection only when needed to test the stated
   bottleneck hypothesis.

Report median and spread for repeated latency. Preserve raw run directories,
package revision, compiler/runtime versions, selected and excluded kernels,
and every negative variant that reaches the device.

## Completion Criteria

This optimization series is complete when either:

- the workflow is within 2x of the BF16 vLLM-Ascend component sum under the
  same boundary; or
- every remaining backlog item misses its acceptance gate and the preserved
  profiles identify no untested bottleneck with at least 5% workflow upside.

No individual item is complete until real-device correctness, repeated
performance, a same-stack baseline rerun, and raw artifacts are available.
