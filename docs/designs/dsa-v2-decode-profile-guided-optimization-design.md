# DSA V2 Decode Profile-Guided Optimization Design

## Status

Approved for staged implementation on 2026-08-01. The launch-geometry-only
experiment was rejected after device measurement. The 1024-thread,
head-parallel Lightning Indexer decode and V3.2 full-score prefill reductions,
the Sparse Attention QK=128 tile, restoration of all 32 fused-kernel warps, and
canonical decode `int32` KV row-offset reuse, P4 Combine weight reuse, and the
two-slot BF16 score producer/consumer pipeline, and single-scan deterministic
Top-K compaction, transposed Value packing, and canonical Value256 tiling are
retained after correctness and performance validation. Corrected canonical
QK256 tiling is also retained on top of Value256 after preserving Query for
all eight selected-token tiles. Pairwise PV coarsening,
2048-thread Key/Value Pack widening, adjacent-value grouped loads, and Value
Pack synchronization contraction were rejected after device measurement. The
current published V2 decode workflow checkpoint is 335.631 us from retained
QK256. The matching published vLLM-Ascend workflow is 169.797 us, so SIMT
currently has 1.977x its latency and 0.506x its performance. The two QK256
selected-workflow measurements are 334.621 and 335.631 us.

The 2x point is an intermediate checkpoint, not the completion target. The
target is at least 0.90x vLLM-Ascend performance under the same selected-kernel
boundary. At the current 169.797 us reference this requires SIMT latency at or
below 188.663 us, another 146.968 us or 43.8% below the published QK256
checkpoint. Later stages remain gated by correctness and repeated device
results.

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

The initial two-slot BF16 prototype was rejected prematurely. A fresh retained
baseline measured score at 147.069 us, radix Top-K at 132.370 us, Indexer at
279.439 us, Sparse Attention plus Combine at 294.868 us, and the complete
workflow at 574.307 us. The failed prototype did expose required Fixpipe
contracts: the call signature is `Fixpipe<bfloat16_t, float>` and FP32-to-BF16
conversion needs `quantPre=F322BF16`.

The follow-up investigation on 2026-08-02 isolated two independent defects:

1. The hanging candidate reused flag IDs 0 and 1 in both directions for ready
   and free handshakes. Ready now uses flags 0 and 1, while free uses flags 2
   and 3. AIC initially owns both slots and waits for free only before a slot is
   reused (`tile_index >= 2`). AIV waits for ready and publishes free only when
   another reuse exists; terminal releases are suppressed.
2. The full Q=2 path failed with runtime error `507015`, device error code 169,
   because BF16 Fixpipe used `mSize=128` and `dualDstCtl=1`. This was an invalid
   Fixpipe parameter, not a synchronization timeout. The retained path disables
   dual destination and issues two 64-row Fixpipe calls with `subBlockId=false`
   and `subBlockId=true`; the second call starts at the second-query L0 offset.

The synchronization protocol was expanded on device before restoring the full
operator: one tile wrote 32 expected outputs, two tiles wrote 64, four tiles
passed the first slot reuse, 64 tiles passed five repeats, and Q=2 with 64 tiles
passed three repeats for both query outputs. The production source then built
with CANN 9.2.0 for `dav-3510` at `-O3`.

Full V3.2 decode accuracy passed three repeats each for canonical, masked-tail,
tied-threshold, near-threshold, and negative-score cases. Validation compared
the unordered Top-K score multiset and required a stable selected index set.

Production `BasicInfo` on Ascend 950PR measured:

| Boundary | Pre-S2 published baseline | Two-slot BF16 | Improvement |
| --- | ---: | ---: | ---: |
| Score kernel | 148.363 us | 82.889 us | 44.1% |
| Radix Top-K | 132.726 us | 131.713 us | 0.8% |
| Lightning Indexer | 0.281089 ms | 0.214602 ms | 23.7% |
| Sparse Attention + Combine | 0.270388 ms | 0.269829 ms | 0.2% |
| DSA workflow | 0.551477 ms | 0.484431 ms | 12.2% |

The workflow result exceeds the 3% retention gate, so the two-owned-slot
producer/consumer pipeline is retained. The local raw CannBench artifact is:

```text
/tmp/cannbench-dsa-v2-s2-owned-slots-results/s2-owned-slots-basicinfo/
```

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

The P1/P2/P4 device-accuracy matrix used a CPU oracle after the original NPU
reference chain repeatedly hit error `507034`. All three variants passed the
canonical valid-index case with zero mismatches across 262,144 output elements
and 512 LSE elements at `atol=rtol=0.05`.

On the valid-index Sparse Attention input, P1 measured 1,215.315 us
(380.076 us QK and 835.239 us PV), while P2 measured 631.005 us
(193.681 us QK, 416.917 us PV, and 20.407 us Combine). The matching P4
baseline was subsequently re-collected at 499.167 to 501.284 us, including a
462.988 to 465.188 us fused kernel and a 36.096 to 36.179 us Combine. P1 and
P2 are therefore slower than P4, so automatic P4 remains selected.

Do not compare the valid-index matrix directly with the historical 294.868 us
workflow-bound Sparse Attention result. The workflow binds Indexer-produced
indices, whereas a direct Sparse Attention case materializes its own indices;
the fused workload and absolute latency are not equivalent. Also discard the
first P4 recollection that omitted `CANNBENCH_SKIP_SIMT_INSTALL=1`, because it
rebuilt and overwrote the package libraries during measurement.

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

The earlier interpretation of `896` active threads as a dav-3510 execution
limit is withdrawn. One Vector Core supports at most 2048 threads, or 64
32-thread warps. The fused VF currently launches 1024 logical threads, or 32
warps, but explicitly discards threads 896 through 1023 and makes the first
four warps cover the missing four rows. That workaround was introduced while
the PV update path still published `kAivToAicReady` before the update VF had
completed. The resulting UB reuse race could make the trailing warps observe
an overwritten PV tile and was not a valid worker-liveness probe. Commit
`8e119a3` subsequently inserted the required `PIPE_V -> PIPE_MTE3` completion
event before publishing the reuse flag, but the `896 / 28` workaround remained.

The post-A0 production source was recompiled on the Ascend 950PR host with the
same `-O3`, `dav-3510`, and dependency inputs plus `--cce-res-usage`. Its source
SHA-256 was
`29c9ca4bf7632b9b8f265b2b4b15ed0ed83413606d1eefc72c0a3779b834d92d`,
matching the local source exactly. Bisheng reported:

| VF | Registers per thread | Stack size |
| --- | ---: | ---: |
| Softmax init | 10 | 0 B |
| Output init | 7 | 0 B |
| Query pack | 21 | 0 B |
| Output update | 19 | 0 B |
| Key pack | 29 | 0 B |
| Value pack | 31 | 0 B |
| Softmax | 32 | 0 B |
| Output write | 31 | 16 B |

The local compiler log is `/tmp/cannbench-sparse-a0-cce-res.log`; the remote
resource-only library is
`/tmp/cannbench-dsa-v2-sparse-a0-f94694b/sparse-a0-cce-res.so`.

#### A3.1: Restore Full 1024-Thread Execution

Restore `1024` active threads and `32` active warps without changing
`launch_bounds`, tile sizes, buffers, or synchronization. Each AIV then maps
one warp directly to each of its 32 local heads or selected rows. Remove the
four-warp second-row workaround and keep the `PIPE_V -> PIPE_MTE3` wait before
every PV reuse publication.

This is a correction of a stale workaround, not a 2048-thread experiment. It
must first pass full V3.2 decode output and LSE accuracy, invalid-index and tail
cases, and repeated launches. Retain it only when two clean-process BasicInfo
runs show at least a 5% fused-kernel improvement and the full workflow does not
regress.

The production `-O3` `dav-3510` build restored 1024 active threads and 32
active warps. It removed the four-warp second-row workaround while preserving
the existing `PIPE_V -> PIPE_MTE3` completion event before PV-buffer reuse.
Bisheng reported zero Stack bytes for every VF:

| VF | Registers per thread | Stack size |
| --- | ---: | ---: |
| Query Pack | 16 | 0 B |
| Key Pack | 19 | 0 B |
| Value Pack | 18 | 0 B |
| Output Update | 14 | 0 B |
| Softmax | 32 | 0 B |
| Output Write | 24 | 0 B |

Full V3.2 decode accuracy passed five repeated launches in one process. Output
and LSE both had zero mismatches at `atol=rtol=0.05`; maximum absolute errors
were 0.0078125 for output and 0.0092830658 for LSE. Coverage included negative,
out-of-range, and causal-future indices.

Two clean-process BasicInfo runs reported:

| Boundary | A0 baseline (us) | 1024/32 run 1 (us) | 1024/32 run 2 (us) | Median change |
| --- | ---: | ---: | ---: | ---: |
| Sparse Attention fused | 327.363 | 283.422 | 283.517 | -13.4% |
| Sparse Attention + Combine | 362.868 | 319.437 | 319.784 | -11.9% |
| Full decode workflow | 642.075 | 598.477 | 598.536 | -6.8% |

The variant passed the 5% fused-kernel gate and was retained in `4b6f38c`.
The measured source SHA-256 was
`4eef8bf63e80abd218c4535876e5cb5e61f6e98d61d7e7480cf5ee19ece86355`.
Raw artifacts are preserved under:

```text
/tmp/cannbench-dsa-v2-sparse-a1-1024-runs/
/tmp/cannbench-dsa-v2-sparse-a1-1024-OZbhzH/  # Ascend 950PR host
```

#### A3.2: KV Offset Reuse (Retained) And 2048-Thread Pack VFs (Rejected)

Ascend 950 assigns at most 16 registers per thread when `launch_bounds` is in
the `1025-2048` tier. A current 1024-tier report below 16 registers is only a
candidate screen: every 2048 variant must be recompiled with its final
`launch_bounds` and retained only when Bisheng reports fewer than 16 registers
and zero Stack bytes.

Do not switch every fused VF to 2048. Softmax keeps its one-warp-per-head
reduction and remains at 1024. The current initialization VFs satisfy the
resource screen but represent too little latency to justify an isolated
experiment. The first material 2048 target was canonical V3.2 decode Key/Value
packing after reducing register pressure and repeated index work.

For each 64-selected-token tile, each AIV owns 32 selected rows. Its first Key
Pack call computes and stores one signed `int32` relative KV row offset per
local row, using `-1` for an invalid context index, and packs the first QK
dimension tile. The following Key Pack calls and all Value Pack calls consume
those offsets. The exact canonical shape keeps every valid relative offset in
the signed `int32` range. This removes repeated index loads, range checks,
batch/query/partition address construction, and long-lived 64-bit
intermediates without adding another VF call. The per-AIV UB cost is 128 bytes
plus alignment.

The fast canonical pack signatures receive preadjusted source and destination
pointers, the row-offset buffer, and the dimension start. Exact V3.2 decode
constants such as context length, selected length, row stride, QK tile, and
Value tile are compile-time values. Generic decode, prefill, tails, and other
shapes keep the existing fallback signatures and semantics.

The evaluated 64-warp Key/Value mapping was:

```text
local selected row = warp / 2
dimension half     = warp % 2
first dimension    = dimension half * 32 + lane
dimension stride   = 64
```

The two warps for a row wrote disjoint dimension intervals and required no
cross-warp reduction. Output Update was screened separately after pointer
preadjustment and fixed 128-dimension specialization; A3.3 records why it did
not pass the profile gate for a 2048-thread experiment. Output Write is
currently 24 registers with zero Stack bytes at the 1024 tier; specialize its
canonical output mode and stop passing complete plan and task objects by value
before considering a wider launch.

Do not use manual unrolling as a register-reduction technique without a
resource report: the occupancy benchmark showed that manual expansion can
increase Stack use. Each register-reduction change must record the final
1024-tier and 2048-tier Bisheng report, full correctness, repeated BasicInfo
latency, and InstrTimeline attribution. A 2048 candidate is rejected on any
Stack allocation, 16 or more registers, less than a 5% fused-kernel gain, or a
workflow regression.

Do not move cross-core waits into an opaque monolithic VF merely to reduce the
visible event count. Accept only a reduction in repeated BasicInfo/Default
latency with unchanged correctness.

The first offset-reuse prototype used signed `int64` offsets and 256 bytes of
UB. It increased the Prepare/Key/Value resource reports to 18/22/20 registers
per thread, so it was not a viable 2048 candidate. Changing only the stored
representation to safe signed `int32` offsets reduced both UB use and register
pressure. The final production report was:

| VF | Registers per thread | Stack size |
| --- | ---: | ---: |
| Key Prepare | 14 | 0 B |
| Fast Key | 14 | 0 B |
| Fast Value | 13 | 0 B |
| Generic Key | 19 | 0 B |
| Generic Value | 18 | 0 B |

Full V3.2 decode accuracy again passed five repeated launches with zero output
and LSE mismatches at `atol=rtol=0.05`. Maximum absolute errors remained
0.0078125 for output and 0.0092830658 for LSE. Generic decode, prefill, tails,
noncanonical shapes, and V1 were unchanged.

Two clean-process BasicInfo runs reported:

| Boundary | 1024/32 baseline (us) | Offset run 1 (us) | Offset run 2 (us) | Median change |
| --- | ---: | ---: | ---: | ---: |
| Sparse Attention fused | 283.470 | 257.467 | 258.487 | -9.0% |
| Sparse Attention + Combine | 319.611 | 294.272 | 294.892 | -7.8% |
| Full decode workflow | 598.507 | 572.848 | 574.461 | -4.2% |

The fused-kernel gain passed the 5% gate, so `int32` offset reuse was retained
in `8561209`. Its measured source SHA-256 was
`c3f53b6ef675f1ec24b9b65be3039023f9c7c21622cdee1acadd91dce00ec8ab`.
The first run was published as 0.572848014 ms in `6c821c2`.

The subsequent 2048-thread experiment widened only the fast Key and Value Pack
VFs, mapping two warps to each selected row. The final 2048-tier build passed
the resource gate: Fast Key used 12 registers, Fast Value used 13, both had
zero Stack bytes, and Key Prepare remained at 1024 threads with 14 registers.
The measured source SHA-256 was
`1ef97d1b746ce6a2a94b0426e89ebf3cc7dddb7dd21be5c29931b524f210145d`.
Accuracy passed, but the first BasicInfo run regressed every retained boundary:

| Boundary | Retained offset run 1 (us) | 2048 Pack run 1 (us) | Change |
| --- | ---: | ---: | ---: |
| Sparse Attention fused | 257.467 | 260.857 | +1.3% |
| Sparse Attention + Combine | 294.272 | 296.750 | +0.8% |
| Full decode workflow | 572.848 | 576.394 | +0.6% |

The 2048 Pack variant therefore failed the performance gate and was reverted
without a second run. The result demonstrates that satisfying the register and
Stack gate is necessary but does not imply useful scaling when the extra warp
only splits one row's dimension work. Do not retry this exact Key/Value mapping
without a different work decomposition or profile evidence.

Raw artifacts are preserved under:

```text
/tmp/cannbench-dsa-v2-sparse-a2-offset-runs/
/tmp/cannbench-dsa-v2-sparse-a3-pack2048-runs/
/tmp/cannbench-dsa-v2-sparse-a2-offset-ZDy60C/   # Ascend 950PR host
/tmp/cannbench-dsa-v2-sparse-a3-pack2048-SOh6z4/ # Ascend 950PR host
```

#### A3.3: 2048-Thread Output Update (Rejected By Profile Gate)

The retained offset-reuse source was profiled before making another launch-
geometry change. The local source and the deployed A2 package both had SHA-256
`c3f53b6ef675f1ec24b9b65be3039023f9c7c21622cdee1acadd91dce00ec8ab`.
CannBench collected separate `Default` and `InstrTimeline` runs for the exact
canonical V3.2 decode workflow. `Default` reported 259.246 us for fused Sparse
Attention and 573.903 us for the full workflow; `InstrTimeline` reproduced the
fused boundary at 259.448 us, so metric perturbation was negligible for this
attribution.

The timeline exposes the repeated eight-selected-tile sequence on every
recorded AIV lane. For each selected tile, the first Value Pack occurs at PC
`0x120053e0116c`; the next three Value Pack calls occur at
`0x120053e01388`; and all four Output Update calls occur at
`0x120053e012ac`. This matches the source pipeline exactly:

```text
first Value Pack
3 x (next Value Pack, wait for AIC PV, Output Update)
wait for AIC PV, final Output Update
```

Across the 12 recorded lanes, each lane had exactly 32 Output Update regions.
Their per-lane sum ranged from 23.416 to 23.978 us, with a 23.628 us mean. The
384 individual calls had a 0.738 us mean, 0.734 us median, and 1.493 us
maximum. In contrast, the 32 Value Pack calls consumed about 111.647 us per
lane on average, including eight first-pack and 24 pipelined-pack regions. The
roughly 3.458 us regions are therefore Value Pack, not Output Update.

The AIC-to-AIV waits immediately preceding Output Update used PC
`0x120053e0127c`. They totaled 6.242 to 7.468 us per lane and remain a separate
dependency cost; widening Output Update cannot remove them and may only expose
more of that wait. Because lanes execute concurrently, the relevant fused
critical-path contribution is the maximum per-lane Output Update sum, not the
sum over all 12 recorded lanes.

Five percent of the 259.448 us fused boundary is 12.972 us. Even an ideal 2x
speedup of the worst recorded 23.978 us Output Update path can save only
11.989 us, or 4.62% of the fused boundary, before accounting for launch,
synchronization, or scheduling overhead. This misses the predeclared 26 us
profile gate. The canonical 2048-thread Output Update specialization was
therefore rejected before source or test changes. Do not retry the identical
two-warps-per-head mapping unless a later profile shows at least 26 us of
Output Update work on the fused critical path or the acceptance threshold is
intentionally changed.

Raw profile artifacts are preserved under:

```text
/tmp/cannbench-dsa-v2-output-update-baseline/
```

### A4: Combine And Output Materialization

Profile whether partition outputs can be reduced with less GM traffic or
whether a direct-output mode is profitable at a lower partition count. Preserve
the output/LSE contract and include any replacement helper kernel in the timing
boundary. The current 36 us Combine stage caps the standalone gain.

The first A4 prototype specializes the canonical P4 Combine path. Each lane
loads the four partial LSE values and computes the four normalized weights once,
then reuses those weights across all 512 output dimensions. Noncanonical paths
keep the existing runtime-partition fallback. The final source needed the
namespace-qualified constant
`aten_dsa_sparse_attention_v2::kHead64OutputPartialFloat`; six focused source
tests pass.

The fixed production package archive SHA-256 is
`29596d93a54a799906eaa6c4ed41fe097a48c360cd8f8128f80c3c9e4ffe7700`.
The baseline and candidate fused libraries both have SHA-256
`7959687fc406d7a584a94bedded98551dd0557483e583bdb984389685c22f010`;
only the Combine library differs (`607642ca...` baseline and `a20089f...`
candidate). A fresh canonical P4 CPU-oracle accuracy run passed with zero
mismatches for all 262,144 output and 512 LSE elements at `atol=rtol=0.05`.

Two alternating clean-process CannBench BasicInfo pairs on the same valid-index
input reported:

| Boundary | Baseline 1 (us) | A4 run 1 (us) | Baseline 2 (us) | A4 run 2 (us) | Median change |
| --- | ---: | ---: | ---: | ---: | ---: |
| Sparse Attention fused | 465.188 | 466.133 | 462.988 | 464.847 | +0.3% |
| Combine | 36.096 | 12.012 | 36.179 | 12.289 | -66.4% |
| Fused + Combine | 501.284 | 478.145 | 499.167 | 477.136 | -4.5% |

A4 therefore had a stable component-level gain above 3%, isolated to Combine.
The first complete workflow collection was blocked when the unchanged Lightning
Indexer path hit `507014` during profiler warmup and also failed to complete a
profiler-free `cannbench internal-run` within 180 seconds. Sparse Attention and
A4 continued to pass immediately afterward, so the failure was not attributed
to the changed A4 boundary.

After the Ascend node restarted, the unchanged Indexer completed at 281.089 us
and Sparse Attention plus the specialized Combine completed at 270.388 us,
including 258.475 us fused and 11.913 us Combine. The resulting V3.2 decode
workflow was 551.477 us, 3.7% below the pre-A4 published 572.848 us checkpoint.
The result passed the workflow retention gate, was retained in `d78b2a6`, and
was published in `8602a74`.

Artifacts are preserved under:

```text
/tmp/cannbench-dsa-v2-a4-sparse-retest-runs/       # controller
/tmp/cannbench-dsa-v2-a4-p4weights-fix-results/   # Ascend host accuracy
/tmp/cannbench-dsa-v2-a4-p4weights-fix-tbRFz2/    # Ascend host package
```

### T0: Distributed Context-Shard Histogram Microbenchmark

Implement only the isolated launch-chain microbenchmark described in the V2
unordered-radix design. Include histogram production, digit reducers, offsets,
and compaction. Proceed to production integration only if the complete chain is
below 105 us, which is a 20% improvement over the current 131.422 us stage.

### T1: Single-Block Radix Refinements

T1 was retained after target-device validation. The two BF16 radix histogram
passes and one-block-per-row launch remain unchanged. The old compaction first
used a contended atomic for every score above the selected threshold, then
visited the context in 1024-element chunks and performed a complete block scan
for each chunk needed to collect threshold-equal values. This repeated scan and
synchronization work after the radix threshold was already known.

The retained compaction gives every one of the 1024 threads its existing 32
strided context elements. Each thread counts values greater than and equal to
the threshold, packs both counts into non-overlapping 16-bit fields, and joins
one ten-step ping-pong inclusive scan. The resulting thread-major exclusive
offsets let each thread write its greater values and its bounded share of equal
values without output-slot atomics. Both complete counts are at most 32768, so
the packed scan cannot carry between fields. Output remains an unordered set of
2048 unique valid indices.

The production `-O3` build targeted Ascend 950PR (`dav-3510`) with CANN 9.2.0.
Five repeats of canonical, masked-tail, tied-threshold, near-threshold, and
negative-score cases all preserved the trusted unordered score multiset and a
stable selected index set. Two clean-process CannBench `BasicInfo` runs gave:

| Boundary | Published baseline | T1 run 1 | T1 run 2 | Improvement |
| --- | ---: | ---: | ---: | ---: |
| Score | 82.889 us | 83.585 us | 83.279 us | unchanged |
| Radix Top-K | 131.713 us | 81.899 us | 81.792 us | 37.8%-37.9% |
| Lightning Indexer | 0.214602 ms | 0.165484 ms | 0.165071 ms | 22.9%-23.1% |
| Sparse Attention + Combine | 0.269829 ms | 0.270273 ms | 0.270519 ms | unchanged |
| DSA workflow | 0.484431 ms | 0.435757 ms | 0.435590 ms | 10.0%-10.1% |

The gain is isolated to Top-K, is stable across both workflow runs, and exceeds
the 3% workflow retention gate. The first valid run, 0.435756979 ms, is the
published workflow record. Context-shard distributed histogram T0 remains
deferred because T1 removed most of the current Top-K gap without adding its
extra launches or GM workspace.

Artifacts are preserved under:

```text
/tmp/cannbench-dsa-v2-topk-t1-results/             # controller raw runs
/tmp/cannbench-dsa-v2-topk-t1-results.tar.gz       # controller archive
/root/cannbench-dsa-v2-topk-t1-results/            # Ascend host raw runs
/root/cannbench-dsa-v2-topk-t1-results.tar.gz       # Ascend host archive
```

The archive SHA-256 is
`f15f47517d28b7adc56f59303870ea8860ffc845572f418d7b8c119a012f6e09`;
the deployed source SHA-256 is
`3cc8d6f56265f636cc40f27192145e28e9cb30f99c95425449e36de4d4517f26`.

### A5: Canonical Value-Pack Load Grouping

After T1, the canonical workflow is 435.757 and 435.590 us in two clean
processes. Sparse Attention plus Combine remains 270.273-270.519 us, including
258.277-258.722 us in the fused kernel and 11.797-11.996 us in Combine. The
retained fused source still has SHA-256
`c3f53b6ef675f1ec24b9b65be3039023f9c7c21622cdee1acadd91dce00ec8ab`,
which is the exact source used by the existing InstrTimeline attribution. That
trace assigns about 111.647 us per recorded AIV lane to the 32 canonical Value
Pack regions, making their gather-and-pack instruction path the next measured
optimization target.

Keep P4, 1024 threads, 32 active warps, QK tile 128, Value tile 128, both L1
gather slots, and the existing producer/consumer protocol unchanged. Modify
only `head64_fused_value_pack_fast_vf`, which is already guarded by the exact
V3.2 BF16 decode predicate. Generic decode, prefill, tails, invalid indices,
and noncanonical shapes continue to use the existing fallback VF.

The first candidate assigns four adjacent BF16 dimensions to each lane. The
canonical row stride is 576 BF16 values, or 1152 bytes; every Value tile starts
at a multiple of 128 BF16 values, or 256 bytes; and lane groups start every
four BF16 values, or eight bytes. The GM address is therefore 64-bit aligned.
Each lane performs one 64-bit load, extracts the four BF16 bit patterns, and
writes them to the four existing NZ destinations. This preserves exact BF16
bits and the complete one-writer-per-element mapping while reducing four GM
load instructions and repeated address calculations to one grouped load.

If the 64-bit candidate does not compile, allocates Stack, or fails accuracy,
evaluate two aligned 32-bit loads per lane. If it is correct but misses the
performance gate, do not stack unrelated scheduling changes on it. A manually
expanded four-scalar-load form is only a diagnostic control because previous
resource experiments show that source unrolling can increase register or
Stack use.

Retain a candidate only when:

- the production `-O3`, `dav-3510` build has zero Stack bytes in the final
  Value Pack VF and no material register-pressure regression;
- full canonical V3.2 decode output and LSE accuracy passes five repeated
  launches, including invalid and causal-future indices;
- two clean-process CannBench `BasicInfo` runs improve the fused kernel by at
  least 5% and the complete workflow by at least 3%; and
- the gain remains isolated to Sparse Attention while Indexer latency stays
  within normal run-to-run variation.

Use the current 435.757/435.590 us workflow pair as the pre-A5 reference. Keep
all raw build, resource, accuracy, and profiler artifacts, including negative
candidates.

The 64-bit candidate was rejected after target-device measurement. Its source
SHA-256 was
`bd43b7f981a31473fa4d6bd171c12c8994d01cb3dc3cb7cb9bbe10e7cf30f36d`.
The production `-O3`, `dav-3510` build completed, and the diagnostic build
reported 11 registers per thread and zero Stack bytes for the fast Value Pack
VF, versus 13 registers and zero Stack bytes for the retained scalar traversal.
Resource pressure therefore did not explain the result.

Five complete V3.2 decode accuracy processes passed every output and LSE row,
including negative, out-of-range, and causal-future indices. Every run had zero
mismatches at `atol=rtol=0.05`; the first run's maximum absolute errors were
0.009765625 for output and 0.009282112 for LSE. Two clean-process CannBench
`BasicInfo` workflows then reported:

| Boundary | Retained run 1 | Retained run 2 | 64-bit run 1 | 64-bit run 2 |
| --- | ---: | ---: | ---: | ---: |
| Indexer | 165.484 us | 165.071 us | 164.992 us | 165.076 us |
| Sparse Attention fused | 258.277 us | 258.722 us | 552.043 us | 552.101 us |
| Combine | 11.996 us | 11.797 us | 12.049 us | 11.946 us |
| DSA workflow | 435.757 us | 435.590 us | 729.084 us | 729.123 us |

The fused boundary regressed by about 113% and the workflow by about 67% while
Indexer, Combine, and the 1650 MHz device frequency remained stable. The
evidence isolates the regression to the grouped fast Value Pack, but BasicInfo
does not distinguish a slow 64-bit GM instruction from the changed per-lane
access and NZ-store schedule. The kernel and source-contract test were reverted.

As a final discriminating control, the requested 32-bit form was also tested.
Each lane loaded dimensions `[2 * lane, 2 * lane + 1]` and
`[2 * lane + 64, 2 * lane + 65]`, producing two warp-coalesced 128-byte rounds
while retaining the existing NZ destinations. The candidate source SHA-256 was
`50bae4751c0ab18dbe219d19b0c9e33e0bca18f27bdbf30da5bd1bd15a9512f1`.
Its production `-O3`, `dav-3510` build passed; the resource report showed 14
registers per thread and zero Stack bytes, versus 13 registers and zero Stack
bytes for the retained traversal.

Five fresh accuracy processes again passed every output and LSE row, including
negative, out-of-range, and causal-future indices, at `atol=rtol=0.05`. Each
run reported zero mismatches; output and LSE maximum absolute errors were
0.009765625 and 0.009282112 respectively. Two clean-process `BasicInfo` runs
reported:

| Boundary | Retained run 1 | Retained run 2 | 32-bit run 1 | 32-bit run 2 |
| --- | ---: | ---: | ---: | ---: |
| Indexer | 165.484 us | 165.071 us | 165.131 us | 164.872 us |
| Sparse Attention fused | 258.277 us | 258.722 us | 353.980 us | 353.490 us |
| Combine | 11.996 us | 11.797 us | 11.962 us | 11.781 us |
| DSA workflow | 435.757 us | 435.590 us | 531.073 us | 530.143 us |

The 32-bit form therefore regressed the fused kernel by 36.6%-37.1% and the
workflow by 21.7%-21.9%, with stable Indexer, Combine, and 1650 MHz device
frequency. This rejects adjacent-value grouped loads at both tested widths.
The extra register alone is not sufficient to explain a roughly 37% fused
regression; generated-instruction or detailed memory-pipeline evidence would
be required before revisiting this load/store mapping. The 32-bit kernel and
its source-contract test were reverted.

The first 32-bit restore check was invalidated before comparison: reinstalling
the retained editable package was insufficient because CannBench's SIMT loader
prepends the active release's operator-local `simt/v2` directory to `sys.path`,
even when `CANNBENCH_SKIP_SIMT_INSTALL=1`. Launching CannBench from the
candidate release therefore loaded the candidate extension again. The profile
made this visible: its dumped `.aicore_binary` SHA-256 remained the candidate
hash `29b2e09f1cdb1c322ca44fc9abfb47ea6f8fb9cf642b0b45949aafc869fac81d`.
This run is deployment diagnostics and is excluded from performance evidence.

Launching the same CannBench command from the retained release selected the
intended scalar extension. The dumped `.aicore_binary` SHA-256 returned to the
historical baseline hash
`e1224bb528dc902cd68374c2729ba47528725a4a034f96e4fa4e86315e0784de`;
fused measured 258.255 us, Combine 11.868 us, and the workflow 435.321 us.
This confirms both the source attribution and the restored target state.

For the earlier 64-bit candidate, reinstalling the retained `c3f53b6...`
source restored the fused kernel to 258.070 us, Sparse Attention plus Combine
to 269.721 us, and the complete workflow to 434.990 us. That same-node A/B
restore independently confirms its candidate attribution.

Artifacts are preserved under:

```text
/tmp/cannbench-dsa-v2-value-pack-a5-u64-controller-vDk0gg/
/root/cannbench-dsa-v2-value-pack-a5-u64-vDk0gg/
/root/cannbench-dsa-v2-value-pack-a5-u64-vDk0gg-results.tar.gz
/root/cannbench-dsa-v2-value-pack-a5-baseline-restore/
/tmp/cannbench-dsa-v2-value-pack-a5-u32-controller-bEueTB/
/root/cannbench-dsa-v2-value-pack-a5-u32-bEueTB/
/root/cannbench-dsa-v2-value-pack-a5-u32-bEueTB-results.tar.gz
/root/cannbench-dsa-v2-value-pack-a5-u32-baseline-restore2/
/root/cannbench-dsa-v2-value-pack-a5-u32-baseline-restore2.tar.gz
```

The 64-bit result archive SHA-256 is
`68bdffc53b65099cc5ba17d07038dac2c8eed8d0a5f03f7fe4653d96344cfe16`;
the 32-bit result archive SHA-256 is
`13e1e0c4ada438fe5a3b801763fbbf5ac8598bf0b2a2f5a3b0075a85d1b7d416`,
and the 32-bit baseline-restore archive SHA-256 is
`22fc8b3f4727990d24153a88fd4dddde3160df359bc555fa25f009f3b533893e`.

### A6: Canonical Value-Pack Tile Transpose

The A5 controls reject wider adjacent GM loads but do not reject changing the
UB destination layout. Both grouped candidates kept the direct SIMT scatter
from coalesced GM dimensions into ZN, where adjacent lanes write with a
16-element stride inside each 16-by-16 block. A6 isolates that store-layout
hypothesis: gather BF16 values into row-major 16-by-16 UB tiles, then use the
public C API `asc_transpose` instruction to generate the existing ZN buffer.

The staging layout is tile-major, not a single row-major 32-by-128 matrix:

```text
tile = (row / 16) * 8 + dim / 16
staging[tile * 256 + (row % 16) * 16 + dim % 16]
```

This distinction is required because `asc_transpose` accepts contiguous
16-by-16 source and destination blocks and has no source-stride argument. The
fast SIMT VF retains one warp per selected row and scalar BF16 GM loads. Within
each half warp, the UB stores for a dimension block are contiguous. After the
VF completes, 2 row blocks times 8 dimension blocks issue 16 transposes:

```text
src = staging + tile * 256
dst = packed + (row / 16) * 16 * 128 + (dim / 16) * 256
```

The transpose maps source `(row_in, dim_in)` to destination
`dim_in * 16 + row_in`, exactly matching the retained ZN offset. The existing
Tensor API ZN-to-ZN UB-to-L1 copy and both L1 gather slots remain unchanged.
One 8,192-byte staging region is appended to the fused UB workspace, increasing
the statically visible workspace from 147,968 to 156,160 bytes.

CANN 9.2.0 Tensor API UB-to-L1 routing was inspected before implementation. It
supports only ND-to-ND, DN-to-DN, NZ-to-NZ, and ZN-to-ZN routes, so a direct
Tensor API ND-to-ZN conversion is not available. Direct per-row GM-to-L1 copies
are also not selected: the rows are gathered by arbitrary indices, and issuing
32 small strided copies per Value Pack would replace one measured problem with
copy-launch and synchronization overhead. A6 uses only the already included
public `c_api/asc_simd.h`, Tensor API, and SIMT API surface; it adds no Basic API
dependency.

The first correctness implementation uses explicit synchronization between the
SIMT producer, vector transposes, and the existing MTE3 copy. Synchronization
may only be reduced after a separate real-device correctness experiment proves
the producer/consumer ordering. The specialization remains behind
`head64_fused_is_canonical_v32_decode`; generic decode, prefill, tails, and V1
keep their existing paths.

Retain A6 only when the same A5 gates hold: the production `-O3`, `dav-3510`
build reports zero Stack and acceptable register use; five full canonical V3.2
decode accuracy processes pass at `atol=rtol=0.05`; and two clean-process
CannBench `BasicInfo` runs improve the fused kernel by at least 5% and the
complete workflow by at least 3%. Preserve build, resource, accuracy, profile,
and rejected-candidate artifacts even when the source is reverted.

A6 passed every retention gate on Ascend 950PR with CANN 9.2.0. The production
source SHA-256 was
`e7876297e29fd6c0ff3131077255132af6795901a71226d8a74763e56eafc3f5`.
The `--cce-res-usage` diagnostic build reported 13 registers per thread and
zero Stack bytes for `head64_fused_value_pack_fast_vf`, matching the retained
scalar path's 13 registers and zero Stack. Five independent full V3.2 decode
accuracy processes passed with zero output and LSE mismatches at
`atol=rtol=0.05`. Every process reported maximum absolute errors of
0.009765625 for output and 0.009282112 for LSE, including negative,
out-of-range, and causal-future indices.

Two clean-process CannBench `BasicInfo` runs at a stable 1650 MHz reported:

| Boundary | Retained run 1 | Retained run 2 | A6 run 1 | A6 run 2 |
| --- | ---: | ---: | ---: | ---: |
| Indexer | 165.484 us | 165.071 us | 165.386 us | 165.204 us |
| Sparse Attention fused | 258.277 us | 258.722 us | 177.510 us | 178.439 us |
| Combine | 11.996 us | 11.797 us | 11.989 us | 12.232 us |
| DSA workflow | 435.757 us | 435.590 us | 354.955 us | 356.116 us |

The fused kernel improved by 31.27% and 31.03%; the complete workflow improved
by 18.54% and 18.25%. Indexer stayed within 0.3 us of the retained runs, and
Combine stayed within 0.5 us, isolating the gain to Sparse Attention. Both
profiles dumped the same fused `.aicore_binary` SHA-256,
`a62f73f25521d9b4788874c171e838fe70335395cc5dd97bf7b55faaa7ea05d6`,
which differs from the retained scalar binary and proves candidate provenance.
The Cast helper measured 3.466 and 3.724 us but remains outside the published
Sparse Attention boundary under the existing contract.

The implementation and all evidence are retained. Raw source, production and
resource builds, five accuracy outputs, and both complete profiles are under:

```text
/tmp/cannbench-dsa-v2-value-pack-a6-controller-i6q7rF/
/root/cannbench-dsa-v2-value-pack-a6-i6q7rF/
/root/cannbench-dsa-v2-value-pack-a6-i6q7rF-results.tar.gz
```

The local and remote result-archive SHA-256 is
`d77e242782a7d50d2964c295dd7c525352e55a6ae044c4ea34927e2e0bb6b18c`.

### A7: Canonical Value-Pack Synchronization Contraction

A7 isolates one synchronization question in the retained A6 path. The fast
Value Pack currently performs a V-to-MTE3 notify/wait immediately after the
SIMT VF, issues sixteen `asc_transpose` operations, and then performs another
V-to-MTE3 notify/wait before the Tensor API UB-to-L1 copy. The first pair
targets MTE3 even though the next consumer is another Vector-side operation.

The candidate removes only that first pair. It relies on issue ordering between
the Vector-side `asc_vf_call<head64_fused_value_pack_fast_vf>` producer and the
following Vector-side `asc_transpose` consumers. The second pair remains
immediately after all transposes and before `Copy(copy_ub_to_l1, ...)`, because
that boundary makes the packed UB result visible to the MTE3/Tensor API
consumer. This is a synchronization contraction, not a pipeline redesign.

P4, the 1024-thread/32-warp launch, ValueTile 128, both L1 gather slots, the
row-major staging layout, all sixteen transposes, and the ZN UB-to-L1 copy stay
unchanged. The specialization remains canonical-decode-only; generic decode,
prefill, tails, and V1 retain their existing paths. No API dependency is added.

The correctness risk is that `asc_vf_call` completion is not implicitly ordered
with subsequent `asc_transpose` reads despite both being issued on the Vector
side. Such a visibility failure may be timing-dependent, so one successful
launch is insufficient evidence. The candidate must pass five independent
full V3.2 decode accuracy processes with zero output and LSE mismatches at
`atol=rtol=0.05`, including invalid and causal-future indices. The production
`-O3`, `dav-3510` build must retain zero Stack and acceptable fast-VF register
use.

Performance retention requires two clean-process CannBench `BasicInfo` runs.
Each complete workflow must improve by at least 3% over the corresponding A6
workflow result of 354.955 and 356.116 us, with stable Indexer and Combine
latencies and attributable candidate binaries. If either workflow misses that
gate, or any correctness/resource gate fails, restore the first synchronization
pair and its source-contract assertion. Preserve the negative result and all
raw artifacts so this synchronization boundary is not retested without new
evidence.

A7 was rejected after target-device validation. The candidate source SHA-256
was `8f178aeb7d705ccb3fad9b7929f6cb56c31386b1c3892bb9943d5d2fc54273eb`.
Its CANN 9.2.0 production `-O3`, `dav-3510` build passed, and the
`--cce-res-usage` build reported 13 registers per thread and zero Stack bytes
for `head64_fused_value_pack_fast_vf`, identical to retained A6.

Five independent full V3.2 decode accuracy processes passed at
`atol=rtol=0.05`. Every process reported zero output and LSE mismatches,
including negative, out-of-range, and causal-future indices. Maximum absolute
errors remained 0.009765625 for output and 0.009282112 for LSE.

Two clean-process CannBench `BasicInfo` workflows at 1650 MHz reported:

| Boundary | A6 run 1 | A6 run 2 | A7 run 1 | A7 run 2 |
| --- | ---: | ---: | ---: | ---: |
| Indexer | 165.386 us | 165.204 us | 165.520 us | 165.347 us |
| Sparse Attention fused | 177.510 us | 178.439 us | 176.542 us | 176.553 us |
| Combine | 11.989 us | 12.232 us | 12.257 us | 11.860 us |
| DSA workflow | 354.955 us | 356.116 us | 354.319 us | 353.760 us |

The fused boundary improved by only 0.55% and 1.06%, while the complete
workflow improved by only 0.18% and 0.66%. Indexer and Combine stayed within
normal variation. After measurement, the retention threshold was relaxed from
3% to a stable 1% complete-workflow gain; A7 still failed because neither run
reached 1%. Both candidate profiles dumped the same fused
`.aicore_binary` SHA-256,
`c6661bb4977dcea9f2ba290d6d92c68b937ee1759967a56e0f862cb0fb0f35e3`,
which differs from A6 and confirms candidate provenance. The data supports
Vector-side producer/transpose ordering for this tested path, but the removed
V-to-MTE3 round trip is not a material latency source. The first synchronization
pair and its source-contract assertion were restored.

Raw source, build logs, resource output, five accuracy results, two complete
profiles, and profiler dumps are preserved under:

```text
/tmp/cannbench-dsa-v2-a7-sync-controller-TpvnnB/
/root/cannbench-dsa-v2-a7-sync-TpvnnB/
/root/cannbench-dsa-v2-a7-sync-TpvnnB-results.tar.gz
```

The local and remote result-archive SHA-256 is
`0f880c765e1957d92216cc2e4b288e83bf94eee410ab9d4bb455e20bb16e760a`.

### A8: Post-A6 Phase-Local L1 Tiling

A6 changed the Sparse Attention instruction mix enough that the pre-A6
InstrTimeline can no longer select the next optimization. Before another source
change, collect separate CannBench `Default` and `InstrTimeline` workflows from
the retained A6 package. Attribute the 40 QK pack/MMAD rounds, 32 Value
pack/PV/update rounds, Softmax work, and exposed cross-pipeline waits. Select
the first larger-tile candidate from that new critical path rather than from
the old 111.647 us Value Pack total.

The post-A6 collection completed on Ascend 950PR with CANN 9.2.0,
`msopprof 26.2.0`, production `-O3`, `dav-3510`, and a stable 1650 MHz device
frequency. The deployed fused source SHA-256 was
`e7876297e29fd6c0ff3131077255132af6795901a71226d8a74763e56eafc3f5`,
which matches retained A6. CannBench ran `Default` and `InstrTimeline` in
separate clean processes through `bench --aic-metrics`; no direct profiler
command or custom profiler option was used. `Default` reported:

| Boundary | Post-A6 Default |
| --- | ---: |
| Indexer Score | 82.743 us |
| Indexer Top-K | 82.031 us |
| Sparse Attention fused | 177.387 us |
| Combine | 12.246 us |
| Selected workflow kernels | 354.407 us |

The Sparse Attention profile also captured the dependency-materialization
Indexer launches and a 4.223 us Cast helper. They remain visible in the raw
rows but are not double-counted in the selected workflow boundary.
`InstrTimeline` reproduced fused Sparse Attention at 177.373 us and Combine at
11.793 us, so the detailed run retained the same kernel-time boundary.

The post-A6 binary symbols and stable call markers identify eight selected
tiles on each of the 12 recorded AIV lanes. Per lane, PC `0x120053e00ee4`
starts the first Key Pack of each tile, `0x120053e01098` starts Softmax,
`0x120053e01180` starts the first Value Pack, and `0x120053e00be4` is the
final output write after the eighth tile. These markers give the following
critical-path segmentation without summing work across concurrently executing
lanes:

| Eight-tile segment | Mean per lane | Minimum | Maximum |
| --- | ---: | ---: | ---: |
| First Key Pack to Softmax | 69.685 us | 65.125 us | 73.019 us |
| Softmax to first Value Pack | 18.721 us | 17.837 us | 19.146 us |
| First Value Pack to next tile/output | 64.447 us | 61.670 us | 67.769 us |
| All eight tile bodies | 152.853 us | 145.686 us | 159.218 us |

A6 reduced the Value Pack SIMT calls themselves to about 20 us per lane: the
eight first-pack calls at PC `0x120053e01180` average about 5 us in total, and
the 24 pipelined calls at PC `0x120053e01524` average about 15 us. The 64
16-by-16 transposes and total PV arithmetic still scale with the 512 output
dimensions, so Value256 cannot claim half of the 64.447 us Value phase as
removable work. QK256 likewise keeps the total 576-dimension MMAD and packing
work; its upside comes from removing two round-level copy, issue, and handshake
sequences per selected tile rather than removing 40% of QK arithmetic.

The profiler warned that some dynamic instrumentation was lost because the
kernel exceeded 108 sub-blocks. Individual long `VTRANSPOSE`, wait, and MMAD
durations therefore cannot be summed reliably. The exact eight-per-lane phase
markers above are retained for candidate selection, while source round counts
and complete-workflow timing remain the acceptance evidence.

QK256 is selected as the first A8 experiment. Its measured QK-to-Softmax
critical segment is consistently larger than the post-A6 Value-to-next-tile
segment, and its 164,352-byte source-visible UB estimate leaves more capacity
margin than Value256's 188,928 bytes. The full 69.685 us QK segment is only a
loose absolute ceiling because data movement and MMAD work remain; the actual
candidate must save at least 3.544 us from the 354.407 us selected workflow to
meet the 1% retention gate. If QK256 misses that gate, its result must not be
used to infer Value256 behavior.

Complete positive, interrupted, and duplicate-install-failure artifacts are
preserved under:

```text
/tmp/cannbench-dsa-v2-a8-profile-controller-7geyqF/
/root/cannbench-dsa-v2-a8-profile-7geyqF/ # Ascend 950PR host
```

The local and remote post-A6 profile archive SHA-256 is
`73c413015d6a1e1be1af9a1cbfc053453794580b04546756ef2e4ef5a7fa5f74`.

The current fused L1 layout reserves all QK and PV buffers simultaneously:

```text
Query 73,728 + Key slots 32,768 + Scores 16,384
+ Probabilities 8,192 + Value slots 32,768 + PV 32,768
= 196,608 bytes
```

QK and PV are sequential within each selected-token tile, but Query is packed
only once per logical task and is reused by all eight selected-token tiles.
Only Key and Score storage becomes dead after each QK phase. Probability stays
live through PV, and Query must remain live until the whole task completes.
A8 therefore reuses only the temporary L1 region after the persistent
73,728-byte Query. This is buffer lifetime reuse, not removal of the existing
QK-to-Softmax-to-PV synchronization protocol.

Two tile candidates are evaluated independently:

- Value256 retains QK128 and both Value gather slots. It reduces four Value
  rounds to two. With phase-local L1 ownership, the QK and PV maxima are
  122,880 and 212,992 bytes rather than the cumulative 262,144-byte layout.
  Its source-visible UB estimate rises from 156,160 to 188,928 bytes. The
  platform exposes 256 KiB of combined hardware resource, with a minimum
  32 KiB DCache allocation and 8 KiB reserved by the system, so this remains
  below the 216 KiB UB limit.
- QK256 retains Value128 and both Key gather slots. It reduces five QK rounds
  to three. Its QK and PV L1 estimates are 155,648 and 147,456 bytes after
  preserving Query, and its source-visible UB estimate is 164,352 bytes. It is
  the lower-capacity-risk alternative, but it is selected first only if the
  post-A6 profile shows more removable QK work.

Both candidates remain canonical V3.2 BF16 decode specializations. Generic
decode, prefill, tails, V1, P4, the 1024-thread/32-warp launch, two gather-slot
ownership, A6 transpose layout, and final visibility fences keep their current
semantics. Key Pack does not receive an A6-style transpose: its current
half-warp UB destination is already contiguous, unlike the old Value Pack NZ
scatter.

Each candidate is a separate build and device experiment. Retention requires
zero Stack in changed VFs, acceptable register use, five independent full
accuracy processes with zero output/LSE mismatches at `atol=rtol=0.05`, and
two clean-process `BasicInfo` workflows that each improve the complete workflow
by at least 1%. If neither tile has enough measured upside or fits the resource
gate, return to the isolated distributed Context-shard histogram microbenchmark
with a reset complete-chain target below 75-78 us; the old 105 us T0 gate is
obsolete after T1 reduced Top-K to about 82 us.

The first QK256 implementation exposed a flaw in the initial lifetime model
before performance measurement. Its production and `--cce-res-usage` builds
succeeded; Key Pack prepare used 14 registers, Key Pack fast used 13, and both
reported zero Stack bytes. The first full accuracy process produced an all-zero
output, 247,842 output mismatches, and 512 `-inf` LSE mismatches. The candidate
had placed PV Probability at L1 offset zero, overwriting Query after the first
selected-token tile even though seven later tiles still consumed it. QK256 was
reverted without workflow timing. This is not evidence against a corrected
QK256 schedule.

The first Value256 build caught two stale Key-slot offset names at compile
time. After that mechanical fix, the same incorrect L1 lifetime model reached
the device and produced 15,056 output mismatches plus 512 LSE mismatches. A
discriminating build reduced source-visible UB from 188,928 to 131,584 bytes
without changing the Value schedule; it reproduced exactly the same mismatch
counts and maximum errors. Together with the 216 KiB UB limit, this rules out
UB capacity as the cause. The persistent Query overwrite was the root cause.

The retained Value256 implementation keeps Query at L1 offset zero for the
entire logical task. QK Key/Score and PV Probability/Value/output buffers reuse
only the region after Query. Canonical Value Pack stages 32 row-major 16-by-16
tiles, issues 32 transposes, and processes Value dimensions in two 256-wide
rounds. Generic decode and prefill remain on Value128. The retained source
SHA-256 is
`8f154a2006fde85554bb66cd07070a6caf0b26366b467ed7009ee2c88b7e84fb`,
and the production fused binary SHA-256 is
`f5887c7aae8b29a7017497c018c219b6382db8cd38bbae7dee652146750af8d5`.

The production and resource builds completed with CANN 9.2.0, `dav-3510`, and
`-O3`. Value Pack fast remained at 13 registers and zero Stack bytes; Key Pack
prepare/fast remained at 14/14 registers and zero Stack bytes. Five independent
full V3.2 decode accuracy processes passed at `atol=rtol=0.05`. Every process
reported zero output and LSE mismatches, including invalid, out-of-range, and
causal-future indices. Maximum absolute errors remained 0.009765625 for output
and 0.009282112 for LSE.

Two clean-process CannBench `BasicInfo` workflows selected only the canonical
V3.2 case and ran at 1650 MHz:

| Boundary | A6 run 1 | A6 run 2 | Value256 run 1 | Value256 run 2 |
| --- | ---: | ---: | ---: | ---: |
| Indexer | 165.386 us | 165.204 us | 165.187 us | 165.186 us |
| Sparse Attention fused | 177.510 us | 178.439 us | 169.088 us | 167.833 us |
| Combine | 11.989 us | 12.232 us | 11.626 us | 12.067 us |
| DSA workflow | 354.955 us | 356.116 us | 345.901 us | 345.086 us |

The fused boundary improved by 4.74% and 5.94%; the complete workflow improved
by 2.55% and 3.10%. Indexer and Combine remained stable, and both workflow runs
exceeded the 1% retention gate. An earlier diagnostic command accidentally
expanded all three realistic decode workflows; its complete V3.2 prefix
measured 344.875 us but is excluded from the two formal retention runs.

Positive and negative Value256 artifacts are preserved under:

```text
/tmp/cannbench-dsa-v2-a8-value256d-controller-2oQO33/
/root/cannbench-dsa-v2-a8-value256d-2oQO33/ # Ascend 950PR host
```

The local and remote Value256 evidence archive SHA-256 is
`8e109ce25411215fa031d4912013f2fa508d085e85e14f7001dcf1943fac1251`.
The QK256 negative-result archive is under
`/tmp/cannbench-dsa-v2-a8-qk256-controller-pKmiNQ/` with SHA-256
`320b8f025bac5796c0d65f9e838baf118bab45c188ba4a599fa77b4c98939d7f`.

The corrected canonical QK256 schedule is retained on top of Value256. Generic
decode and prefill remain QK128/Value128. Canonical decode executes QK rounds
of 256, 256, and 64 dimensions, while Key Pack receives the tail-safe
`current_k`. Query stays live at L1 offset zero for all eight selected-token
tiles; PV Probability begins after the 73,728-byte persistent Query region.
P4, 1024 threads, 32 active warps, two gather slots, the current synchronization
protocol, and transposed Value packing are unchanged.

The corrected layout has the following source-visible resource maxima:

```text
QK L1: 155,648 bytes
PV L1: 212,992 bytes
UB:    197,120 bytes
```

UB therefore retains 24,064 bytes of headroom under the tested 216 KiB limit.
The production and resource builds used CANN 9.2.0, `dav-3510`, `-O3`, and
1650 MHz on Ascend 950PR. Key Pack prepare used 14 registers, Key Pack fast
used 13, and Value Pack fast used 13; all three reported zero Stack bytes. The
candidate source SHA-256 is
`5a932044f91db7d85102865585ce81eb4f54f05b9e948c0e42b21277055759ef`,
and the production fused binary SHA-256 is
`fce0e96ab9dcbf483a9750b656b590440abe9d49596907f81331c778077ff039`.

Five independent full V3.2 decode accuracy processes with seeds 7 through 11
passed at `atol=rtol=0.05`. Every process reported zero output and LSE
mismatches while covering negative, out-of-range, valid-past, and
causal-future indices. Output maximum absolute error ranged from 0.0078125 to
0.009765625; LSE maximum absolute error ranged from 0.00924 to 0.00940.

Two clean-process CannBench `BasicInfo` workflows selected only the canonical
V3.2 case and ran at 1650 MHz:

| Boundary | Value256 run 1 | Value256 run 2 | QK256 run 1 | QK256 run 2 |
| --- | ---: | ---: | ---: | ---: |
| Indexer | 165.187 us | 165.186 us | 165.274 us | 165.355 us |
| Sparse Attention fused | 169.088 us | 167.833 us | 157.565 us | 158.414 us |
| Combine | 11.626 us | 12.067 us | 11.782 us | 11.862 us |
| DSA workflow | 345.901 us | 345.086 us | 334.621 us | 335.631 us |

The fused boundary improved by 6.82% and 5.61%; the complete workflow improved
by 3.26% and 2.74%. Both runs exceed the 1% retention gate, while Indexer and
Combine remain stable. Generated benchmark records report 335.345 and
335.932 us because they use the separately profiled Indexer launch. The design
and published-data convention instead uses the dependency Indexer launches
captured inside the Sparse Attention workflow profile, producing the selected
334.621 and 335.631 us sums above.

The complete QK256 evidence is preserved under:

```text
/tmp/cannbench-dsa-v2-a9-qk256-controller-5uBRXP/
/root/cannbench-dsa-v2-a9-qk256-5uBRXP/ # Ascend 950PR host
```

The local and remote evidence archive SHA-256 is
`15241f23981977cb306deb6d752aeb6cec180e61b4c031f9915d5e63b36f940b`.

QK256 is an incremental checkpoint toward the 0.90x target, not a reason to
stop near 2x latency. The next candidate should use the isolated Context-shard
distributed histogram chain with the post-T1 75-78 us gate. Exhausting these
local tiling candidates without approaching the target requires reopening
fusion and ownership boundaries rather than declaring the series complete.

### W0: Workflow-Level Cleanup

Keep dependency materialization and helper launches visible in raw profiles.
The 3.7-4.2 us `Cast_*` helper visible immediately before the selected Sparse
Attention kernel is benchmark input preparation, not an Indexer-to-Attention
conversion or an Attention output conversion. The canonical Indexer already
returns `int32` indices. The canonical Head64 path receives BF16 Query/shared
KV, allocates BF16 raw output, and returns it directly because its dtype already
matches Query.

The helper appears because the canonical shared-KV tensor has 37,748,736
elements, below the current 67,108,864-element direct-device threshold. The
deterministic materializer therefore supplies an `array('f')` FP32 host buffer
to `torch.tensor(..., device=npu, dtype=bfloat16)`, and the NPU runtime performs
the FP32-to-BF16 conversion. `msopprof` observes process-wide setup launches,
even though the operator callable is prepared outside the selected-kernel
timing boundary. Real decode normally consumes already-resident BF16 Query and
KV tensors, so this Cast is not expected once per decode token. Keep it visible
in raw evidence, but do not add it to or claim its removal from the published
workflow boundary without an intentional contract change.

Consider score-to-histogram fusion only after S0-S2 and T0 establish that
avoiding the BF16 score workspace is worth the added ownership and
synchronization complexity.

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

This optimization series is complete when the workflow reaches at least 0.90x
of BF16 vLLM-Ascend performance under the same boundary. Equivalently, SIMT
latency must be no greater than `vLLM latency / 0.90`; with the current
169.797 us reference, the limit is 188.663 us. Reaching 2x, or exhausting the
current local backlog, does not satisfy completion. If the backlog is
exhausted first, use the preserved profiles to define the next architectural
or fusion stage.

No individual item is complete until real-device correctness, repeated
performance, a same-stack baseline rerun, and raw artifacts are available.
