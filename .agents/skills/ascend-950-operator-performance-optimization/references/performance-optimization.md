# Performance Optimization

## Contents

- [Optimize The Algorithm Before The Schedule](#optimize-the-algorithm-before-the-schedule)
- [Split Workloads By Behavior](#split-workloads-by-behavior)
- [Find The Dominant Resource](#find-the-dominant-resource)
- [Choose Fusion Boundaries Deliberately](#choose-fusion-boundaries-deliberately)
- [Tune Launch Geometry And Reductions](#tune-launch-geometry-and-reductions)
- [Treat Layout As Part Of The Algorithm](#treat-layout-as-part-of-the-algorithm)
- [Treat Source-Level Scheduling As A Hypothesis](#treat-source-level-scheduling-as-a-hypothesis)
- [Preserve Comparison Fairness](#preserve-comparison-fairness)

## Optimize The Algorithm Before The Schedule

Estimate work, bytes, temporary storage, and launch count per output. If the
algorithm repeats work asymptotically, fix that before tuning instructions.

Warning signs include:

- repeatedly sorting a padded maximum-size buffer after each small input chunk
- rescanning an entire context for every query when reusable structure exists
- writing a large intermediate to GM only for the next kernel to reread it
- launching one tiny kernel per reduction stage or candidate group
- allocating and clearing workspace far larger than live data

For incremental Top-K, compare streaming selection, hierarchical local Top-K,
radix selection, and bounded merge strategies. Do not default to a full bitonic
sort of `current_topk + new_candidates` for every chunk without measuring its
total comparison count and padding cost.

The observed many-query case in
[observed-optimization-results.md](observed-optimization-results.md) is a useful
scale check: a repeated padded bitonic merge made Top-K and synchronization
dominate the workflow by orders of magnitude. Consult that reference before
spending time on launch geometry around an asymptotically poor boundary.

## Split Workloads By Behavior

Prefill and decode often require different paths:

- Prefill has many queries and exposes row/head parallelism, but can amplify
  repeated context scans, sorting, and workspace traffic.
- Decode has few queries and may need context sharding, persistent work, or
  launch reduction to use the device.

Also specialize only when evidence supports differences in dtype, head size,
context length, Top-K, layout, or tail behavior. Keep a general correct path as
the fallback and make dispatch thresholds measurable.

## Find The Dominant Resource

Use profiles and simple bounds to classify the path:

- memory-bound: bytes and latency dominate; improve coalescing, reuse, packing,
  and intermediate placement
- compute-bound: instruction or unit utilization dominates; improve algorithm,
  vector width, reduction tree, and instruction mix
- launch-bound: device work is small relative to launch and framework overhead;
  fuse compatible stages or batch independent work
- synchronization-bound: barriers, flags, or imbalance serialize progress;
  repartition work or shorten shared lifetimes

Do not infer the class from wall time alone.

## Choose Fusion Boundaries Deliberately

Fuse when it removes material GM traffic or launch overhead and the stages share
compatible tiling. Reject or limit fusion when it:

- exceeds UB/L1/L0 or DCache budgets
- increases register pressure enough to reduce useful occupancy
- forces incompatible cube/vector layouts
- introduces cross-core coordination
- duplicates expensive work across output units
- removes a stable correctness/debug boundary without measurable benefit

Test an unfused twin to separate algorithmic benefit from compiler side effects.

## Tune Launch Geometry And Reductions

- Map independent rows, heads, batches, or context shards before adding cores.
- Balance per-core work; avoid many idle blocks and one long tail block.
- Use hierarchical reductions with defined neutral values and stable accumulation.
- Minimize global atomics and cross-core merge stages.
- Verify that the launch form and actual block size match compiler assumptions.

Measure both main-kernel time and total operator time. A faster kernel can lose
overall if it adds packing, initialization, or merge launches.

Batch VF work at the natural ownership boundary. Repeatedly entering a VF and
performing cache publication for one row can dominate a mixed kernel even when
the physical AIC/AIV launch geometry is unchanged. Count dynamic VF calls and
DCCI or fence operations per physical worker, then compare per-row, multi-row,
and per-tile publication as controlled variants. Keep the visibility operation
at the first true cross-pipeline or cross-core consumer boundary; do not simply
delete it to make the microbenchmark faster.

For multi-stage launch fusion, compare both the sum of selected device rows and
one interval that contains launch gaps. If a device-wide barrier is permitted
by the operator's API boundary, every launched task must reach every barrier in
the same order, including tasks masked out of reducer work. Publish producer GM
writes with the platform-required cache/visibility sequence before the barrier.
The repository's API rules override an attractive historical implementation.

Do not equate legal resource occupancy with useful parallelism. In the recorded
950PR experiments, removing a spill was latency-neutral, 2048-thread variants
could pass register and Stack gates yet regress, and a 32-block by 512-thread
geometry beat 64 blocks by 256 threads for one fixed small workload. Use
resource analysis to reject impossible candidates, then benchmark the feasible
ones for the actual shape.

## Treat Layout As Part Of The Algorithm

For UB-resident stages, derive the address pattern for one emitted instruction,
not only the logical tensor layout. Check whether active lanes reach the same
bank-group/subbank at different addresses. Prefer a capacity-neutral producer
layout or inline format conversion when it removes repeated conflicts. Padding,
physical-order writes, packed accesses, and shuffle-based traversal can all cost
more than the conflict they remove.

Measure the conversion, transpose, padding, or remap inside the changed kernel
boundary. Recorded fused-workflow experiments include both a large win from
inline NZ-to-column-major output and several regressions from otherwise
plausible padding/remap schemes; see
[observed-optimization-results.md](observed-optimization-results.md).

## Treat Source-Level Scheduling As A Hypothesis

Loop unrolling, manual load grouping, and hand ordering can improve or regress
performance because they change registers, spills, instruction scheduling, and
compiler decisions. Compare generated artifacts or instruction timelines when
available, but use repeated device time as the acceptance metric.

Use a controlled matrix such as:

```text
traversal: global-stride | block-contiguous
launch: pure SIMT | mixed SIMD/SIMT
loop: compiler loop | pragma unroll | manual expansion
```

Keep work, data, outputs, warmup, and toolchain identical. Run the full matrix
more than once; do not conclude from one short microsecond-scale sample.

Apply the same rule to packed scalar types. `half2` or `bfloat16x2_t` helps only
when the generated path reduces useful loads, stores, conversions, or arithmetic
at a measured bottleneck. It can be neutral or slower when values are unpacked
immediately for FP32 math, when the original access was already conflict-free,
or when register lifetime and tail handling grow. Test each VF boundary
independently before propagating a packed type through an operator.

Prefer ownership-local reuse before shared metadata. A few lane-local values or
validity bits can survive from one reduction pass to the next without a shared
UB producer or block barrier. Shared caching is justified only when the avoided
work exceeds its synchronization and memory traffic on the measured path.

## Preserve Comparison Fairness

Match semantics, inputs, dtype, layout, workspace accounting, output materialization,
and synchronization. A baseline using a compressed dtype, a different phase
path, precomputed metadata, or excluded helper stages is not directly comparable
until the boundary is normalized and disclosed.

Accept an optimization only after correctness, repeated performance, and the
full supported case set remain valid.

For prior real-device outcomes and their exact applicability boundaries, use
[observed-optimization-results.md](observed-optimization-results.md). Never add
percentages from separate experiments to predict a combined workflow gain.
