# Performance Optimization

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

## Preserve Comparison Fairness

Match semantics, inputs, dtype, layout, workspace accounting, output materialization,
and synchronization. A baseline using a compressed dtype, a different phase
path, precomputed metadata, or excluded helper stages is not directly comparable
until the boundary is normalized and disclosed.

Accept an optimization only after correctness, repeated performance, and the
full supported case set remain valid.
