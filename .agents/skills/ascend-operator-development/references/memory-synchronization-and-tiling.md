# Memory, Synchronization, And Tiling

## Contents

- [Make A Capacity Worksheet](#make-a-capacity-worksheet)
- [Validate Tiles Against Real Limits](#validate-tiles-against-real-limits)
- [Define Work Ownership](#define-work-ownership)
- [Choose Traversal By Measurement](#choose-traversal-by-measurement)
- [Align Copies And Handle Tails](#align-copies-and-handle-tails)
- [Design Pipeline Ownership](#design-pipeline-ownership)
- [Debug Memory And Synchronization Together](#debug-memory-and-synchronization-together)

## Make A Capacity Worksheet

Do not size a tile from the largest tensor alone. For every stage, calculate:

```text
live_bytes = sum(live_elements_i * sizeof(dtype_i))
required_bytes = live_bytes + scratch + alignment_padding + reserved_runtime
```

Track GM, L1, L0A, L0B, L0C, and UB separately. Include double buffers,
transpose or format-conversion scratch, reduction scratch, compiler-generated
temporaries, and any memory reserved for the launch model.

For SIMT or mixed kernels, budget dynamic UB and SIMT DCache together according
to the tested platform's rules. A source-level UB array is not the full resource
picture.

## Validate Tiles Against Real Limits

- Read limits from the installed SDK, compiler output, or target documentation.
- Calculate bytes after padding and physical layout conversion.
- Check each live stage, not only total tensor size.
- Use smaller tiles to discriminate a capacity error from an addressing error.
- Reinspect generated metadata when unrelated template instantiation changes a
  runtime resource error.

Treat a resource error code as a symptom. Confirm the effective resource
request before redesigning the algorithm.

## Define Work Ownership

Write the mapping explicitly:

```text
logical_task -> core/block -> thread -> vector lane -> output interval
```

Prove:

- every logical element is produced exactly once, unless a reduction protocol
  intentionally combines writers
- no task reads outside valid input or initialized padding
- tails cannot reach a fast path that assumes a full tile
- actual threads per launch agree with kernel assumptions and `__launch_bounds__`
- grid size follows useful independent work, not a fixed habit

Low core utilization can be correct for a tiny workload. Increasing the grid
without increasing independent work may add overhead or races.

## Choose Traversal By Measurement

Compare at least these layouts when applicable:

- global/grid-stride traversal
- block-contiguous traversal
- one output row or head per block
- multiple rows or heads per block

Reason about adjacent thread addresses, transaction alignment, loop trip count,
register pressure, and tail divergence. Then confirm on the target. A traversal
that looks less contiguous at the block level can still generate a better
instruction schedule or memory pattern.

## Align Copies And Handle Tails

- Separate logical length from padded transfer length.
- Use aligned bulk copies only for regions proven valid.
- Mask or scalar-handle the tail; do not read beyond allocation merely because
  the hardware transfer is aligned.
- Initialize padded values to the operation's neutral element, such as negative
  infinity for a maximum, only when those values may enter computation.
- Verify byte offsets after dtype conversion and layout packing.

## Design Pipeline Ownership

For each buffer, name its producer, consumer, lifetime, and reuse point. In a
mixed pipeline, specify whether UB, L1, or GM carries each boundary and whether
the next stage sees the required format.

Use the narrowest synchronization scope that proves correctness:

- instruction or copy ordering inside one stage
- kernel-local producer/consumer synchronization
- block/core-local barrier
- inter-core protocol only when partitioning cannot remove it

Do not use a flag or barrier as a substitute for unclear ownership. Extra
synchronization can hide a race while destroying overlap.

## Debug Memory And Synchronization Together

When a memory fault or hang appears:

1. Reduce to one task, one core, and the smallest failing tile.
2. Replace complex arithmetic with identifiable loads and stores.
3. Verify offsets and capacity independently.
4. Add the next pipeline stage without changing tile geometry.
5. Increase core count last.

This order separates address/capacity bugs from visibility and coordination bugs.
