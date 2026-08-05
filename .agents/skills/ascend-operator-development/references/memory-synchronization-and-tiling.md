# Memory, Synchronization, And Tiling

## Contents

- [Make A Capacity Worksheet](#make-a-capacity-worksheet)
- [Validate Tiles Against Real Limits](#validate-tiles-against-real-limits)
- [Define Work Ownership](#define-work-ownership)
- [Choose Traversal By Measurement](#choose-traversal-by-measurement)
- [Align Copies And Handle Tails](#align-copies-and-handle-tails)
- [Audit UB Bank Conflicts](#audit-ub-bank-conflicts)
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

### Account For Optional UB Reserved Regions

On the tested Ascend 950PR and CANN 9.2 toolchain, Bisheng leaves 216 KiB of UB
available by default. Compiling every ASC translation unit with both of these
options releases two compiler-reserved regions and raises the observed usable
limit to 224 KiB (229376 bytes):

```text
--cce-disable-vf-stack-reserved-ubuf
--cce-disable-asc-reserved-ubuf
```

Treat this as a toolchain-specific build contract, not a portable hardware
constant. Verify that the installed compiler accepts both options and validate
the resulting binary on the target device.

Do not size data arrays to the full 224 KiB. Apply the capacity worksheet to all
simultaneously live allocations:

```text
data_bytes <= 229376 - reduction_scratch - pipeline_state
              - alignment_padding - compiler_margin
```

For a double-buffered in-place tile, `data_bytes` is
`2 * tile_elements * sizeof(dtype)`. For separate input/output buffers, include
both arrays; for independently double-buffered input and output, include all
four slots. Keep the usable-UB constant, tile constants, and compile options in
sync, add compile-time capacity assertions where the language permits, and
recheck compiler resource metadata whenever any live UB allocation changes.

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

## Audit UB Bank Conflicts

Analyze one warp and one emitted memory instruction at a time. On the tested
950PR model, the useful reasoning unit was the bank-group/subbank address, not
just the bank number. Same-address reads can merge; different addresses mapped
to the same resource serialize.

For every suspicious loop:

1. Write the byte address reached by lane `l` for one unrolled instruction.
2. Reduce it to the documented bank-group/subbank mapping for the target.
3. Count different addresses per resource; do not count inactive lanes or
   separate instructions as one conflict.
4. Compare a capacity-neutral traversal/layout first.
5. Include transpose, padding, remap arithmetic, and changed store order in the
   measured boundary.

Do not assume that fewer predicted conflicts means lower latency. Recorded
experiments found a large gain when Fixpipe emitted a capacity-neutral
column-major producer layout, but several physical-order staging writes, a
shuffle-based reducer, and several padding schemes were neutral or slower. See
[observed-optimization-results.md](observed-optimization-results.md) before
choosing a remediation.

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

For a two-slot producer/consumer pipeline, use distinct ready and free state in
each direction. State initial ownership, the first reuse that must wait, and
whether the terminal release has a consumer. Reusing the same flag IDs for both
directions caused a recorded device hang. Validate one tile, two tiles, first
slot reuse, full tile count, and multi-query behavior in that order.

Double buffering is useful only when the launch gives each physical worker
multiple items and there is exposed copy/compute overlap. A two-slot source
array with one logical row per physical block is not a pipeline. Conversely,
for very short rows the event and VF overhead can exceed the saved direct-GM
traffic. Benchmark the actual trip count per block.

## Debug Memory And Synchronization Together

When a memory fault or hang appears:

1. Reduce to one task, one core, and the smallest failing tile.
2. Replace complex arithmetic with identifiable loads and stores.
3. Verify offsets and capacity independently.
4. Add the next pipeline stage without changing tile geometry.
5. Increase core count last.

This order separates address/capacity bugs from visibility and coordination bugs.
