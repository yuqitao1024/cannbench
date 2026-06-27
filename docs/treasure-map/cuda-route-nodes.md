# CUDA Treasure Map Route Nodes

## Scope

This document records the current CUDA treasure map content shown by the frontend so the generated background image and the frontend overlays can be maintained together.

## Main Route Order

1. `Profile the Truth`
2. `Guard Correctness`
3. `Shape Parallel Work`
4. `Cut Data Motion`
5. `Fix Global Access`
6. `Stage Through Shared`
7. `Tune Launch Geometry`
8. `Polish Instructions`

## Main Nodes

### Profile the Truth

- Guide sections: `4`, `9`
- Route IDs: `O1`, `O2`
- Summary: Measure device-side reality before changing kernel code.
- Details:
  - Use GPU-side timing and profiler traces to find the real hotspot.
  - Anchor optimization work to measured bottlenecks, not intuition.
  - Capture bandwidth and latency symptoms before tuning.

### Guard Correctness

- Guide sections: `7`
- Route IDs: `O5`
- Summary: Keep numerical intent and validation alongside performance work.
- Details:
  - Check optimized kernels against trusted reference outputs.
  - Track acceptable floating-point error bounds.
  - Do not trade away correctness by accident.

### Shape Parallel Work

- Guide sections: `5`, `6.1`, `6.3`
- Route IDs: `O3`, `O4`
- Summary: Expose enough parallel work before low-level tuning.
- Details:
  - Choose decomposition that gives threads enough independent work.
  - Prefer tuned libraries if the workload already matches them.
  - Avoid optimizing serial structure with CUDA-specific tricks.

### Cut Data Motion

- Guide sections: `10.1`
- Route IDs: `O6`
- Summary: Reduce host-device movement and redundant intermediate traffic.
- Details:
  - Batch transfers and keep intermediates on GPU when possible.
  - Treat transfer reduction as a first-class optimization target.
  - Only optimize copy overlap after reducing copy volume.

### Fix Global Access

- Guide sections: `10.2.1`
- Route IDs: `O9`, `O10`
- Summary: Repair coalescing, stride, and alignment before instruction micro-tuning.
- Details:
  - Make adjacent threads touch adjacent memory.
  - Rewrite strided and misaligned access patterns.
  - Use bandwidth waste as the trigger for layout changes.

### Stage Through Shared

- Guide sections: `10.2.3.2`, `10.2.3.3`
- Route IDs: `O12`
- Summary: Move into shared memory only when reuse or access repair justifies it.
- Details:
  - Tile reused inputs through shared memory.
  - Use shared staging to repair global memory access shape.
  - Treat shared memory as a bandwidth tool, not a default.

### Tune Launch Geometry

- Guide sections: `11.1`, `11.4`
- Route IDs: `O17`, `O18`
- Summary: Balance threads, blocks, occupancy, registers, and shared memory together.
- Details:
  - Tune block shape after memory behavior is understood.
  - Do not chase occupancy blindly.
  - Use resource balance to hide latency, not single-metric maximization.

### Polish Instructions

- Guide sections: `12.1`, `12.2`, `13.1`, `13.2`
- Route IDs: `O21`, `O22`, `O23`, `O24`, `O25`
- Summary: Apply arithmetic and control-flow micro-optimizations after memory and launch tuning.
- Details:
  - Reduce expensive arithmetic where accuracy allows it.
  - Use intrinsics and precision flags deliberately.
  - Cut divergence and use predication or unrolling when it helps.

## Branch Nodes

### Pinned / Async / Streams

- Branch from: `Cut Data Motion`
- Guide sections: `10.1.1`, `10.1.2`
- Route IDs: `O7`, `O19`
- Summary: Use overlap and pinned memory when transfers are unavoidable.
- Details:
  - Use pinned host memory for faster async copies.
  - Overlap copies with compute through streams.
  - Reserve this path for unavoidable transfer-heavy pipelines.

### L2 persistence

- Branch from: `Fix Global Access`
- Guide sections: `10.2.2`
- Route IDs: `O11`
- Summary: Treat hot memory regions as persistent only when reuse is predictable.
- Details:
  - Use access-policy windows for predictable reuse regions.
  - Apply only when the workload benefits from stable cache residency.

### Bank conflicts

- Branch from: `Stage Through Shared`
- Guide sections: `10.2.3.1`, `10.2.3.3`
- Route IDs: `O13`
- Summary: Pad and lay out shared tiles to avoid conflict serialization.
- Details:
  - Use padding in transpose-style tiles.
  - Validate that shared-memory speedups are not lost to conflicts.

### Async G2S copy

- Branch from: `Stage Through Shared`
- Guide sections: `10.2.3.4`
- Route IDs: `O14`, `O15`
- Summary: Pipeline global-to-shared movement when it helps throughput and pressure.
- Details:
  - Use async copy paths to reduce register pressure.
  - Overlap staging with useful work where the architecture supports it.

### Concurrent kernels

- Branch from: `Tune Launch Geometry`
- Guide sections: `11.5`
- Route IDs: `O19`
- Summary: Use independent streams only when the workload really permits overlap.
- Details:
  - Launch independent kernels concurrently when resources allow it.
  - Prefer this for multi-stage pipelines or independent work queues.

### Target GPU build

- Branch from: `Polish Instructions`
- Guide sections: `15-18`
- Route IDs: `O26`
- Summary: Make sure optimized kernels are built for the GPU targets that matter.
- Details:
  - Compile for relevant compute capabilities.
  - Check deployment/runtime compatibility before trusting the optimization result.

## Completeness Check

Current frontend route content contains:

- `8` main nodes
- `6` branch nodes
- total `14` displayed nodes

Based on the current route manifest, the image and the frontend node set are complete. No node is missing from the current visual route definition.

## Node To Optimization IDs

| Node | Optimization IDs |
| --- | --- |
| Profile the Truth | O1, O2 |
| Guard Correctness | O5 |
| Shape Parallel Work | O3, O4 |
| Cut Data Motion | O6 |
| Pinned / Async / Streams | O7, O19 |
| Fix Global Access | O9, O10 |
| L2 persistence | O11 |
| Stage Through Shared | O12 |
| Bank conflicts | O13 |
| Async G2S copy | O14, O15 |
| Tune Launch Geometry | O17, O18 |
| Concurrent kernels | O19 |
| Polish Instructions | O21, O22, O23, O24, O25 |
| Target GPU build | O26 |

## Optimization ID To Node

| Optimization ID | Node |
| --- | --- |
| O1 | Profile the Truth |
| O2 | Profile the Truth |
| O3 | Shape Parallel Work |
| O4 | Shape Parallel Work |
| O5 | Guard Correctness |
| O6 | Cut Data Motion |
| O7 | Pinned / Async / Streams |
| O9 | Fix Global Access |
| O10 | Fix Global Access |
| O11 | L2 persistence |
| O12 | Stage Through Shared |
| O13 | Bank conflicts |
| O14 | Async G2S copy |
| O15 | Async G2S copy |
| O17 | Tune Launch Geometry |
| O18 | Tune Launch Geometry |
| O19 | Pinned / Async / Streams; Concurrent kernels |
| O21 | Polish Instructions |
| O22 | Polish Instructions |
| O23 | Polish Instructions |
| O24 | Polish Instructions |
| O25 | Polish Instructions |
| O26 | Target GPU build |

## Notes On Non-Unique Mapping

- The mapping is intentionally not one-to-one.
- One displayed node can aggregate multiple optimization IDs.
- One optimization ID can appear in more than one displayed node if that concept is relevant in multiple route contexts.
- In the current route, `O19` appears in both:
  - `Pinned / Async / Streams`
  - `Concurrent kernels`
