# Softmax V3 Five-Path Performance Experiments

## Goal

Evaluate five general Softmax V3 dispatch and kernel specializations on
Ascend 950PR. Retain only candidates with repeatable device gains and no
correctness or neighboring-path regressions. Record rejected candidates in
this document after restoring their source changes.

The experiments extend the CUDA/PyTorch dispatch investigation without copying
CUDA thresholds mechanically. Dispatch may depend on dtype, reduction width,
outer rows, generated iteration count, alignment, tile count, or measured
launch geometry. It must not depend on case id, model name, or a complete fixed
input shape.

## Repository Boundary

All implementation and source tests remain under:

```text
src/cannbench/operators/builtin/softmax/
```

This design document owns the experiment matrix and conclusions. Retained
measurements update the existing Softmax V3 published records without changing
their schema, run ids, order, accuracy fields, or unrelated cases.

No common backend, CLI, core, or result-layer change is permitted.

## Target And Measurement Contract

- Target: port 20002, Ascend 950PR `Ascend950PR_9589`, `dav-3510`.
- Toolchain: CANN 9.2.0, Bisheng 15.0.5, production `-O3`.
- Timing boundary: CannBench `msopprof BasicInfo`, selected Softmax V3 device
  launches aggregated per operator call.
- Frequency gate: current and rated frequency must both be 1650 MHz.
- Provenance: record source revision, source diff, imported extension path,
  clean extension hash, post-profiler hash, selected kernel names, launch count,
  and raw run root.
- Comparison: baseline and candidate use identical prepared inputs, dtype,
  warmup, profiler mode, and aggregation.
- Repetition: collect at least two fresh baseline/candidate pairs for retention.

The clean build artifact must be preserved before profiling because `msopprof`
has previously modified the extension in place.

## Retention And Rejection Rules

A candidate is retained only when all of the following hold:

1. Focused FP16 and FP32 boundary/tail correctness passes.
2. The complete canonical FP16 Softmax accuracy suite passes.
3. Each named primary shape improves in both paired collections, with at least
   3% median device-latency improvement or a clearly larger absolute saving.
4. Control shapes from unchanged neighboring dispatch ranges do not regress by
   more than 1% outside normal repeated-run noise.
5. The final combined branch repeats the gain after earlier retained candidates
   have been integrated.

If a candidate misses a gate, restore its implementation and source-test
changes. Add its exact source hypothesis, shapes, raw roots, measured result,
and rejection reason to this document, then commit only that experiment record.

For a retained candidate, commit its implementation, operator-local tests,
this document's result section, V3 implementation notes, and affected published
records as one independently reviewable change.

## Experiment 1: Exact Width 256

### Hypothesis

The existing width-256 UB pipeline still accepts runtime `dim_size` and retains
element-validity branches in fully unrolled loops. An exact-width entry can make
all eight lane iterations compile-time valid and use proven `int32_t` tile-local
indices while retaining wide GM offsets.

### Candidate

- Add an exact-256 VF or exact template mode in an isolated translation unit.
- Preserve the current two-slot MTE2/V/MTE3 protocol, 32 rows per tile, 1024
  threads, physical grid limit, FP32 accumulation, and actual DMA bytes.
- Dispatch every width-256 row to the exact path; widths 225-255 remain on the
  current capacity bucket.

### Matrix

- Primary: realistic width 256 with outer 131072.
- Synthetic outer controls: 1024, 16384, 65536, and 262144 at width 256.
- Dispatch controls: widths 224, 225, 255, and 257.
- Dtypes: FP16 and FP32 correctness; FP16 performance retention.

## Experiment 2: Large-Row Selector

### Hypothesis

The current selector sends all FP16 widths 8192-56320 to whole-row UB staging,
making the CUDA-style direct aligned/shifted fast paths unreachable. Whole-row
DMA and events may not dominate direct GM vector access uniformly across row
width, alignment, outer rows, and tile count.

### Candidates

Compare existing implementations without changing semantics:

- whole-row UB with 32768-element capacity;
- whole-row UB with 56320-element capacity;
- direct FP16 x4 for divisible-by-four widths;
- shifted x2 for even or row-shifted widths;
- scalar/ILP fallback for odd tails;
- physical grid limits 32 and 64 where the implementation uses a persistent
  grid-stride pipeline.

Derive a selector only if the winning regions form stable boundaries expressible
through width, row bytes, alignment, outer rows, or tile count. Do not dispatch
by published case identity.

### Matrix

- Published widths: 30522, 32005, 32128, 50005, 50265, and 50272.
- Boundary widths: 8191, 8192, 32768, 32769, 56320, and 56321.
- Outer rows: 1024, 4096, and 8192 where memory permits.
- Controls: one 2K-8K register-cache row and one width above 56320.

## Experiment 3: Width 33-64 Two-Dimensional Selector

### Hypothesis

The rejected UB pipeline improved a small-outer width-48 case but regressed a
large-outer width-49 case. A stable selector may exist in total rows or tiles per
physical core, rather than width alone.

### Candidate

Recreate the general 33-64 two-slot UB pipeline in an isolated experiment and
compare it with the current direct-GM persistent path. If results are monotonic,
dispatch by a derived outer-row or tile-count threshold. If no stable threshold
survives boundary tests, reject the path again.

### Matrix

- Widths: 33, 47, 48, 49, 63, and 64.
- Outer rows: 1024, 3072, 4096, 16384, 65536, 200704, and 262144.
- Published anchors: width 48 at outer 3072 and width 49 at outer 200704.
- Neighbor controls: widths 32 and 65.

## Experiment 4: Very-Short-Row Launch Policy

### Hypothesis

The generic persistent path uses 1024 threads for most shapes even when
`next_power_of_two(dim_size)` is small. CUDA varies warp batch and generated
work with the power-of-two class. On Ascend 950PR, fewer VF threads or a
different rows-per-warp policy may reduce scheduling and register cost without
adding a new data-movement pipeline.

### Candidates

- VF threads: 128, 256, 512, and 1024.
- Rows per warp: retain one and two as the only supported candidates; the prior
  four-row candidate remains rejected.
- Physical grid: derive from rows per block and cap at 32 or 64.

Select by `next_power_of_two(dim_size)` and outer rows only if the matrix shows
a stable region. Do not add an exact full-shape branch.

### Matrix

- Widths: 1, 9, 16, 31, and 32.
- Outer rows: 1024, 49152, 65536, and 442368.
- Primary published shape: width 9 at outer 49152.
- Controls: widths 33 and 48.

## Experiment 5: Width 129-224 Launch Policy

### Hypothesis

Width-197 with small outer rows is already near CANN Ops, while width-196 with
very large outer rows retains a gap. The remaining difference is more likely
tile scheduling, grid size, or event amortization than another capacity bucket.

### Candidates

- Physical grid limits: 32, 64, and 128.
- Rows per tile: 16 and 32 where UB capacity and generated resources permit.
- Preserve the existing 160 and 224 capacity buckets and event protocol unless
  the experiment changes only one named factor.

### Matrix

- Widths: 129, 144, 160, 161, 196, 197, 204, and 224.
- Outer rows: 1024, 16320, 262144, 401408, and 802816.
- Neighbor controls: widths 128, 225, and 256.

## Isolation And Integration

Each experiment runs in its own Git worktree and branch created from this spec
commit. It uses a unique remote build and profile root. Agents must not edit or
benchmark another experiment's worktree.

Completed experiment commits are integrated into the main experiment branch in
priority order. After each retained implementation is integrated, its primary
and controls are remeasured against the new combined baseline. Rejected-result
documentation commits can be integrated without source changes.

After all five experiments, run the complete repository test suite, canonical
Softmax FP16 accuracy, focused FP32 boundary coverage, and a final published
collection for every retained dispatch range.

## Results

### Experiment 1: Exact Width 256 - Retained

The exact-width candidate is retained. It adds a distinct width-256 VF and
kernel entry in `persistent_256.asc`; widths 225-255 continue to use the
capacity-256 entry. The exact VF has eight unconditionally valid compile-time
iterations and `int32_t` UB tile indices while GM and outer-row offsets remain
`int64_t`. DMA byte counts, two-slot event order, 32-row tiles, 1024 threads,
physical grid policy, and FP32 accumulation are unchanged.

The candidate passed FP16 and FP32 widths 224, 225, 255, 256, and 257 with 65
tail rows. Maximum absolute error was `9.54e-07` for FP16 and `1.49e-08` for
FP32; the canonical FP16 suite passed 40/40 cases.

Two fresh remote-locked standard CannBench BasicInfo pairs at 1650/1650 MHz
measured one selected kernel per call. CannBench passed no application warmup
argument; each log confirms the profiler default `Warm Up enabled. times:5`:

| Shape | Baseline R1 | Candidate R1 | Gain R1 | Baseline R2 | Candidate R2 | Gain R2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TrOCR `131072x256` | 118.053 us | 73.665 us | 37.60% | 118.038 us | 69.383 us | 41.22% |

Three valid control pairs retained their existing kernel families. CrossViT
width 197 had baseline/candidate medians `4.278/4.389 us`; its `0.111 us`
difference lies within the candidate's `4.268-4.456 us` spread. GPT-J width
128 had medians `3.907/3.914 us`; one valid `37.355 us` profiler outlier is
retained in its reported `3.852-37.355 us` candidate spread. The unchanged
paths do not show a repeatable candidate-specific regression.

The canonical TrOCR publication collection measured `69.735001 us`. Accepted
raw data is under `standard-final-o`, `standard-final-controls-extra`, and
`standard-final-publication` in the final remote Task 1 root. Full hashes and
raw CSV paths are recorded in the Task 1 report.
