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

### Experiment 2: Rejected

The experiment added temporary, operator-local compile-time controls for the
FP16 whole-row capacity (production/32768), direct implementation (automatic,
x4, shifted x2, or scalar), and persistent grid limit (32/64). The source test
first failed without these controls and then passed with production-preserving
defaults. The focused operator suite passed 39 tests, the complete operator
SIMT suite passed 41 tests, and the focused profile suite passed 22 tests.

All nine clean relinked extensions had distinct SHA-256 hashes. The hashes used
for screening were:

- production whole-row selector (32768/56320), grid 64:
  `63b55140de0ec494ac4cb2a1fbd55fb2eea6db231281a2fcd839fee0d5c02d3c`;
- whole-row 32768/grid 64:
  `eba1203dc64abd582a4ad28441137fe26e77dc1646505d0f995592109a2b7b2a`;
- direct x4/grid 64:
  `32125660ebe14a02dff5eecf8cfa35605e72477c46962321ddf1c030a719d2e0`;
- direct x2/grid 64:
  `5e75b132343a45ca3e51a218c81d2f319b4ef82d2bfac73a4d28f65be94908cb`;
- direct scalar/grid 64:
  `41bdb99a59cfc557df8ac793db5cbc6522a086f21f3b6224d70ffbf297ccc88a`.

The corresponding grid-32 variants were built and discriminated by hashes
`4b7221d2b8c067bf388aa82239a851ceb694a8904aea67ef11c5d57920783596`,
`8091c9beab688bdf37fbd3e11c9fc7e8cfa3461c210e07dc900aedc5d0212516`,
`40abee08a57c867659fac0ea016ee63d075322afdb47f6728076d9362956cc5b`,
and `ab832354ab9145fe680103abc7faacec7898bbd1a03f7ef68fa2f9a327c55d94`
for whole-row 32768, x4, x2, and scalar respectively. An earlier summary that
read stale inplace extensions after relinking was marked `INVALID` and was not
used; under remote root
`/root/cannbench-softmax-v3-large-selector-task2-20260812-b`, the authoritative
artifact list is `evidence/valid-artifact-hashes.log`.

The experiment source revision was
`f511cff5dc0667dd6b8bd0a4e563d8c90116e709`; its source diff SHA-256 was
`02bc0feda870f93d437b566a8c625cd7095a2bdf0075250babbc3602599f03ed`.
Each profile `run.log` records the imported inplace extension path and its hash
before launch. A read-only audit after all profiles, at
`2026-08-12T15:26:46+08:00`, found profile directories only for baseline,
whole32768_g64, direct_x4_g64, direct_x2_g64, and direct_scalar_g64. For each,
the currently loaded inplace copy still had the hash listed above and matched
the clean `build/lib` copy byte-for-byte, with unchanged size and mtime. This is
retrospective post-profile verification, not a contemporaneous per-run
post-profile capture. Grid-32 variants were not profiled and have no
post-profile claim. Profiles used `--launch-count=1`; raw BasicInfo rows select
the kernel, while `run.log` records the imported path
`<remote-root>/variants/<variant>/src/src/cannbench/operators/builtin/softmax/simt/v3/aten_softmax_v3/_C.cpython-311-x86_64-linux-gnu.so`.
The selected kernel names were
`_ZN15aten_softmax_v312_GLOBAL__N_150row_softmax_fast_large_row_inplace_pipeline_kernelIDhfLl32768EEEvPT_PKS2_ll`
or
`_ZN15aten_softmax_v312_GLOBAL__N_150row_softmax_fast_large_row_inplace_pipeline_kernelIDhfLl56320EEEvPT_PKS2_ll`
(block 64), and
`_ZN15aten_softmax_v312_GLOBAL__N_131row_softmax_fast_forward_kernelIDhfDhEEvPT1_PKT_llll`
(block 4096 for automatic direct and block 64 for forced direct).

Fresh-process correctness passed for the baseline (14 cases, including FP16
widths 8191, 8192, 32768, 32769, 56320, 56321 and FP32 boundaries) and for each
grid-64 candidate (12 FP16 boundary, published-width, and tail cases). A
previous multi-process run stalled during device initialization; a locked
single-case diagnostic loaded the intended baseline hash and completed the
8192-width reference/candidate comparison with `allclose=True`, so it was
treated as environmental contamination rather than candidate evidence.

Raw `msopprof BasicInfo` screening at 1650/1650 MHz showed that unchanged-path
controls below the proposed 32768 threshold remained within 0.30%, but switching
to the direct path was substantially slower:

| Outer x width | Existing whole-row (us) | Candidate | Candidate (us) | Change |
|---|---:|---|---:|---:|
| 4096 x 32769 | 527.904968 | automatic direct | 1226.911987 | +132.4% |
| 4096 x 50005 | 741.446045 | automatic direct | 1764.466919 | +138.0% |
| 4096 x 50265 | 748.242004 | automatic direct | 1767.493042 | +136.2% |
| 4096 x 50272 | 706.226013 | automatic direct | 1412.397095 | +100.0% |
| 4096 x 32128 | 490.988007 | forced x4 | 923.638000 | +88.1% |
| 4096 x 50272 | 706.226013 | forced x4 | 1381.910034 | +95.7% |
| 4096 x 30522 | 502.796997 | forced shifted x2 | 1083.557983 | +115.5% |
| 4096 x 32005 | 520.099976 | forced scalar | 1915.894043 | +268.4% |
| 4096 x 50005 | 741.446045 | forced scalar | 2902.718018 | +291.5% |
| 4096 x 50265 | 748.242004 | forced scalar | 2909.327148 | +288.8% |

The unchanged-path control pairs at widths 30522, 32005, 32128, and 32768
were `502.796997/504.262024`, `520.099976/520.739014`,
`490.988007/491.332977`, and `497.518005/497.579987` us (baseline/candidate).
Raw roots are
`/root/cannbench-softmax-v3-large-selector-task2-20260812-b/screen-r1`
and
`/root/cannbench-softmax-v3-large-selector-task2-20260812-b/forced-direct-r1`.

No direct mode produced a plausible winning region, including at the first
general width boundary. Therefore no selector could be derived. Grid-32
profiling, the wider outer-row matrix, canonical accuracy, and two paired
retention rounds were pruned under the rejection rule because grid-32 was not
plausible enough to profile after grid-64 lost by 88-292%; its result remains
unmeasured. All temporary source and source-test changes were restored; no
implementation, selector,
published record, or README change is retained.

### Experiment 3: Width 33-64 Two-Dimensional Selector - Rejected

The experiment recreated a general two-slot, 32-row MTE2/V/MTE3 UB pipeline
for FP16 and FP32 widths 33-64. Its temporary selector used only width and an
outer-row cap. It did not use case ids, model names, or complete fixed shapes.
All implementation, selector, link-order, and source-test changes were restored
after rejection.

Focused FP16/FP32 correctness passed 16/16 cases at widths 32, 33, 47, 48, 49,
63, 64, and 65. The canonical FP16 suite passed 40/40. The clean baseline and
candidate extension SHA-256 values were respectively
`cecd7362326058eeec4584f4b713297c28d5dd01b0a604644fb0e4415c34a6ca` and
`f9b5869d74feb3007ed8dbdcb7ab60d3292f6428f4297d2c1f250846dfb4f56d`.

After the device reboot, the clean binaries completed two standard CannBench
pairs at 1650/1650 MHz. No profiler warmup option was passed; every row used
the default `Warm Up enabled. times:5`.

| Round | Case | Width / outer rows | Baseline | Candidate | Candidate change |
| --- | --- | --- | ---: | ---: | ---: |
| R1 | `xcit_attention` | 48 / 3072 | 4.441000 us | 4.064000 us | 8.49% faster |
| R2 | `xcit_attention` | 48 / 3072 | 4.483000 us | 4.025000 us | 10.22% faster |
| R1 | `swin_window_attention` | 49 / 200704 | 70.728996 us | 80.675003 us | 14.06% slower |
| R2 | `swin_window_attention` | 49 / 200704 | 70.387001 us | 80.823997 us | 14.83% slower |
| R1 | width-8 control | 8 / 64 | 2.473000 us | 2.516000 us | 1.74% slower |
| R2 | width-8 control | 8 / 64 | 2.772000 us | 2.695000 us | 2.78% faster |
| R1 | width-128 control | 128 / 2048 | 3.875000 us | 4.075000 us | 5.16% slower |
| R2 | width-128 control | 128 / 2048 | 3.818000 us | 3.895000 us | 2.02% slower |

The rated-frequency evidence root is
`/root/cannbench-softmax-v3-task3-standard-ab-post-reboot-20260812`; its raw
CSV hash manifest has SHA-256
`73c2b55b1efafbe8c2fdb0f00368f159c7716d6dfcf472712989fa6e458e2bf7`.

The broad selector is rejected. It repeatably helps the small primary but
regresses the large primary by about 10 us and 14%-15%. The two measured outer
sizes do not establish a safe monotonic cutoff between 3072 and 200704, so no
narrower threshold is retained without inventing an unmeasured boundary.

### Experiment 4: Very-Short-Row Launch Policy - Rejected

The experiment compared three orthogonal changes against production's 1024
threads, two rows per warp, and grid limit 64 for power-of-two widths up to 32:

- 256 threads with the other settings unchanged;
- one row per warp with the other settings unchanged;
- grid limit 32 with the other settings unchanged.

Focused device correctness covered the baseline and all three candidates for
FP16 and FP32 widths 1, 9, 16, 31, 32, 33, and 48. All 56 comparisons passed
without NaN, Inf, or tolerance mismatch.

Standard CannBench measurements of `convbert_local_kernel` used no explicit
warmup option. Every accepted row reported 1650/1650 MHz and the profiler
default `Warm Up enabled. times:5`:

| Configuration | Duration | Change vs. 8.246 us baseline |
| --- | ---: | ---: |
| Production baseline | 8.246 us | 0.0% |
| 256 threads | 17.435 us | 111.4% slower |
| One row per warp | 10.101 us | 22.5% slower |
| Grid limit 32 | 13.088 us | 58.7% slower |

All candidates regress materially, so none is retained. The temporary policy
controls and source tests were removed. Raw evidence is preserved under
`/root/cannbench-softmax-v3-very-short-task4-20260812-a`, with the accepted
screen under `evidence/post-reboot-standard-screen-20260812-b`.

### Experiment 5: Width 129-224 Launch Policy - Rejected

The experiment compared 16 rows per tile, grid limit 32, and grid limit 128
independently with production's 32 rows per tile and grid limit 64. The width
160/224 capacity buckets, two-slot event protocol, and FP32 accumulation were
unchanged. No case, model, or complete-shape dispatch was added.

The four clean extensions passed 88/88 FP16 and FP32 device comparisons across
widths 128, 129, 144, 160, 161, 196, 197, 204, 224, 225, and 256. Maximum
absolute error was `1.22070312e-4` for FP16 and `5.96046448e-8` for FP32.

Standard CannBench used no explicit warmup option. All accepted rows reported
1650/1650 MHz and the profiler default `Warm Up enabled. times:5`. The initial
width-197 screen was:

| Configuration | Duration | Change vs. baseline |
| --- | ---: | ---: |
| Production baseline | 4.244 us | 0.0% |
| 16 rows, grid 64 | 4.165 us | 1.86% faster |
| 32 rows, grid 32 | 4.391 us | 3.46% slower |
| 32 rows, grid 128 | 4.327 us | 1.96% slower |

After this screen, the retention threshold was adjusted from 3% to a
repeatable improvement of at least 1% in every fresh pair, with correctness
and neighboring controls unchanged. The 16-row candidate advanced:

| Pair | Production | 16 rows | Improvement |
| --- | ---: | ---: | ---: |
| 1 | 4.407 us | 4.220 us | 4.24% |
| 2 | 4.313 us | 4.294 us | 0.44% |

The second pair missed the adjusted threshold. The published large-outer
width-196 case regressed from `607.084045 us` to `761.452026 us`, or 25.43%.
Neighbor controls measured width 128 at `3.627 -> 3.490 us` and width 225 at
`4.641 -> 4.693 us`; width 225 was 1.12% slower.

No candidate satisfies the adjusted contract across the affected shape
family. All temporary implementation controls, headers, tests, and remote
datasets were removed. No README or published record was changed. Evidence is
preserved under `/root/cannbench-softmax-v3-mid-launch-task5-20260812-a`.
