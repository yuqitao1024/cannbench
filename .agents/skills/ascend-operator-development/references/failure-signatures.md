# Failure Signatures

## Contents

- [Compile Rejects A Type, Intrinsic, Or Annotation](#compile-rejects-a-type-intrinsic-or-annotation)
- [Link Fails Or A Symbol Is Undefined](#link-fails-or-a-symbol-is-undefined)
- [Library Loads But The Operator Is Missing](#library-loads-but-the-operator-is-missing)
- [Compilation Passes But Registration Or Loading Fails](#compilation-passes-but-registration-or-loading-fails)
- [Runtime Reports A UB Or DCache Resource Error](#runtime-reports-a-ub-or-dcache-resource-error)
- [Runtime Reports A Memory, Stack, Or Cache Fault](#runtime-reports-a-memory-stack-or-cache-fault)
- [Kernel Hangs Or Stream Synchronization Never Completes](#kernel-hangs-or-stream-synchronization-never-completes)
- [Output Is All Zero Or Contains Untouched Sentinels](#output-is-all-zero-or-contains-untouched-sentinels)
- [One Core Passes But Multiple Cores Fail](#one-core-passes-but-multiple-cores-fail)
- [Only One Shape Family Or Threshold Fails](#only-one-shape-family-or-threshold-fails)
- [Profile Contains Unexpected Fill, Zero, Cast, Or Copy Kernels](#profile-contains-unexpected-fill-zero-cast-or-copy-kernels)
- [Profile Launch Counts Or Aggregated Time Look Impossible](#profile-launch-counts-or-aggregated-time-look-impossible)
- [An Optimized Baseline Is Orders Of Magnitude Faster](#an-optimized-baseline-is-orders-of-magnitude-faster)
- [Profiler Produces No Data Or Changes Behavior](#profiler-produces-no-data-or-changes-behavior)
- [Unrolling Or Manual Instruction Ordering Is Slower](#unrolling-or-manual-instruction-ordering-is-slower)

Use this index to choose discriminating checks. Error codes are not unique root
causes; always preserve compiler output, device logs, launch metadata, and the
smallest failing input.

## Compile Rejects A Type, Intrinsic, Or Annotation

Likely classes: unsupported toolchain feature, wrong header/API layer, wrong
compile mode, unavailable dtype spelling, or host/device code in the wrong
translation unit.

Check the installed headers and matching SDK examples, then compile a one-kernel
repro using the same annotation and flag. Do not guess replacement type names.

## Link Fails Or A Symbol Is Undefined

Likely classes: missing source in the build, host/device symbol mismatch,
visibility or namespace mismatch, wrong library order, or stale generated files.

Inspect the exact link command and symbols in every input/output library. Prove
the source defining the symbol was compiled for the intended target.

## Library Loads But The Operator Is Missing

Likely classes: registration source omitted, schema mismatch, backend key
mismatch, load failure hidden by fallback, or an older library shadowing the
new build.

Resolve the loaded path, inspect registration symbols, disable fallback in a
diagnostic call, and retry in a fresh process.

## Compilation Passes But Registration Or Loading Fails

Likely classes: unsupported generated kernel form, dtype-specific toolchain
limitation, unresolved device metadata, or ABI incompatibility.

Reduce to FP16/BF16 or pure/mixed paired repros and identify the exact first
failing boundary. Do not debug numerical code before the operator can load.

## Runtime Reports A UB Or DCache Resource Error

For errors such as observed `507035` resource failures, possible classes include
real static/dynamic UB overcommit, SIMT DCache reservation, double-buffer or
compiler scratch, launch metadata mismatch, and translation-unit/template
resource inflation.

Calculate the full live budget, reduce tile and dynamic memory independently,
inspect compiler metadata, and compare a split-translation-unit variant. Accept
a cause only when one controlled change removes the failure.

## Runtime Reports A Memory, Stack, Or Cache Fault

Likely classes: out-of-bounds offset, wrong dtype byte arithmetic, invalid tail
copy, overlapping per-core scratch, ABI mismatch, excessive local state, or a
visibility race.

Reduce to one core and one tile, replace arithmetic with bounded copies, verify
every interval in bytes, then restore stages and cores separately.

## Kernel Hangs Or Stream Synchronization Never Completes

Likely classes: barrier divergence, missing producer/consumer signal, mismatched
participant count, stale flag, inter-core dependency cycle, or a very large
unexpected loop.

Check whether all paths reach synchronization in the same order. Run one core,
bound loops, add debug progress markers, and remove cross-core dependencies.

## Output Is All Zero Or Contains Untouched Sentinels

Likely classes: only an initialization kernel ran, stale module, early return,
wrong dispatch, empty indexed region, incomplete writes, or missing visibility.

Verify the intended kernel launch, loaded artifact, dispatch parameters, and
changed output intervals before changing math.

## One Core Passes But Multiple Cores Fail

Likely classes: overlapping output/scratch ranges, incorrect task-to-core map,
cross-core flag protocol, shared counter initialization, or launch-bound mismatch.

Test two cores, print or calculate intervals, remove shared state, and scale only
after disjoint ownership passes repeatedly.

## Only One Shape Family Or Threshold Fails

Likely classes: wrong dispatch condition, different template instantiation,
capacity threshold, missing tail handling, shape-specific layout, or integer
overflow in offsets.

Test values immediately below, at, and above the threshold and invoke each path
directly when possible.

## Profile Contains Unexpected Fill, Zero, Cast, Or Copy Kernels

Likely classes: output/workspace materialization, dtype/layout adaptation,
framework fallback, or profile selection that includes setup work.

List all raw launches, map them to source-level allocations and transforms, and
state whether they belong to the comparison boundary.

## Profile Launch Counts Or Aggregated Time Look Impossible

Likely classes: application and profiler warmup interaction, multiple staged
kernels, repeated operator calls, wrong name filter, or division by the wrong
launch count.

Write the expected launch sequence and reconcile it row by row before publishing
an aggregate.

## An Optimized Baseline Is Orders Of Magnitude Faster

Likely classes: different algorithmic complexity, phase-specific path, dtype or
compression difference, precomputed metadata, excluded helper stages, or a
measurement-boundary error.

Normalize semantics and boundary first, then compare component times and work
counts. Investigate the dominant component instead of averaging the workflow.

## Profiler Produces No Data Or Changes Behavior

Likely classes: profiler/CANN version mismatch, unsupported metric group, wrong
kernel type, collection filter, insufficient launches after warmup, or altered
compile flags.

Confirm the operator runs without profiling, collect the lowest-overhead metric
set, inspect tool logs, and test the profiler against a minimal known kernel.

## Unrolling Or Manual Instruction Ordering Is Slower

Likely classes: register pressure, spills, lost compiler scheduling, code size,
changed memory issue pattern, or measurement noise.

Use a controlled factorial microbenchmark, repeat runs, and compare generated
resource metadata or instruction timelines. Treat source appearance as a
hypothesis, not evidence of the emitted schedule.
