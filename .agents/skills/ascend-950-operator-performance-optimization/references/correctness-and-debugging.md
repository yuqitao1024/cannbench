# Correctness And Debugging

Use this reference to validate or recover an Ascend 950 performance candidate.
Do not treat it as a generic functional-development workflow.

## Contents

- [Establish A Trustworthy Oracle](#establish-a-trustworthy-oracle)
- [Build A Behavioral Case Matrix](#build-a-behavioral-case-matrix)
- [Localize The Failing Stage](#localize-the-failing-stage)
- [Distinguish Initialization From Computation](#distinguish-initialization-from-computation)
- [Isolate Multicore Failures](#isolate-multicore-failures)
- [Debug Races And Hangs](#debug-races-and-hangs)
- [Eliminate Stale-Code Explanations](#eliminate-stale-code-explanations)
- [Report Correctness Evidence](#report-correctness-evidence)

## Establish A Trustworthy Oracle

- Compare semantic outputs, not merely buffer shapes or successful launches.
- Match masking, padding, invalid indices, tie handling, and output ordering.
- Use FP32 accumulation for reductions when the contract permits it, then cast
  at the defined boundary.
- Choose tolerances from operation depth and dtype. Also report maximum error,
  mismatch count, and the first mismatching coordinate.
- For Top-K, compare sets when ordering is unspecified; compare ordered pairs
  only when ordering is part of the contract.

## Build A Behavioral Case Matrix

Cover cases that activate different risks:

- zero, minimum, typical, threshold, maximum, odd, and tail dimensions
- aligned and deliberately unaligned logical lengths
- all equal, monotonic, duplicate, negative, very large, and non-finite values
- single-core and multicore launches
- every phase, dtype, layout, and dispatch family
- repeated execution in one process and a fresh-process execution

Use deterministic seeds and save the first failing input.

## Localize The Failing Stage

For a multi-stage operator, validate stage boundaries independently:

1. Fill intermediate storage with a recognizable sentinel.
2. Run one stage and verify exactly which region changed.
3. Copy or expose the intermediate only in a debug build.
4. Compare that stage with a host or framework reference.
5. Continue until the first divergent stage is known.

Prefer this over rewriting several stages from intuition. For fused paths,
temporarily restore an observable boundary or build an unfused twin.

## Distinguish Initialization From Computation

Unexpected zero or fill kernels often come from output or workspace creation,
not the main algorithm. Determine whether initialization is semantically
required, defensive, or dead work. If a kernel overwrites the complete output,
allocate uninitialized storage only after proving every element is written,
including tails and early-return paths.

Zero output can mean:

- the intended kernel never launched or the old module was loaded
- an output was initialized but not fully written
- indexing selected an empty region
- synchronization exposed stale data
- numerical conversion or masking removed all values

Inspect launch evidence and changed memory regions before changing arithmetic.

## Isolate Multicore Failures

When one core passes and multiple cores fail:

- compute each core's exact input, output, and scratch interval
- prove intervals are disjoint or document the reduction protocol
- verify core count, logical task count, and launch geometry independently
- test two cores before scaling to the full device
- remove cross-core coordination and assign independent output units when possible
- check that flags, counters, and scratch buffers are initialized once per use

A launch-bound mismatch is suspicious but not automatically causal. Correct it,
then rerun the minimal failing case before naming it the root cause.

## Debug Races And Hangs

- Identify the producer and consumer of every shared region.
- Record the required visibility point and whether the primitive provides it.
- Ensure every participating path reaches barriers in the same order.
- Do not place a barrier behind data-dependent control flow unless participation
  is proven identical.
- Distinguish kernel-local pipeline synchronization from inter-core coordination.
- Add bounded progress markers in debug builds instead of waiting indefinitely.

For intermittent errors, repeat the same input many times and vary only core
count or scheduling pressure.

For seed-specific pipeline failures, preserve one failing seed and one passing
seed. Change one ordering edge without changing buffers, work, or flag count;
repeat both in fresh processes. If moving an older wait before the next wait or
reuse point fixes only the failing path, that is stronger evidence for a state
ordering defect than broad synchronization or layout changes.

## Eliminate Stale-Code Explanations

Before deep kernel debugging, verify:

- the built library contains the expected symbol or identifying string
- the deployed path and imported module path match the new artifact
- no older package shadows it in `PYTHONPATH` or the environment
- a fresh process reproduces the result
- build caches and generated outputs correspond to the current source revision

Same-process module replacement is not reliable evidence that new device code
was loaded.

## Report Correctness Evidence

Record hardware and stack versions, revision, artifact identifier, exact cases,
reference, tolerance, mismatch summary, and whether a fresh process was used.
State paths that were compiled but not executed separately from paths verified
on the device.
