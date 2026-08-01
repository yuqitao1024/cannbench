# Development Workflow

## Contents

- [Use An Evidence Ladder](#use-an-evidence-ladder)
- [Record A Reproducible Baseline](#record-a-reproducible-baseline)
- [Define The Contract First](#define-the-contract-first)
- [Build The Smallest Real Path](#build-the-smallest-real-path)
- [Expand Coverage By Behavior](#expand-coverage-by-behavior)
- [Iterate One Hypothesis At A Time](#iterate-one-hypothesis-at-a-time)
- [Escalate With Smaller Repros](#escalate-with-smaller-repros)
- [Completion Criteria](#completion-criteria)

## Use An Evidence Ladder

Prefer evidence in this order:

1. Reproducible target-device results with preserved commands and artifacts.
2. Compiler diagnostics, device logs, kernel metadata, and raw profiler rows.
3. Focused tests and minimal repros that isolate one boundary.
4. Source and generated-code inspection.
5. Architectural intuition and analogies to another backend.

Lower levels generate hypotheses. They do not override contradictory device
evidence.

## Record A Reproducible Baseline

Before editing, record:

- repository revision and dirty state
- exact build and package identifiers
- hardware, CANN, compiler, driver, firmware, framework, and Python versions
- environment setup, compile flags, launch API, and loaded shared-library path
- input shapes, dtype, layout, strides, seed, and semantic parameters
- reference implementation and comparison rule
- warmup, repetitions, measurement command, and raw output path

Keep production code, temporary experiments, and deployed packages distinguishable.
Use a clean process after rebuilding a custom operator unless same-process
reload behavior is itself under test.

## Define The Contract First

Write down the complete output contract before choosing a kernel structure:

- supported shape ranges and layouts
- dtype of every input, output, workspace, and accumulator
- numerical tolerance and exceptional-value behavior
- reduction dimensions and empty-input behavior
- Top-K ordering, tie behavior, index type, padding, and invalid-candidate rules
- aliasing, mutability, workspace, stream, and synchronization expectations

Use a simple trusted implementation as the oracle. If the oracle performs
extra casting, sorting, masking, or padding, make that behavior explicit.

## Build The Smallest Real Path

Bring up one representative case end to end:

1. Register the host entry and verify its schema.
2. Validate dtype, rank, shape, layout, device, and workspace assumptions.
3. Launch a minimal kernel through the intended production launch mechanism.
4. Synchronize and compare against the oracle on the real device.
5. Preserve a fallback while specialized paths are incomplete.

Start with independent work ownership and a correctness-first algorithm. Add
mixed pipelines, fusion, multicore exchange, and persistent state separately.

## Expand Coverage By Behavior

Partition cases by behaviors that change generated code or runtime structure:

- prefill versus decode, or other large-query versus small-query phases
- dtype and accumulation path
- head or channel dimension family
- contiguous versus strided layout
- single-tile versus multi-tile and aligned versus tail tiles
- single-core versus multicore execution
- Top-K or reduction ranges that change the algorithm

Select one representative and one boundary case for each path. Include minimum,
maximum, odd, tail, and dispatch-threshold values.

## Iterate One Hypothesis At A Time

For every performance or correctness change:

1. State the expected mechanism and metric.
2. Change one independent factor.
3. Rebuild from a known revision and deploy a uniquely identifiable artifact.
4. Run focused correctness before timing.
5. Repeat measurements and compare distributions, not only the best sample.
6. Keep negative results; they prevent repeated false assumptions.

Use factorial microbenchmarks when factors interact. For example, compare
traversal x launch form x loop form using identical work and generated outputs.

## Escalate With Smaller Repros

Reduce dependencies in stages:

1. Full framework workflow.
2. Direct registered-operator call.
3. Single custom-op package and one shape.
4. Standalone host launch plus one ASC source.
5. One translation unit, one kernel, one launch, and fixed data.

Keep the failing and passing variants side by side. Remove a construct only
after verifying whether its removal changes the failure.

## Completion Criteria

Do not claim completion until:

- all declared paths pass representative and boundary correctness cases
- real-device runtime was exercised for device-dependent work
- performance uses a defined and fair boundary with raw artifacts retained
- the baseline was rerun under the same stack
- unsupported shapes, dtypes, phases, and devices are stated
- temporary flags, packages, workspaces, and debug paths are not mistaken for
  the final implementation
