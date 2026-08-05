# Profiling And Benchmarking

## Contents

- [Define The Timing Boundary](#define-the-timing-boundary)
- [Audit Every Kernel First](#audit-every-kernel-first)
- [Aggregate Multi-Stage Implementations Correctly](#aggregate-multi-stage-implementations-correctly)
- [Separate Application And Profiler Warmup](#separate-application-and-profiler-warmup)
- [Collect Metric Groups Deliberately](#collect-metric-groups-deliberately)
- [Use Repeated Measurements](#use-repeated-measurements)
- [Compare Baselines Fairly](#compare-baselines-fairly)
- [Interpret Profiles Conservatively](#interpret-profiles-conservatively)

## Define The Timing Boundary

Before collecting data, state whether the result measures:

- one device kernel
- one registered operator including helper kernels
- a fused or multi-operator workflow
- host dispatch plus device execution
- initialization, packing, workspace clearing, copies, and synchronization

Use the same semantic boundary for every implementation. If a stage is excluded,
name it and explain why it is outside the contract.

## Audit Every Kernel First

List all raw profiler rows before applying name filters. Classify each launch as:

- input/output initialization
- layout conversion or packing
- main computation
- reduction, merge, or postprocess
- framework/runtime bookkeeping
- repeated warmup or replay

Unexpected `Fill`, `ZerosLike`, cast, or copy kernels are clues about materialization,
not automatically bugs. Determine who launched them and whether the comparison
boundary includes them.

## Aggregate Multi-Stage Implementations Correctly

For each selected kernel family, record:

- expected launches per operator call
- observed launch count
- per-launch duration distribution
- sum per operator call
- number of application calls captured

Do not divide a selected total by an unrelated global launch count. A staged
implementation may require summing several differently named kernels; a fused
implementation may use one. Validate aggregation against the raw timeline.

When one operator call emits separate profiler files or replay groups, decide
explicitly whether they are stages of the same call or repeated samples. Sum
stages per call before computing a distribution across calls. One recorded
multi-stage row reduction averaged `stats` and `write` as if they were repeated
samples, materially underreporting the operator and creating a false regression
when the aggregation was later corrected.

## Separate Application And Profiler Warmup

Understand which component actually emits launches:

- application warmup executes the operator before measured iterations
- profiler warmup usually skips or treats early emitted launches specially
- replay or launch-count options may add another selection layer

Write the expected launch sequence, then compare it with raw rows. Do not assume
two warmup settings independently create the same number of extra executions.

## Collect Metric Groups Deliberately

Profiler metric groups can require separate runs and can perturb execution.
Use a low-overhead timing-oriented run for latency and separate runs for
instruction timeline, arithmetic utilization, memory behavior, or other detailed
counters supported by the installed profiler.

Do not compare detailed-metric latency directly with a default run unless the
tool documents equivalent overhead. Preserve the exact profiler version and
command because option names and availability can change.

## Use Repeated Measurements

- Warm the runtime, module, allocator, and caches consistently.
- Run enough repetitions to distinguish a small gain from launch jitter.
- Report median and a spread measure; retain individual samples for short kernels.
- Recollect suspicious results in a new process and, when practical, a second run.
- Avoid setting unusual warmup or compiler flags solely to make collection work
  without documenting the altered behavior.
- Use a unique run name or a clean output directory for every collection. If a
  parser discovers both old and new profiler directories, discard the aggregate
  and rebuild it from an auditable clean tree.

For very short kernels, enlarge work or repeat inside a controlled harness, then
divide only when each repetition performs identical independent work.

## Compare Baselines Fairly

Confirm both paths use the same:

- semantic inputs and output contract
- dtype and compression policy
- phase and shape
- precomputed metadata and workspace lifecycle
- stream synchronization point
- initialization, packing, and postprocess boundary
- device frequency and target core type

An orders-of-magnitude gap warrants a boundary audit before an architectural
explanation.

Prove package provenance for both sides. Resolve the source root inserted by
the loader, the imported extension path, and a source or binary hash in the
same process that runs the benchmark. `PYTHONPATH` alone is insufficient when
the framework prepends an operator-local directory or an earlier import has
already populated the module cache. Re-run in a fresh process after replacing
an extension.

## Interpret Profiles Conservatively

A profile can show where time is spent and which kernels execute. It may not
prove why a kernel is slow. Combine timing with workload counts, memory traffic,
resource metadata, and detailed metrics. If the evidence cannot distinguish
algorithmic work from instruction scheduling or memory stalls, say so and design
the next discriminating collection.

Retain raw CSV/timeline output, selected-kernel rules, aggregation script or
formula, environment manifest, and the exact source/package revision.
