---
name: ascend-950-operator-performance-optimization
description: Use when profiling, benchmarking, or performance-tuning custom operators on Ascend 950 or 950PR, including 950-specific tiling, launch geometry, AIC/AIV/SIMT/VF pipelines, memory placement, DCCI/cache visibility, synchronization, fusion, reduction, Top-K, performance regressions, or compiler/runtime/correctness failures that prevent a trustworthy performance comparison.
---

# Ascend 950 Operator Performance Optimization

Optimize custom operators for Ascend 950 and 950PR from measured evidence. Use
compiler/runtime and correctness diagnostics only to establish or protect a
valid performance experiment.

## Enforce The Scope

- Confirm the target is Ascend 950 or 950PR before applying this skill as the
  primary workflow.
- Do not use it for generic operator scaffolding, framework integration,
  registration-only bring-up, or functional implementation without a stated
  performance objective.
- On another Ascend generation, use 950 observations only as hypotheses and
  re-establish hardware limits, compiler behavior, correctness, and timing.
- Diagnose build, load, synchronization, accuracy, or runtime failures here
  only when they block a benchmark or invalidate a performance candidate.

## Freeze The Context

Collect existing facts before proposing a design or changing code:

- exact Ascend 950 variant and SoC target, CANN and Bisheng versions,
  firmware/driver version, framework, and `torch_npu` versions
- build mode, kernel launch form, loaded library path, package provenance, and
  whether the process may already have loaded an older module
- operator phase, dtype, complete shape family, layout, strides, Top-K or other
  semantic parameters, and expected output contract
- trusted reference, acceptable tolerance, current failing cases, and known
  passing cases
- exact performance boundary, baseline implementation, input equivalence,
  warmup, repetitions, profiler mode, and raw artifact locations

Discover these facts from the workspace and artifacts when available. Ask only
for facts that cannot be recovered safely.

## Follow The Evidence-First Loop

1. Reproduce a correct Ascend 950 baseline with a defined timing boundary.
2. Specify semantics, including edge behavior, ordering, ties, padding, dtype,
   accumulation, layout, and supported shape ranges.
3. Validate representative, boundary, tail, and dispatch-switching cases. A
   local source test or successful compilation does not prove device behavior.
4. Enumerate every launched kernel and establish where time is spent before
   optimizing.
5. State one bottleneck hypothesis, change one factor, and retain the source,
   command, environment, raw output, and repeated measurements.
6. If the experiment fails, classify it as build/load, launch/ABI,
   memory/synchronization, numerical correctness, or measurement before fixing
   it.
7. Re-run correctness and the original 950 baseline under the same environment.
   Accept a gain only when both remain valid.

Use
[performance-optimization-workflow.md](references/performance-optimization-workflow.md)
for a complete 950 tuning campaign, remote-device iteration, or retention
criteria.

## Route The Investigation

- When a 950 performance candidate produces wrong values, NaNs, zeros,
  ordering errors, shape-specific failures, multicore-only failures, or hangs,
  read
  [correctness-and-debugging.md](references/correctness-and-debugging.md).
- For UB/L1/L0 capacity, alignment, tails, tiling, launch geometry, data
  traversal, pipelines, DCCI/cache visibility, multi-flag ordering, races, or
  barriers, read
  [memory-synchronization-and-tiling.md](references/memory-synchronization-and-tiling.md).
- For slow kernels, excessive launches, poor algorithms, fusion decisions,
  reductions, Top-K, prefill/decode specialization, or microbenchmarks, read
  [performance-optimization.md](references/performance-optimization.md).
- Before estimating the benefit of a familiar optimization, or when deciding
  whether to retry a previously tested idea, read
  [observed-optimization-results.md](references/observed-optimization-results.md).
  It records scenario-bound gains, regressions, and no-gain experiments from
  prior real-device work. Use it as prior evidence, not as a portable promise.
- For kernel attribution, timing boundaries, launch gaps, multi-process
  workflow replay, launch aggregation, warmup, metric-set selection, variance,
  or baseline fairness, read
  [profiling-and-benchmarking.md](references/profiling-and-benchmarking.md).
- For `msopprof` commands, `--aic-metrics` selection, the meaning of
  `Default` or "all metrics," CSV versus MindStudio Insight artifacts,
  metric-specific build requirements, replay restrictions, or 950-only
  timeline and SIMT stall analysis, read
  [msopprof.md](references/msopprof.md) before the general profiling guidance.
- When compiler, linker, registration, loader, ABI, translation-unit,
  packaging, or version-dependent behavior blocks or contaminates a 950
  performance comparison, read
  [compiler-and-runtime-pitfalls.md](references/compiler-and-runtime-pitfalls.md).
- When a performance experiment starts from an error code or visible symptom,
  read
  [failure-signatures.md](references/failure-signatures.md) first, then load the
  reference it points to.

Load only the references required by the current problem.

## Keep Claims Calibrated

- Mark untested explanations as hypotheses and name the test that would
  discriminate them.
- Describe compiler, profiler, and loader quirks as observations tied to the
  tested versions. Reproduce locally before generalizing.
- Treat measured gains as specific to the recorded 950 variant, CANN/compiler
  stack, frequency, shape, and timing boundary. Revalidate even across 950
  variants or software versions.
- Do not infer resource use from source declarations alone; compiler template
  instantiation and launch metadata can change the effective budget.
- Do not infer a workflow latency from one kernel row or assume every visible
  auxiliary kernel belongs inside the comparison boundary.
- Do not import CUDA tuning rules mechanically. Re-test traversal, unrolling,
  occupancy, and fusion assumptions on the target Ascend generation.

## Finish With Verifiable Evidence

Report the exact 950 variant, cases, reference, tolerance, and correctness
result. Also report raw profile locations, selected and excluded kernels,
launch counts, warmup, repetitions, distribution summary, baseline parity, and
any untested dtype, phase, shape, or 950 variant.
