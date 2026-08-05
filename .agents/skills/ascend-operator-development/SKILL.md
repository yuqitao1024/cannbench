---
name: ascend-operator-development
description: Evidence-driven Ascend custom-operator development, debugging, profiling, and performance optimization. Use when Codex needs to implement or review an Ascend C/C++, Tensor API, SIMT, or SIMD/SIMT mixed kernel; choose tiling, launch geometry, memory placement, synchronization, fusion, reduction, or Top-K strategies; diagnose compile, link, registration, loading, launch, device-runtime, accuracy, race, hang, or resource errors; interpret msprof or msopprof data; build a minimal compiler/runtime repro; or compare and tune prefill, decode, and other shape-dependent workloads on Ascend hardware.
---

# Ascend Operator Development

Develop from evidence gathered on the target stack. Treat source inspection,
compiler output, device logs, correctness results, and profiles as different
forms of evidence; do not let an early hypothesis become a conclusion.

## Freeze The Context

Collect existing facts before proposing a design or changing code:

- Ascend product and SoC, CANN and Bisheng versions, firmware/driver version,
  framework and `torch_npu` versions
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

1. Specify semantics, including edge behavior, ordering, ties, padding, dtype,
   accumulation, layout, and supported shape ranges.
2. Build or isolate the smallest correct real-device path. Avoid adding fusion,
   multicore coordination, and aggressive specialization at the same time.
3. Validate representative, boundary, tail, and dispatch-switching cases. A
   local source test or successful compilation does not prove device behavior.
4. Classify a failure before fixing it: build, link, registration/load,
   launch/ABI, memory/synchronization, numerical correctness, or measurement.
5. Define the measured boundary, enumerate every launched kernel, and establish
   where time is spent before optimizing.
6. State one bottleneck hypothesis, change one factor, and retain the source,
   command, environment, raw output, and repeated measurements.
7. Re-run correctness and the original baseline under the same environment.
   Accept a gain only when both remain valid.

Use [development-workflow.md](references/development-workflow.md) for a new
operator, a major rewrite, remote-device iteration, or completion criteria.

## Route The Investigation

- For wrong values, NaNs, zeros, ordering errors, shape-specific failures,
  multicore-only failures, or hangs, read
  [correctness-and-debugging.md](references/correctness-and-debugging.md).
- For UB/L1/L0 capacity, alignment, tails, tiling, launch geometry, data
  traversal, pipelines, races, or barriers, read
  [memory-synchronization-and-tiling.md](references/memory-synchronization-and-tiling.md).
- For slow kernels, excessive launches, poor algorithms, fusion decisions,
  reductions, Top-K, prefill/decode specialization, or microbenchmarks, read
  [performance-optimization.md](references/performance-optimization.md).
- Before estimating the benefit of a familiar optimization, or when deciding
  whether to retry a previously tested idea, read
  [observed-optimization-results.md](references/observed-optimization-results.md).
  It records scenario-bound gains, regressions, and no-gain experiments from
  prior real-device work. Use it as prior evidence, not as a portable promise.
- For kernel attribution, timing boundaries, launch aggregation, warmup,
  metric-set selection, variance, or baseline fairness, read
  [profiling-and-benchmarking.md](references/profiling-and-benchmarking.md).
- For compiler, linker, operator registration, loader, ABI, translation-unit,
  packaging, or version-dependent failures, read
  [compiler-and-runtime-pitfalls.md](references/compiler-and-runtime-pitfalls.md).
- When starting from an error code or visible symptom, read
  [failure-signatures.md](references/failure-signatures.md) first, then load the
  reference it points to.

Load only the references required by the current problem.

## Keep Claims Calibrated

- Mark untested explanations as hypotheses and name the test that would
  discriminate them.
- Describe compiler, profiler, and loader quirks as observations tied to the
  tested versions. Reproduce locally before generalizing.
- Do not infer resource use from source declarations alone; compiler template
  instantiation and launch metadata can change the effective budget.
- Do not infer a workflow latency from one kernel row or assume every visible
  auxiliary kernel belongs inside the comparison boundary.
- Do not import CUDA tuning rules mechanically. Re-test traversal, unrolling,
  occupancy, and fusion assumptions on the target Ascend generation.

## Finish With Verifiable Evidence

For correctness work, report the exact cases, reference, tolerance, device, and
result. For performance work, also report raw profile locations, selected and
excluded kernels, launch counts, warmup, repetitions, distribution summary,
and baseline parity. State any untested dtype, phase, shape, or device explicitly.
