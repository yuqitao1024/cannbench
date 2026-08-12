# Compiler And Runtime Pitfalls

Use this reference only when a compiler, loader, launch, or runtime issue blocks
or contaminates an Ascend 950 performance experiment. It is not a general
operator bring-up guide.

## Contents

- [Locate The Failing Boundary](#locate-the-failing-boundary)
- [Verify Compile Mode And Launch Form](#verify-compile-mode-and-launch-form)
- [Check Host/Kernel ABI Mechanically](#check-hostkernel-abi-mechanically)
- [Watch Translation-Unit Resource Effects](#watch-translation-unit-resource-effects)
- [Isolate Dtype-Specific Loader Failures](#isolate-dtype-specific-loader-failures)
- [Prove Which Artifact Is Loaded](#prove-which-artifact-is-loaded)
- [Construct A Minimal Repro Ladder](#construct-a-minimal-repro-ladder)
- [Handle Version Drift](#handle-version-drift)

## Locate The Failing Boundary

Separate the path into:

1. ASC/C++ compilation
2. device and host linking
3. shared-library loading
4. operator schema and backend registration
5. host argument preparation and launch
6. device execution and stream synchronization

Capture the first failing boundary. A successful build says nothing about
registration, ABI correctness, or device execution.

## Verify Compile Mode And Launch Form

Confirm that source annotations, compiler mode, generated kernel kind, launch
API, block dimensions, dynamic memory, and host argument layout belong to the
same programming model. Pure SIMT launches and SIMD/SIMT mixed calls may require
different entry annotations and flags.

Do not change optimization flags as a profiling convenience. If a flag change
is required to collect data, treat it as a different build and re-establish the
baseline.

In mixed AIC/AIV launches, distinguish logical AIC tasks, physical subblocks,
and host grid units. On tested `MIX_AIC_1_2` code, AIV block indices include the
subblock dimension, the logical AIC id is derived using the task ratio, and the
host grid must cover both AIV subblocks. A grid sized only to logical AIC count
can execute half the Vector work while looking superficially plausible. Verify
actual AIC/AIV work rows, not only `Block Dim` metadata.

## Check Host/Kernel ABI Mechanically

Compare host and kernel definitions field by field:

- argument count and order
- pointer address space and constness
- scalar width, signedness, and enum representation
- shape and stride integer width
- dynamic UB or workspace byte count
- grid, block, and launch-bound assumptions

Use a minimal fixed-value launch to expose ABI errors before testing full data.

## Watch Translation-Unit Resource Effects

Observed toolchains have allowed an unused template instance in the same ASC
translation unit to change effective UB/resource metadata or the performance of
a kernel that was actually launched. One measured row-reduction build left the
width-256 source and dispatch unchanged, yet adding width-160 and width-224
instances to its translation unit caused a repeatable regression of about
`1.1%`. Moving the new instances to a separate translation unit restored the
width-256 control to baseline.

When a resource error or unexplained neighboring-range regression appears after
adding a dispatch family:

1. Keep the failing launch fixed.
2. Remove only the unused instantiation.
3. Put the instantiation in a separate ASC translation unit.
4. Preserve the relative order of existing objects and append new
   specialization objects while testing, so link-layout movement is not mixed
   into the source experiment.
5. Compare compiler metadata and runtime behavior, including a control shape
   from an unchanged neighboring dispatch range.

If split-TU passes while single-TU fails, preserve both as a compiler repro.
Do not claim that all same-TU templates are unsafe; bind the observation to the
tested compiler and source form.

## Isolate Dtype-Specific Loader Failures

When FP16 works and BF16 fails, do not immediately rewrite the operator. Build
the smallest pair with identical launch form and control flow, changing only
the dtype and required intrinsic/type spelling. Determine whether failure occurs
at compile, link, registration/load, launch, or synchronization.

Type names and mixed-programming support vary by toolchain. Verify them against
the installed headers and examples rather than guessing an alias. Record a
version-specific limitation if the minimal BF16 form fails while FP16 passes.

## Prove Which Artifact Is Loaded

- Resolve the imported Python module and shared-library paths.
- Inspect symbol presence and timestamps or content hashes.
- Print or expose a build identifier in a diagnostic path.
- Use a fresh process after replacing a custom operator.
- Check environment ordering for older wheels, build trees, and copied packages.
- When device-program caching is plausible, use a unique kernel and launcher
  symbol for the discriminator in addition to checking host-library hashes.

Rebuilding successfully does not prove the remote process loaded that build.

Treat profiler instrumentation as a separate artifact-producing step. In one
observed `msopprof` run, profiling modified the input shared library in place:

```text
clean build:          4c8b4878a2046181a0afeaecae10c52611fa7dc66ccb14b95a920b896843a254
post-instrumentation: 1a94b31efc8f73b99a4a8dbb77410a7c4329cd6702c9229b927bd2df7b0d8b6b
```

Record build and post-instrumentation hashes separately, retain an untouched
build artifact, and do not use an in-place instrumented library as the clean
candidate in a later timing comparison.

## Construct A Minimal Repro Ladder

Reduce the failing system without changing the failure boundary:

- call the registered op directly instead of a workflow
- remove unrelated operators, allocations, and framework transforms
- fix one shape, dtype, grid, and block
- reduce to one kernel source and one host launcher
- remove the framework and use the runtime launch API when possible

Keep passing and failing variants together with exact build/run commands,
expected return codes, device-log excerpts, and version information. Avoid
including credentials or machine-specific endpoints.

When direct execution passes but profiling fails, reproduce with the smallest
one-entry and multi-entry device binaries before blaming the operator. Recorded
profiler combinations produced `RegisterFuncSymbol`, `507046`, empty data, or
`Kernel binary register failure` under replay while compatible versions
profiled the same one- and two-VF controls. Application replay can be a useful
version-bound workaround, but baseline and candidate must use the same replay
mode and the unprofiled direct run must still validate.

## Handle Version Drift

Record compiler, CANN, profiler, driver, and firmware versions for every repro.
If a package update changes behavior, rebuild the same source with both stacks
before attributing the change to source. A profiler built for a mismatched stack
can fail to collect or interpret launches even when the operator runs correctly.
