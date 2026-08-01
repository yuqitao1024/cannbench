# Compiler And Runtime Pitfalls

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
translation unit to change effective UB/resource metadata for a kernel that was
actually launched. When a resource error appears after adding a dispatch family:

1. Keep the failing launch fixed.
2. Remove only the unused instantiation.
3. Put the instantiation in a separate ASC translation unit.
4. Compare compiler metadata and runtime behavior.

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

Rebuilding successfully does not prove the remote process loaded that build.

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

## Handle Version Drift

Record compiler, CANN, profiler, driver, and firmware versions for every repro.
If a package update changes behavior, rebuild the same source with both stacks
before attributing the change to source. A profiler built for a mismatched stack
can fail to collect or interpret launches even when the operator runs correctly.
