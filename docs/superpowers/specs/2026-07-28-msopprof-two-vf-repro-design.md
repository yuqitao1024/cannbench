# msopprof Two-VF Standalone Reproduction Design

## Purpose

Provide a small source package that can be handed to msopprof maintainers to
investigate profiling failures when one Ascend device ELF contains two SIMT VF
entries. The package must reproduce the ELF layout independently of CannBench,
Python, PyTorch, and torch_npu.

## Location And Contents

Keep the diagnostic under the owning operator package:

```text
src/cannbench/operators/builtin/lightning_indexer/simt/test/
  msopprof_two_vf_repro/
    CMakeLists.txt
    two_vf_repro.asc
    single_vf_control.asc
    README.md
```

`two_vf_repro.asc` builds one executable and one device ELF containing two
distinct `__simt_vf__` entries. Its normal run launches only the first VF's
kernel; the `--launch-second` switch launches the second kernel as well.
Keeping the default runtime workload to one launch isolates the number of VF
entries in the ELF from the number of executed kernels.

`single_vf_control.asc` performs the same default copy operation and validation
but contains only one `__simt_vf__` entry. It is the control for confirming that
separate compilation avoids the profiler failure.

## Runtime Behavior

Both executables use ACL directly to:

1. initialize device 0 and create a stream;
2. allocate small input and output buffers;
3. launch a one-block SIMT copy kernel;
4. synchronize and copy the result to the host;
5. validate the output and release all resources.

They print a stable target name, synchronization return code, and validation
result. A direct run must exit zero before an msopprof result is interpreted.

The reproduction uses only ACL plus the Ascend SIMT API in `.asc` sources. It
does not import or link Python, PyTorch, torch_npu, or CannBench code.

## Build And Profiling Interface

`CMakeLists.txt` uses the repository's existing standalone ASC pattern,
defaults `CMAKE_ASC_ARCHITECTURES` to `dav-3510`, and builds:

- `two_vf_repro`
- `single_vf_control`

The README provides exact commands for CMake configure/build, direct execution,
and separate BasicInfo msopprof collections. It also lists the files and logs
to return to the profiler team.

## Expected Diagnostic Result

The sample does not make the profiler failure a build or repository-test
requirement because the behavior belongs to a specific msopprof version. Its
expected comparison is:

- both executables build, run, and validate successfully outside msopprof;
- `single_vf_control` produces profile data;
- `two_vf_repro` exposes the missing-data or `RegisterFuncSymbol` failure on an
  affected profiler build;
- the updated profiler may make both collections succeed, which is also a
  valid result and demonstrates that the minimal reproducer no longer fails.

Operator-local source tests verify that the sample remains standalone and that
the two targets contain exactly one and two `__simt_vf__` declarations,
respectively.

## Relationship To The Prefill Change

This diagnostic is an independent test artifact. It does not change Lightning
Indexer dispatch or kernel behavior. After it is built and exercised on the
port-20002 device, the 32-mixed-task V3.2 prefill correctness and performance
gate resumes unchanged.
