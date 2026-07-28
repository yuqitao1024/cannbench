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
    copy_vf.asc
    add_one_vf.asc
    two_vf_main.asc
    single_vf_main.asc
    README.md
```

`copy_vf.asc` and `add_one_vf.asc` are separate ASC translation units linked
into one shared device library. Both use a 1024-thread `__simt_vf__` behind a
pure `__global__ __vector__` entry. `two_vf_main.asc` normally launches only
the copy kernel; the `--launch-second` switch launches the add-one kernel as
well. Keeping the default runtime workload to one launch isolates the number
of VF entries in the shared ELF from the number of executed kernels.

`single_vf_main.asc` links a control shared library containing only
`copy_vf.asc`. It performs the same default copy operation and validation with
one `__simt_vf__` entry.

## Runtime Behavior

Both executables use ACL directly to:

1. initialize device 0 and create a stream;
2. allocate small input and output buffers;
3. launch a one-block, 1024-thread SIMT copy kernel;
4. synchronize and copy the result to the host;
5. validate the output and release all resources.

They print a stable target name, synchronization return code, and validation
result. A direct run must exit zero before an msopprof result is interpreted.

The reproduction uses only ACL plus the Ascend SIMT API in `.asc` sources. It
does not import or link Python, PyTorch, torch_npu, or CannBench code.

## Build And Profiling Interface

`CMakeLists.txt` uses the repository's existing standalone ASC pattern,
defaults `CMAKE_ASC_ARCHITECTURES` to `dav-3510`, and builds:

- `libtwo_vf_kernels.so`
- `libsingle_vf_kernel.so`
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
- compare whether `single_vf_control` and `two_vf_repro` produce profile data;
- record any missing data, `RegisterFuncSymbol`, or binary-registration error;
- treat success as evidence that VF count and pure-Vector ELF layout are not a
  sufficient trigger for that profiler build.

The July 28 target run succeeded with both the installed 26.0 build and a
fresh 26.1 branch build. A mixed-CV control also succeeded. The exact affected
historical profiler remains necessary to determine whether the failure was
version-specific or depended on another property of the original operator
ELF.

Operator-local source tests verify that the sample remains standalone and that
the two targets contain exactly one and two `__simt_vf__` declarations,
respectively.

## Relationship To The Prefill Change

This diagnostic is an independent test artifact. It does not change Lightning
Indexer dispatch or kernel behavior. After it is built and exercised on the
port-20002 device, the 32-mixed-task V3.2 prefill correctness and performance
gate resumes unchanged.
