# SIMT Add Traversal Reproduction Design

## Purpose

Provide a standalone Ascend reproduction for measuring three independent
binary factors in a fixed in-place `float` accumulation workload:

- direct SIMT versus SIMD/SIMT hybrid execution;
- global-stride versus block-contiguous traversal;
- a four-iteration loop versus manual expansion.

The full matrix contains eight cases. The reproduction is independent of
Python, PyTorch, torch_npu, and the CannBench runtime.

## Source Boundary

Keep the reproduction under the owning operator implementation test tree:

```text
src/cannbench/operators/builtin/lightning_indexer/simt/test/
  simt_add_traversal_repro/
    CMakeLists.txt
    DESIGN.md
    README.md
    main.asc
```

`main.asc` owns the device functions, host launchers, deterministic input
generation, and full-output validation. `CMakeLists.txt` builds both execution
models from that source. `DESIGN.md` describes the current implementation and
acceptance criteria. `README.md` records commands and, when available for the
current workload, measurements and conclusions.

## Fixed Workload

Every case uses the same compile-time configuration:

- element type: `float`;
- operation: `output[i] += input_x[i] + input_y[i]`;
- grid dimension: 64 blocks;
- block dimension: 2048 threads;
- launch bound: `__launch_bounds__(2048)`;
- elements per thread: 4;
- total elements: `64 * 2048 * 4`, or 524288 elements.

The fixed shape maps exactly to the launch configuration, so the device
functions do not need an input-dependent boundary branch.

## Factor Matrix

The executable name selects the execution model and the single command-line
argument selects traversal and source form:

| Execution model | Mode | Traversal | Source form |
| --- | --- | --- | --- |
| direct SIMT | `global-stride` | global stride | loop |
| direct SIMT | `global-stride-unrolled` | global stride | manual expansion |
| direct SIMT | `block-contiguous` | block contiguous | loop |
| direct SIMT | `block-contiguous-unrolled` | block contiguous | manual expansion |
| hybrid | `global-stride` | global stride | loop |
| hybrid | `global-stride-unrolled` | global stride | manual expansion |
| hybrid | `block-contiguous` | block contiguous | loop |
| hybrid | `block-contiguous-unrolled` | block contiguous | manual expansion |

## Execution Models

Both targets compile the same four traversal bodies. Compile-time definitions
change only their device declaration and launch boundary.

The direct target defines `ADD_TRAVERSAL_DIRECT_SIMT`, declares each body as a
`__global__ __launch_bounds__(2048)` kernel, compiles with `--enable-simt`, and
launches with:

```text
<<<64, 2048, 0, stream>>>
```

The hybrid target defines `ADD_TRAVERSAL_HYBRID`, declares each body as an
independent `__simt_vf__ __launch_bounds__(2048)` function, and does not use
`--enable-simt`. A corresponding `__global__ __vector__` outer kernel calls:

```text
asc_vf_call<traversal_vf>(dim3(2048, 1, 1), input_x, input_y, output)
```

The host launches 64 outer Vector kernels with zero dynamic UB bytes. Each
mode has a distinct kernel name so msopprof can report it without a device-side
mode branch.

## Traversal Forms

Global stride starts at the global thread index and advances by the total
number of threads:

```text
global_tid = blockIdx.x * blockDim.x + threadIdx.x
index = global_tid + iteration * (gridDim.x * blockDim.x)
```

Block-contiguous traversal assigns one `4 * 2048` element region to each
block and advances by one block of threads:

```text
block_base = blockIdx.x * (4 * blockDim.x)
index = block_base + threadIdx.x + iteration * blockDim.x
```

Both loop variants preserve the four-iteration `for` loop. Both manually
expanded variants use the same load-all ordering:

1. compute `index0` through `index3`;
2. load `input_x0` through `input_x3`;
3. load `input_y0` through `input_y3`;
4. accumulate into and store `output0` through `output3`.

This alignment keeps traversal indexing as the only intended difference
between the two manually expanded bodies. No explicit unroll pragma,
`volatile`, or inline assembly is used.

## Host Flow

Each invocation:

1. initializes device 0 and creates one stream;
2. allocates two inputs and one in-place output in device memory;
3. copies deterministic inputs and nonzero initial output values to the device;
4. launches only the selected kernel;
5. synchronizes and copies the complete output to the host;
6. infers the actual uniform accumulation count from one fixed nonzero probe;
7. requires that count to equal the independent expected count from the
   executable argument, which defaults to one;
8. validates every element after applying `output += input_x + input_y` the
   expected number of times on the host;
9. releases all ACL resources.

The program reports the execution model, mode, launch dimensions, element
count, mismatch count, and validation result. It exits zero only when all
elements match.

## Build And Profile

`CMakeLists.txt` uses `find_package(ASC REQUIRED)` and builds:

- `simt_add_traversal_direct` with `--npu-arch=dav-3510 --enable-simt`;
- `simt_add_traversal_hybrid` with `--npu-arch=dav-3510`.

The previously validated profiling setup used the installed CANN 9.1 compiler,
runtime, and msopprof. Its direct SIMT binary used the ordinary
`aclrtLaunchKernelWithHostArgs` path, which that profiler intercepts. The
tested CANN 9.2 direct binary uses `aclrtLaunchSIMTKernelWithHostArgs`, which
the installed msopprof injection library does not intercept; the hybrid binary
remains profileable.

Profile every case independently with:

```text
msopprof --output=<output> --aic-metrics=Default --launch-count=1 \
  ./<executable> <mode> 3
```

Use the profiler's default replay and warmup behavior. Kernel replay can apply
the stateful accumulation repeatedly to the same output buffer, so the host
infers the actual uniform positive accumulation count and requires it to match
the independent executable argument. The trailing `3` is an application
argument matching the observed CANN 9.1.0 default replay count, not a profiler
option. Use `Task Duration(us)` from `OpBasicInfo.csv` as the primary metric and
the mean `aiv_total_cycles` across 64 blocks as supporting data. Record fresh
two-round results in the reproduction README after all accumulation cases pass
full-output validation.

## Acceptance

The reproduction is complete when:

1. the two targets build with their target-specific compile options;
2. all eight executable/mode invocations report `mismatch_count=0` and
   `validation=pass`;
3. every compared profile emits exactly one matching kernel record;
4. global and block manual expansions retain identical instruction ordering;
5. the operator-local source contract test confirms the in-place accumulation
   and initialized-output setup;
6. the README records the exact environment, commands, measurements,
   variability, and conclusions after the accumulation workload is profiled.

Target compilation, full-output validation, and profiler artifacts remain the
real-device verification.
