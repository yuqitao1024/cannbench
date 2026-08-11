# Softmax Dim-128 Persistent Pipeline Design

## Goal

Improve the Ascend 950PR Softmax V3 `dim_size == 128` realistic family while
preserving the existing plugin boundary, numerical behavior, fallback paths,
and published-data contract.

The current FP16 published results are 1.64x to 2.81x slower than the CANN ops
library for the five width-128 cases. The highest-value cases share the same
generic persistent direct-GM implementation:

| Case | Outer rows | V3 | CANN ops | Ratio |
| --- | ---: | ---: | ---: | ---: |
| `gptj_attention` | 2,048 | 6.193 us | 3.787 us | 1.64x |
| `bert_pytorch_attention` | 24,576 | 19.950 us | 10.308 us | 1.94x |
| `gptneo_attention` | 65,536 | 50.688 us | 18.762 us | 2.70x |
| `mobilebert_attention` | 65,536 | 51.036 us | 18.557 us | 2.75x |
| `pegasus_attention` | 65,536 | 52.383 us | 18.668 us | 2.81x |

## Scope

The implementation stays under:

```text
src/cannbench/operators/builtin/softmax/simt/v3/
```

It may add an exact-width kernel translation unit, update operator-local
dispatch, extend operator-local tests, and document measured results. It does
not change CLI, common backends, core configuration, result schemas, or other
operators.

## Selected Architecture

Add a dedicated `persistent_128.asc` translation unit for exact
`dim_size == 128`. Keep the generic persistent implementation as the fallback
and keep the existing 129-256, 512, and 1024 specializations unchanged.

The dedicated path uses the existing mixed SIMD/SIMT execution model:

```text
GM input
  -> two-slot MTE2 copy into input UB
  -> SIMT VF max, exp-sum, normalize
  -> two-slot MTE3 copy from output UB to GM
```

One 32-lane warp owns one row. Each lane owns four elements, so a 1024-thread
VF processes 32 rows per tile. This replaces lane-level GM accesses with
aligned bulk transfers while keeping the VF small enough for trustworthy
`BasicInfo` profiling on the target toolchain.

The input and output double buffers each hold two `32 x 128` FP16/FP32 tiles.
The maximum combined allocation is 64 KiB for FP32 and remains below the
available 224 KiB UB budget. The FP16 path uses 32 KiB.

Because the path accepts only width 128, element bounds and warp iteration
counts are compile-time facts. VF loop and address indices use `int32_t`; outer
row and GM offsets remain `int64_t`.

## Dispatch

`row_persistent_fallback.asc` dispatches exact width 128 to the new FP16 or
FP32 entry point. All other dimensions retain their current behavior.

The dedicated path is used for every `outer_size`. Both paired collections
improved the 2,048-row case, so no shape threshold is required.

## Alternatives Rejected

Extending the existing 256 bucket down to width 128 was built as a control. It
allocates eight values per lane while only four are live and performs inactive
iterations. Its synchronized 65,536-row latency was 59.19 us versus 34.74 us
for the direct-GM baseline, so it was rejected.

The first dedicated candidate assigned two rows to every warp and processed
64 rows per tile. Its synchronized 65,536-row latency improved to 27.57 us,
but `msopprof BasicInfo` expanded the three large realistic cases to about
389 us. The retained single-row VF measured 24.59 us on the synchronized
boundary and about 24.4 us on `BasicInfo`, avoiding that measurement cliff.

Changing only generic launch geometry preserves direct-GM traffic. It is a
useful control for attribution but has a lower expected ceiling and does not
address the observed memory schedule.

## Correctness

The specialized path preserves stable Softmax semantics: FP32 accumulation,
row maximum subtraction before exponentiation, one normalized output per
input element, and no cross-row state.

Validation covers:

- all five realistic FP16 width-128 cases;
- representative FP32 width-128 input;
- tail row counts below and not divisible by 32;
- exact dispatch selection for width 128;
- unchanged dispatch for widths 127, 129, 256, 512, and 1024;
- the complete existing Softmax accuracy suite before retention.

## Performance Experiment

Create separate baseline and candidate directories on the Ascend 950PR node.
Both use the same device, CANN 9.2.0 toolchain, frequency, dtype, manifests,
CannBench default parameters, and BasicInfo timing boundary. Commands do not
set warmup or iteration overrides.

For each build, record:

- source commit and working-tree diff;
- imported package and extension paths;
- host extension SHA-256;
- selected kernel name and launch count;
- current and rated device frequency;
- raw profiler directory and parsed CannBench records.

The first pass measures the five width-128 cases. A second fresh-process pass
is required for any candidate close to the retention boundary.

## Retention Gates

Retain the specialized path only when:

- all required FP16 and FP32 correctness checks pass;
- `gptneo_attention`, `mobilebert_attention`, and `pegasus_attention` each show
  a repeatable improvement over the current V3 baseline;
- `gptj_attention` has no material regression; use a measured shape threshold
  if pipeline fixed cost makes the small case slower;
- the complete realistic Softmax suite has no unexplained regression;
- imported extension provenance matches the intended source directory.

Do not retain a candidate based only on compiler resources or one profiler
sample. Compiler register and Stack reports are diagnostic evidence; repeated
device time and correctness decide retention.

## Measured Result

The retained candidate was built on `Ascend950PR_9589` with CANN 9.2.0,
Bisheng 15.0.5, `dav-3510`, and `torch_npu 2.11.0.dev20260414`. No CannBench
warmup, iteration, or metric overrides were passed. The selected kernel count
was one for every target case, and every retained row reported 1650 MHz current
and rated frequency.

| Case | Baseline R1 | Candidate R1 | Baseline R2 | Candidate R2 |
| --- | ---: | ---: | ---: | ---: |
| `bert_pytorch_attention` | 20.312 us | 11.059 us | 19.987 us | 11.187 us |
| `gptj_attention` | 5.428 us | 3.997 us | 5.396 us | 4.132 us |
| `gptneo_attention` | 48.986 us | 24.471 us | 49.090 us | 24.407 us |
| `mobilebert_attention` | 48.733 us | 24.689 us | 48.632 us | 24.383 us |
| `pegasus_attention` | 48.298 us | 24.636 us | 48.412 us | 24.486 us |

The candidate passed 40/40 FP16 canonical accuracy cases. FP32 `(63,128)`
passed with `max_abs_error=7.45e-09`; FP16 `(65,128)` matched exactly.

Provenance and raw evidence:

- baseline remote root: `/root/cannbench-softmax-dim128-baseline-Ptf8R0`;
- candidate remote root: `/root/cannbench-softmax-dim128-single-row-jFxg8x`;
- baseline extension SHA-256: `d8edbb47a67dd98f89e0a521418fadf6648b5f54f43391b3b2b5d44ef9a8806f`;
- candidate extension SHA-256: `4d3c14450189a3c0e36aecf02eac9a96bea67ea7ce81f4cf035500c6cd14bf49`;
- paired R1 local roots: `/tmp/cannbench-softmax-dim128-single-row-r1-5j1Cd1`
  and `/tmp/cannbench-softmax-dim128-baseline-clean-r2-0XEsho`;
- paired R2 local roots: `/tmp/cannbench-softmax-dim128-baseline-paired-r2-QpDC69`
  and `/tmp/cannbench-softmax-dim128-single-row-r2-3hRGxi`.

## Publication

After acceptance, update only the affected records in:

```text
published/opbench-ascend-950pr-simt-v3-softmax-realistic-float16/meta/benchmark-records.json
```

Preserve record ordering, canonical run IDs, accuracy fields, and all unrelated
records. Raw profiler artifacts remain in the isolated local and remote run
directories and are not copied into `published/`.
