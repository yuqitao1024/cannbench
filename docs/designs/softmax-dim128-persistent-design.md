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

One 32-lane warp owns two rows. Each lane owns four elements per row, so a
1024-thread VF processes 64 rows per tile. This preserves the useful row
parallelism of the current generic path while replacing lane-level GM accesses
with aligned bulk transfers.

The input and output double buffers each hold two `64 x 128` FP16/FP32 tiles.
The maximum combined allocation is 128 KiB for FP32 and remains below the
available 224 KiB UB budget. The FP16 path uses 64 KiB.

Because the path accepts only width 128, element bounds and warp iteration
counts are compile-time facts. VF loop and address indices use `int32_t`; outer
row and GM offsets remain `int64_t`.

## Dispatch

`row_persistent_fallback.asc` dispatches exact width 128 to the new FP16 or
FP32 entry point. All other dimensions retain their current behavior.

The initial candidate uses the dedicated path for every `outer_size`. If the
remote paired benchmark shows a material regression for `gptj_attention`, an
`outer_size` threshold may select the old generic path for small workloads.
Such a threshold must be derived from measured crossover data and expressed in
terms of shape, never a concrete case name.

## Alternatives Rejected

Extending the existing 256 bucket down to width 128 is simpler but allocates
eight values per lane while only four are live, performs inactive iterations,
and processes 32 rather than 64 rows per tile. It is retained as a quick
control experiment, not the preferred production design.

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
- tail row counts below and not divisible by 64;
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

## Publication

After acceptance, update only the affected records in:

```text
published/opbench-ascend-950pr-simt-v3-softmax-realistic-float16/meta/benchmark-records.json
```

Preserve record ordering, canonical run IDs, accuracy fields, and all unrelated
records. Raw profiler artifacts remain in the isolated local and remote run
directories and are not copied into `published/`.
