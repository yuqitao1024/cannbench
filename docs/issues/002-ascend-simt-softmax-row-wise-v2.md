# Issue 002: Ascend SIMT softmax row-wise path and V2 follow-up

## Status

Open

## Context

The first Ascend SIMT softmax implementation was originally ported from the
PyTorch CUDA spatial softmax kernel shape. It used one implementation for all
flattened softmax layouts:

```text
outer_size x dim_size x inner_size
```

The initial implementation worked for small reductions, but large realistic
attention shapes exposed correctness problems when `dim_size >= 64`. A typical
failing case was:

```text
case: t5_attention
shape: (4, 8, 1024, 1024)
dim: -1
flattened: outer_size=32768, dim_size=1024, inner_size=1
```

The old path launched the spatial kernel even when `inner_size == 1`. That is
not how PyTorch CUDA dispatches softmax.

## CUDA Reference Behavior

PyTorch CUDA separates the dispatch:

- `inner_size == 1`: use a row-wise softmax path. For common transformer
  attention and logits shapes, this goes through persistent/warp-style softmax
  implementations such as `dispatch_softmax_forward` in
  `PersistentSoftmax.cuh`, or other row-wise fast paths.
- `inner_size != 1`: use the spatial softmax kernel, where the 2-D launch
  parallelizes both outer and inner dimensions and only uses `threadIdx.x` to
  reduce over the softmax dimension.

The important design point is that PyTorch CUDA does not use the spatial kernel
as the universal softmax implementation. Large `dim=-1` attention cases are
row-wise cases, not spatial cases.

## V1 Change Implemented

V1 now adds a dedicated row-wise forward kernel:

```text
row_softmax_forward_kernel
```

The dispatch is split as follows:

```text
inner_size == 1  -> row_softmax_forward_kernel
inner_size != 1  -> existing spatial softmax kernel
```

The row-wise kernel:

- treats each row as one softmax application;
- computes max in float accumulation type;
- computes exp/sum in float accumulation type;
- writes fp16 output for the fp16 no-cast path;
- keeps the previous fp32 output behavior for `half_to_float=True`;
- limits `threadIdx.x` to a CUDA-style warp-lane shape with
  `kCudaWarpLaneLimit = 32`;
- caps row-wise `grid.x` with `kRowSoftmaxGridXLimit = 32768`.

The `32`-lane limit was chosen because CUDA persistent softmax treats
`threadIdx.x` as a warp lane, not as a full 1024-thread block. A direct
1024-thread block reduction on Ascend SIMT produced finite but incorrect
outputs for the large row-wise case.

The `32768` grid cap was accepted as the V1 row-wise launch policy after
correctness testing. The kernel already strides rows by `gridDim.x`, so the cap
does not reduce coverage. It avoids oversized one-block-per-row launches for
large `outer_size` cases and lets each block process multiple rows.

## Validation Evidence

### Correctness fixed for the original large case

After adding the row-wise path and using the 32-lane row reduction, the original
large fp16 case passed accuracy comparison against the CANN ops library:

```text
case: t5_attention
shape: (4, 8, 1024, 1024)
dim: -1
dtype: float16
allclose: True
max_abs_error: 7.6293945e-06
max_rel_error: 0.001814882
nan: 0
inf: 0
```

The same case also passed for fp32:

```text
case: t5_attention
shape: (4, 8, 1024, 1024)
dim: -1
dtype: float32
allclose: True
max_abs_error: 7.4505806e-09
max_rel_error: 3.4987272e-07
nan: 0
inf: 0
```

### Profiler target validated

`msprof op` confirmed that the SIMT test was profiling the SIMT row-wise
kernel, not CANN library softmax and not a Cast kernel:

```text
row_softmax_forward_kernel
```

For `t5_attention` fp16, the measured device-side durations were:

```text
CANN ops library kernel: SoftmaxV2_float16_high_precision_1000
Task Duration(us): 58.563999

SIMT kernel: row_softmax_forward_kernel
Task Duration(us): 6661.099121
```

This confirms that V1 is now measuring the intended SIMT kernel, but the
current SIMT implementation is still far slower than the optimized CANN ops
library implementation.

## Additional Finding: grid.x cap

Some realistic row-wise cases still failed when the launch used:

```cpp
<<<dim3(outer_size), dim3(block_x), ...>>>
```

Examples:

```text
electra_attention: outer_size=65536, dim_size=512, inner_size=1
layoutlm_attention: outer_size=98304, dim_size=512, inner_size=1
levit_global_attention: outer_size=802816, dim_size=196, inner_size=1
```

The symptom was:

```text
SIMT output finite: yes
nan: 0
inf: 0
finite_min: 0
finite_max: 0
row_sum_error: 1
```

That indicates the kernel completed but did not correctly write the expected
softmax output.

The accepted V1 policy caps `grid.x` at `32768`:

```cpp
<<<dim3(std::min<int64_t>(outer_size, 32768)), dim3(block_x), ...>>>
```

The kernel already loops over rows:

```cpp
for (int64_t row = blockIdx.x; row < outer_size; row += gridDim.x)
```

So capping `grid.x` does not reduce the amount of work. It only reduces the
number of simultaneously launched blocks and lets each block process multiple
rows.

With this grid cap, all 30 softmax realistic fp16 cases passed accuracy
comparison:

```text
summary: total=30 passed=30 failed=0
```

The cap should still be revisited in V2. It is a correctness-preserving launch
policy for V1, but it is not yet a principled occupancy model.

## Accuracy Regression Script

The repository includes a reusable accuracy validation script:

```text
scripts/ascend_softmax_accuracy.py
```

It compares CANN ops library softmax and the installed SIMT softmax operator on
Ascend. The script is self-contained and hard-codes the canonical 40 softmax
cases, so it can be copied to an Ascend machine and run without the dataset JSON
files.

Recommended V2 correctness gate:

```bash
python scripts/ascend_softmax_accuracy.py \
  --mode both \
  --dataset ALL \
  --case ALL \
  --dtype float16 \
  --warmup 0 \
  --iters 1
```

Dataset coverage:

```text
smoke: 3 cases
realistic: 30 cases
stress: 7 cases
total: 40 cases
```

The script prints one CSV-style row per case and returns a non-zero exit code if
any case fails `torch.allclose` with the configured tolerances.

## Open Questions for V2

- What is the correct maximum `grid.x` for Ascend SIMT row-wise operators?
- Should V2 keep the fixed `32768` cap, or derive the limit from vector core
  count, occupancy, and active block limits?
- Is the all-zero output caused by a hardware/runtime launch limit, compiler
  behavior, or a missing launch validation check?
- Can row-wise softmax use a more CUDA-like persistent layout with multiple rows
  per block instead of one logical row per block?
- Can row-wise reductions keep intermediate values in registers/UBUF more
  efficiently to close the large performance gap versus CANN ops library?

## V2 Direction

V2 should not simply keep optimizing the current V1 row kernel in place. It
should explicitly design the row-wise softmax launch policy:

1. Use the PyTorch CUDA dispatch split as the reference:
   `inner_size == 1` and `inner_size != 1` need different kernels.
2. Add an Ascend-specific row-wise launch policy that caps or tiles `outer_size`
   based on measured safe limits. V1 currently uses `32768` as the row-wise
   `grid.x` cap.
3. Preserve correctness first by requiring all 30 realistic fp16 cases to pass.
4. Use `scripts/ascend_softmax_accuracy.py` as the V2 correctness regression
   gate across smoke, realistic, and stress cases.
5. Add profiler validation to ensure SIMT runs profile the SIMT kernel name.
6. Then optimize performance against:
   - CANN ops library softmax on Ascend 950PR;
   - PyTorch CUDA softmax on H800;
   - previous SIMT versions.

## Current Policy

Treat V1 as a correctness-oriented baseline for the row-wise path, not as the
final performance design. The V1 row-wise policy is:

```text
threadIdx.x lane cap: 32
grid.x cap: 32768
```

Use this issue and `scripts/ascend_softmax_accuracy.py` as the input checklist
for the next SIMT softmax V2 implementation.
