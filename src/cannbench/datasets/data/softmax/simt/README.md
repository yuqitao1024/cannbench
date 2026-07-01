# Softmax Dispatch Shape Notes

This note compares the forward dispatch paths in PyTorch CUDA softmax and the
current Ascend SIMT V1 softmax implementation. It only describes shape-based
branching. Kernel internals are intentionally omitted.

References:

- PyTorch CUDA: `aten/src/ATen/native/cuda/SoftMax.cu`
- PyTorch CUDA persistent path: `aten/src/ATen/native/cuda/PersistentSoftmax.cuh`
- Ascend SIMT V1: `aten_softmax/csrc/simt/spatial_softmax.asc`

## PyTorch CUDA Softmax Forward Pseudocode

```text
cuda_softmax_forward(input, dim, half_to_float, dtype):
    input = contiguous(input)
    if input.dim == 0:
        input = view_as_1d(input)

    dim = wrap_dim(dim, input.dim)
    outer_size = product(input.shape[0:dim])
    dim_size = input.shape[dim]
    inner_size = product(input.shape[dim + 1:])

    if input.numel == 0:
        return empty output, using float output when half_to_float is true

    if inner_size == 1:
        # Row-wise softmax. The reduction dimension is the innermost contiguous
        # dimension. Typical shapes: [rows, vocab], [batch, heads, q, k] with dim=-1.

        if dim_size <= 2048 and dim_size * sizeof(input_dtype) <= 8192:
            # Small/medium row length.
            # Persistent warp softmax.
            # One warp processes one or two rows depending on dim_size.
            # Compiled variants cover dim_size rounded up to powers of two from 1 to 2048.
            launch persistent_warp_softmax

        else:
            if normal softmax path enables fast softmax:
                # Large row length, normal softmax only.
                # Uses 512 threads per block.
                if dim_size is aligned for ILP vector loads:
                    launch fast_global_memory_softmax
                else:
                    launch fast_unaligned_softmax

            else:
                # Large row length, generic row-wise path.
                block_x = next warp-multiple up to min(dim_size, 1024)

                if output_dtype == input_dtype and estimated register count is small:
                    # Row fits the register-specialized path.
                    launch register_softmax

                else if row fits shared memory and input/output are aligned and ILP-aligned:
                    # Row can be staged in shared memory.
                    launch shared_memory_softmax

                else:
                    # Fallback for large or unaligned row-wise shapes.
                    launch generic_row_softmax

    else:
        # Spatial softmax. The reduction dimension has a non-trivial inner tail.
        # Typical shapes: dim is not the last dimension, for example channel-wise
        # activation maps [N, C, H, W] with dim=1.

        block_y = min(inner_size, 1024)
        block_x = largest power-of-two-like dim parallelism such that
                  block_x * block_y <= 1024,
                  used only when block_y <= 64 and dim_size >= 64
        grid_y = tiles over inner_size, capped by occupancy
        grid_x = tiles over outer_size, capped by occupancy

        launch spatial_softmax
```

## Ascend SIMT V1 Softmax Forward Pseudocode

```text
simt_v1_softmax_forward(input, dim, half_to_float, dtype):
    converted = input
    if dtype is provided and not half_to_float:
        converted = input.to(dtype)

    if converted.dim <= 0:
        fallback to at::_softmax

    # V1 currently treats every non-scalar input as supported by the SIMT path.
    # should_use_spatial_softmax_path only checks inner_size >= 1, which is always
    # true for valid tensor shapes.
    dispatch to simt_v1_spatial_softmax_forward(converted, dim, half_to_float)


simt_v1_spatial_softmax_forward(input, dim, half_to_float):
    input = contiguous(input)
    dim = wrap_dim(dim, input.dim)
    outer_size = product(input.shape[0:dim])
    dim_size = input.shape[dim]
    inner_size = product(input.shape[dim + 1:])

    if input.numel == 0:
        return empty output, using float output when half_to_float is true

    if half_to_float:
        # Only Half input is accepted.
        # Input is first converted to float, and output dtype is float.
        compute_input = input.to(float)

        if inner_size == 1:
            # Row-wise float output path.
            launch row_softmax_forward with float input/output
        else:
            # Spatial float output path.
            launch spatial_softmax_forward with float input/output

    else:
        if input dtype is float16:
            if inner_size == 1:
                # Row-wise fp16 output path.
                # block_x is next power of two up to min(dim_size, 32).
                # grid_x is min(outer_size, 32768).
                launch row_softmax_forward with fp16 input/output and float accumulation
            else:
                # Spatial fp16 output path.
                # block_y = min(inner_size, 1024)
                # block_x grows with dim_size while block_x * block_y <= 1024
                # grid_x/grid_y tile outer_size and inner_size using occupancy estimate.
                launch spatial_softmax_forward with fp16 input/output and float accumulation

        else if input dtype is float32:
            if inner_size == 1:
                # Row-wise fp32 path.
                # Same row-wise block/grid policy as fp16.
                launch row_softmax_forward with fp32 input/output
            else:
                # Spatial fp32 path.
                # Same spatial block/grid policy as fp16.
                launch spatial_softmax_forward with fp32 input/output

        else:
            reject unsupported dtype
```

## Main Dispatch Differences

```text
CUDA:
    inner_size == 1 has multiple row-wise subpaths:
        persistent warp path for dim_size <= 2048 and small row bytes
        fast global-memory path for large normal softmax
        register path for suitable large rows
        shared-memory path for aligned rows that fit shared memory
        generic row-wise fallback

SIMT V1:
    inner_size == 1 has one row-wise path:
        block_x <= 32, grid_x <= 32768

CUDA:
    inner_size > 1 uses spatial path.

SIMT V1:
    inner_size > 1 uses a CUDA-like spatial path shape:
        block_y follows inner_size
        block_x follows dim_size only when inner_size is small enough
        block_x * block_y <= 1024

CUDA:
    dtype and half_to_float select output dtype and kernel template, but do not
    remove the row-wise subpath diversity.

SIMT V1:
    dtype and half_to_float select fp16/fp32 input-output handling, but row-wise
    still has only one SIMT kernel path.
```
