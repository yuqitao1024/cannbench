# Softmax Dispatch Shape Notes

本文对比 PyTorch CUDA softmax 前向路径和当前 Ascend SIMT V1 softmax
实现的 shape 分发逻辑。这里只描述 shape 相关分支，不展开 kernel 内部实现。

参考：

- PyTorch CUDA: `aten/src/ATen/native/cuda/SoftMax.cu`
- PyTorch CUDA persistent path: `aten/src/ATen/native/cuda/PersistentSoftmax.cuh`
- Ascend SIMT V1: `aten_softmax/csrc/simt/spatial_softmax.asc`

## PyTorch CUDA Softmax 前向伪代码

```text
cuda_softmax_forward(input, dim, half_to_float, dtype):
    input = contiguous(input)

    如果 input.dim == 0:
        input = view_as_1d(input)

    dim = wrap_dim(dim, input.dim)
    outer_size = input.shape[0:dim] 的乘积
    dim_size = input.shape[dim]
    inner_size = input.shape[dim + 1:] 的乘积

    如果 input.numel == 0:
        返回空 output；如果 half_to_float=true，则 output 用 float

    如果 inner_size == 1:
        # Row-wise softmax。
        # 归约维度是最后一个连续维度。
        # 典型 shape: [rows, vocab]，或 [batch, heads, q, k] 且 dim=-1。

        如果 dim_size <= 2048 且 dim_size * sizeof(input_dtype) <= 8192:
            # 小/中等长度 row。
            # 使用 persistent warp softmax。
            # 一个 warp 处理一行或两行，取决于 dim_size。
            # 编译期变体覆盖 dim_size 向上取 2 的幂，范围 1 到 2048。
            launch persistent_warp_softmax

        否则:
            如果 normal softmax path 启用了 fast softmax:
                # 大 row length，仅 normal softmax 场景。
                # 每个 block 使用 512 threads。
                如果 dim_size 满足 ILP 向量化 load 对齐:
                    launch fast_global_memory_softmax
                否则:
                    launch fast_unaligned_softmax

            否则:
                # 大 row length，通用 row-wise 路径。
                block_x = 向上取 warp multiple，最大不超过 min(dim_size, 1024)

                如果 output_dtype == input_dtype 且预估寄存器压力较小:
                    # row 适合寄存器专用路径。
                    launch register_softmax

                否则如果 row 能放入 shared memory，且 input/output/ILP 对齐:
                    # row 可以暂存在 shared memory。
                    launch shared_memory_softmax

                否则:
                    # 大 row 或非对齐 row 的 fallback 路径。
                    launch generic_row_softmax

    否则:
        # Spatial softmax。
        # 归约维度后面还有非平凡 inner tail。
        # 典型 shape: dim 不是最后一维，例如 [N, C, H, W] 且 dim=1。

        block_y = min(inner_size, 1024)
        block_x = 类似 2 的幂增长的 dim 并行度，
                  要求 block_x * block_y <= 1024，
                  仅在 block_y <= 64 且 dim_size >= 64 时使用
        grid_y = 按 inner_size 分 tile，并受 occupancy 限制
        grid_x = 按 outer_size 分 tile，并受 occupancy 限制

        launch spatial_softmax
```

## Ascend SIMT V1 Softmax 前向伪代码

```text
simt_v1_softmax_forward(input, dim, half_to_float, dtype):
    converted = input

    如果 dtype 被指定且 half_to_float=false:
        converted = input.to(dtype)

    如果 converted.dim <= 0:
        fallback 到 at::_softmax

    # V1 当前把所有非标量输入都视为可走 SIMT 路径。
    # should_use_spatial_softmax_path 只检查 inner_size >= 1，
    # 对合法 tensor shape 来说这个条件恒成立。
    dispatch 到 simt_v1_spatial_softmax_forward(converted, dim, half_to_float)


simt_v1_spatial_softmax_forward(input, dim, half_to_float):
    input = contiguous(input)
    dim = wrap_dim(dim, input.dim)
    outer_size = input.shape[0:dim] 的乘积
    dim_size = input.shape[dim]
    inner_size = input.shape[dim + 1:] 的乘积

    如果 input.numel == 0:
        返回空 output；如果 half_to_float=true，则 output 用 float

    如果 half_to_float:
        # 只接受 Half 输入。
        # 输入先转成 float，输出 dtype 也是 float。
        compute_input = input.to(float)

        如果 inner_size == 1:
            # Row-wise float output 路径。
            launch row_softmax_forward，float input/output
        否则:
            # Spatial float output 路径。
            launch spatial_softmax_forward，float input/output

    否则:
        如果 input dtype 是 float16:
            如果 inner_size == 1:
                # Row-wise fp16 output 路径。
                # block_x = next_power_of_two(min(dim_size, 32))
                # grid_x = min(outer_size, 32768)
                launch row_softmax_forward，fp16 input/output，float accumulation
            否则:
                # Spatial fp16 output 路径。
                # block_y = min(inner_size, 1024)
                # block_x 随 dim_size 增长，但要求 block_x * block_y <= 1024
                # grid_x/grid_y 按 outer_size 和 inner_size 分 tile，并结合 occupancy 估计
                launch spatial_softmax_forward，fp16 input/output，float accumulation

        否则如果 input dtype 是 float32:
            如果 inner_size == 1:
                # Row-wise fp32 路径。
                # block/grid 策略和 fp16 row-wise 一样。
                launch row_softmax_forward，fp32 input/output
            否则:
                # Spatial fp32 路径。
                # block/grid 策略和 fp16 spatial 一样。
                launch spatial_softmax_forward，fp32 input/output

        否则:
            reject unsupported dtype
```

## 主要分发差异

```text
CUDA:
    inner_size == 1 有多条 row-wise 子路径:
        dim_size <= 2048 且 row bytes 较小时，走 persistent warp path
        大 normal softmax 场景，可能走 fast global-memory path
        合适的大 row，可能走 register path
        对齐且能放入 shared memory 的 row，可能走 shared-memory path
        其他大 row 或非对齐 row，走 generic row-wise fallback

SIMT V1:
    inner_size == 1 只有一条 row-wise 路径:
        block_x <= 32, grid_x <= 32768

CUDA:
    inner_size > 1 走 spatial path。

SIMT V1:
    inner_size > 1 使用接近 CUDA 的 spatial shape 策略:
        block_y 跟随 inner_size
        仅当 inner_size 足够小时，block_x 才跟随 dim_size 增长
        block_x * block_y <= 1024

CUDA:
    dtype 和 half_to_float 会选择输出 dtype 和 kernel template，
    但不会移除 row-wise 子路径的多样性。

SIMT V1:
    dtype 和 half_to_float 会选择 fp16/fp32 输入输出处理，
    但 row-wise 仍然只有一条 SIMT kernel 路径。
```

## 线程和网格配置对比

| Shape 场景 | PyTorch CUDA | Ascend SIMT V1 | 配置是否相同 |
| --- | --- | --- | --- |
| `inner_size == 1`, `dim_size <= 2048` | Persistent warp softmax。一个 warp 处理一行或两行，取决于 `dim_size`。 | 单一 row-wise SIMT kernel。`block_x <= 32`，`grid_x <= 32768`。 | 不相同。高层场景一致，但 CUDA 使用 persistent warp 变体，SIMT V1 使用固定 row-wise 策略。 |
| `inner_size == 1`, 大 row | 多条可能路径：512-thread fast path、register path、shared-memory path、generic row-wise path。通用路径里 `block_x` 可增长到 `1024`。 | 单一 row-wise SIMT kernel。`block_x <= 32`，`grid_x <= 32768`。 | 不相同。这是当前最大差异。 |
| `inner_size > 1`, spatial softmax | `block_y = min(inner_size, 1024)`。`block_x * block_y <= 1024`。`grid_x` 和 `grid_y` 按 `outer_size` 与 `inner_size` 分 tile，并受 occupancy 限制。 | 形态接近的 spatial 策略。`block_y = min(inner_size, 1024)`。`block_x * block_y <= 1024`。`grid_x` 和 `grid_y` 按 `outer_size` 与 `inner_size` 分 tile，并结合 occupancy 估计。 | 形态接近，但不保证完全相同，因为 CUDA 和 Ascend 的 occupancy 与 launch 约束不同。 |
| dtype / `half_to_float` 分支 | 选择输出 dtype 和 kernel template，但不会移除 CUDA 的 row-wise 子路径多样性。 | 选择 fp16/fp32 输入输出处理，但 row-wise 仍然只有一条 SIMT kernel 路径。 | 不相同。高层 dtype 行为类似，但 row-wise dispatch 多样性不同。 |

关键结论：主要差异在 row-wise softmax。大多数 realistic attention 和
logits case 都是 `inner_size == 1`，CUDA 可以从多条 row-wise 优化 kernel
中选择，而 SIMT V1 当前只使用一条 `block_x <= 32` 的 row-wise SIMT kernel。

## V2 第一步 Row-Wise 拆分

V2 先把 row-wise 路径拆成 CUDA 风格的分发分支：

```text
inner_size == 1:
    如果 dim_size <= 2048 且 dim_size * sizeof(dtype) <= 8192:
        row_softmax_persistent_forward
    否则如果选择 large-row fast path:
        row_softmax_fast_forward
    否则:
        row_softmax_generic_forward
```

V2 已对齐所有 row-wise 路径的 dispatch 和 launch policy。其中 persistent
路径也已对齐 CUDA persistent kernel 的内部结构：

| V2 路径 | Launch policy | 目的 |
| --- | --- | --- |
| `row_softmax_persistent_forward` | `block_x <= 32`, `block_y = 128 / block_x`, `WARP_BATCH = 2 if next_power_of_two <= 128 else 1` | 对齐 CUDA persistent softmax：`log2_elements` dispatch、寄存器驻留 elements、`asc_shfl_xor` warp reduction。 |
| `row_softmax_fast_forward` | `block_x = 32`, `block_y = 16` | 使用 CUDA-like 512 总线程 fast path 形态，同时把每行归约限制在一个 32-lane group 内。 |
| `row_softmax_generic_forward` | `block_x = 32`, `block_y = 32` | 使用 CUDA-like 1024 总线程 generic 形态，同时把每行归约限制在一个 32-lane group 内。 |

后续 V2 迭代再逐步替换 fast/generic 的 correctness-first 内部实现，包括
block-wide reduction、ILP/vectorized load，以及 Ascend SIMT 支持场景下的
shared/register row buffering。

spatial 路径保持 CUDA `cunn_SpatialSoftMaxForward` 结构：`block.x` 沿
softmax 维度做归约，`block.y` 覆盖独立 inner 位置，`grid.x/grid.y`
分别切分 outer 和 inner 范围。
