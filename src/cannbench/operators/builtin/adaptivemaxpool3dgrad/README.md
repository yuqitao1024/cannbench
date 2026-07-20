# AdaptiveMaxPool3DGrad

`AdaptiveMaxPool3DGrad` 是三维自适应最大池化
`AdaptiveMaxPool3D` 的反向梯度算子。它根据前向池化时记录的最大值位置，
把输出梯度写回原输入张量对应的位置，从而得到 `grad_input`。

## 功能说明

前向算子 `AdaptiveMaxPool3D` 接收形状如下的输入张量：

```text
[N, C, D_in, H_in, W_in]
```

并输出固定空间大小的张量：

```text
[N, C, D_out, H_out, W_out]
```

对输出张量中的每一个空间坐标，前向过程都会在输入张量中选择一个池化区域，
取该区域的最大值写入 `output`，并把最大值在原输入中的位置记录到
`indices`。

反向算子 `AdaptiveMaxPool3DGrad` 使用这些位置记录完成梯度回传：

```text
grad_output + indices -> grad_input
```

其中：

- `grad_output` 的形状是 `[N, C, D_out, H_out, W_out]`。
- `indices` 的形状是 `[N, C, D_out, H_out, W_out]`。
- `grad_input` 的形状是 `[N, C, D_in, H_in, W_in]`。

`grad_output` 中的每个值都会根据同坐标的 `indices` 写回 `grad_input`
中的指定位置。前向过程中没有被选为最大值的位置，其梯度为 0。

## 池化区域规则

自适应池化不是直接指定固定的 `kernel_size`，而是指定输出大小
`output_size`，再由算子反推每个输出位置对应的输入区域。

对任意一个空间维度，第 `i` 个输出位置对应的输入区间为：

```text
start = floor(i * input_size / output_size)
end   = ceil((i + 1) * input_size / output_size)
```

区间是 `[start, end)`，即包含 `start`，不包含 `end`。

三维池化会分别对 `D`、`H`、`W` 三个维度应用这条规则，三个维度的区间组合
起来就是一个三维池化区域。

## indices 的含义

对每个 `[N, C]` 切片，`indices` 通常保存最大值在原始
`[D_in, H_in, W_in]` 空间内展开成一维后的下标：

```text
linear_index = d * H_in * W_in + h * W_in + w
```

因此，`indices` 中保存的不是最大值本身，而是前向过程中被选中的最大值位置。

## 示例

假设有一个 `[D, H, W] = [2, 2, 2]` 的输入切片：

```text
D = 0:
[[9, 2],
 [7, 1]]

D = 1:
[[3, 8],
 [4, 6]]
```

当 `output_size = (1, 2, 2)` 时，前向输出为：

```text
[[[9, 8],
  [7, 6]]]
```

原输入空间展开成一维后的下标为：

```text
D = 0:
[[0, 1],
 [2, 3]]

D = 1:
[[4, 5],
 [6, 7]]
```

因此前向过程记录的最大值位置为：

```text
indices =
[[[0, 5],
  [2, 7]]]
```

如果反向时后一层传回来的梯度为：

```text
grad_output =
[[[10, 20],
  [30, 40]]]
```

则 `AdaptiveMaxPool3DGrad` 会把这些梯度写回 `indices` 指定的位置：

```text
grad_input:

D = 0:
[[10, 0],
 [30, 0]]

D = 1:
[[0, 20],
 [0, 40]]
```

总结来说，`AdaptiveMaxPool3DGrad` 不需要重新计算最大值。它依赖前向过程
保存的 `indices`，把 `grad_output` 路由回原输入张量的对应位置。

## 目录文件说明

当前目录是 `adaptivemaxpool3dgrad` 在 CannBench 中的算子插件目录。各文件
职责如下：

```text
adaptivemaxpool3dgrad/
  __init__.py
  cases.py
  materialize.py
  README.md
  cpu_demo/
    CMakeLists.txt
    main.cpp
  cuda_source/
    AdaptiveMaxPooling3d.cu
  cann_source/
    op_kernel/
    op_host/
    op_api/
  data/
    smoke.json
    realistic.json
    stress.json
  simt/
    v1/
    v2/
    v3/
    v4/
    v5/
    test/
  test/
    test_adaptivemaxpool3dgrad_plugin.py
```

### `__init__.py`

负责把该算子注册为 CannBench 的 `OperatorPlugin`。

这个文件中定义了：

- 算子名：`adaptivemaxpool3dgrad`
- 支持的数据类型：`float32`、`float16`、`bfloat16`
- 数据集加载函数
- 输入生成函数
- PyTorch baseline callable
- profile kernel 匹配规则

当前 callable 会按 `AdaptiveMaxPool3D` 的前向池化规则在 CannBench 侧生成合法
的 `indices`，然后 benchmark PyTorch 暴露的默认反向算子：

```text
aten::adaptive_max_pool3d_backward
```

在 NVIDIA backend 下，这条路径对应 PyTorch 默认 CUDA 实现。在 Ascend backend
且 `--implementation cann_ops_library` 时，这条路径会通过 `torch_npu` 调用
CANN 默认实现。

根据 CANN ops-info，`AdaptiveMaxPool3DGrad` 的 CANN 默认实现要求：

- `input/self`、`grad_output`、`grad_input` 支持 `float16`、`float32`、`bfloat16`
- `indices/argmax` 使用 `int32`
- 数据格式为 `NCDHW`

因此 Ascend 路径会在调用 backward 前把 `indices` 转为 `int32`。`indices`
不依赖 Ascend 前向池化算子生成，避免把前向 `AdaptiveMaxPool3D` 的性能或兼容性
混入 `AdaptiveMaxPool3DGrad` 的反向 benchmark。

常用运行命令：

```bash
python -m cannbench bench \
  --backend nvidia \
  --op adaptivemaxpool3dgrad \
  --dataset realistic \
  --case-id ct_resnet_global_pool_backward \
  --dtype float16 \
  --output-dir runs
```

```bash
python -m cannbench bench \
  --backend ascend \
  --implementation cann_ops_library \
  --op adaptivemaxpool3dgrad \
  --dataset realistic \
  --case-id ct_resnet_global_pool_backward \
  --dtype float16 \
  --output-dir runs
```

### `cases.py`

负责定义 benchmark case 的数据结构和数据集加载逻辑。

每个 case 描述一次要测试的形状和来源信息，例如：

```text
input_shape = [N, C, D_in, H_in, W_in]
output_size = [D_out, H_out, W_out]
```

它还会校验 case 是否合法，例如：

- `input_shape` 必须是 5 维 `[N, C, D, H, W]`
- `output_size` 必须是 3 维 `[D_out, H_out, W_out]`
- `output_size` 不能超过输入张量的空间维度

CannBench 运行时会通过 `cases.py` 从 `data/*.json` 中找到指定的 case。

### `materialize.py`

负责根据 case 生成实际输入数据。

对该算子来说，主要生成：

- `input_values`：前向池化使用的输入张量数据
- `grad_output_values`：反向传播时后一层传回来的输出梯度
- `indices`：按 adaptive max pool 前向规则计算出的最大值位置
- `input_shape`
- `output_shape`
- `output_size`

`indices` 在每个 `[N, C]` 切片内按 `[D, H, W]` 展开后一维坐标保存。这样
benchmark 只测试反向算子本身，不把前向 `AdaptiveMaxPool3D` 的运行时间混入
`AdaptiveMaxPool3DGrad`。

### `cpu_demo/`

提供一个纯 C++ 的 CPU 教学版本，不依赖 PyTorch、CANN 或 CannBench 运行框架。

它演示 `AdaptiveMaxPool3DGrad` 的核心反向逻辑：

```text
grad_input[indices[out_pos]] += grad_output[out_pos]
```

其中 `indices[out_pos]` 是在对应 `[N, C]` 切片内按 `[D, H, W]` 展开后的一维
坐标。示例还包含重复 `indices` 的情况，用来说明为什么反向实现需要累加；在
CUDA 中，池化区域可能重叠时这个累加通常需要 atomic add。

编译和运行：

```bash
cmake -S src/cannbench/operators/builtin/adaptivemaxpool3dgrad/cpu_demo \
  -B /tmp/adaptivemaxpool3dgrad_cpu_demo
cmake --build /tmp/adaptivemaxpool3dgrad_cpu_demo -j
/tmp/adaptivemaxpool3dgrad_cpu_demo/adaptivemaxpool3dgrad_cpu_demo
```

### `data/smoke.json`

保存小规模测试用例，用于快速验证插件注册、case 加载、输入生成和 callable
构造是否正常。

### `data/realistic.json`

保存更接近生产场景的常用形状，例如：

- 医学 CT/MRI 3D CNN 的全局池化反向
- 视频理解模型的时空特征池化反向
- 3D backbone 中间特征池化反向
- 3D 分割模型上下文 head 的池化反向
- 体素特征的全局池化反向

这个数据集是默认用于观察 PyTorch CUDA baseline 性能的主要集合。

### `data/stress.json`

保存更大的边界或压测形状，用于观察大输入、大通道数或较大输出尺寸下的性能。

### `test/test_adaptivemaxpool3dgrad_plugin.py`

保存该算子的 operator-local 测试。

测试内容包括：

- realistic 数据集是否包含预期生产形状
- materialize 生成的数据长度是否匹配输入/输出形状
- `adaptivemaxpool3dgrad` 是否注册为可用的 CannBench operator plugin

## SIMT 自定义实现版本演进

`simt/v1` 到 `simt/v5` 使用相同的 Python 调用接口，核心计算都是根据
`indices` 将 `grad_output` scatter 回 `grad_input`。版本迭代主要围绕窗口重叠
处理、确定性计算和 kernel block 配置展开。

| 版本 | 主要变化 | gradient kernel block 策略 |
| --- | --- | --- |
| v1 | 基础 atomic scatter 实现 | `min(ceil(output_numel / 1024), 64)` |
| v2 | 增加 overlap/non-overlap 和 deterministic 分支 | 非 deterministic 使用 `min(output_numel, 64)`；deterministic 使用 `min(ceil(N*C / 512), 64)` |
| v3 | 按实际 gradient 工作量计算 block 数 | `min(ceil(work_items / 512), 64)` |
| v4 | 提高 gradient kernel 的 block 并行度 | `min(ceil(work_items / 128), 64)` |
| v5 | `empty_like` 后由设备清零，再执行 v3 scatter | 清零与 scatter 分别按工作量计算，最多 64 block |

表中的 `work_items` 在非 deterministic 模式下是 `output_numel`，在
deterministic 模式下是 `N * C`。所有版本都会保证至少启动一个 block。

### v1：基础 atomic scatter

v1 是最直接的实现。每个线程处理一个或多个 `grad_output` 元素，计算对应的
`[N, C]` 切片和输入空间下标，然后无条件执行：

```cpp
asc_atomic_add(&grad_input[input_offset], grad_output[out_offset]);
```

这一版的主要特点是：

- kernel 使用 1024 个线程/block，最多启动 64 个 block；
- 无论池化窗口是否重叠都使用 atomic add；
- 不区分 deterministic 和 non-deterministic 执行模式；
- kernel 内的元素数量和空间下标使用 `int32_t`；
- host 侧通过 `at::zeros_like` 创建并清零 `grad_input`，自定义扩展只启动
  scatter kernel。

v1 的优点是实现简单，并且 atomic add 可以正确处理重复 `indices`。局限是非重叠
场景也承担 atomic 开销，重叠场景的浮点累加顺序不固定，同时 `int32_t` 下标限制了
超大 tensor 的寻址范围。

### v2：增加 overlap 和 deterministic 路径

v2 将 CANN arch35 SIMT 分支的核心调度方式迁移到 CannBench 扩展。相对 v1，
主要变化包括：

- 线程数从 1024 调整为 512；
- offset 和元素数量计算改为 `int64_t`/`uint64_t`；
- 根据 `D/H/W` 输入尺寸能否被输出尺寸整除判断池化窗口是否可能重叠；
- 非重叠场景直接执行普通 `+=`，避免不必要的 atomic add；
- 重叠且未要求 deterministic 时继续使用 atomic add；
- 重叠且要求 deterministic 时，一个线程负责一个 `[N, C]` 切片，并按固定的
  输出空间顺序完成累加。

v2 的 deterministic 路径避免了多个线程对同一地址进行无序浮点累加，但并行度
受 `N * C` 限制。其 gradient block 配置为：

```text
non-deterministic: block_num = clamp(output_numel, 1, 64)
deterministic:     block_num = clamp(ceil(N * C / 512), 1, 64)
```

小输出的 non-deterministic 场景没有先除以 512，可能为 gradient kernel 启动
过多 block。v2 最初曾在 `at::zeros_like` 之后再次运行
`init_output_zero_kernel`；后续统一采用方案 A，保留 host 侧 `at::zeros_like`，并
从 v2-v4 删除了重复的设备清零 kernel。

### v3：修正梯度计算的 block 策略

v3 不改变 v2 的 scatter、overlap 和 deterministic 算法，主要修正 kernel launch
配置。gradient block 数改为按实际工作量除以每 block 线程数：

```text
grad_work_items = deterministic ? N * C : output_numel
grad_block_num = clamp(ceil(grad_work_items / 512), 1, 64)
```

相对 v2，v3 解决了小输出时 gradient block 过度启动的问题。设备侧 gradient
kernel 的计算逻辑没有变化。方案 A 清理后，v3 不再包含 `zero_block_num`，输出
清零完全由 host 侧 `at::zeros_like` 负责。

### v4：提高 gradient kernel 的 block 并行度

v4 是针对 gradient kernel block 数的调度实验。相对 v3，唯一影响计算调度的源码
变化是：

```cpp
// v3
packed_work_items_per_block = 512;

// v4
packed_work_items_per_block = 128;
```

设备侧 `.asc` kernel 在消除版本命名差异后与 v3 相同，每个 block 仍然包含 512
个线程。因而 v4 没有改变 scatter 算法或单线程处理逻辑，只会在达到 64 block
上限前，将 gradient kernel 的 block 数最多提高到 v3 的 4 倍。

但是，设备 kernel 仍按下面的公式计算线程起点：

```text
thread_start = blockIdx.x * 512 + threadIdx.x
```

host 侧按每 block 128 个任务计算 grid，并没有同步改变设备侧任务映射。因此 v4
新增的 block 不会重新分担前面 block 的工作。例如 `output_numel=512` 时启动 4
个 block，但只有 block 0 的 `thread_start` 小于 512；`output_numel=8192` 时启动
64 个 block，也只有前 16 个 block 有实际 scatter 任务。新增 block 只执行 kernel
前置逻辑后退出，实际效果是增加空 block 和调度开销，而不是提高有效并行度。

仓库已有的 Ascend 950PR float16 结果显示，提高 block 数不一定能带来收益：

| realistic case | v3 block 数 | v4 block 数 | v3 latency | v4 latency |
| --- | ---: | ---: | ---: | ---: |
| `spatiotemporal_mid_feature_pool_backward` | 1 | 4 | 5.748 us | 6.008 us |
| `volumetric_segmentation_head_pool_backward` | 16 | 64 | 6.591 us | 7.220 us |

独立复测中，以上两个 case 的 v4 延迟也分别比 v3 高约 7% 和 12%。对于
`output_numel <= 128`、v3 和 v4 都只启动一个 gradient block 的 case，两版代码
的实际调度相同，观察到的小幅性能差异应视为测量波动。基于当前有限数据，v4 更适合
视为“增加 block 并行度”的实验版本，而不能直接视为对 v3 的全面性能提升。

### v5：设备侧单次清零与完整设备时间

v5 用于验证与 CANN normal 更接近的完整设备工作量。它恢复一个自定义清零 kernel，
但不再使用 `at::zeros_like`，而是通过 `at::empty_like` 只分配输出：

```text
empty_like 分配 grad_input
-> adaptive_max_pool3d_grad_v5_zero_kernel
-> adaptive_max_pool3d_grad_v5_scatter_kernel
```

两个 kernel 在同一 stream 上顺序启动，因此 scatter 开始前清零 kernel 已完成，
不需要在纯 SIMT kernel 内实现不安全的跨 block barrier。清零覆盖
`grad_input.numel()`，block 数为 `min(ceil(input_numel / 512), 64)`；scatter 沿用
v3 的算法和 block 策略。与 v2-v4 的历史双重清零不同，v5 不会先执行
`ZerosLike`，完整设备路径中只有一次清零。

v5 还增加了原生 `bfloat16_t` clear/scatter/atomic launcher，不再沿用 v3/v4 的
FP32 累加后转 BF16 fallback。10 组 nondivisible overlap seed 的 NPU 对比均与
CANN BF16 输出逐元素一致。

v5 的 profile 选择同时匹配 zero 和 scatter kernel，并启用跨 profiler 文件求和，
因此 published latency 表示：

```text
v5_operator_device_total = zero_kernel + scatter_kernel
```

v5 仍然是两个 stream 顺序 kernel，不是 CANN arch35 那种使用 `SyncAll()` 的单个
AICore 外层任务。它的目的首先是统一设备工作范围；与方案 A
（`ZerosLike + SIMT scatter`）并列测试后，才能判断自定义 SIMT 清零和 kernel
launch 开销是否优于现有 `ZerosLike`。

Ascend 950PR realistic float16 的 12 个 case 已完成实测，v5 record 的 latency 是
zero 与 scatter 两个 kernel duration 之和：

| Case | CANN kernel | v5 zero + scatter | CANN/v5 |
| --- | ---: | ---: | ---: |
| CT global | 16.514 us | 11.900 us | 1.388 |
| Video global | 6.417 us | 6.503 us | 0.987 |
| Spatiotemporal | 9.192 us | 9.266 us | 0.992 |
| Volumetric segmentation | 14.610 us | 13.316 us | 1.097 |
| PointPillar global | 7.762 us | 7.230 us | 1.074 |
| R3D18 global | 10.531 us | 10.296 us | 1.023 |
| Regular grid | 11.279 us | 11.025 us | 1.023 |
| Nondivisible grid | 11.629 us | 10.398 us | 1.118 |
| Anisotropic grid | 25.511 us | 20.022 us | 1.274 |
| Large volume | 79.721 us | 58.661 us | 1.359 |
| Large batch global | 29.695 us | 21.329 us | 1.392 |
| Single-sample nondivisible | 8.654 us | 8.916 us | 0.971 |

几何平均 `CANN/v5=1.131`，即 v5 按设备 kernel 总时间整体快约 1.13x；v5 赢
9 个 case，CANN 赢 3 个 case。所有 12 个 accuracy record 均通过且最大绝对误差
为 0。与 scatter-only 的 2.75x 相比，这组结果更能说明完整清零成本会显著缩小
SIMT 与 CANN normal 的差距。

前五个 case 中，v5 实测总时间比方案 A 的 `ZerosLike + scatter` 估算慢约
16%-58%，表明连续大输出清零更适合现有 `ZerosLike`/SIMD 路径，而 direct scatter
更适合 SIMT。当前结果支持保留方案 A 作为实际实现，v5 作为统一计时边界的实验
基线。需要注意，v5 latency 是两个 device kernel duration 的和，不包含两次 launch
之间的 host gap，仍不等于完整端到端 latency。

## CUDA 与 CANN 源码实现特点

本目录下额外保留了两份参考源码：

- `cuda_source/AdaptiveMaxPooling3d.cu`：PyTorch ATen CUDA 风格源码。
- `cann_source/`：从 CANN/ops-nn 获取的 `AdaptiveMaxPool3DGrad` 算子工程源码快照。

这两份源码都不是 CannBench 默认执行路径本身。CannBench 运行 CUDA baseline 时
仍然通过 PyTorch 调用已安装的 CUDA 实现；运行 CANN baseline 时通过 `torch_npu`
调用已安装的 CANN 默认实现。`simt/v1` 到 `simt/v5` 才是 CannBench 当前可编译、
可修改的自定义实现。

### PyTorch CUDA 实现

CUDA backward 的核心逻辑是根据前向保存的 `indices` 做 scatter：

```text
grad_input 先清零
对每个 grad_output 元素:
  argmax = indices[output_pos]
  grad_input[argmax] += grad_output[output_pos]
```

CUDA 源码在 host 侧根据输入/输出空间维度判断是否可能出现池化窗口重叠：

```text
atomic = (D_in % D_out != 0) ||
         (H_in % H_out != 0) ||
         (W_in % W_out != 0)
```

如果 `atomic == false`，说明每个输入位置最多只会被一个输出窗口写回，kernel
直接执行：

```text
gradInput[argmax] += grad_delta
```

如果 `atomic == true`，多个输出位置可能把梯度写回同一个输入位置，CUDA kernel
使用 atomic add：

```text
gpuAtomicAddNoReturn(&gradInput[argmax], grad_delta)
```

CUDA 的 backward kernel 使用固定的二维 `blockDim`：

```text
blockDim = (32, 8, 1)
```

`gridDim` 根据 shape 计算。对 5D `NCDHW` 来说，源码里大致把：

```text
totalZ = N * C * D_out
```

映射到 `gridDim.x`，每个 `blockIdx.x` 负责一个输出 `Z plane`，`threadIdx.y`
和 `threadIdx.x` 再覆盖该 plane 内的 `H_out`、`W_out`。如果 `totalZ` 超过单次
launch 的 `gridDim.x` 限制，源码通过 `offsetZ` 分批 launch，`o_plane =
blockIdx.x + offsetZ` 表示当前 block 对应的全局 plane 编号。

这种实现的优点是结构简单；缺点是当 `H_out`、`W_out` 很小时，一个 block 中
很多线程会空转。例如全局池化反向的 `D_out=H_out=W_out=1` 场景下，每个 block
内实际工作的线程很少，整体并行度主要依赖 `N * C * D_out`。

### CANN 默认实现

CANN 源码入口 `op_kernel/adaptive_max_pool3d_grad.cpp` 根据架构和 tiling key
选择不同实现路径：

```text
arch35:
  AdaptiveMaxPool3dGradSimt

其他路径:
  tilingKey = 0   -> AdaptiveMaxPool3DGradNormal
  tilingKey = 100 -> AdaptiveMaxPool3DGradNormal overlap
  tilingKey = 2   -> AdaptiveMaxPool3DGradScatter
  tilingKey = 102 -> AdaptiveMaxPool3DGradScatterOverlap
```

`AdaptiveMaxPool3DGradNormal` 不是简单的一元素一线程 scatter。它会使用
AscendC 的 UB buffer、transpose、mask/select、workspace、copy out 等逻辑，
把 `indices`、`grad_output` 和输出写回过程组织成更适合 Ascend vector core
的高性能路径。

之前在 CANN 默认 `float16` realistic case 上观察到的 `507035` 设备异常，日志
显示落在类似：

```text
AdaptiveMaxPool3DGrad_fp16_high_performance_16
tilingKey = 0
```

也就是 CANN 默认 normal 高性能路径，而不是 CannBench 自定义 `simt/v1` 到
`simt/v4`。

### 异步执行与输出张量生命周期

排查 `cann_ops_library` 路径时还发现一个容易误判为 CANN kernel 问题的
benchmark 调用层风险：`torch_npu` 上的算子执行是异步的，Python 调用返回时
device kernel 可能还没有真正执行完。

如果 benchmark 代码只调用算子但不持有返回的输出张量，例如：

```python
operator()
synchronize()
```

那么 `operator()` 返回的 NPU tensor 可能在同步前就被 Python 释放。对异步
device 任务来说，这会导致输出内存生命周期过早结束，进而可能触发类似
`507035` 的 device error。

当时在 `realistic` 后续 case 上观察到的典型报错类似：

```text
npuSynchronizeDevice:../torch_npu/csrc/core/npu/NPUStream.cpp:565 NPU function error:
device error type 3, error code is 507035
[ERROR] ERR00100 PTA call acl api failed
```

也可能出现在 host/device copy 路径上：

```text
copy_between_host_and_device_opapi:../torch_npu/csrc/aten/ops/op_api/CopyKernelOpApi.cpp:57
NPU function error: device error type 3, error code is 507035
[ERROR] ERR00100 PTA call acl api failed
```

更稳妥的写法是把输出保存到局部变量，等同步完成后再释放：

```python
output = operator()
synchronize()
del output
```

这个经验和 `AdaptiveMaxPool3DGrad` 的数学逻辑无关，也不是对 CANN 默认二进制
算子内部实现的修复；它修复的是 benchmark / framework 调用层对异步输出
tensor 生命周期的管理。也就是说，如果 `cann_ops_library` 之前在某些 case 上
失败、而保存输出后不再失败，不能直接说明 CANN 默认 kernel 已被修好，只能说明
原先的失败至少可能包含调用层生命周期问题。

CANN 默认实现的 block 并行度由 host tiling 侧设置：

```text
context_->SetBlockDim(blockNum)
```

device 侧通过 `GetBlockIdx()`、`GetBlockNum()` 分配任务。这和 CUDA 的
`gridDim.x/y/z`、`blockDim.x/y/z` 不同，更接近一维 block 任务分配模型。

### CANN arch35 SIMT 分支与 CannBench v2

`cann_source/op_kernel/arch35/adaptive_max_pool3d_grad_simt.h` 是 CANN 源码中
最接近 CannBench SIMT extension 的实现。它同样采用 scatter 思路：

```text
非 overlap:
  xGradData[inputIdx] += gradVal

overlap:
  asc_atomic_add(&xGradData[inputIdx], gradVal)
```

它还包含 `deterministicFlag` 分支：

- 非 deterministic：按 output 元素并行，overlap 时使用 atomic add，性能较好，
  但同一地址的浮点累加顺序不固定。
- deterministic：按 `N * C` 切片并行，每个切片内部按固定 `DHW` 顺序遍历并累加，
  避免 atomic 顺序不确定，但当 `N * C` 较小时并行度较低。

CannBench 的 `simt/v2` 不是调用 CANN 默认二进制，也不是完整复刻
`tilingKey=0` 的 normal 高性能路径。它是把 CANN arch35 SIMT 分支的核心思想
改造成当前 CannBench 可编译、可注册的 Python extension：

```text
--implementation simt --implementation-version v2
```

因此 `simt/v2` 的定位是：

- 保留 CANN arch35 SIMT 分支的 overlap / non-overlap / deterministic 结构。
- 避免进入已安装 CANN 默认二进制中的问题 kernel。
- 方便在 CannBench 内直接修改 `.cpp` / `.asc`、重新编译并 benchmark。

它不等价于 CANN 默认 high-performance normal kernel，性能特征也不同。

## CUDA、CANN 与 SIMT v3 性能分析

本节记录当前 `published/` 中 `AdaptiveMaxPool3DGrad` realistic 数据集的跨实现
测试结论。测试平台和实现为：

- NVIDIA H800 PyTorch CUDA；
- Ascend 950PR CANN Ops Library；
- Ascend 950PR CannBench SIMT v3。

三个记录文件各包含相同的 12 个 case，记录内部的 dtype 都是 `float16`。36 条
accuracy 记录全部通过，`max_abs_error` 和 `max_rel_error` 都是 0。当前 CUDA
published 目录名包含 `cuda-library` 和 `bfloat16`，但 record 内实际标记为
`cuda-pytorch` 和 `float16`；分析时应以 record 内容为准，发布前需要修正目录名和
run metadata。

### Published kernel 延迟结果

下表单位为微秒。粗体表示当前 published 指标下该 case 的最快实现。

| Case | H800 CUDA | 950PR CANN | 950PR SIMT v3 |
| --- | ---: | ---: | ---: |
| CT global | 3.712 | 16.514 | **2.723** |
| Video global | 3.808 | 6.417 | **2.803** |
| Spatiotemporal | **3.744** | 9.192 | 5.888 |
| Volumetric segmentation | **4.032** | 14.610 | 6.602 |
| PointPillar global | 3.808 | 7.762 | **3.025** |
| R3D18 global | 7.360 | 10.531 | **6.270** |
| Regular grid | **4.320** | 11.279 | 6.061 |
| Nondivisible grid | 5.536 | 11.629 | **5.016** |
| Anisotropic grid | **4.160** | 25.511 | 6.881 |
| Large volume | **4.064** | 79.721 | 6.704 |
| Large batch global | 26.240 | 29.695 | **8.647** |
| Single-sample nondivisible | 7.584 | 8.654 | **6.156** |

按 12 个 case 的几何平均计算：

```text
SIMT v3 相对 CANN published kernel：2.75x
CUDA 相对 CANN published kernel：   2.72x
SIMT v3 相对 CUDA published kernel：1.01x
```

SIMT v3 与 CUDA 的几何平均基本持平：SIMT 赢 7 个 case，CUDA 赢 5 个 case。
但是，以上比值只能用于比较当前 profiler 选中的 kernel，不能直接解释为完整
backward 算子的端到端加速比。

### 三种实现的计时边界

三种实现执行的设备工作和 published 计入的工作并不相同：

| 实现 | 实际 backward 设备工作 | 当前 published 计入 |
| --- | --- | --- |
| CUDA | `gradInput.zero_()` + CUDA scatter | 只匹配 `adaptivemaxgradinput` 或 atomic 版本 |
| SIMT v3 | `ZerosLike` + SIMT scatter | 只匹配 `adaptive_max_pool3d_grad_kernel` |
| CANN | 单个 `AdaptiveMaxPool3DGrad` kernel | 记录整个被选中的 CANN kernel |

CUDA 源码在 scatter 前调用 `gradInput.zero_()`，但 profile kernel pattern 只包含
`adaptivemaxgradinput`、`atomicadaptivemaxgradinput` 等名称，所以单独的 memset、
fill 或 zero kernel 不会进入结果。SIMT profile 同样不匹配 `ZerosLike`。CANN
的清零、同步和梯度计算位于同一个被选中的设备 kernel 内，无法用当前 kernel-name
过滤方式拆开。

因此，CUDA 与 SIMT 的结果主要是 scatter-kernel 对比；CANN 与二者不是完全一致
的工作范围。尤其对于输入很大而 `output_numel` 很小的 case，当前口径会明显放大
CUDA/SIMT 相对 CANN 的表面优势。

### CUDA 与 SIMT v3 的 shape 特征

PyTorch CUDA backward 使用 `(32, 8)`，即 256 个线程的二维 block，并令
`grid.x = N * C * D_out`。每个 block 负责一个输出 H/W plane。SIMT v3 使用
512 个线程的一维 block，并按下面的方式覆盖扁平化输出：

```text
simt_blocks = min(ceil(output_numel / 512), 64)
```

全局池化的 `H_out=W_out=1` 场景下，CUDA 每个 256-thread block 只有一个线程
处理有效输出，单个 block 的有效线程比例约为 `1/256 = 0.39%`。SIMT 则把不同
`[N, C]` 的输出集中到少量 block 中。五个 global case 均由 SIMT 获胜：

```text
CT global:          SIMT 快 1.36x
Video global:       SIMT 快 1.36x
PointPillar global: SIMT 快 1.26x
R3D18 global:       SIMT 快 1.17x
Large batch global: SIMT 快 3.03x
```

其中 large-batch global case 的 `N*C=32768`。CUDA 启动 32768 个 block、约
839 万个线程，但实际只有 32768 个线程处理输出；SIMT 启动 64 个 block、32768
个线程，线程任务密度明显更高。

对于规则的非全局网格，CUDA 连续的 `threadIdx.x` 对应连续 `W_out`，能够自然形成
连续的 `grad_output` 和 `indices` 读取；`[N,C,D_out]` 已由 block 坐标表达，也不
需要像 SIMT 一样为每个输出执行 64 位 `output_index / output_spatial`。CUDA 在
五个规则非全局 case 中快约 1.40x 到 1.65x。

两个 nondivisible case 会进入 atomic 路径，SIMT 分别快 1.10x 和 1.23x。可能的
有利因素包括一维输出映射更紧凑，以及 Ascend 路径使用 int32 indices，而 CUDA
源码读取 int64 indices。不过 published 中没有保留原始 NCU counter，无法从当前
数据进一步区分 atomic contention、L2 cache 和整数指令成本。

### CANN kernel 的额外工作

现有 raw profile 中 CANN kernel 名为
`AdaptiveMaxPool3DGrad_fp16_high_performance_16`，测试的前五个 case 都使用 32
个 block。仓库随附的 CANN normal 源码显示，该路径除了梯度写回外还会执行：

- 完整 `grad_input` 或 overlap workspace 清零和全核同步；
- grad/indices 搬入 UB 和 transpose；
- indices 到 D/H/W 坐标的转换；
- 窗口索引生成、compare、select 和 reduce；
- 输出 copy；float16 overlap 路径还可能使用 FP32 workspace 并最终 cast。

raw profile 的 GM 写流量验证了计时边界差异：

| Case | CANN kernel GM 写入 | SIMT scatter kernel GM 写入 |
| --- | ---: | ---: |
| CT global | 约 14.4 MiB | 约 4 KiB |
| Video global | 约 448 KiB | 约 8 KiB |
| Volumetric segmentation | 约 7.9 MiB | 约 0.91 MiB |
| PointPillar global | 约 1.46 MiB | 约 16 KiB |

CANN kernel 在处理完整输出初始化，而 SIMT published 指标主要只写被 `indices`
命中的位置。这也是 large-volume case 中 CANN published 延迟达到 79.721 微秒、
SIMT scatter 只有 6.704 微秒的主要原因之一。这个 11.89x 比值不能解释为两个
完整 backward 实现之间的真实差距。

### SIMT v3 的双重清零问题及修复

SIMT v2-v4 原始实现的 C++ host 使用 `at::zeros_like` 创建 `grad_input`，随后
自定义设备代码又运行一次 `init_output_zero_kernel`。两次操作都会遍历完整
`grad_input`，第二次清零是重复工作。

已有前五个 case 的修复前 raw profile 可以将设备 kernel 时间拆开。下表中的
“SIMT 修复前总和”是 `ZerosLike + init_output_zero_kernel + scatter` 的设备时间
之和，不包含 host launch gap；“SIMT 方案 A”是保留较快的 `ZerosLike`、去掉重复
init 后的估算值。

| Case | CANN kernel | SIMT scatter | SIMT 修复前总和 | SIMT 方案 A |
| --- | ---: | ---: | ---: | ---: |
| CT global | 16.455 | 3.044 | 17.050 | 7.535 |
| Video global | 6.055 | 2.660 | 8.738 | 5.202 |
| Spatiotemporal | 8.972 | 5.748 | 11.308 | 7.998 |
| Volumetric segmentation | 14.940 | 6.591 | 16.679 | 10.096 |
| PointPillar global | 7.424 | 3.066 | 9.841 | 5.829 |

单位均为微秒。按修复前双重清零代码的设备 kernel 总和，前五个 case 中 CANN 比
SIMT 快约 4% 到 31%。采用方案 A 后，估算结果变为 SIMT 比 CANN 快约 1.12x 到
2.18x。当前 v2-v4 源码已经删除 `init_output_zero_kernel` 和相关
`zero_block_num` 接口，只保留 `at::zeros_like`。三个扩展重新编译后的 smoke
数值对比均已通过，v3 smoke profiler 也确认设备任务中不再出现自定义清零 kernel；
仍需重新运行 realistic 性能测试，以实测值替换表中的方案 A 估算值。

### 与 `index_add` 的性能差异对照

`index_add` 和 `AdaptiveMaxPool3DGrad` 最终都有 scatter 写回，但两者的主要工作量
并不相同，不能仅凭“都是 scatter 类型”推断 CANN/SIMT 应有相近的性能比。

| 对比项 | `index_add` | `AdaptiveMaxPool3DGrad` |
| --- | --- | --- |
| 输出初始状态 | `clone(self)` 后累加 | 创建与输入同形状的全零 `grad_input` |
| realistic 更新密度 | `source_numel / output_numel` 为 0.25-128 | 11/12 case 不超过 1.5625%，最低约 0.000763% |
| 写冲突 | 任意重复 index 合法，通用实现必须保留 atomic 语义 | argmax 受 pooling window 约束，非重叠窗口可直接写，只有 overlap 路径需要 atomic |
| 当前 published 口径 | profile 主要选择 IndexAdd 更新 kernel，通常不包含独立 clone | SIMT 只选择 scatter，CANN 单 kernel 同时包含清零和梯度计算 |
| 主要瓶颈 | 大量随机写、atomic 冲突和内存局部性 | 完整算子常由大输出清零主导，scatter 本身可能只有少量更新 |

`index_add` 的 15 个 realistic case 中，SIMT v2/CANN 几何平均为 1.303，即 CANN
整体约快 1.30x；SIMT 赢 5 个 case，CANN 赢 10 个 case。这个汇总值掩盖了明显的
shape 分化：GNN neighbor case 中 SIMT 快 4.17x，而 T5 hidden case 中 CANN 快
2.68x，大尺寸 1D 和最后一维随机写 case 则基本接近。其原因是双方都必须执行大量
scatter/atomic 更新，性能主要受 index 分布、冲突程度、连续 slice 大小和随机 GM
写限制，某一方不能省略主工作量。

相比之下，CT global case 需要清零 4,194,304 个 `grad_input` 元素，却只执行 32
次 scatter。当前 published 将 SIMT 的 `ZerosLike` 排除在选中 kernel 之外，却将
CANN kernel 内的完整清零计入，因此 AdaptiveMaxPool3DGrad 的 2.75x 表面差距主要
来自更新密度和计时边界不一致，不是纯 scatter 实现能力的差距。`index_add` 的
CANN/SIMT 差距较小，核心原因是双方统计到的工作更接近，而且共同承担了高密度
随机 atomic scatter 这一主要瓶颈。

### SIMT 与 SIMD 的适用场景

CANN normal 是典型的 AscendC SIMD/vector 路径：将 grad 和 indices 搬入 UB，
经过 transpose、坐标转换、窗口索引生成以及 compare/select/reduce 后连续写回。
CannBench SIMT 则将每个 `grad_output` 元素作为独立任务，读取 argmax 后直接写入
目标地址，只有 overlap 时才使用 atomic。两种模型的适用范围可以概括为：

| 特征 | 更适合 SIMT | 更适合 SIMD |
| --- | --- | --- |
| 更新密度 | 低，只需处理少量有效位置 | 高，向量准备成本可被大量更新摊薄 |
| 目标地址 | 索引驱动、随机或难以批量化 | 连续、块状、容易合并搬运 |
| 单元素计算 | 很少，主要是地址计算和写回 | 较多且能使用向量指令批处理 |
| pooling window | 大窗口但每个输出只命中一个位置 | 小而规则，窗口数据和索引可复用 |
| 写冲突 | non-overlap 或低冲突 overlap | 高冲突，可在 UB 中先聚合再写回 |
| 数据复用和对齐 | 复用少、尾块多、有效 lane 少 | UB 复用高、对齐好、向量 lane 利用率高 |

因此，SIMT 的优势不是适用于所有 scatter，而是能低成本表达稀疏、不规则、
索引驱动的独立更新；SIMD 的优势则是在规则、密集、可复用的数据上，通过批量
搬运和局部归约降低全局访存及 atomic 成本。对本算子更合理的是混合路径：使用
`ZerosLike` 或专用 SIMD/memset 完成输出清零，稀疏 non-overlap 和低冲突 overlap
使用 SIMT scatter，密集或高冲突场景考虑 SIMD UB 聚合。当前方案 A
（`ZerosLike + SIMT scatter`）正是这种混合方式。

当前 published 的 SIMT/CANN 2.75x 比值工作范围不一致，不能作为 SIMT 普遍优于
SIMD 的证据。方案 A 前五个 case 的完整设备时间估算显示 SIMT 快约 1.12x-2.18x，
只能作为低更新密度场景更适合 SIMT 的初步证据；最终仍需在相同清零和计时边界下，
对 CANN normal、CANN scatter 与 SIMT scatter 进行实测验证。

### 性能结论与后续验证要求

基于当前数据，可以得到以下结论：

1. SIMT v3 的 scatter 映射适合 global pooling 和部分 overlap case；CUDA 的二维
   plane 映射更适合规则的非全局网格。
2. SIMT v3 与 CUDA 的 scatter-kernel 几何平均性能基本持平，但这是 H800 与
   Ascend 950PR 的跨硬件结果，不能解释为单纯的软件实现优劣。
3. 当前 published 中 SIMT 相对 CANN 的 2.75x 是不等工作量的 kernel 比值，不是
   端到端算子加速比。
4. SIMT v3 原始实现的两次清零抵消了 scatter kernel 的大部分优势；当前源码已按
   方案 A 删除重复清零，等待重新 benchmark 验证完整设备时间。
5. v4 只增加空 block，没有解决清零或规则网格下的索引开销，因此不构成对 v3 的
   有效优化。
6. v5 使用 `empty_like + zero kernel + v3 scatter` 后，12 个 realistic case 的
   设备 kernel 总时间相对 CANN 几何平均快 1.13x，说明旧 2.75x 主要由不一致的
   计时边界放大；同时 v5 清零慢于方案 A 的 `ZerosLike`，更支持 SIMD 清零与
   SIMT scatter 的混合路径。

后续 benchmark 应同时发布两类指标：

```text
scatter_kernel_latency
operator_device_total_latency
```

`operator_device_total_latency` 必须覆盖输出分配后的完整清零和 scatter，并在三种
实现上使用一致的边界。还应保留原始 NCU/msprof 文件、增加多次重复统计，并修正
CUDA published 目录、canonical `run_id` 和 `published/index.json`，再形成正式的
跨平台性能结论。
