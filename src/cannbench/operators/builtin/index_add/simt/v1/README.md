# index_add SIMT v1

v1 是 `index_add` 在 CannBench 中接入 Ascend SIMT 的初始版本。它的目标不是做形状特化优化，而是先建立一条完整、可验证、可发布的 SIMT operator 集成路径。

## 目的

v1 主要解决三件事：

1. 在 Ascend NPU 上提供一个可运行的 `index_add` SIMT 实现。
2. 保持和 PyTorch `index_add` 接近的语义，尤其是重复 index 时必须用 atomic add 保证累加语义。
3. 把 SIMT 实现以 operator-local 插件形式接入 CannBench，避免在公共 backend/CLI 中增加 `index_add` 专用分支。

因此，v1 采用一个通用 kernel 覆盖不同 rank 和 dim：

- 输入 `self`、`index`、`source` 会先转为 contiguous。
- `index` 会转为 `int32`。
- 每个线程处理一个 `source` 元素。
- 通过线性 `thread_id` 计算 `outer/index/inner` 坐标。
- 根据 `index[j]` 计算目标地址。
- 使用 `asc_atomic_add` 写回输出，保证重复 index 的语义。

## 实现方式

核心 SIMT kernel 位于：

```text
aten_index_add/csrc/simt/index_add.asc
```

核心地址计算逻辑：

```cpp
const int32_t k = thread_id % inner_stride;
const int32_t j = (thread_id / inner_stride) % index_size;
const int32_t i = thread_id / (inner_stride * index_size);

const int32_t dst_index =
    i * self_dim_size * inner_stride + index[j] * inner_stride + k;
const scalar_t val = scale_index_add_value(source[thread_id], alpha);
asc_atomic_add(&output[dst_index], val);
```

这个公式对应 PyTorch `index_add(self, dim, index, source)` 的通用布局：

- `i`：`dim` 之前的 outer 坐标。
- `j`：index 维度上的位置。
- `k`：`dim` 之后的 inner 坐标。
- `index[j]`：写回到 `self` 在 `dim` 维上的目标位置。

## 支持范围

v1 支持：

- `float32`
- `float16`
- 任意 rank，只要 `self` 和 `source` rank 相同
- `index` 为一维
- `source.size(dim) == index.numel()`
- 重复 index 的累加语义

对于非 `float32/float16` 类型，C++ 包装层会转成 `float32` 计算后再转回原 dtype。当前 CannBench 的 index_add SIMT 性能测试主要关注 `float16` 和 `float32`。

## 为 v1 新增的文件及作用

v1 的文件都放在：

```text
src/cannbench/operators/builtin/index_add/simt/v1/
```

### `pyproject.toml`

定义 v1 SIMT 扩展包的 Python 构建元信息：

- 包名：`aten_index_add`
- 构建后端：`setuptools`
- 运行依赖：`torch`、`torch_npu`
- setuptools 包列表：`aten_index_add`

这个文件让 v1 可以作为一个独立的 editable Python 扩展包安装。

### `setup.py`

定义 C++/ASC 扩展的编译和链接流程。

主要职责：

- 查找 torch、torch_npu、Python、Ascend SIMT 相关 include/lib 路径。
- 使用 `bisheng -x asc --enable-simt` 编译 `.asc` 源码。
- 使用 C++ 编译器编译 `.cpp` 源码。
- 链接生成 `aten_index_add._C` 扩展库。
- 根据 `NPU_ARCH` 选择目标 NPU 架构。

这是 v1 能在 Ascend 环境中被构建成 `.so` 的核心文件。

### `install.sh`

顶层安装入口。

它只负责定位当前脚本目录，然后调用：

```text
scripts/install.sh
```

这样外部框架只需要执行 `simt/v1/install.sh`，不需要知道内部脚本布局。

### `scripts/common.sh`

构建环境准备脚本。

主要职责：

- 查找并 source Ascend CANN 环境脚本。
- 设置 `PIP_NO_BUILD_ISOLATION=1`，避免隔离构建时找不到本机 torch/torch_npu/CANN 环境。
- 自动检测 `NPU_ARCH`，例如 `dav-3510`。
- 如果无法自动检测，则提示用户手动设置 `NPU_ARCH`。

### `scripts/install.sh`

实际安装脚本。

主要流程：

1. source `scripts/common.sh`
2. 准备 Ascend 构建环境
3. 进入 v1 工程根目录
4. 执行：

```bash
python -m pip install -e . --no-build-isolation --no-deps
```

这会触发 `setup.py` 编译并安装 `aten_index_add` 扩展。

### `aten_index_add/__init__.py`

Python 包入口。

主要职责：

- 尝试 import 编译生成的 `_C` 扩展。
- 暴露 `ops` 子模块。

导入 `aten_index_add` 时，会间接加载 C++ 注册逻辑，使 `torch.ops.aten_index_add.index_add_forward` 可用。

### `aten_index_add/ops.py`

Python 调用封装。

主要职责：

- 检查 `_C` 扩展是否加载成功。
- 检查 `torch.ops.aten_index_add.index_add_forward` 是否注册成功。
- 提供 Python 函数：

```python
index_add_forward(self, dim, index, source, alpha=1.0)
```

CannBench 的 SIMT callable 最终会调用这个函数。

### `aten_index_add/csrc/register.cpp`

最小 Python C extension 注册文件。

它定义 `PyInit__C`，让 Python 能 import `aten_index_add._C`。真正的 torch operator 注册逻辑在 `index_add.cpp` 中完成。

### `aten_index_add/csrc/index_add.cpp`

PyTorch/torch_npu 侧的 operator 包装层。

主要职责：

- 定义 C++ operator：`aten_index_add::index_add_forward`
- 校验输入设备、rank、dtype、shape
- 处理负数 dim 到 wrapped dim
- 将 `source` 和 `index` 转为 contiguous
- 将 `index` 转为 `int32`
- 计算：
  - `self_dim_size`
  - `index_size`
  - `inner_stride`
  - `outer_size`
  - `total_length`
- 根据 dtype 调用：
  - `launch_index_add_float`
  - `launch_index_add_half`
- 将 kernel 放入当前 NPU stream 执行
- 注册到 `torch.ops.aten_index_add.index_add_forward`
- 同时注册 `aten::index_add` 的 PrivateUse1 实现

这个文件连接了 PyTorch 张量世界和底层 SIMT kernel。

### `aten_index_add/csrc/simt/index_add.asc`

SIMT kernel 源码。

主要职责：

- 定义 `float` 和 `half` 的 alpha 缩放逻辑。
- 定义通用 `index_add_kernel`。
- 使用 `asc_atomic_add` 实现重复 index 的正确累加。
- 提供 C ABI launcher：
  - `launch_index_add_float`
  - `launch_index_add_half`
- 配置每 block 1024 线程，最多 64 个 block。

这是 v1 真正执行 `index_add` 计算的文件。

## 使用方式

在仓库根目录生成 release：

```bash
make release
```

进入 release 目录运行 v1：

```bash
cd dist/cannbench-release

python -m cannbench bench \
  --backend ascend \
  --implementation simt \
  --implementation-version v1 \
  --op index_add \
  --dataset realistic \
  --dtype float16 \
  --output-dir runs \
  --run-name opbench-ascend-950pr-simt-v1-index_add-realistic-float16 \
  --warmup 0 \
  --iterations 1
```

如果不显式指定 `--implementation-version`，当前插件默认会使用 v1：

```bash
python -m cannbench bench \
  --backend ascend \
  --implementation simt \
  --op index_add \
  --dataset realistic \
  --dtype float16 \
  --output-dir runs \
  --run-name opbench-ascend-950pr-simt-v1-index_add-realistic-float16 \
  --warmup 0 \
  --iterations 1
```

## v1 的定位

v1 是 baseline 版本，价值主要在于：

- 建立 `index_add` 的 SIMT 接入方式。
- 提供正确的 atomic 累加语义。
- 为 v2/v3 后续优化提供可对比基线。
- 保持实现简单，便于定位性能瓶颈。

v1 的主要局限是：所有形状都走同一个 generic kernel，因此对 1D、`dim=0` slice、最后一维等常见形状没有做地址计算和访存模式特化。v2 的形状特化就是在 v1 的基础上解决这部分问题。
