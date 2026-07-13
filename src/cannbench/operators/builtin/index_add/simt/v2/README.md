# index_add SIMT v2

v2 是 `index_add` Ascend SIMT 实现的第一版形状特化优化版本。它保留 v1 的通用语义，在不修改公共 backend/CLI 框架的前提下，只在 `index_add` operator 包内增加针对常见形状的 fast path。

## 目的

v1 使用一个通用 kernel 覆盖所有 rank 和 dim：

- 每个线程处理一个 `source` 元素。
- 通过除法和取模从线性 `thread_id` 反推出 `outer/index/inner` 坐标。
- 根据 `index[j]` 计算目标地址。
- 使用 `asc_atomic_add` 写回输出。

这个写法通用性强，但对一些常见场景有额外开销：

- 1D `dim=0` 不需要 `outer/inner` 坐标计算。
- `dim=0` 的 2D/3D 场景可以把目标地址计算简化为 `index[j] * slice_size + offset`。
- 4D 最后一维 `dim=3` 可以避免 generic 路径里的多次除法和取模。
- 某些 `dim=0` 大 slice 场景中，generic 路径会让多个 index 维度的更新交织在一起，特化后更容易形成连续 slice 内的并行写。

v2 的目标是验证：在保持 atomic 语义不变的前提下，通过形状特化和更简单的地址计算，能否提升 realistic 数据集里的实际 workload 性能。

## 改了什么

v2 新增了独立模块：

```text
src/cannbench/operators/builtin/index_add/simt/v2/
  aten_index_add_v2/
```

插件中通过 `--implementation-version v2` 路由到 `aten_index_add_v2`，不影响 v1。

v2 在 `csrc/index_add.cpp` 中根据 rank 和 dim 选择 kernel：

```cpp
if (rank == 1 && wrapped_dim == 0) {
  launch_index_add_1d_dim0_half(...);
}
if (rank == 2 && wrapped_dim == 0 && shape.inner_stride <= 256) {
  launch_index_add_2d_dim0_half(...);
}
if (rank == 3 && wrapped_dim == 0) {
  launch_index_add_3d_dim0_half(...);
}
if (rank == 4 && wrapped_dim == 3) {
  launch_index_add_4d_dim3_half(...);
}
```

不满足这些条件时回退到 generic kernel。

当前 v2 包含这些 kernel：

- `index_add_generic_kernel`
- `index_add_1d_dim0_kernel`
- `index_add_2d_dim0_kernel`
- `index_add_3d_dim0_kernel`
- `index_add_4d_dim3_kernel`

其中：

- generic kernel 使用 `__launch_bounds__(1024)`，launch 时每 block 1024 线程。
- 1D/2D/3D `dim=0` 特化 kernel 使用 `__launch_bounds__(2048)`。
- 4D `dim=3` 特化 kernel 使用 `__launch_bounds__(1024)`。
- 所有正确实现路径仍然使用 `asc_atomic_add`，没有改变 `index_add` 对重复 index 的语义。

## 为什么这样改

### 1D `dim=0`

1D 情况下，源元素 `source[j]` 只需要写到 `output[index[j]]`：

```cpp
asc_atomic_add(&output[index[j]], val);
```

相比 generic 路径，可以去掉 `inner_stride`、`outer`、除法和取模。

实际结果显示，这个特化对 1D 随机场景帮助有限。后续 v3 诊断也证明，1D 慢的主要原因不是无冲突 `asc_atomic_add` 指令本身，而更像是 Ascend 上随机 global 写路径的瓶颈。

### 2D/3D `dim=0`

这类场景里，`index[j]` 选择的是一个完整 row 或 slice。目标地址可以简化为：

```cpp
dst_index = index[j] * slice_size + offset;
```

这个特化的收益来自两个方面：

- 减少 generic 路径里的除法/取模和多维坐标反推。
- 让同一个 `index[j]` 对应的 slice 内更新更集中，降低不同 index 更新交织带来的调度和访存开销。

实测中，v2 的主要收益就集中在这类 `dim=0` 大 slice 或块稀疏场景。

### 4D `dim=3`

最后一维 `dim=3` 的目标地址可以简化为：

```cpp
outer = thread_id / index_size;
j = thread_id - outer * index_size;
dst_index = outer * self_dim_size + index[j];
```

这避免了 generic 路径的部分取模和多维坐标计算。不过实测显示，最后一维随机写仍然主要受随机 global 写影响，特化收益有限。

## 使用方式

在仓库根目录重新生成 release 包：

```bash
make release
```

在 release 目录运行 realistic 数据集：

```bash
cd dist/cannbench-release

python -m cannbench bench \
  --backend ascend \
  --implementation simt \
  --implementation-version v2 \
  --op index_add \
  --dataset realistic \
  --dtype float16 \
  --output-dir runs \
  --run-name opbench-ascend-950pr-simt-v2-index_add-realistic-float16 \
  --warmup 0 \
  --iterations 1
```

运行单个 case：

```bash
python -m cannbench bench \
  --backend ascend \
  --implementation simt \
  --implementation-version v2 \
  --op index_add \
  --dataset realistic \
  --case-id basic_gnn_gin_neighbor_index_add \
  --dtype float16 \
  --output-dir runs \
  --run-name opbench-ascend-950pr-simt-v2-index_add-basic_gnn_gin_neighbor-float16 \
  --warmup 0 \
  --iterations 1
```

如果使用 release 中已经预生成的 prepared 输入，也可以指定 `--prepared-dir`，但不能同时指定 `--dataset` 或 `--case-id`：

```bash
python -m cannbench bench \
  --backend ascend \
  --implementation simt \
  --implementation-version v2 \
  --op index_add \
  --prepared-dir prepared/index_add/realistic \
  --output-dir runs \
  --run-name opbench-ascend-950pr-simt-v2-index_add-realistic-float16 \
  --warmup 0 \
  --iterations 1
```

## 测试结果

以下结果来自 `published/` 目录下的 realistic float16 数据，包含 15 个 case：

- `published/opbench-ascend-950pr-simt-v1-index_add-realistic-float16`
- `published/opbench-ascend-950pr-simt-v2-index_add-realistic-float16`
- `published/opbench-ascend-950pr-cannops-index_add-realistic-float16`
- `published/opbench-nvidia-h800-cuda-pytorch-index_add-realistic-float16`

整体几何平均：

| 对比 | ratio | 解释 |
| --- | ---: | --- |
| v2/v1 | 0.934 | v2 相比 v1 几何平均约快 1.07x |
| v2/CANN | 1.303 | v2 整体仍比 CANN 慢 |
| v2/CUDA | 1.058 | v2 整体略慢于 CUDA |

v2 相比 v1 提升明显的 case：

| case | v2/v1 | 主要原因 |
| --- | ---: | --- |
| `basic_gnn_gin_neighbor_index_add` | 0.648 | 3D `dim=0` slice 特化收益明显 |
| `bigbird_block_index_add` | 0.673 | `dim=0` 块/slice 更新更适合特化路径 |
| `block_sparse_mid_index_add` | 0.765 | `dim=0` 地址计算和 slice 写更简单 |

v2 相比 CUDA 仍明显偏慢的 case：

| case | v2/CUDA | 主要瓶颈 |
| --- | ---: | --- |
| `allenai_longformer_large_index_add_inplace` | 3.406 | 1D 随机 global atomic/scatter 写 |
| `longformer_medium_1d_index_add_inplace` | 3.073 | 1D 随机 global atomic/scatter 写 |
| `last_dim_memory_medium_index_add` | 2.254 | 最后一维随机写 |
| `xlnet_memory_index_add` | 1.988 | 4D 最后一维随机写 |

部分 hidden/MoE/embedding 类 case 中，v2 比 CUDA 快但比 CANN 慢，说明 SIMT 实现对这些形状没有明显胜过 CANN 的专用实现：

| case | v2/CANN | v2/CUDA |
| --- | ---: | ---: |
| `bert_hidden_index_add` | 1.713 | 0.756 |
| `gpt_decoder_hidden_index_add` | 1.680 | 0.757 |
| `llama_hidden_large_batch_index_add` | 1.798 | 0.772 |
| `llama_moe_token_combine_index_add` | 1.999 | 0.907 |

## 结论

v2 的形状特化是有效的，但收益不均匀。

有效的场景：

- `dim=0` 且每个 index 对应较大连续 slice。
- GNN、block sparse、部分块状更新场景。
- 特化后能减少地址计算，并让 slice 内并行写更集中。

收益有限或不明显的场景：

- 1D 随机 index。
- 最后一维随机 index。
- hidden/MoE/embedding 类高冲突或随机写场景。

最重要的判断是：v2 仍然保留 atomic 语义，问题不在于“代码里用了 `asc_atomic_add` 就一定慢”。后续 `atomic_1d` 和 v3 诊断显示：

- 连续无重复写时，Ascend SIMT v2 并不慢。
- 随机无重复写时，Ascend 相比 CUDA 迅速拉开差距。
- 把 1D atomic 改成普通非 atomic 写的 v3 没有变快，反而更慢。

因此，1D 和最后一维场景后续优化的重点不应该是简单替换 `asc_atomic_add`，而应该是：

- 减少随机 global 写次数。
- 改善 index 的局部性。
- 在写回 global memory 前对重复 index 做局部聚合。
- 针对真实 workload 的 index 分布设计更有约束的特化，而不是对完全随机 index 盲目优化。
