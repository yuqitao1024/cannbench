# index_add SIMT v3

v3 是一个用于 `index_add` 1D atomic 性能分析的诊断版本，不是用来替代 v2 的通用正确实现。

## 目的

v3 用来验证一个具体假设：Ascend SIMT 在 1D `index_add` 上相比 NVIDIA CUDA 慢，是否主要是因为 `asc_atomic_add` 指令本身开销太高。

实验主要对比 v2 和 v3 在 `atomic_1d` 数据集上的表现：

- `atomic_1d_unique_contiguous`：index 无重复，写地址连续。
- `atomic_1d_unique_random_permutation`：index 无重复，写地址随机。
- `atomic_1d_random_with_replacement`：index 随机且有重复，存在写冲突。

其中只有前两个 case 对 v3 具有正确性意义，因为它们没有重复写同一个输出地址。

## 和 v2 的差异

v3 从 v2 复制而来，只修改了 1D `dim=0` 的特化 kernel。

v2：

```cpp
asc_atomic_add(&output[index[j]], val);
```

v3：

```cpp
output[index[j]] = output[index[j]] + val;
```

其它路径保持和 v2 一致：

- generic 路径仍然使用 `asc_atomic_add`
- 2D `dim=0` 仍然使用 `asc_atomic_add`
- 3D `dim=0` 仍然使用 `asc_atomic_add`
- 4D `dim=3` 仍然使用 `asc_atomic_add`

## 正确性适用范围

v3 只在 1D `dim=0` 路径下没有重复目标地址时才是正确的。

适合用 v3 观察的 case：

- `atomic_1d_unique_contiguous`
- `atomic_1d_unique_random_permutation`

不应该用 v3 做正确性判断的场景：

- `atomic_1d_random_with_replacement`
- realistic 或 stress 数据集中可能存在重复 index 的 case
- 任何生产式 `index_add` 工作负载，只要可能有多个元素写同一个输出地址

原因是：对于重复 index，v3 的 1D `dim=0` 路径是非 atomic 的读-改-写，会产生数据竞争。

## 使用方式

在仓库根目录重新生成 release 包：

```bash
make release
```

进入 release 目录，运行完整 `atomic_1d` 数据集：

```bash
cd dist/cannbench-release

python -m cannbench bench \
  --backend ascend \
  --implementation simt \
  --implementation-version v3 \
  --op index_add \
  --dataset atomic_1d \
  --dtype float16 \
  --output-dir runs \
  --run-name opbench-ascend-950pr-simt-v3-index_add-atomic_1d-float16 \
  --warmup 0 \
  --iterations 1
```

也可以只运行两个无重复 index 的 case：

```bash
python -m cannbench bench \
  --backend ascend \
  --implementation simt \
  --implementation-version v3 \
  --op index_add \
  --dataset atomic_1d \
  --case-id atomic_1d_unique_contiguous \
  --dtype float16 \
  --output-dir runs \
  --run-name opbench-ascend-950pr-simt-v3-index_add-atomic_1d-unique-contiguous-float16 \
  --warmup 0 \
  --iterations 1

python -m cannbench bench \
  --backend ascend \
  --implementation simt \
  --implementation-version v3 \
  --op index_add \
  --dataset atomic_1d \
  --case-id atomic_1d_unique_random_permutation \
  --dtype float16 \
  --output-dir runs \
  --run-name opbench-ascend-950pr-simt-v3-index_add-atomic_1d-unique-random-float16 \
  --warmup 0 \
  --iterations 1
```

## 实测结果

在 Ascend 950PR、`float16` 下，v3 的实测延迟如下：

| case | v3 latency ms |
| --- | ---: |
| `atomic_1d_unique_contiguous` | 0.010631 |
| `atomic_1d_unique_random_permutation` | 0.115586998 |
| `atomic_1d_random_with_replacement` | 0.228259003 |

和此前 v2 atomic 版本对比：

| case | v2 atomic ms | v3 non-atomic ms | v3/v2 |
| --- | ---: | ---: | ---: |
| `atomic_1d_unique_contiguous` | 0.008368 | 0.010631 | 1.27x |
| `atomic_1d_unique_random_permutation` | 0.067346 | 0.115586998 | 1.72x |
| `atomic_1d_random_with_replacement` | 0.133512 | 0.228259003 | 1.71x |

其中 `atomic_1d_random_with_replacement` 对 v3 只具备性能参考意义，不能作为正确性对比。

## 结论

这组实验不支持“无冲突 `asc_atomic_add` 是 1D 慢的主要原因”这个假设。

如果 atomic 指令本身是主要瓶颈，那么在无重复 index 的两个 case 上，v3 应该明显快于 v2。但实测结果相反：v3 在两个无重复 case 上都比 v2 更慢。

更合理的解释是：1D 随机场景的主要瓶颈是 Ascend 上随机 global 写，尤其是随机 scatter 风格写入。对于这种访问模式，atomic 路径至少不比普通读-改-写路径差，甚至更好。

因此，后续优化不应该优先尝试把 `asc_atomic_add` 替换成普通 store，而应该关注：

- 减少随机 global 写次数
- 改善 index 局部性
- 在写回 global memory 前对重复 index 做局部聚合
