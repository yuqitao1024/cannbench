# Row Width 行为分桶与 Exact-shape 反例

## 概述

本样例对应 Ascend 950 性能案例候选 3.16/P0。它用简单的整数 row sum + normalize，隔离三种 shape 策略：单一通用路径、按行为区间 dispatch、以及 exact width 过拟合。重点是可审计的路由和相邻 shape 对照，不是为某个未实测阈值背书。

| Case | 编译开关 | Dispatch | 每次调用 |
|:---:|:---|:---|:---:|
| 0 | `SCENARIO_NUM=0` | small/medium/wide 全部走固定 512-thread 通用 kernel | 1 次 kernel launch |
| 1 | `SCENARIO_NUM=1` | `<=256` 为 256 threads，`<=4096` 为 512 threads，`>4096` 为 1024 threads | 1 次 kernel launch |
| 2 | `SCENARIO_NUM=2` | exact width 1024 走无 guard 专用 kernel，其他 shape 走 padded fallback | 1 次 kernel launch |

每次程序调用打印 `launch_count=1 kernel=<name>`。run 脚本同时在 direct correctness log 和 profiler log 中核对该字段，再从 raw CSV 提取对应 kernel rows。

## 固定语义

输入是固定 seed `20260812` 生成的正 `uint32`，64 行，运行时 width 范围为 1 到 8192。物理 row stride 固定为 8192，以便所有策略共享相同分配和 tail 检查。每行输出：

```text
row_sum[row] = sum(input[row, 0:width])
normalize[row, column] = input[row, column] * 1000000 / row_sum[row]
```

整数除法向下取整，因此 host oracle 可精确比较，无浮点归约顺序差异。每行 `[width, 8192)` 在 launch 前初始化为 `0xa5a5a5a5`，完成后必须保持不变。

## 三种策略

### Case 0：单一通用路径

`generic_row_normalize_kernel` 对所有 width 使用 512 threads。预期瓶颈随 width 改变：短行可能有大量闲置线程，宽行则增加每线程循环次数；这些只是待验证假设。

### Case 1：行为区间分桶

路由只看会改变工作分解的区间：

| 区间 | Kernel | Geometry |
|:---|:---|---:|
| `<=256` | `small_row_normalize_kernel` | 256 threads |
| `<=4096` | `medium_row_normalize_kernel` | 512 threads |
| `>4096` | `wide_row_normalize_kernel` | 1024 threads |

阈值是实验起点，不是通用结论。必须比较 255/256/257 与 4095/4096/4097，确认 dispatch switch 两侧正确且性能没有断崖式回退，再决定是否调整边界。

### Case 2：Exact-shape 过拟合反例

exact width 1024 使用 256 threads，每 lane 固定读取四个元素，源码不保留 runtime width guard。1023 和 1025 等所有邻近 shape 则进入 `exact_overfit_fallback_kernel`：1024 threads 固定扫描 padded width 8192，只对逻辑 width 内元素求和和写回。

该不对称是刻意设计，用来展示只优化单点可能把 padding/回退代价推给邻近 shape，并增加 kernel 和路由维护成本。不能在没有 Ascend 950 raw profile 的情况下声称 exact kernel 更快或 fallback 更慢。

## 编译与运行

```bash
cmake -S . -B build/scenario_1 \
  -DCMAKE_ASC_ARCHITECTURES=dav-3510 \
  -DSCENARIO_NUM=1
cmake --build build/scenario_1 --parallel
./build/scenario_1/row_width_dispatch 4096
```

成功时程序打印选中 kernel、`launch_count=1` 和 `Verification PASSED`。

## 边界与邻近 Shape Profile

```bash
./scripts/run.sh
```

每个场景先完成 width 1、255/256/257、1023/1024/1025、4095/4096/4097、8192 的 correctness，之后再 profile 边界与邻近 shape。msopprof 使用默认参数，不显式设置 warmup 或 metric group。raw 数据保留在：

```text
profiles/scenario_<N>/width_<W>/raw/<RUN_ID>/
```

解析行写到同级 `parsed/<RUN_ID>/kernel_rows.csv`。比较边界是单次调用的唯一目标 kernel `Task Duration`；不同 width、scenario 或 profiler collection 是独立样本，不能混合求一个无 shape 权重的平均值。

## 性能结果

2026-08-12 在 20002 节点、CANN 9.2.0、`dav-3510`、1650 MHz 完成单次采集，全部
correctness 和 30 个 profiler run 通过。单位均为 us：

| Width | 通用 512T | 行为分桶 | Exact/回退 | 分桶相对通用 |
|---:|---:|---:|---:|---:|
| 255 | 5.829 | 5.451 | 6.677 | -6.49% |
| 256 | 5.576 | 5.852 | 6.257 | +4.95% |
| 257 | 5.815 | 5.698 | 6.682 | -2.01% |
| 1023 | 6.063 | 6.352 | 6.721 | +4.77% |
| 1024 | 6.166 | 6.072 | 6.688 | -1.52% |
| 1025 | 6.044 | 6.376 | 6.694 | +5.49% |
| 4095 | 8.438 | 8.364 | 8.004 | -0.88% |
| 4096 | 8.369 | 8.527 | 7.673 | +1.89% |
| 4097 | 8.560 | 7.587 | 7.927 | -11.37% |
| 8192 | 11.201 | 9.343 | 9.177 | -16.59% |

负号表示降低。分桶只在本实验宽行明显获益，多个中小宽度出现回退；exact 1024 本身比
通用路径慢 `8.47%`，因此不能仅凭无 guard 或固定循环保留单点特化。raw 根目录为
`profiles/scenario_0..2/width_<W>/raw/20260812-062030`。以上是单次结果，正式阈值选择还需
多 run 离散度和真实 shape 权重。
