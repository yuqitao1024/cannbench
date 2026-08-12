# 长序列 Sharded Histogram Top-K 实践

## 概述

本样例用一个固定的 256-bin Top-K threshold 工作负载，展示长 context 从“单 block 串行长尾”拆为多个独立 shard 后的并行化方式。两个 Case 输入和输出完全一致，仅改变 context 的工作所有权和归约边界。

| Case | 编译开关 | 实现 | 每次调用 |
|:---:|:---|:---|:---:|
| 0 | `SCENARIO_NUM=0` | 单 block 扫描全部 16M 分数，生成 histogram 并求 threshold | 1 次 kernel launch |
| 1 | `SCENARIO_NUM=1` | 64 个 shard blocks 生成局部 histogram，再由 1 个 block 层次归约 | 2 次 kernel launch |

样例面向 Ascend 950PR/Ascend 950DT，默认 `dav-3510`。设备源码只使用 C API 和 SIMT API；host 使用 C 风格函数、POD struct、`malloc/calloc/free` 与裸 ACL。

## 输出语义

输入是固定 seed 生成的 `16 * 1024 * 1024` 个 `uint8` 分数。host oracle 与 device 都输出：

- 全部 256 个 histogram bins；
- 所有 bins 的总数；
- 降序第 `TOP_K=4096` 个元素所在的 `threshold`；
- 为凑满 Top-K，需要从 threshold bin 取得的 `threshold_tail`。

其中：

```text
threshold_tail = TOP_K - count(score > threshold)
```

direct run 会逐项核对 256 个 bins，并单独核对总数、threshold 和 threshold tail。固定 seed、前 256 个强制覆盖全部 bin，以及整数精确比较使 A/B 具有相同可复现语义。

## 工作分解

Case 0 只有 512 个线程处理完整长序列，每个线程承担约 32768 个输入，最后由 lane 0 扫描 256 个 bins 求阈值。它是用于暴露长 context 串行 tail 的对照路径。

Case 1 将 `[0, context_count)` 按整数边界划成 64 个互不重叠 shard。每个 block 只处理自己的 shard，并写出一个 `256 * uint32` 的 partial histogram。第二个 kernel 用 256 个线程按 bin 归约全部 64 个 partials，之后由 thread 0 求阈值。这样把占主导的长序列扫描分散到 64 个 block，同时保留明确、可审计的归约阶段。

逻辑所有权如下：

```text
Case 0: context -> 1 block -> 512 threads -> final histogram -> threshold
Case 1: context -> 64 shards/blocks -> 512 threads each -> 64x256 partials
        -> 1 reducer block/256 threads -> final histogram -> threshold
```

## 编译与正确性验证

配置 CANN 环境后分别编译：

```bash
cmake -S . -B build/case0 -DCMAKE_ASC_ARCHITECTURES=dav-3510 -DSCENARIO_NUM=0
cmake --build build/case0 --parallel
./build/case0/sharded_histogram_topk

cmake -S . -B build/case1 -DCMAKE_ASC_ARCHITECTURES=dav-3510 -DSCENARIO_NUM=1
cmake --build build/case1 --parallel
./build/case1/sharded_histogram_topk
```

成功时程序退出码为 0 并打印 `Verification PASSED`。

## 性能采集与聚合

运行：

```bash
SAMPLES=5 ./scripts/run.sh
```

脚本对每个 Case 先 direct 验证，再在独立 profiler 输出目录中采集 5 次完整调用。每次脚本执行使用唯一 `profiles/<run-id>/`，已存在的 run id 会直接失败，不删除历史 raw。Case 0 的 profile launch count 为 1；Case 1 为 2。Case 1 的 shard 与 reduce 行属于同一次算子调用，必须先相加为 `stage_sum_us`，不能把两个阶段当作两个样本求平均。脚本在 `profiles/<run-id>/operator_calls.csv` 保留每次调用的阶段值与和，并在同目录的 `performance_summary.txt` 输出跨调用中位数和范围。

`stage_sum_us` 仅表示选中 device kernel rows 的和，不包含 Case 1 两次 launch 之间的 gap。正式结论还需在相同输入和同步条件下，用 ACL Event 或同步 wall interval 覆盖完整 launch 序列，并记录 Ascend 950 具体型号、CANN/Bisheng、驱动、固件与频率。

## 性能结果

2026-08-12 在 20002 节点完成 5 次重复采集：Ascend 950 系列 `dav-3510`、CANN 9.2.0、
Bisheng/Clang 15.0.5、driver `7.0.t9.0.B791`、1650 MHz。两个场景的 direct run 和
profiler 内运行均输出 `Verification PASSED`。Scenario 1 每次先将 shard 和 reduce 两个
kernel row 相加，再跨调用统计。

| Case | selected kernel rows | launch count | median `stage_sum_us` | range `stage_sum_us` | 加速比 |
|:---:|:---|---:|---:|---:|---:|
| 0 | baseline | 1 | 16328.507812 | 16327.749023-16331.936523 | 基准 |
| 1 | shard + reduce | 2 | 292.433994 | 292.352021-292.878998 | 55.84x |

Scenario 1 的选中 kernel 阶段和相对 Scenario 0 降低 `98.21%`。该次远端 raw 数据位于
`profiles/scenario_0/sample_1..5` 和 `profiles/scenario_1/sample_1..5`；后续脚本采集位于
`profiles/<run-id>/scenario_N/sample_1..5`。msopprof 对 64 block
场景提示动态插桩 sub-block 数可能超过 108；BasicInfo kernel row 完整，但解读其他动态
指标时必须保留该限制。完整序列耗时尚未用 ACL Event 或同步 wall interval 采集；表中
Scenario 1 只包含两个 kernel row 的阶段和，不包含两次 launch 之间的 gap。

不得用 source-level 工作量估计替代设备数据。是否保留 Case 1，取决于正确性通过后在同一 Ascend 950 环境得到的完整调用测量。
