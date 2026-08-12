# Top-K 算法选择优化实践

## 概述

本案例用同一个 Top-K 问题展示优化顺序：先消除重复全排序和 padding 造成的算法工作量，再考虑指令调度、局部数组放置、占用率等微架构优化。两个实现位于同一个 `.asc` 中，由 `SCENARIO_NUM` 编译选择。

| Case | 编译选项 | 算法 |
|:---:|:---|:---|
| 0 | `SCENARIO_NUM=0` | 每 16 个新候选与 Top-32 合并，补齐 64 项并重复执行 bitonic merge |
| 1 | `SCENARIO_NUM=1` | 单遍扫描，维护容量 32 的 worst-root heap，最后只整理有效 Top-32 |

Case 1 的教学重点是把扫描阶段从重复 padded bitonic 网络改为 `O(N log K)` 的流式/有界 Top-K，且不生成 padding；它不是对 Case 0 做 unroll、launch geometry 或缓存参数微调。

## 样例规格

| 项目 | 规格 |
|:---|:---|
| 输入 | `[32, 1024]`，`float32`，连续 ND |
| 输出值 | `[32, 32]`，`float32` |
| 输出索引 | `[32, 32]`，`uint32` |
| Top-K | 逐行 Top-32 |
| 排序语义 | 值降序；值相等时原始索引升序 |
| 输入 | 固定随机种子 `20260811`，两种 scenario 生成相同输入，并刻意包含 tie |
| 正确性 | C 风格 host oracle 使用 `qsort`，值和索引逐项精确检查 |
| Launch | 1 个 block，512 个 SIMT 线程；每线程处理 2 行 |
| 目标产品 | Ascend 950PR / Ascend 950DT，`dav-3510` |

实现仅使用 C API、Tensor API/SIMD 头文件和 SIMT API 边界。host 侧使用 POD、`malloc/free` 和裸 ACL 分配/释放，不用 `class`、STL 容器或设备资源 RAII 封装。

## 预期瓶颈

- Case 0：每行 256 次 64 项 bitonic 网络，每次都有 16 个 sentinel padding；预期瓶颈是重复 compare-exchange 和无效 padding 工作。
- Case 1：扫描只维护 32 项 heap；预期瓶颈变为 GM 候选读取、heap 数据依赖和线程局部数组资源。

这些是待 profile 验证的假设。不能从源码数组大小直接推断寄存器、spill 或占用率，也不能把不同 metric 配置下的 Task Duration 混合比较。

## 编译与正确性验证

先配置 CANN 环境，例如：

```bash
source /usr/local/Ascend/cann/set_env.sh
```

分别构建和运行：

```bash
cmake -S . -B build/scenario_0 -DSCENARIO_NUM=0 -DCMAKE_ASC_ARCHITECTURES=dav-3510
cmake --build build/scenario_0 -j
./build/scenario_0/topk_algorithm_selection

cmake -S . -B build/scenario_1 -DSCENARIO_NUM=1 -DCMAKE_ASC_ARCHITECTURES=dav-3510
cmake --build build/scenario_1 -j
./build/scenario_1/topk_algorithm_selection
```

成功时输出包含：

```text
Verification PASSED
```

也可运行完整流程：

```bash
./scripts/run.sh
```

脚本对两个场景分别构建，先在 msopprof 外运行并检查正确性，再用独立目录采集。每次采集保留在 `profiles/scenario_N/raw/<run-id>/`；匹配目标 kernel 的原始 CSV 行写入 `profiles/scenario_N/parsed/<run-id>/kernel_rows.csv`。解析文件只是可审计摘录，raw 结果不会被删除或改写。

## 性能边界

性能表使用 msopprof 中目标 Top-K kernel 的 `Task Duration`。该边界不包含 host oracle、ACL 初始化、内存分配、H2D、D2H 和 host 校验。两个场景的 shape、dtype、固定随机输入、输出语义、1 个 kernel launch 和 launch geometry 完全一致。

## Ascend 950 性能记录

2026-08-12 在 20002 节点完成一次设备复核：Ascend 950 系列 `dav-3510`，CANN 9.2.0，
Bisheng/Clang 15.0.5，driver `7.0.t9.0.B791`，采集时频率为 1650 MHz。msopprof 使用
默认 warmup（日志为 5 次），两个 direct run 和 profiler 内运行均输出 `Verification PASSED`。

| Scenario | Kernel | Task Duration (us) | 原始结果目录 | 正确性 |
|:---:|:---|:---:|:---|:---:|
| 0 | `topk_repeated_padded_bitonic_kernel` | 182261.843750 | `profiles/scenario_0/raw/20260812-002736` | PASS |
| 1 | `topk_streaming_bounded_kernel` | 7616.294922 | `profiles/scenario_1/raw/20260812-002736` | PASS |

本次单次 raw 结果中，Scenario 1 相对 Scenario 0 降低 `95.82%`，约 `23.93x`。这是该固定
shape 和工具链下的算法选择证据；正式发布结论仍应补充多 run 中位数和离散程度，不能把
该倍数外推到其他 shape、Top-K 或实现。
