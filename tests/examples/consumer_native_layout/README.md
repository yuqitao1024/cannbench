# Consumer-native Layout 优化实践

## 概述

当下游 consumer 需要 16x16 blocked layout 时，producer 先写 row-major 再做显式 packing 会产生额外 GM workspace、读写和 launch。本案例比较完整生产边界：

| Case | 编译选项 | 完整路径 |
|:---:|:---|:---|
| 0 | `SCENARIO_NUM=0` | row-major producer kernel + explicit blocked pack kernel |
| 1 | `SCENARIO_NUM=1` | producer 直接按 consumer blocked offset 写一个 kernel |

这里有意使用纯 SIMT/C 地址映射，不强依赖工具链版本不确定的 ND2NZ API。案例展示的是“在真正消费边界生成布局”，不是宣称某种具体硬件 copy-format 一定更快。

## 数据合同

| 项目 | 规格 |
|:---|:---|
| 逻辑输入 | `[4096, 256]`，`float32`，row-major |
| producer | 每元素 `input + 1.0` |
| consumer layout | 16x16 blocked，物理 `[256, 16, 16, 16]` |
| 轴顺序 | `[row_block, col_block, row_inner, col_inner]` |
| 正确性 | 固定随机种子 `20260812`，C 风格 host oracle 验证完整物理 output |
| Launch | 每个 kernel 64 个 block、512 个 SIMT 线程 |
| 目标 | Ascend 950PR / Ascend 950DT，`dav-3510` |

Case 0 分配 4 MiB row-major GM workspace，先后 launch 两个 kernel。Case 1 不分配该 workspace，只 launch 一个 kernel。两者输入、producer 运算、最终 consumer-layout 输出和同步点相同。

## 预期瓶颈

- Case 0：row-major GM workspace 的完整写回和重读、显式 packing 地址计算、第二次 launch。
- Case 1：blocked output 的 store 地址模式、producer 运算以及单次 launch 固定成本。

Case 1 虽然删除中间转换，但直接 blocked store 可能改变访问合并和指令调度，因此“更少 kernel/bytes”只是待 profile 验证的瓶颈假设，不是性能结论。

## 编译和正确性

配置 CANN 环境后执行：

```bash
cmake -S . -B build/scenario_0 -DSCENARIO_NUM=0 -DCMAKE_ASC_ARCHITECTURES=dav-3510
cmake --build build/scenario_0 -j
./build/scenario_0/consumer_native_layout

cmake -S . -B build/scenario_1 -DSCENARIO_NUM=1 -DCMAKE_ASC_ARCHITECTURES=dav-3510
cmake --build build/scenario_1 -j
./build/scenario_1/consumer_native_layout
```

两个场景都必须输出：

```text
Verification PASSED
```

完整采集：

```bash
./scripts/run.sh
```

脚本先在 profiler 外做正确性检查，再运行 msopprof。每个 run 使用唯一目录，raw 数据保存在 `profiles/scenario_N/raw/<run-id>/`；解析目录保留 `kernel_rows.csv` 和 `aggregate.csv`，前者列出全部 selected kernel 原始行，后者记录 expected launch count、observed rows、captured calls 和完整生产边界均值。

## 性能边界

Case 0 必须将 `row_major_producer_kernel` 与 `explicit_blocked_pack_kernel` 的 `Task Duration` 按一次 application call 相加；不能把两行当作重复样本平均。Case 1 的完整边界是 `consumer_native_layout_kernel`。由于 launch count 和 GM workspace 不同，本表比较的是完整生产边界的 device-work sum，不是单 kernel 微基准，也不包含 launch gap 或 host wall time。脚本不自行设置 profiler warmup 参数。

## Ascend 950 性能记录

2026-08-12 在 20002 节点、CANN 9.2.0、`dav-3510`、1650 MHz 完成单次设备采集；两个
direct run 和 profiler 内运行均正确。

| Scenario | Selected kernels | 完整边界 Task Duration (us) | Raw profile | 正确性 |
|:---:|:---|:---:|:---|:---:|
| 0 | producer + explicit pack | 29.364000 | `profiles/scenario_0/raw/20260812-004441` | PASS |
| 1 | consumer-native producer | 15.025000 | `profiles/scenario_1/raw/20260812-004441` | PASS |

Scenario 0 由 producer `14.632000 us` 和 pack `14.732000 us` 相加；Scenario 1 相对完整
基线降低 `48.83%`，约 `1.95x`。结果不含两次 launch 间 gap，正式发布仍需补充同步完整
边界、多 run 中位数和离散度。
