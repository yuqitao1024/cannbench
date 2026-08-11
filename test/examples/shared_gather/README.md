# 跨消费者 Shared Gather 优化实践

## 概述

两个 consumer 需要同一组离散选择的数据时，分别 gather 会重复地址计算和不规则 GM 读取。本案例保持一个 kernel、相同 launch、相同输入和两个完整输出，只把“两次独立 gather”改为“一次 gather 后由 owner thread 复用”，用于隔离数据复用本身的效果。

| Case | 编译选项 | 数据路径 |
|:---:|:---|:---|
| 0 | `SCENARIO_NUM=0` | consumer A/B 各自从同一 selected address 执行一次 volatile GM gather |
| 1 | `SCENARIO_NUM=1` | producer gather 一次，thread-local value 交给 consumer A/B |

复用发生在同一 SIMT thread 内，不使用共享 UB、中间 GM workspace、flag、barrier 或跨核协议。

## 样例规格

| 项目 | 规格 |
|:---|:---|
| source | `[65536, 128]`，`float32`，row-major |
| selected indices | `[8192]`，`uint32`，允许重复 row |
| consumer A | `[8192, 128]`，`float32`，gathered value `+ 1.0` |
| consumer B | `[8192, 128]`，`float32`，gathered value `- 1.0` |
| 正确性 | 固定随机种子 `20260812`，C 风格 host oracle 逐项验证两个完整输出 |
| Launch | 64 个 block，每 block 512 个 SIMT 线程，单 kernel |
| 目标 | Ascend 950PR / Ascend 950DT，`dav-3510` |

Case 0 使用两个显式 `volatile GM` 读取函数，避免编译器把教学基线中的重复 gather 自动合并。Case 1 的 producer 只读取一次，再分别执行两个 consumer 变换。

## 预期瓶颈

- Case 0 预期瓶颈：相同离散地址的重复 GM load、地址生成和访存延迟。
- Case 1 预期瓶颈：一次离散 source load、两个完整输出的 GM store，以及可能的 launch 固定成本。

这些只是待设备 profile 验证的假设。源码中的 load 数量不能证明实际流量、cache 命中或性能收益，必须结合目标工具链生成物和 raw profile 判断。

## 编译和正确性

配置 CANN 环境后，在案例目录执行：

```bash
cmake -S . -B build/scenario_0 -DSCENARIO_NUM=0 -DCMAKE_ASC_ARCHITECTURES=dav-3510
cmake --build build/scenario_0 -j
./build/scenario_0/shared_gather

cmake -S . -B build/scenario_1 -DSCENARIO_NUM=1 -DCMAKE_ASC_ARCHITECTURES=dav-3510
cmake --build build/scenario_1 -j
./build/scenario_1/shared_gather
```

两种场景都必须输出：

```text
Verification PASSED
```

完整采集流程：

```bash
./scripts/run.sh
```

脚本分别构建两个 scenario，先在 profiler 外验证，再运行 msopprof。每次运行使用唯一 run id；完整原始证据保存在 `profiles/scenario_N/raw/<run-id>/`，目标 kernel 的原始 CSV 行摘录到 `profiles/scenario_N/parsed/<run-id>/kernel_rows.csv`。解析不会删除或改写 raw 文件。

## 性能边界

性能表只记录目标 gather kernel 的 `Task Duration`，不包括输入生成、host oracle、ACL 初始化、分配、H2D、D2H 和 host 校验。两个 scenario 都只有一个物理 kernel launch；脚本不自行设置 profiler warmup 参数。

## Ascend 950 性能记录

2026-08-12 在 20002 节点完成一次设备复核：Ascend 950 系列 `dav-3510`、CANN 9.2.0、
Bisheng/Clang 15.0.5、driver `7.0.t9.0.B791`、1650 MHz。两个 direct run 和 profiler
内运行均输出 `Verification PASSED`。

| Scenario | Kernel | Task Duration (us) | Raw profile | 正确性 |
|:---:|:---|:---:|:---|:---:|
| 0 | `shared_gather_independent_kernel` | 22.200001 | `profiles/scenario_0/raw/20260812-003942` | PASS |
| 1 | `shared_gather_reuse_kernel` | 18.719999 | `profiles/scenario_1/raw/20260812-003942` | PASS |

Scenario 1 相对 Scenario 0 降低 `15.68%`，约 `1.19x`。这是单次短 kernel raw 结果，
正式发布仍需补充多 run 中位数和离散度。64 block profile 有动态插桩 sub-block 数提示，
BasicInfo 时延完整，其他动态指标需谨慎解释。
