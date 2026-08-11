# Phase-aware Tile 与传输粒度实践

本案例固定同一输入、输出和 host oracle，对比三个 tile 生命周期方案。每个元素的语义始终是：

```text
intermediate = input * operand + 1
output = intermediate + operand
```

每个 block 独立处理 row，不使用跨核同步。输入为固定生成的 `128 x 8192` float32，host 验证全部输出、64 个输出 guard 和 device 回传的逻辑工作量计数。

## 三个场景

| 场景 | 结构 | 教学目的 | 实测延迟 |
| --- | --- | --- | --- |
| `SCENARIO_NUM=0` | 1024 tile，persistent operand + 两个独立 phase scratch；通过 GM workspace 交接 phase | 展示按各 phase buffer 容量保守求和如何限制 tile，并增加 tile、copy 和 barrier 轮数 | 88.457001 us |
| `SCENARIO_NUM=1` | 2048 tile，persistent operand + 一个跨 phase 复用 scratch；原地计算 | 展示按真实 live-set 峰值复用不重叠 scratch，扩大 tile 并保留持久 operand | 64.139999 us |
| `SCENARIO_NUM=2` | 4096 outer tile + 1024 streamed subtile；每个 subtile 重放整块 operand | 展示过大 live-set 和粗粒度 copy 合并可能带来的占用或重复搬运代价 | 81.013000 us |

程序打印以下静态逻辑计数并与 host 期望逐项比较：outer round、`asc_syncthreads` 动态轮数、逻辑 GM 读写元素、逻辑 UB copy 元素、源码声明 UB 字节和 phase worksheet 的 peak live-set 字节。这些计数描述源码工作量，不等于 cache 后的物理流量、编译器资源元数据或实际耗时。

## 独立因子边界

容量、同步轮数、传输粒度是三个独立因子。A/B/C 用于把三类现象同时放在一个可审计案例中，但 A 到 B 或 B 到 C 的总延迟差不能归因于某一个因素：tile 大小会改变同步轮数，scratch 复用会改变 workspace 流量，outer/subtile 组合又会改变 copy 粒度与声明 live-set。

若要形成因果结论，必须补充固定另外两个因子的控制组，例如固定 tile 只切换 scratch alias、固定 buffer layout 只改变 tile、固定 compute subtile 只改变 coarse copy 范围。本文不把“copy 次数更少”预设为更快。

2026-08-12 在 20002 节点、CANN 9.2.0、`dav-3510`、1650 MHz 完成单次采集，三个
direct/profiler run 均正确。Scenario 1 相对 0 降低 `27.49%`；Scenario 2 比 1 回退
`26.31%`，保留了过大 live-set 与粗粒度 replay 的有效反例。raw 目录分别为
`profiles/scenario_0_tuNtp2`、`scenario_1_J82ySE`、`scenario_2_egof6O`。

## 构建与正确性

```bash
cmake -S . -B build/scenario_0 -DSCENARIO_NUM=0
cmake --build build/scenario_0 -j
./build/scenario_0/phase_aware_tile
```

将 `SCENARIO_NUM` 改为 1 或 2 可分别构建其余场景。成功时输出 `Verification PASSED`。

完整流程：

```bash
./scripts/run.sh
```

脚本对三个场景独立 build，先执行正确性，再用 msopprof 采集。每次采集使用唯一目录，完整 `raw` 输出、stdout 和环境摘要均保留；脚本不设置 warmup，采用当前 msopprof 默认行为。

## Profile 解释

当前三个场景每次应用调用都只有 1 个相关 kernel。应先核对 raw timeline 的应用调用数和 replay 行，再按调用统计相关 kernel 的 Task Duration。Task Duration 是 device-work 边界，不包含 host dispatch；若以后变为多 kernel，必须按同一次调用聚合所有 stage，并额外采集相同同步点的端到端区间。

需要同时记录具体 Ascend 950/950PR 型号、CANN/Bisheng、driver/firmware、编译选项、频率、重复次数和离散程度。源码声明字节不能代替编译器资源报告，逻辑 copy 计数也不能代替 profiler 的物理流量证据。
