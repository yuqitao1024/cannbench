# Lane-local 不变量复用实践

## 概述

本样例对应 Ascend 950 性能案例候选 3.9/P0，比较小型 lane-owned metadata 在多个 pass 间的三种生命周期。实验不是为了预先证明 lane-local 一定更快，而是把重复 GM 工作、shared UB 访存与同步、lane-local 活跃值带来的寄存器压力放在同一语义和 launch 边界下测量。

| Case | 编译开关 | 不变量生命周期 | 每次调用 |
|:---:|:---|:---|:---:|
| 0 | `SCENARIO_NUM=0` | 每个 pass 重复加载 input/raw metadata，并重算 metadata/value/validity | 1 次 kernel launch |
| 1 | `SCENARIO_NUM=1` | 写入 shared UB，一次 block barrier 后每 pass 共享加载 | 1 次 kernel launch |
| 2 | `SCENARIO_NUM=2` | 每 lane 保留 value/metadata/validity mask 并跨 pass 复用 | 1 次 kernel launch |

三个场景均为 64 blocks、每 block 512 threads。每 lane 固定拥有最多 4 个元素，并按相同顺序执行 8 个 pass。输出是逐元素 `uint32` 确定性累加结果，固定 seed 为 `20260812`，host oracle 做精确比较。

## 工作所有权与 tail

单个 block 覆盖 2048 个逻辑元素：

```text
block -> lane(threadIdx.x) -> [lane, lane+512, lane+1024, lane+1536]
```

运行时 `element_count` 决定 validity。默认正确性套件覆盖一个元素、`tile-1`、完整 tile、`tile+1`、`capacity-13` tail 和满容量。每次 launch 前，完整输出以 `0xa5a5a5a5` sentinel 初始化；host oracle 除逐项检查有效输出外，还检查 `[element_count, capacity)` 保持 sentinel，验证三条路径没有写出有效范围。

## 三组对照

### Case 0：重复加载与重算

pass loop 内的 GM input/raw metadata load 使用 volatile，随后重复生成 compact metadata 和 prepared value。其预期瓶颈是重复 GM 指令与整数计算；volatile 是受控实验的一部分，否则编译器可能自动把不变量提升到 loop 外，破坏基线。

### Case 1：shared UB 缓存

每 lane 先把 value、metadata 和 validity 写到相同 local index 的 shared UB。一次 `asc_syncthreads()` 保证同一 SIMT block 的生产完成，再进入 8 个 pass；pass 内使用 volatile shared UB load 保留共享访存成本。三个数组合计 24 KiB 动态 UB。

这里不使用 Mutex：producer/consumer 都在同一个 VF 和同一个 SIMT block，block barrier 已足够，没有需要 Mutex 表达的跨 pipeline 顺序。`AscendC::Mutex::Lock/Unlock` 的例外只适用于确有必要的 kernel-local pipeline 同步，本样例不满足该条件。

### Case 2：lane-local 复用

每 lane 在 pass loop 前构造 `value[4]`、`metadata[4]` 和 4-bit validity mask，之后只读取自己的局部状态。源码没有共享 UB 或新增 barrier。预期收益来自删除重复 load/recompute；潜在代价是更长的活跃区间和寄存器压力。源码局部数组不等于已证明使用寄存器，设备实验还必须检查 compiler resource metadata 和 spill。

## 编译与运行

配置 CANN 环境后，可单独编译任一场景：

```bash
cmake -S . -B build/scenario_2 \
  -DCMAKE_ASC_ARCHITECTURES=dav-3510 \
  -DSCENARIO_NUM=2
cmake --build build/scenario_2 --parallel
./build/scenario_2/lane_local_reuse
```

不带参数会运行完整边界/tail 套件；传入 `1..131072` 的 `element_count` 时只运行一个 shape。全部输出匹配后打印 `Verification PASSED`。

## 正确性后 profile

```bash
./scripts/run.sh
```

脚本依次构建三个场景，先运行完整 host oracle 验证，再 profile `element_count=131059`。msopprof 使用默认参数，不显式设置 warmup 或 metric group。每次运行创建唯一的：

```text
profiles/scenario_<N>/raw/<RUN_ID>/
profiles/scenario_<N>/parsed/<RUN_ID>/kernel_rows.csv
```

raw 目录完整保留，`kernel_rows.csv` 只用于目标 kernel 归属审计。三个场景每次调用均只有一个相关 kernel，因此比较边界是各自同一次调用的 `Task Duration`；不能把 profiler 初始化、正确性套件的多个 shape 或不同 metric group 混入比较。

## 性能结果

2026-08-12 在 20002 节点、CANN 9.2.0、`dav-3510`、1650 MHz 完成单次设备采集；全部
边界/tail direct run 和 profiler 内运行均正确。

| Case | 预期瓶颈 | Task Duration | compiler resources | 相对结果 |
|:---:|:---|---:|:---|:---|
| 0 | 重复 GM load 与重算 | 8.312000 us | 未采集 | 基准 |
| 1 | shared UB load 与 barrier | 6.038000 us | 未采集 | 降低 27.36% |
| 2 | lane-local 活跃值与潜在寄存器压力 | 4.468000 us | 未采集 | 降低 46.25% |

Scenario 2 比 shared UB 路径再降低 `26.00%`。raw 位于
`profiles/scenario_0..2/raw/20260812-004212`。compiler resource/spill 尚未采集，因此该结果
支持保留候选，但解释寄存器压力仍需补充资源元数据和多 run 离散度。
