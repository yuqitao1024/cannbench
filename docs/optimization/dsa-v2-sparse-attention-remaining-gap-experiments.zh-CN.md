# DSA V2 Sparse Attention 剩余差距实验设计

## 1. 目标与基线

本实验从 `main@c7b23cf` 的 V3.2 decode shared-gather rolling kernel 出发，
逐项验证以下四类开销是否仍有可测收益：

1. selected128 rolling 流水中的细粒度同步；
2. 两个 Value256 阶段造成的两次 PV handoff 和 output update；
3. online softmax 对同一 tile 的 indices/validity 重复读取；
4. 随机 KV gather 的 `GM -> UB -> shared GM -> L1` 中转和搬运粒度。

canonical case 固定为：

```text
B=2, Q=2, H=128, C=32768, selected=2048, Dqk=576, Dv=512, BF16
```

比较边界是 Sparse Attention fused kernel。所有候选保持相同输入、输出语义、
任务映射、selected128 outer tile、三槽 rolling 结构和现有 operator plugin。

## 2. 实验方法

四个方向不一次性混改。每个候选从同一基线独立产生，先验证精度，再采集性能；
只有单项确认有收益后，才允许进入组合实验。无收益或不稳定的候选保留证据并撤出
后续组合，避免无法归因。

性能采集必须走 CannBench 框架现有 Ascend profiling 路径。msopprof 仅使用：

```text
--aic-metrics=BasicInfo
--launch-count=10
```

除 `--output` 外，不手工增加或覆盖 replay、warm-up、kill、kernel-name 等参数，
其余行为全部使用框架和 msopprof 默认值。每个候选至少采集两次独立 framework run，
核对实际 kernel、launch 数和频率后再比较中位数与离散范围。

## 3. 候选一：Gather32 批量写回

当前四个 AIV 共同产生 selected128 tile，每个 AIV 负责 32 行，但独立 gather staging
只有 `16 x 576 BF16`。因此每个 AIV、每个 tile 要执行两轮 16-row gather、
`UB -> shared GM` 搬运及对应 MTE event。

第一候选把 staging 扩为 `32 x 576 BF16`：随机行仍逐行 `GM -> UB`，但 32 行只做
一次 `UB -> shared GM` 搬运。该实验保持 shared-GM 数据量和四 AIV ownership 不变，
只验证更粗搬运粒度能否减少 MTE event 与搬运控制开销。

UB 墍量为 `16 x 576 x 2 = 18432 B`，需要重新核算动态 UB 总量、编译器保留区和
运行时资源元数据。若超过当前有效预算，不通过缩减其他业务 buffer 强行实现。

## 4. 候选二：逻辑 PV512 Handoff

不尝试单次物理 `[64,128] x [128,512]` MMAD，因为 Value512 与 FP32 result512 会
超过现有 L0B/L0C 单阶段容量。Cube 侧仍执行两个 Value256 MMAD，但把两个结果写入
同一个逻辑 `32 x 512 FP32` AIV staging，完成后只发布一次 ready，并由 AIV 做一次
512 维 output update。

该候选单独验证减少一次 PV ready/free 协议和一次 output-update VF boundary 的收益。
它会比当前 `32 x 256 FP32` staging 增加 32768 B UB，因此与 Gather32 先独立测试；
只有二者分别有收益且组合后的实际 UB 预算成立，才测试组合版本。

## 5. 候选三：Tile-local Validity Metadata

softmax max/exp 两遍需要相同的 selected128 validity/causal 判断。候选在每个 active
AIV 的私有 UB 中为当前 tile 缓存 128 个 index 或等价 validity metadata，避免第二遍
再次读取 GM indices。

仓库历史实验曾因 metadata 生命周期与 rolling slot 复用不正确而出现精度失败。
本次必须让 metadata 与 slot/generation 一一对应，先验证单 tile、slot 首次复用、
完整 16 tile 和多 query，再进行性能采集。不得让两个 AIV 共享未同步的 UB metadata。

## 6. 候选四：同步收敛

同步优化不以直接删除 flag/event 为起点。先根据前三个候选确定每个 buffer 的 producer、
consumer、ready、free 和 slot reuse 边界，只合并因 Gather32 或逻辑 PV512 自然消失的
重复 handoff，或经指令时间线证明没有独立依赖的本地 MTE event。

不新增 C++ Basic API 依赖，不新增 CrossCore 协议。算子源码继续收敛在 C API、
Tensor API 和 SIMT API 边界；现存过渡性同步调用不作为扩展接口使用。

## 7. 验证与接受标准

每个候选依次通过：

1. operator-local 源码契约测试；
2. 远端干净编译和实际加载路径核对；
3. canonical decode seed 7、19 完整精度，output/LSE mismatch 均为 0；
4. CannBench framework BasicInfo profiling，两次独立 run、每次 launch-count 10；
5. 原始 CSV 中 kernel 归属、launch 数、Block/Mix Block Dim 和频率审计。

接受性能候选要求精度完整通过，并且两次 framework run 的中位数均优于同条件基线。
若收益落在 profiler 抖动范围内，则记录为无明确收益，不进入组合版本。

## 8. 实验记录

### 8.1 Gather32：拒绝

Gather32 候选把每个 AIV 的 gather staging 从 `16 x 576 BF16`
扩大为 `32 x 576 BF16`，动态 UB 从 178816 B 增至 197248 B。远端干净编译通过，
seed 7、19 的 output/LSE mismatch 均为 0。

性能严格走 CannBench framework，msopprof 只显式收到 `BasicInfo` 和
`launch-count=10`，其余参数为默认值。两轮结果如下：

| 实现 | Run 1 | Run 2 | 两轮均值 |
| --- | ---: | ---: | ---: |
| main baseline | 98.033997 us | 97.447998 us | 97.740998 us |
| Gather32 | 99.214996 us | 98.678001 us | 98.946499 us |

Gather32 平均回退 1.205501 us，约 1.23%。说明减少一次 16-row UB-to-GM 搬运和
一组 MTE event 不足以抵消更大 staging/单次搬运带来的代价。该候选已从最终代码中
移除，不进入后续组合。

本机原始 framework artifacts 位于：

```text
/tmp/cannbench-dsa-v2-gap-baseline-r1
/tmp/cannbench-dsa-v2-gap-baseline-r2
/tmp/cannbench-dsa-v2-gap-gather32-r1
/tmp/cannbench-dsa-v2-gap-gather32-r2
```

### 8.2 逻辑 PV512 Handoff：保留

逻辑 PV512 候选保留两个 Value256 MMAD/Fixpipe，只把两块
`32 x 256 FP32` 结果放入同一个 `32 x 512 FP32` staging。每个 selected128 tile
由 Cube 完成两段结果后只发布一次 PV ready，AIV 只做一次 512 维 output update，
因此不改变矩阵计算量，只减少一次 ready/free handoff 和一次 output-update VF 边界。

动态 UB 为 211584 B，低于 216 KiB 上限；operator-local 测试 `43 passed`，远端
干净编译通过。seed 7、19 的完整 decode 精度中，output/LSE mismatch 均为 0。

性能沿用与 baseline 完全相同的 CannBench framework 路径。msopprof 只显式收到
`BasicInfo` 和 `launch-count=10`，其余参数保持默认。两轮目标 fused kernel 都是
`Block Dim=16`、`Mix Block Dim=32`、1650 MHz：

| 实现 | Run 1 | Run 2 | 两轮均值 |
| --- | ---: | ---: | ---: |
| main baseline | 98.033997 us | 97.447998 us | 97.740998 us |
| 逻辑 PV512 handoff | 91.224998 us | 90.707001 us | 90.966000 us |

两轮分别降低 6.808999 us 和 6.740997 us，平均降低 6.774998 us，约 6.93%。
该候选满足“两轮均优于 baseline”的接受标准，予以保留，并作为后续增量候选的
新对照点。

本机原始 framework artifacts 位于：

```text
/tmp/cannbench-dsa-v2-gap-pv512-r1
/tmp/cannbench-dsa-v2-gap-pv512-r2
```

### 8.3 Slot-private Validity Metadata：拒绝

Validity metadata 候选在 UB 末尾增加三个 slot-private 的 `128 x uint32` validity
数组，动态 UB 从 211584 B 增至 213120 B。每个 tile 由 warp 0 读取一次 GM
indices 并生成 validity，经过一次 `__syncthreads()` 后，32 个 head warp 的 max 和
exp 两遍只读取 UB metadata。

operator-local 目标测试 `43 passed`，远端干净编译通过。seed 7、19 都覆盖 negative、
out-of-range、causal-future 和完整 16 tiles，output/LSE mismatch 均为 0。

两轮 CannBench framework BasicInfo 结果如下，目标 kernel 均为 `Block Dim=16`、
`Mix Block Dim=32`、1650 MHz：

| 实现 | Run 1 | Run 2 | 两轮均值 |
| --- | ---: | ---: | ---: |
| 逻辑 PV512 handoff | 91.224998 us | 90.707001 us | 90.966000 us |
| PV512 + validity metadata | 91.082001 us | 91.253998 us | 91.168000 us |

Run 1 快 0.142997 us，但 Run 2 慢 0.546997 us；两轮均值回退 0.202000 us，
约 0.22%。减少重复 GM indices 读取的收益不足以抵消 validity UB 访问和新增块内同步，
且结果不满足“两轮均优于对照”的接受标准。该候选已从最终代码中移除。

本机原始 framework artifacts 位于：

```text
/tmp/cannbench-dsa-v2-gap-validity-r1
/tmp/cannbench-dsa-v2-gap-validity-r2
```

### 8.4 同步收敛边界

逻辑 PV512 已把两个 Value256 阶段的两次 PV ready/free 和两次 output update 自然
收敛为一次。剩余 AIV `output update -> V/MTE3 event -> PV free` 保证 Cube 不会在
AIV 读完 PV UB 前覆写结果，AIC 侧 `PV free -> 两段 MMAD/Fixpipe -> PV ready` 则
保证 AIV 只消费完整的 512 维结果。这两条仍是实际 buffer ownership 边界，不能在
不改变协议或缺少额外时序证据时删除，因此不再构造无依赖证明的删同步候选。

### 8.5 PV512 阶段接受结果

最终源码只保留经过验证的逻辑 PV512 handoff operator source/test。移除 validity
候选后重新干净编译，seed 7、19 的完整 decode output/LSE mismatch 再次均为 0。

最终两轮 CannBench framework BasicInfo 结果为：

| 实现 | Run 1 | Run 2 | 两轮均值 |
| --- | ---: | ---: | ---: |
| main baseline | 98.033997 us | 97.447998 us | 97.740998 us |
| 最终 PV512 | 90.723000 us | 91.130997 us | 90.926999 us |

最终均值降低 6.813999 us，约 6.97%。两轮目标 fused kernel 均为
`Block Dim=16`、`Mix Block Dim=32`、1650 MHz，framework failure count 均为 0。
msopprof 仍只显式使用 `BasicInfo` 和 `launch-count=10`，其他参数保持框架默认。

本机最终 artifacts 位于：

```text
/tmp/cannbench-dsa-v2-gap-final-r1
/tmp/cannbench-dsa-v2-gap-final-r2
```

operator-local 目标测试为 `43 passed`，`git diff --check` 通过，公共 backend/core/CLI
无修改。相对 `main@c7b23cf`，Basic API include、Set/Wait 和 CrossCore 调用数量均未
增加。完整 SIMT 测试目录仍有一个基线已有失败：ABI 后缀测试要求所有 launch symbol
以 `_v2` 结尾，而现有 rolling symbol 为
`launch_sparse_attention_head64_fused_hd576_bf16_v2_rolling_restored`；该问题不由本次
优化引入，也未在本分支顺带修改。

按照 V2 decode published lane 现有的“两轮 provenance-audited 结果取较低 workflow”
约定，最终 checkpoint 由已发布的 fused Indexer `0.084852998 ms` 和本实验较低的
Sparse Attention `0.090723000 ms` 相加，得到 `0.175575998 ms`。canonical run
`opbench-ascend-950pr-simt-v2-dsa_decode-realistic-bfloat16` 的 schema、run id 和其他
字段保持不变。

### 8.6 PV512 后续关键路径：Lane-local Softmax 复用

在 PV512 checkpoint 上重新采集同一 canonical case。新的 CannBench framework
BasicInfo 基线为 91.004997 us；独立的 InstrTimeline 和 PipeTimeline 归因采集分别为
90.666 us 和 90.861 us。详细时间线只用于归因，不与 BasicInfo 延迟混算。

PipeTimeline 中 Vector 关键路径约 85 us，Cube 为 27.452 us，AIV MTE2 约 18 us。
稳态 selected128 round 的周期约 4.95--5.05 us，其中 Vector 区间约
4.73--4.85 us；两段 gather MTE2 合计约 1.0--1.3 us，基本被 Vector 覆盖。因此
没有继续修改 gather 中转或跨核协议，而是转向仍在关键路径上的 softmax VF。

每个 selected128 tile 中，一个 lane 固定处理最多 4 个位置。原实现的 max pass 和
exp pass 都会读取同一份 GM indices，并再次读取相同 UB score。保留候选在 max pass
中把 4 个 scaled score 缓存在 lane-local `float[4]` 中，同时用一个 4-bit mask 记录
validity；exp pass 直接复用这些值。该修改不增加共享 UB、不增加 `__syncthreads()`，
也不改变 gather、CrossCore 协议、Cube 逻辑、UB 布局或 API 边界。

这与 8.3 中拒绝的 slot-private validity metadata 不同：旧候选由 warp 0 生成共享
UB metadata，其他 head warp 经块内同步后读取；本候选由每个 head warp 的 lane
自行读取一次 indices/score，并在 lane-local 生命周期内跨两遍 softmax 复用，避免了
共享 UB 访问和新增同步。

远端 clean build 成功。seed 7、19 均覆盖 negative、out-of-range、causal-future 和
完整 16 tiles，output/LSE mismatch 均为 0。两轮独立 CannBench framework
BasicInfo 结果如下：

| 实现 | Run 1 | Run 2 | 两轮均值 |
| --- | ---: | ---: | ---: |
| PV512 checkpoint | 90.723000 us | 91.130997 us | 90.926999 us |
| PV512 + lane-local softmax 复用 | 81.133003 us | 81.343002 us | 81.238003 us |

候选相对 PV512 checkpoint 两轮均值降低 9.688996 us，约 10.66%；相对本轮新采集的
91.004997 us BasicInfo 基线降低 9.766995 us，约 10.73%。两轮目标 fused kernel
均为 `Block Dim=16`、`Mix Block Dim=32`、1650 MHz，framework failure count 为 0。
msopprof 仍只显式使用 `BasicInfo` 和 plugin 提供的 `launch-count=10`，其他参数保持
框架默认。该候选满足精度和两轮性能接受标准，予以保留。

本机原始 artifacts 位于：

```text
/tmp/cannbench-dsa-pv512-critical-basic-20260810
/tmp/cannbench-dsa-pv512-critical-instr-20260810
/tmp/cannbench-dsa-pv512-critical-pipe-20260810
/tmp/cannbench-dsa-pv512-lane-cache-r1-20260810
/tmp/cannbench-dsa-pv512-lane-cache-r2-20260810
```

按照现有 published lane 的“两轮 provenance-audited 结果取较低 workflow”约定，
本轮 checkpoint 由已发布的 fused Indexer `0.084852998 ms` 和本实验较低的
Sparse Attention `0.081133003 ms` 相加，得到 `0.165986001 ms`。canonical run
`opbench-ascend-950pr-simt-v2-dsa_decode-realistic-bfloat16` 的 schema、run id 和其他
字段保持不变。
