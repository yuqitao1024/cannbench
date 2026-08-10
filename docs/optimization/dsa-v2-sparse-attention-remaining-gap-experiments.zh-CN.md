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
