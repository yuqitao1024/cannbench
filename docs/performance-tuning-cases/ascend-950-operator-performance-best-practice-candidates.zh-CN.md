# Ascend 950 算子性能优化最佳实践案例候选

## 1. 文档目的

本文从 `/root/aiagent/cannbench` 和 `/root/aiagent/cannbench-2` 关联的 Codex
会话、仓库设计文档、提交记录和已保留性能结果中，提取有代表性且可以独立验证的
Ascend 950 算子性能优化方法，归纳为 22 个独立优化项，说明每类方法的问题瓶颈、
优化方案、实测结果和适用边界，并筛选适合继续开发为性能调优最佳实践的案例。

本文同时对照 asc-devkit 的以下目录，标记已经存在的案例，避免重复开发：

- [03_simt_api](https://gitcode.com/cann/asc-devkit/tree/master/examples/03_simt_api)
- [05_simd_simt_hybrid](https://gitcode.com/cann/asc-devkit/tree/master/examples/05_simd_simt_hybrid)

对照基于 asc-devkit `master` 提交
`f33e814d7795961e351497b7a90ee0511ece2160`，检查日期为 2026-08-11。

## 2. 证据范围和口径

本次精确匹配工作目录为 `/root/aiagent/cannbench` 或
`/root/aiagent/cannbench-2` 的 Codex rollout 文件共 177 份。已有性能归档截至
2026-08-10 覆盖 163 份 rollout；2026-08-11 新增记录主要是技能整理、文档任务及其
子会话，没有新增上板性能实验。

初次归档从 154 份相关 rollout 中重建了 43 个逻辑会话。详细实验结果和负向结果见
[Observed Performance Patterns](../../.agents/skills/ascend-950-operator-performance-optimization/references/observed-optimization-results.md)。

多数 DSA 多阶段延迟实验的环境如下：

```text
设备：Ascend 950PR，dav-3510
CANN：9.2.0
编译：production -O3
频率：1650 MHz
主要数据类型：BF16
主要 shape：batch=2, query rows=2, context rows=32768, groups=64
             feature width=128, selected rows=2048
             consumer groups=128, secondary widths=576/512
```

Softmax 行归约实验使用相同设备家族和工具链，但 dtype、行宽、外层行数和测量方法
会随实验变化。

本文使用以下测量边界：

- **Kernel**：一个选定设备 kernel 的耗时。
- **Component**：一个注册算子，包括其辅助 kernel。
- **Workflow**：包含多个组件的完整工作流。

不同边界上的百分比不能相加，也不能把 kernel 收益直接当作端到端收益。

## 3. 代表性独立优化项

### 3.1 Top-K 算法复杂度优化

**问题瓶颈**

多 query 路径在每个小输入块后，对 `current_topk + new_candidates` 补齐到 4096
元素并重复执行 bitonic sort。该路径约执行 985088 次 merge 和 7684 万个同步阶段，
算法复杂度和 padding 成本远高于实际有效数据量。

**优化方案**

使用流式选择、radix selection、分层局部 Top-K 或有界 merge；先减少总比较次数、
同步次数和 padding，再考虑指令级优化和 launch geometry。

**代表性结果**

选择阶段为 `3134.926 ms`，优化库实现为 `5.929 ms`，存在 `528.7x` 差距。

**现有样例对照**：asc-devkit 未覆盖。

**案例建议**：强烈推荐。适合作为“算法复杂度优先于微架构调度”的总览案例。

### 3.2 Warp 协作并行主循环

**问题瓶颈**

kernel 已启动大量线程，但 64-channel 主循环仍由单线程串行执行。仅减少线程数只能
减少空闲线程，不能缩短关键路径。

**优化方案**

将一个 32-lane Warp 映射到一个 32-position tile 项，让 lane 协作处理 64 个通道并
完成归约。

**代表性结果**

- Decode producer kernel 降低 `49.4%`，component 降低 `34.1%`，workflow 降低
  `17.2%`。
- 同一映射迁移到多 query 路径后，producer 降低 `50.4%`，workflow 降低
  `21.6%`。
- 只把 VF 线程数从 1024 改为 32/64/128/256 时，workflow 最多改善 `0.8%`。

**现有样例对照**：`03_simt_api/.../warp_divergence` 已使用一个 Warp 协作处理一行。

**案例建议**：通用机制已有样例，不建议重复开发。

### 3.3 单遍确定性压缩

**问题瓶颈**

Top-K 阈值确定后，仍通过 atomic 和多次 threshold scan 做候选压缩，反复读取同一批
候选并产生不必要的原子竞争。

**优化方案**

对每个候选只扫描一次，通过 packed count 构造确定性 offset，直接写入压缩结果。

**代表性结果**

选择链降低 `37.8%-37.9%`，component 降低 `22.9%-23.1%`，workflow 降低约
`10.0%`。

**现有样例对照**：asc-devkit 未覆盖。

**案例建议**：强烈推荐。A/B 边界清晰，能够同时展示原子竞争、重复扫描和确定性输出。

### 3.4 分片直方图和层次归约

**问题瓶颈**

长度 32768 的 context 由一个 block 串行完成选择；低字节 reducer 的 shard tail 和
offset 构造也只使用少量线程，形成明显串行尾部。

**优化方案**

把 context 划分为 16 个 shard，先生成 shard-local histogram，再分层归并；并行化
reducer 的 tail、offset 和 256-bin threshold scan。

**代表性结果**

- 完整选择链由约 `82.031 us` 降至 `56.759-57.279 us`，降低
  `30.2%-30.8%`，workflow 降低 `7.85%-7.98%`。
- 低字节 reducer 由 `28.904 us` 降至 `14.423-14.444 us`，workflow 降低
  `5.56%-5.62%`。

**现有样例对照**：asc-devkit 未覆盖。

**案例建议**：强烈推荐。可作为长序列选择和层次归约的独立案例。

### 3.5 多阶段 Kernel Launch 融合

**问题瓶颈**

分布式 Top-K 由五个短 kernel 串行组成，设备工作很短，launch 和阶段间间隔占比高。

**优化方案**

融合为一个 64-task kernel，在四个阶段边界执行生产者 GM 可见性发布和 device-wide
barrier。包括不参加 reducer 计算的任务在内，所有物理任务必须以相同顺序到达每个
barrier。

**代表性结果**

Top-K 降低 `19.9%-21.0%`，完整 Indexer 降低 `7.6%-8.5%`，workflow 降低
`3.6%-4.3%`。

**现有样例对照**：`simd_simt_grid_dim_config` 已展示 VF 调用次数的固定开销，但未
覆盖带 GM 可见性协议的多阶段融合。

**案例建议**：条件性推荐。历史实验使用了任务级 API 例外；CannBench 当前 API 边界
不允许通过该案例重新引入跨核同步。更适合作为 SDK 或平台能力限定案例。

### 3.6 跨消费者复用离散 Gather

**问题瓶颈**

两个 Head64 consumer 使用同一组 selected-128 KV，却分别执行离散 gather，重复产生
不规则 GM 访问和布局准备。

**优化方案**

由四个 AIV producer 共享执行一次 KV gather，再把结果交给两个 consumer。

**代表性结果**

已发布 workflow 由 `208.847 us` 降至 `191.162 us`，降低 `8.47%`。

**现有样例对照**：官方有基础 Gather 和 SIMD/SIMT Gather+Adds，但没有展示跨消费者
复用。

**案例建议**：强烈推荐。该案例直接回答离散访存使用 SIMT 后如何通过数据复用获得
端到端收益。

### 3.7 批量 VF 调用和可见性发布

**问题瓶颈**

对 8192 个 selected row 分别调用 QK/V VF，并逐行执行 DCCI。VF 入口、地址生成和
cache publication 成本远大于每行有效计算。

**优化方案**

在 VF 内批量处理多行，在 tile 或 phase 的第一个真实跨流水消费者边界执行一次可见性
发布。

**代表性结果**

逐行方案使 sparse main kernel 从 `54.483 us` 回退到 `882.803-883.764 us`，慢
`16.21x`，workflow 回退 `40.62%`。

**现有样例对照**：`simd_simt_grid_dim_config` 已证明 VF 调用次数会增加固定开销，但
没有覆盖 DCCI 发布粒度。

**案例建议**：部分推荐。只开发“逐行 DCCI 与按 tile DCCI”这一官方尚未覆盖的差异。

### 3.8 基于真实生命周期调整 Tile 与传输粒度

**问题瓶颈**

容量规划把不同时存活的 L1/UB buffer 全部相加，导致 tile 过小，重复 packing、状态
更新、copy 和同步次数过多。

**优化方案**

按 phase 计算 live-set 峰值，复用不重叠的片上存储；保留持久化 operand，并调整
matrix-product、selected-token 或 outer tile。必要时使用大 outer tile 加小 streamed
subtile，而不是让全部状态同时驻留。传输粒度、预取深度和 copy 数量需要作为独立因子
测量，不能把“copy 次数更少”直接视为收益。

**代表性结果**

- 多次 tile 扩大使 workflow 降低 `2.5%-9.4%`。
- selected-token tile 从 64 增至 128 后，workflow 降低约 `9.4%`。
- outer tile 增至 512 但重放 persistent operand 三次时，仅改善 `0.24%-0.67%`。
- Gather staging 从 16 行扩大到 32 行，虽然删除了一次 UB-to-GM copy 和 MTE event，
  sparse fused 均值仍回退 `1.23%`。
- 删除一个 L1 score handoff 后，fused kernel 只改善 `0.81%-0.87%`，workflow 在两组
  配对中分别回退 `0.33%` 和改善 `0.50%`。
- 把 Indexer packet 扩大到 128-context 并增加双槽 L1 prefetch 后，Score 回退
  `3.40%-4.33%`，workflow 回退 `0.92%-1.55%`。

**现有样例对照**：现有 Matrix Transpose 涉及固定 tile 和双缓冲，但没有展示多阶段
生命周期 worksheet。

**案例建议**：强烈推荐。适合展示容量、生命周期、同步轮数、传输粒度和 tile 大小的
完整权衡，并保留“减少 copy 但性能回退”的反例。

### 3.9 不变量、紧凑元数据和 Lane-local 复用

**问题瓶颈**

多个 phase 重复计算 row offset、range mask、merge weight 和 softmax 中间值。把小型
元数据放入共享 UB 又会引入共享访存和 block barrier。

**优化方案**

- 在重复 gather phase 间保留紧凑 `int32` row offset。
- 每个 lane 只计算一次 merge weight，并跨 512 个输出元素复用。
- 在 lane 寄存器中保留 softmax value 和 validity bit，跨 max/exp pass 复用。

**代表性结果**

- offset 复用使 workflow 降低 `4.2%`。
- merge weight helper 降低 `66.4%`，workflow 降低 `3.7%`。
- lane-local softmax 复用使 fused kernel 降低 `10.66%`，workflow 降低 `5.46%`。
- shared-UB validity cache 的均值反而回退 `0.22%`。

**现有样例对照**：asc-devkit 未覆盖 ownership-local 与 shared metadata 的对照。

**案例建议**：强烈推荐。适合形成“重复计算、共享缓存、lane-local 缓存”三组 A/B。

### 3.10 直接生成消费者布局

**问题瓶颈**

显式 UB transpose、AIV packing、UB-to-L1 copy 和 CrossCore handoff 只是为了生成下游
Cube 需要的布局，造成重复中间转换。

**优化方案**

- Value 保持 row-major，使用 L1-to-L0B copy 的格式转换能力完成转置。
- Query 使用 AIC GM-to-L1 ND2NZ 直接生成 consumer layout。
- 对离散 gather 先连续 staging，再以 16x16 tile 转为目标 blocked layout。

**代表性结果**

- L1-to-L0B 转换使 workflow 降低 `5.00%`。
- GM-to-L1 ND2NZ 使 fused kernel 降低约 `5.4%`，workflow 降低
  `2.22%-2.42%`。
- 连续 staging 后转置使 workflow 降低 `18.25%-18.54%`。

**现有样例对照**：Matrix Transpose 已覆盖通用 GM/UB 转置，但没有覆盖 Cube copy
格式转换和直接生成消费布局。

**案例建议**：推荐。新案例应聚焦硬件 copy-format 边界，避免重复通用转置样例。

### 3.11 合并生产者 Handoff

**问题瓶颈**

两个 Value-256 Cube 结果拥有同一 consumer 和共同生命周期，却分别执行 ready/free
交换和输出更新 VF。

**优化方案**

合并为一个逻辑 PV-512 handoff，只执行一次 ready/free 交换和一次输出更新 VF。

**代表性结果**

Sparse fused kernel 降低约 `6.97%`，对应 workflow checkpoint 降低约 `3.61%`。

**现有样例对照**：asc-devkit 未覆盖。

**案例建议**：适合作为融合算子的进阶案例，优先级低于通用数据复用和布局案例。

### 3.12 双缓冲生产者/消费者流水

**问题瓶颈**

数据搬入、计算和后处理按 tile 串行执行，64 个重复 tile 之间存在可以隐藏的等待。

**优化方案**

使用两槽 buffer，把 ready 和 free 状态分方向管理，使当前 tile 计算与下一 tile 搬运
重叠。同步协议必须同时满足：

- ready 和 free 两个方向使用不同状态或 flag ID，不能复用同一状态表达相反所有权；
- 对多个 flag ID 明确 producer publication 和 consumer wait 的偏序，不能假设等待后
  发布的 flag 会顺带消费先前 flag；
- 在发布 buffer 可复用之前，保留 `PIPE_V -> PIPE_MTE3` 等本地流水完成边；
- device-wide barrier 只保证控制收敛时，必须另行执行平台要求的 GM cache publication；
- 所有物理任务以相同顺序到达每个全局 barrier，包括被 mask 掉计算的任务。

仅当每个物理 worker 有多个重复项并且确实存在 copy/compute overlap 时采用。

**代表性结果**

BF16 producer 由 `148.363 us` 降至 `82.889 us`，降低 `44.1%`；workflow 降低
`12.2%`。相同方法用于九元素短行时，固定事件和 VF 成本使中位数回退 `22.8%`。
初版两槽协议复用 ready/free flag ID 时发生设备 hang；删除本地 V-to-MTE3 完成边后
kernel 能结束，但 output 错误 `247866/262144`，LSE 错误 `512/512`。

**现有样例对照**：SIMT/混合 Matrix Transpose 和 `simd_simt_hash_table_mte_queue`
已完整覆盖。

**案例建议**：不建议重复开发。

### 3.13 UB 布局与 Bank Conflict

**问题瓶颈**

逻辑布局使同一 Warp 的一条访问指令命中相同 bank group/subbank 的不同地址，UB 访问
串行化。

**优化方案**

使用 padding、capacity-neutral column-major producer 或改变单条指令的 lane stride；
必须把 padding copy、地址计算、store pattern 和 transpose 成本纳入测量边界。

**代表性结果**

- capacity-neutral column-major producer 使 kernel 降低约 `32%`，workflow 降低
  `10.6%-11.2%`。
- 连续 staging 后转置使 workflow 降低 `18.25%-18.54%`。
- 多种预测可减少冲突的 padding、remap 和 shuffle 方案实际无收益或回退。

**现有样例对照**：SIMT 和混合 Matrix Transpose 已提供完整案例。

**案例建议**：不建议重复开发。

### 3.14 Packed FP16/BF16 访问

**问题瓶颈**

标量 load/store 和转换指令过多，但盲目扩大访问宽度可能增加寄存器生命周期、tail
处理和 unpack 成本。

**优化方案**

只在相邻且事务确实合并的边界使用 `half2` 或 BF16 packed 类型，继续使用 FP32
累加；每个 VF 边界独立验证。

**代表性结果**

- FP16 packed reduction 在 30-case 新包上平均降低 `12.07%`。
- 一个 BF16 packing 边界使 workflow 降低 `1.32%`。
- 其他 BF16 边界从仅改善 `0.41%` 到回退 `2.73%`；保持 scattered destination 的
  32/64-bit 宽访问可使 workflow 回退 `21.7%-67%`。

**现有样例对照**：`short_vector_add` 已覆盖 `half2`。

**案例建议**：不建议重复开发；BF16 可作为现有案例的 dtype 扩展，而非新主题。

### 3.15 Row Reduction 扫描融合与在线统计

**问题瓶颈**

宽行 Softmax 多次扫描同一行并启动多个 kernel；超宽行无法整体驻留 UB；最终写回
路径在部分 shape 上由 direct-GM bulk traffic 主导。

**优化方案**

- 对 `32768 < width <= 51200` 融合 max/sum stats 扫描和 launch。
- 对 `width > 51200` 使用数值稳定的 tiled online `(max,sum)`。
- 仅在 profile 证明写回阶段占主导时，使用 tiled MTE2/compute/MTE3 写回流水。
- 融合后仍需优化 inner schedule；保留 unfused control。

**代表性结果**

- stats fusion 降低 `25.30%-37.67%`。
- online `(max,sum)` 降低 `17.16%-21.84%`。
- 完整写回 component 降低约 `27.3%` 和 `40.5%`，被修改的写回阶段降低 `44%-56%`。
- 仅融合而保持标量 normalize/write 时回退约 `32%`；packed inner schedule 后才获得
  `21.21%-25.15%` 收益。

**现有样例对照**：asc-devkit 未覆盖在线稳定归约和多扫描融合。

**案例建议**：强烈推荐。可以同时展示容量边界、数值稳定性、kernel fusion 和流水化。

### 3.16 按行为区间做 Shape 特化

**问题瓶颈**

一个通用路径不能同时适配短行、中行、整行驻留和多 tile；逐 exact shape 生成专用
kernel 又难以维护，并容易只优化单点。

**优化方案**

按会改变算法和内存调度的行为分桶，例如 129-256、512/1024、8192-32768、
32768-51200 和超宽行；保留通用 fallback，并实测 dispatch threshold。

**代表性结果**

- 129-256 桶的六个 case 合计由 `1950.873 us` 降至 `1427.170 us`，降低
  `26.84%`。
- width-1024 两个 case 由 `810.992 us` 降至 `738.867 us`，降低 `8.89%`。
- 固定 `49152 x 9` shape 的实验虽达到约 `4.7-5.3 us`，但因不具备通用 dispatch
  价值被回退。

**现有样例对照**：asc-devkit 未覆盖行为分桶 dispatch。

**案例建议**：强烈推荐。应强调行为分桶，而不是逐 shape 硬编码。

### 3.17 Grid、线程数和寄存器联合调优

**问题瓶颈**

最大核数、最大线程数或消除少量 spill 不等于最快。增加资源可能带来调度开销、寄存器
压力和端到端辅助路径回退。

**优化方案**

联合扫描 `blocks x threads x launch_bounds`，检查寄存器和 Stack，但以真实设备端到端
耗时作为保留标准。

**代表性结果**

- 固定 16384 元素扫描中，`32x512` 比 `64x256` 快 `6.3%`。
- 把 32 B Stack 降为 0 后，两个版本中位数为 `98.219 us` 和 `98.253 us`，基本相同。
- 某 many-row kernel 加速 `3.59x`，但同步公共边界反而回退 `48.5%`。

**现有样例对照**：`grid_dim_config`、`simd_simt_grid_dim_config` 和
`sincos_compute` 已完整覆盖。

**案例建议**：不建议重复开发。

### 3.18 遍历方式、执行模型与循环形态

**问题瓶颈**

global-stride 和 block-contiguous 会改变相邻线程地址、循环步长和寄存器压力；direct
SIMT 与 SIMD/SIMT hybrid 会改变 launch 与编译调度；手工 unroll、load grouping 和
语句排序还会改变寄存器生命周期和 Stack。源代码看起来更连续或更并行并不代表生成
代码更快。

**优化方案**

在 work、数据、输出和工具链完全相同的前提下，构造以下 `2 x 2 x 2` 因子矩阵：

```text
执行模型：direct SIMT | SIMD/SIMT hybrid
遍历方式：global-stride | block-contiguous
代码形态：compiler loop | manual expansion
```

需要更多编译器对照时，再为 loop 增加 `pragma unroll` 变体。完整矩阵至少采集两轮，
短 kernel 的数个百分点差异必须结合方差判断。

**代表性结果**

- 当前原地 float 累加 8-case 中，loop 在四组一一配对中均比 manual expansion 快
  `15.3%-21.2%`，总体低 `19.0%`。
- Hybrid 在四组一一配对中均比 direct SIMT 快 `8.0%-15.3%`，总体低 `10.8%`。
- Global-stride 总体均值比 block-contiguous 低 `5.7%`，但这是最弱因子，部分两轮方向
  不一致，不能推广为固定结论。
- 其他历史配对中，compiler loop 比 manual expansion 快约 `20.4%`（direct SIMT）
  和 `34.2%`（mixed）。

**现有样例对照**：`simd_simt_high_performance` 部分覆盖线程映射和执行模型，但没有提供
完整的三因子受控矩阵。

**案例建议**：适合低成本反例案例，帮助开发者避免把遍历直觉、执行模型或源代码形式
直接当作性能结论。

### 3.19 重新验证并撤销历史正确性 Workaround

**问题瓶颈**

早期 PV buffer 复用存在同步竞态，因此实现屏蔽 1024 个线程中的后 128 个线程，只让
28 个 Warp 正常处理，并由前四个 Warp 补做第二行。同步竞态修复后，这个正确性
workaround 仍留在生产路径，长期限制有效并行度。

**优化方案**

每当相关同步、编译器或硬件条件变化后，重新审计历史 correctness workaround。将
workaround 的移除作为单因素实验，保持 tile、buffer、launch bounds 和其余同步不变，
先覆盖原始失败 case 和重复 launch，再测量完整边界。

**代表性结果**

恢复 1024 active thread 和 32 active Warp 后，Sparse Attention fused 由
`327.363 us` 降至 `283.422-283.517 us`，降低 `13.4%`；完整 workflow 由
`642.075 us` 降至 `598.477-598.536 us`，降低 `6.8%`。

**现有样例对照**：asc-devkit 未覆盖历史 workaround 的生命周期管理。

**案例建议**：强烈推荐。该案例可以展示正确性修复、保守降级和性能恢复之间的关系。

### 3.20 重计算与中间结果缓存的权衡

**问题瓶颈**

开发者容易把昂贵函数的重复计算视为必然浪费，并尝试把 exp 等中间结果缓存到 UB。
但缓存会增加 UB 读写、占用容量、延长 buffer 生命周期，并可能破坏搬运与计算重叠。

**优化方案**

对比 recompute、lane-local register、shared UB 和 GM materialization 四种边界。只有在
减少的计算成本大于新增访存、同步和容量机会成本时才保留缓存。对于能整行驻留 UB 的
Softmax，可采用 whole-row recompute；超宽行使用在线统计和分阶段 workspace。

**代表性结果**

在 30K-32K 行宽上缓存 FP16 exponential 约慢 `28%`，缓存 FP32 exponential 慢
`22%-23%`。这说明“中间结果能放入 UB”只是可行性结论，不是性能结论。

**现有样例对照**：asc-devkit 未覆盖 recompute 与中间结果物化的对照。

**案例建议**：推荐作为 Row Reduction 案例的独立子案例，保留缓存回退的负向结果。

### 3.21 释放编译器保留 UB 与建立容量契约

**问题瓶颈**

在已测试工具链中，编译器默认保留 VF Stack 和 ASC runtime UB 区域，可用 UB 为
216 KiB。大行或双缓冲路径可能因这部分保留空间无法采用更合适的 tile，但把数据数组
直接顶到硬件标称容量又会挤占 reduction scratch、对齐和编译器状态。

**优化方案**

在确认工具链支持后，对每个 ASC translation unit 同时使用：

```text
--cce-disable-vf-stack-reserved-ubuf
--cce-disable-asc-reserved-ubuf
```

已测试环境中可用 UB 提高到 224 KiB。容量规划必须满足：

```text
data_bytes <= 224 KiB - reduction_scratch - pipeline_state
              - alignment_padding - compiler_margin
```

编译选项、容量常量和静态断言需要保持一致，并重新检查最终二进制资源信息。

**代表性结果**

FP32 `(1024,49152)` 的两 kernel MTE/UB tiled 路径由 `0.853744 ms` 降至
`0.436659 ms`，提升约 `1.95x`；两个 FP16 大 shape 基本持平。收益具有明显 dtype 和
调度边界，不应把 224 KiB 当作跨版本硬件常量。

**现有样例对照**：asc-devkit 当前相关目录未覆盖编译器保留 UB 的释放和容量契约。

**案例建议**：条件性推荐。适合作为工具链特定资源案例，必须明确 CANN/Bisheng 版本。

### 3.22 编码前计算关键路径收益上界

**问题瓶颈**

局部 kernel 或 VF 看起来耗时较高，但它可能只占并发 critical path 的一小部分。未先
计算物理收益上界就实现候选，容易投入大量时间优化无法达到 retention gate 的路径。

**优化方案**

从 timeline 计算每个物理 lane 上该阶段的总耗时，取并发关键路径上的最大值，而不是
跨 lane 求和。再用候选的理想加速比估算 workflow 或 fused boundary 的理论最大收益，
若上界低于预先声明的保留阈值，则在改代码前淘汰候选。

**代表性结果**

Output Update 最慢 lane 的总耗时为 `23.978 us`。即使理想加速 `2x`，最多节省
`11.989 us`，只占 `259.448 us` fused boundary 的 `4.62%`，低于预声明 gate，因此
没有实现对应 2048-thread specialization。

**现有样例对照**：asc-devkit 未覆盖基于 critical-path share 的候选筛选。

**案例建议**：强烈推荐。它适合作为所有性能案例之前的分析案例，可以显著减少无效开发。

## 4. 推荐开发清单

### 4.1 P0：优先开发

| 案例 | 建议的 A/B 结构 | 主要教学价值 |
| --- | --- | --- |
| Top-K 算法选择 | 重复 padded bitonic / 流式或 radix selection | 先优化算法复杂度，再优化微架构 |
| 单遍确定性压缩 | atomic+重复扫描 / packed count+offset | 消除原子竞争和重复内存扫描 |
| 分片直方图 Top-K | 单 block / shard-local histogram+层次归约 | 长序列工作分解和串行尾部并行化 |
| 跨消费者共享 Gather | 两次独立 gather / 一次 gather 多消费者复用 | SIMT 离散访存的端到端复用收益 |
| Phase-aware Tile 与传输粒度 | 保守容量求和 / live-set 峰值复用 / 过大 tile 或 copy 合并反例 | 片上容量、传输粒度和同步轮数权衡 |
| Lane-local 复用 | 重复加载 / shared UB+barrier / lane-local register | 所有权局部复用优先于共享缓存 |
| 消费者原生布局 | 显式 transpose/packing / copy-format 或 ND2NZ | 在真正消费边界完成格式转换 |
| 在线稳定行归约 | 多次扫描 / fused stats / tiled online max-sum | 宽行容量、数值稳定性和 launch 融合 |
| 行宽行为分桶 | 单通用路径 / 行为区间 dispatch / exact-shape 反例 | 可维护的 shape specialization |
| 历史 Workaround 复验 | 保留旧降级 / 修复根因后撤销降级 | 避免正确性 workaround 永久限制并行度 |
| 关键路径收益上界 | timeline critical share / 理想加速上界 / retention gate | 在编码前淘汰不可能达标的候选 |

### 4.2 P1：进阶或条件性案例

| 案例 | 条件或限制 |
| --- | --- |
| VF 与 DCCI 发布粒度 | 仅开发官方未覆盖的逐行发布与按 tile 发布对照 |
| PV Handoff 融合 | 适合有共同 consumer 和共同生命周期的融合算子 |
| 重计算与中间结果缓存 | 作为 Row Reduction 子案例，比较 recompute/register/UB/GM |
| 编译器保留 UB 释放 | 工具链特定，必须绑定 CANN/Bisheng 版本和 dtype |
| 遍历、执行模型与循环形态 | 使用完整 `2 x 2 x 2` 因子矩阵，不把弱因子差异过度推广 |
| 多阶段 Launch 融合 | 需要平台支持的 barrier/visibility API；不得突破 CannBench 当前 API 边界 |

### 4.3 不建议重复开发

以下主题已经被 asc-devkit 完整覆盖：

- GM 合并访问和 UB 中转；
- Matrix Transpose、UB padding 和 bank conflict；
- 双缓冲及 MTE/SIMT producer-consumer 流水；
- `half2` 短向量；
- DCache hint 和类型对齐；
- `gridDim`、线程数、`launch_bounds` 和 register spill；
- Warp divergence 和 Warp 协作；
- VF 调用次数的固定开销；
- SIMD/SIMT 线程映射；
- SIMT producer 与 MTE task queue；
- 固定除数快速除法。

## 5. 案例验收要求

每个新增最佳实践案例至少应满足以下条件：

1. 固定 Ascend 950/950PR 型号、CANN 和编译器版本、频率、dtype、shape 和布局。
2. 提供正确性 oracle，覆盖代表、边界、tail 和 dispatch threshold case。
3. 每次只改变一个独立因素；组合优化必须重新测量，不能累加孤立收益。
4. 同时保留 kernel、component 和 workflow 边界，明确 selected/excluded kernel。
5. 列出全部物理 launch，核对每次 operator call 的预期和实际 launch count。
6. 固定 warmup、重复次数和同步点，报告 median、离散度和原始样本。
7. 用唯一 run name 和干净 profiling 目录，避免旧结果被 parser 重复发现。
8. 在 benchmark 进程内记录加载模块、共享库、device ELF 路径和 hash。
9. 性能采集与详细 metric 采集分开，避免 metric group 改变延迟。
10. 保留负向结果、正确性失败和恢复基线的证据。

历史会话中已经出现过以下测量失真：

- 把不同 stage 文件当作重复样本，修正聚合后六个 case 的已发布结果表现为
  `52.6%` 回退；
- 复用 run name 导致旧 profile 目录被重复聚合，结果接近真实值的两倍；
- operator-local 路径或 stale editable build 覆盖候选二进制，使整组性能比较失效；
- 只累加五个设备 kernel row，却把结果误称为端到端 launch latency；
- workflow 分进程 profiling 时，下游输入绑定重新执行 producer，但聚合规则没有反映
  真实物理执行序列。

因此，测量可信度是所有性能最佳实践案例的共同前置条件，而不是可选的附加步骤。

## 6. 相关资料

- [Ascend 950 Performance Optimization Skill](../../.agents/skills/ascend-950-operator-performance-optimization/SKILL.md)
- [Observed Performance Patterns](../../.agents/skills/ascend-950-operator-performance-optimization/references/observed-optimization-results.md)
- [DSA V2 Decode Profile-guided Optimization](../designs/dsa-v2-decode-profile-guided-optimization-design.md)
- [Fused Distributed Top-K](../designs/lightning-indexer-fused-distributed-topk.md)
- [Softmax V3 优化总结](../../src/cannbench/operators/builtin/softmax/simt/v3/README.zh-CN.md)
- [Sparse Attention 剩余差距实验](../optimization/dsa-v2-sparse-attention-remaining-gap-experiments.zh-CN.md)
