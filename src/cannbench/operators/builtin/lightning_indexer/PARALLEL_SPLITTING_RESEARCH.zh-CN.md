# Lightning Indexer 并行拆分方案调研

本文记录 DeepSeek、DeepGEMM 和 vLLM-Ascend 中 Lightning Indexer 相关实现的
任务拆分方式，并与 CannBench 当前 Ascend SIMT 实现对比。本文是优化讨论的事实
基线，不代表 CannBench 已经选定最终方案。

调研日期：2026-07-27。

## 1. 调研范围和术语

本文主要讨论 decode case：

```text
deepseek_v32_flashmla_decode_b2_q2_ctx32768_top2048
```

简写如下：

| 符号 | 当前值 | 含义 |
| --- | ---: | --- |
| `B` | 2 | 相互隔离的 batch/序列槽位数 |
| `Q` | 2 | 每条序列本次已有的 Query token 数 |
| `C` | 32768 | 每条序列的 Context/KV token 数 |
| `G` | 64 | Indexer Query 的 head 数 |
| `D` | 128 | 每个 Indexer head 的向量维度 |
| `K` | 2048 | 每个 Query 最终保留的 TopK 数 |

需要区分：

- 逻辑任务数不等于物理 AIC、AIV 或 GPU SM 数。逻辑任务最终由设备调度到物理核。
- host 设置的 `blockDim` 不等于有效工作核数。kernel 内部可以让多余的核直接退出。
- `Q=2` 表示 kernel 收到两个已经存在的 Query 向量，不表示 Indexer 自己生成两个 token。

## 2. `B=2, Q=2` 的来源

这个 shape 不是 CannBench 随意构造的，也不是 DeepSeek 模型架构写死的唯一运行
配置。它来自 FlashMLA 官方 V3.2 production decode 测试：

```python
RawTestParam(
    b=0,
    h_q=128,
    s_q=2,
    s_k=32768,
    topk=2048,
    d_qk=576,
)
batch_sizes = [2, 64, 74, 128]
```

其中：

- `Q=2` 是该 V3.2 production decode 模板的 `s_q`。
- `B=2` 是官方性能测试提供的多个 batch size 之一，CannBench 选择了这个测试点。
- 实际在线服务中的 `B` 会随调度变化，普通单 token decode 也可能使用 `Q=1`。

CannBench case 定义位于
[`data/realistic_decode.json`](data/realistic_decode.json)，详细来源记录位于
[`docs/datasets/dsa-real-sources.md`](../../../../../docs/datasets/dsa-real-sources.md)。

## 3. CannBench 当前实现

当前 `family_64x128` 的任务单位是一行 `(batch, query_token)`：

```cpp
total_rows = batch_size * query_count;
used_core_num = min(total_rows, kMaxUsedCoreNum);
```

源码位于
[`lightning_indexer_fused_family_64x128.asc`](simt/v1/aten_dsa_lightning_indexer/csrc/simt/lightning_indexer_fused_family_64x128.asc)。

每个有效 AIC：

1. 取得一行 `(batch_index, query_index)`。
2. 将该 Query 的 `[G,D] = [64,128]` 数据装入 L1/L0A。
3. 按 128 个 Context token 一个 tile，串行遍历该行全部 Context。
4. 每个 tile 执行 `[64,128] x [128,128]` 点积。

对应 AIV对每个 128-token tile执行 ReLU、head 加权和 TopK 合并。TopK 状态在同一
行的所有 Context tile 之间持续保留，因此不需要跨 AIC 的最终归约。

当前 case 的调度为：

```text
total_rows = B * Q = 2 * 2 = 4
used_core_num = 4
每个 AIC 的 Context 循环数 = 32768 / 128 = 256
```

因此有效计算受 `B*Q` 限制为 4 个 AIC。MIX kernel 声明的 AIC:AIV 比例是 `1:2`，
但当前 AIV 路径只允许每组中的第一个 AIV继续执行，所以有效工作量是 4 AIC 加
4 AIV，而不是 4 AIC 加 8 AIV。

这个实现的优点是没有完整 score 的 GM 往返，也没有跨 AIC TopK merge。主要问题
是长 Context 仍由单个 AIC 串行处理；在小 `B*Q` decode case 中，大部分 AIC 空闲。

## 4. DeepSeek 教学实现

DeepSeek-V3.2-Exp 的 TileLang `fp8_index_kernel` 使用三维 grid：

```python
T.Kernel(batch, query_count, ceildiv(context_count, 512))
```

逻辑任务是：

```text
(batch, query_token, context_512_token_shard)
```

每个 program 负责 512 个 Context token，内部再按 128 token 做 4 次流水。对于
当前 case：

```text
program 数 = 2 * 2 * ceil(32768 / 512) = 256
```

这表示产生 256 个可调度 program，不表示设备同时拥有或使用 256 个物理 SM。

该 kernel 将全部 score 写到 `[B,Q,C]` 输出，之后由 PyTorch 单独执行
`topk(dim=-1)`。它通过完整 score 中间结果消除了跨 program TopK 合并，但增加了
完整 logits 的 GM 写回、读取和一个独立 TopK kernel。

固定版本源码：

- [`inference/kernel.py`](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/87e509a2e5a100d221c97df52c6e8be7835f0057/inference/kernel.py#L199-L251)
- [`inference/model.py`](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/87e509a2e5a100d221c97df52c6e8be7835f0057/inference/model.py#L480-L486)

## 5. DeepGEMM 当前生产型 GPU 路径

CannBench CUDA adapter 使用 DeepGEMM 的 `fp8_paged_mqa_logits`。当前 DeepGEMM
实现采用持久化 kernel 和预生成调度 metadata：

```text
SPLIT_KV = 256
每个请求的 segment 数 = ceil(context_len / 256)
```

metadata 将全部请求的 segment 尽量均匀地分给设备可用 SM。主 kernel 启动
`num_sms` 个 CTA，每个 CTA 根据 metadata 连续领取自己的任务范围。并行度不受
`B*Q` 限制。

对于 `Q=2`，scheduler 将两个 Query token 组成一个 Query atom：

```text
next_n_atom = 2
num_next_n_atoms = ceil(Q / 2) = 1
```

当前 case 的 segment 数为：

```text
每个 batch：32768 / 256 = 128
两个 batch：2 * 128 = 256
```

一个 CTA 的任务同时覆盖这个 Query atom 的两个 Query token，从而复用同一个 KV
分片。CTA 内有 4 个 math warp group，每组计算 64 个 KV token：

```text
4 * 64 = 256 KV token
```

DeepGEMM 也输出完整 logits，CannBench adapter 随后调用 `torch.topk`，而不是在
Paged MQA logits kernel 内做跨 CTA TopK 归约。

固定版本：DeepGEMM `559d79fb6994a58b8a15b4b93bf13ccc16edf247`。

关键源码：

- [`sm90_paged_mqa_logits.cuh`](https://github.com/deepseek-ai/DeepGEMM/blob/559d79fb6994a58b8a15b4b93bf13ccc16edf247/deep_gemm/include/deep_gemm/scheduler/sm90_paged_mqa_logits.cuh)
- [`sm90_fp8_paged_mqa_logits.cuh`](https://github.com/deepseek-ai/DeepGEMM/blob/559d79fb6994a58b8a15b4b93bf13ccc16edf247/deep_gemm/include/deep_gemm/impls/sm90_fp8_paged_mqa_logits.cuh)
- CannBench adapter：[`src/cannbench_cuda_dsa_flashmla_deepgemm.py`](../../../../cannbench_cuda_dsa_flashmla_deepgemm.py)

## 6. vLLM-Ascend 的三种拆分策略

本节基于工作区中的 `vllm-ascend-upstream/csrc/attention/` 源码快照。不同目录代表
不同代或不同数据类型路径，不能将它们概括成唯一的“vLLM-Ascend 拆分方式”。

### 6.1 当前 BF16 `lightning_indexer`

arch35 路径定义：

```text
S1_BASE_SIZE = 4
S2_BASE_SIZE = 128
isLDOpen = false
```

host tiling 使用设备全部物理 AIC/AIV 计算 `blockDim`，但 kernel 的有效逻辑块数由
`GetTotalBaseBlockNum()` 决定。由于 `isLDOpen=false`，每个
`(batch, S1-group, KV-head)` 的全部 S2/Context 只计为一个逻辑块：

```text
logical AIC blocks = B * ceil(Q / 4) * N2
```

当前 `N2=1`，所以：

```text
logical AIC blocks = 2 * ceil(2 / 4) * 1 = 2
```

每个 AIC 仍在核内循环全部 `ceil(C/128)=256` 个 S2 tile。每个 AIC 对应两个
AIV，AIV 按 Query 行拆分工作：`Q=2` 时两个 AIV各负责一个 Query。因此该路径在
当前 case 中预计是 2 个有效 AIC和 4 个有效 AIV。

这个版本不沿 Context 跨 AIC 拆分，避免了跨 AIC TopK 归约。代价是长 Context、
小 Batch decode 时 AIC 并行度较低。

关键路径：

```text
csrc/attention/lightning_indexer/op_host/lightning_indexer_tiling.cpp
csrc/attention/lightning_indexer/op_kernel/lightning_indexer_common.h
csrc/attention/lightning_indexer/op_kernel/arch35/lightning_indexer_kernel.h
csrc/attention/lightning_indexer/op_kernel/arch35/lightning_indexer_service_vector.h
```

### 6.2 旧 `lightning_indexer_vllm`

旧版将 S2 纳入逻辑块数：

```text
S2_BASE_SIZE = 512
logical blocks包含 ceil(S2 / 512)
```

`SplitCore()` 将这些基本块均匀分给可用 AIC。同一 Query 行被多个 AIC拆开后：

1. 每个分片计算局部 TopK。
2. 局部 TopK 和描述信息写入 workspace。
3. `ProcessLD()` 读取多个分片结果，执行最终 TopK merge。

这个方案用额外 workspace 和归约换取 Context 方向的 AIC 并行度。

关键路径：

```text
csrc/attention/lightning_indexer_vllm/op_kernel/lightning_indexer_kernel.h
csrc/attention/lightning_indexer_vllm/op_kernel/lightning_indexer_service_vector.h
```

### 6.3 量化 `vllm_quant_lightning_indexer`

量化版重新采用 Context 拆分。Ascend 950 arch35 的基本 S2 tile 是 128。单独的
AICPU metadata kernel 根据估算 cost 按以下层次分配任务：

1. 尽量按完整 batch 分配。
2. 再按 Query row/S1 group 分配。
3. 必要时继续按 S2 block 分配。

metadata 为每个 AIC记录 `(batch/N2, S1-group, S2-start, S2-end)`。当一行被多个
AIC拆开时，还会为最终归约分配 AIV 工作。metadata 格式为最多 36 个 AIC和
72 个 AIV预留槽位，实际核数由平台传入。

关键路径：

```text
csrc/attention/vllm_quant_lightning_indexer/op_kernel/arch35/quant_lightning_indexer_kernel.h
csrc/attention/vllm_quant_lightning_indexer/op_kernel/vllm_quant_lightning_indexer_metadata.h
csrc/attention/vllm_quant_lightning_indexer_metadata/op_kernel_aicpu/
    vllm_quant_lightning_indexer_metadata_aicpu.cpp
```

## 7. 方案对比

| 路径 | 主要逻辑任务 | Context 跨计算核 | 两个 Query 是否共享 KV 加载 | TopK 处理 |
| --- | --- | --- | --- | --- |
| CannBench 当前 v1 | `(B,Q)` | 否 | 否 | 同一行核内持续合并 |
| DeepSeek TileLang | `(B,Q,C/512)` | 是 | 否 | 完整 logits 后独立 TopK |
| DeepGEMM | `(B,query-atom,C/256)` | 是 | 是 | 完整 logits 后独立 TopK |
| vLLM-Ascend BF16 | `(B,ceil(Q/4),N2)` | 否 | 是 | 完整行计算后 TopK |
| vLLM-Ascend 旧版 | 包含 `(C/512)` | 是 | 取决于 S1 group | 局部 TopK 加最终 merge |
| vLLM-Ascend 量化版 | AICPU cost-based metadata | 是 | 按 S1 group | 分片结果加 AIV归约 |

从这些实现可以归纳出两个主要选择。

### 7.1 写完整 score，再单独 TopK

优点：

- Context 可以自由拆成大量独立计算任务。
- 不需要在 score kernel 内协调不同计算核。
- 调度和负载均衡相对直接。

代价：

- 需要写回并重新读取 `[B,Q,C]` score。
- 至少增加一个 TopK kernel launch。
- Ascend 上需要确认完整 score GM 流量是否抵消 AIC 并行收益。

### 7.2 每个 Context 分片做局部 TopK，再最终 merge

优点：

- workspace 只保存局部候选，理想情况下小于完整 score。
- 可以保留当前边计算边筛选的思路。

代价：

- 需要第二阶段 merge 或可靠的核间归约机制。
- `K` 很大、Context 分片很小时，局部 TopK 几乎不能压缩候选。
- 需要处理变长 Context、空分片、相同分数 tie policy 和全局 index 偏移。

当前 case 的 `K=2048` 尤其重要：

```text
假设两个 Query 组成一个 atom，每个 AIC 处理该 atom 的一个 Context shard：

每个 batch 使用 8 个 Context shard：shard_len = 4096，总计 2 * 8 = 16 AIC
局部候选最多为 8 * 2048 = 16384 / Query

每个 batch 使用 16 个 Context shard：shard_len = 2048，总计 2 * 16 = 32 AIC
局部 TopK 等于完整 shard，最终仍需处理 32768 个候选 / Query
```

所以“直接使用 32 个 AIC”不一定自动优于 8 或 16 个 AIC，必须把局部 TopK、
workspace、最终 merge 和 GM 流量一起测量。如果仍采用单 Query task，逻辑任务
数还要再乘 `Q`，但两个 Query 会重复读取同一批 Key。

### 7.3 理论成本模型能确定什么

当前 case 的总点积量与拆分方式无关：

```text
B * Q * C * G * D
= 2 * 2 * 32768 * 64 * 128
= 1,073,741,824 MAC
```

拆分方案改变的是每个任务的工作量、有效 AIC 数、Key 读取量和 TopK 归约量。
可以用下面的近似模型比较方案：

```text
T_total ~= T_AIC_score
          + T_AIV_postprocess
          + T_local_topk
          + T_final_merge
          + T_GM
          + T_launch_and_sync
```

理论分析可以排除明显不合理的分片大小，并给出计算量、数据量和并行度上下界，
但不能单独确定最终最快方案，原因包括：

- 16 AIC 和 32 AIC 的实际 MMAD 效率不等于理论峰值的固定比例。
- 两个独立 Query 重复读取 Key 时，实际 GM 流量取决于缓存命中和并发行为。
- `4096 -> 2048` 局部 TopK 与更大候选集合 final merge 的硬件效率不同。
- MIX kernel 中 AIC、AIV、GM 搬运和同步可能重叠，也可能互相等待。
- Q atom 对 M 维形状、L0/L1 占用和流水深度的影响需要在目标设备确认。

因此应先用理论模型限定候选，再用少量硬件微基准校准以下参数：

1. Q=1 和 Q=2 atom 的单核 MMAD 吞吐。
2. 16 AIC 和 32 AIC 的纯 score 计算耗时。
3. 16384 和 32768 候选的 TopK/final merge 耗时。

校准后可以将实测吞吐和带宽代入模型，再决定最终默认配置，而不必长期维护三套
独立实现。

## 8. CannBench 候选优化方向

以下只是下一轮讨论的候选项，不是本文结论。

### 8.1 保留单 Query task，增加 Context shard

```text
task = (batch, query_token, context_shard)
```

优点是对当前 `[64,128] x [128,128]` MMAD 形状改动较小。缺点是同一个 batch 的
两个 Query 会分别加载同一份 Key tile。

### 8.2 将两个 Query 组成 atom，再拆 Context

```text
task = (batch, query_atom, context_shard)
query_atom_size = 2
```

这更接近 DeepGEMM，可以让两个 Query 复用 Key tile。但需要评估：

- 将 M 从 64 扩成 128 的 MMAD 形状和 L0/L1 占用；或
- Key 保留在片上时串行计算两个 Query，是否仍有收益；
- 两个 Query 不同有效 Context 长度时的尾部处理。

### 8.3 选择完整 score 或局部 TopK merge

这两个方向应分别实现最小原型并测量，而不能只凭计算量判断。至少需要记录：

- AIC/AIV 实际有效核数和利用率；
- Indexer score 计算耗时；
- TopK 或 merge 耗时；
- workspace 大小和 GM 读写量；
- 端到端 `dsa_decode` 延迟；
- `B/Q/C/K` 改变后的稳定性。

### 8.4 三组对照方案

第一轮实现和测试保留以下三组配置。表中的 Key 读取量是假设 BF16 Key、跨任务
没有缓存复用时的上界；merge 候选数是 4 个 Query row 的总数。

| 方案 | Query 组织 | Context shard | 逻辑 AIC task | 每任务 MAC | Key 读取上界 | merge 候选数 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| A：高并行 atom | Q=2 atom | 2048 | 32 | 33,554,432 | 约 16 MiB | 131072 |
| B：低归约 atom | Q=2 atom | 4096 | 16 | 67,108,864 | 约 16 MiB | 65536 |
| C：单 Query 对照 | Q=1 task | 4096 | 32 | 33,554,432 | 约 32 MiB | 65536 |

三组方案验证不同瓶颈：

- A 对比 B：32 AIC 的 score 并行收益能否覆盖增加的 final merge 成本。
- B 对比 C：Q atom 的 Key 复用能否覆盖任务数从 32 降到 16 的利用率损失。
- A 对比 C：在相同 32 个逻辑任务下，Key 复用与不同 merge 规模的综合影响。

建议同时记录一个不含 TopK 的 score-only 微基准，以便把 AIC 扩展效率与归约成本
分开分析。

### 8.5 三组方案的代码复用约束

三组方案不应复制三套设备核函数。目标结构是一套共享的 score、局部候选生成和
final merge 设备函数，差异由 host 根据测试参数计算并传入 launch plan：

```text
host inputs:
    B, Q, valid_context_lengths, top_k
    query_atom_size        # 1 or 2
    context_shard_size     # 2048 or 4096

host-derived launch plan:
    query_atom_count
    context_shard_count per batch/row
    task_count and task-to-(batch, atom, shard) mapping
    score loop start/end
    local candidate count
    final reduce group start/count
    workspace offsets
```

kernel 只消费 launch plan 和张量地址，不根据“方案 A/B/C”名称分支。循环次数、
是否需要局部筛选、final reduce 的输入范围以及尾部分片，都由通用参数推导。host
测试通过同一个入口改变 `query_atom_size` 和 `context_shard_size`，从而形成三组
可直接对照的 profiler 数据。

第一轮只需要覆盖当前 V3.2 decode case 使用的 Q atom size 1/2、Context shard
2048/4096 和 `K=2048`，不提前设计任意 atom size 或任意 shard size 的泛化接口。

新的 Ascend SIMT 设计还必须遵守本仓库 API 边界：算子源码只使用 C API、Tensor
API 和 SIMT API；仅在片内流水无法表达时使用 kernel-local Mutex，不新增 C++
Basic API 或跨核同步依赖。

## 9. 下一轮需要决定的问题

1. 第一轮以 V3.2 `B=2,Q=2,C=32768,K=2048` 为校准 case，何时扩展到其他 decode case。
2. 三组对照完成后，默认选择固定配置还是根据 `B/Q/C/K` 运行时自适应。
3. Q atom 在设备侧采用 M=128 一次 MMAD，还是复用 Key 后执行两次 M=64 MMAD。
4. TopK 采用完整 score 加独立 kernel，还是局部 TopK 加最终 merge。
5. 最终 merge 由第二个 kernel 完成，还是由同一 MIX kernel 的 AIV阶段完成。
6. 如何让 `1:2` MIX 配置中的两个 AIV都有稳定、对称的有效工作。
7. 如何处理变长 Context、因果 Query 行和不足一个 shard 的尾块。

这些问题确定后，再形成 CannBench operator-local 的实现方案和测试矩阵。
