# Sparse Attention 并行拆分与占核调研

本文记录 CannBench 当前 Ascend SIMT Sparse Attention、vLLM-Ascend
SparseFlashAttention 和 DeepSeek FlashMLA sparse decode 的任务拆分方式，并给出
后续优化讨论的事实基线。本文不是最终实现方案；其中标为“候选”或“待验证”的内容
必须经过目标设备 benchmark 才能成为默认策略。

调研日期：2026-07-27。

## 1. 调研范围

本文主要讨论 DeepSeek V3.2 realistic decode case：

```text
deepseek_v32_flashmla_decode_b2_q2_ctx32768_top2048
```

CannBench case 定义位于
[`data/realistic_decode.json`](data/realistic_decode.json)，对应 shape 为：

| 符号 | 当前值 | 含义 |
| --- | ---: | --- |
| `B` | 2 | batch/序列槽位数 |
| `H` | 128 | Query head 数 |
| `KV_H` | 1 | 共享 Key/Value head 数 |
| `Q` | 2 | 每条序列本轮已有的 Query token 数 |
| `C` | 32768 | Context/KV cache token 数 |
| `S` | 2048 | Indexer 选出的 sparse token 数 |
| `Dqk` | 576 | Query/Key 点积维度 |
| `Dv` | 512 | Value/输出维度 |

需要先区分两个序列轴：

- `C` 是 Indexer 搜索前的完整 Context 长度。
- `S` 是 Sparse Attention 实际读取和计算的 selected token 数。

Sparse Attention 已经拿到了 `indices[B,Q,S]`，因此它不再遍历全部 `C`。对该算子
做 Split-KV 时，应该拆 `S`，而不是拆原始 `C`；`C` 只参与校验 index 和计算
KV cache 地址。

## 2. CannBench 当前实现

当前 V3.2 BF16 decode 走单个 fused MIX kernel：

[`sparse_attention_score_family_hd512.asc`](simt/v1/aten_dsa_sparse_attention/csrc/simt/sparse_attention_score_family_hd512.asc)

虽然文件名使用 `hd512`，host 会把实际 `qk_head_dim=576` 写入 tiling shape，
`value_head_dim=512` 仍单独传入。

### 2.1 当前任务单位

Host 计算：

```cpp
total_rows = batch_size * query_heads * query_tokens;
used_core_num = min(total_rows, 32);
```

一个 row 是一个独立的：

```text
(batch, query_head, query_token)
```

当前 case：

```text
total_rows = B * H * Q = 2 * 128 * 2 = 512
used_core_num = 32
每核 row 数 = 512 / 32 = 16
```

AIC 和有效 AIV都按下面的步长遍历 row：

```cpp
for (row_index = block_idx;
     row_index < total_rows;
     row_index += used_core_num)
```

因此当前分核没有沿 `S` 拆分。每个 row 都完整处理 2048 个 selected tokens，
同一个 `(batch, query_token)` 的共享 K/V会被 128 个 query heads 分别 gather。

### 2.2 每核循环和 profiler 对照

当前 tile：

```text
M = 1
baseN = 64 selected tokens
baseK = 64 dimensions

N loop = 2048 / 64 = 32
K loop = 576 / 64 = 9
每 row MMAD = 32 * 9 = 288
每 AIC MMAD = 16 * 288 = 4608
```

2026-07-27 realistic decode roofline profile 中，32 个 `cube0` 的
`aic_cube_total_instr_number` 都是 4608，与源码循环完全一致。Profile 的关键数据：

```text
kernel: sparse_attention_fused_family_hd512_kernel
task duration: 271405.44 us
block dim: 32
mix block dim: 64

cube0 rows: 32
vector0 rows: 32
vector1 rows: 32

cube0 average time: 269625.51 us
cube utilization: 0.0166%

vector0 average time: 271071.79 us
vector utilization: 8.54%
scalar utilization: 97.40%

vector1 average time: about 2.28 us
```

原始 profile 目录名为
`cannbench-v32-decode-all-metrics-20260727-114901`，主要证据来自
`OpBasicInfo.csv`、`ArithmeticUtilization.csv`、`PipeUtilization.csv` 和
`ResourceConflictRatio.csv`。

这些数据说明：

- 32 个 AIC都被调度并长期占用，但 Cube 流水没有被有效利用。
- MIX 声明为 `1 AIC : 2 AIV`，物理上调度了 64 个 AIV。
- AIV入口显式让 `GetSubBlockIdx()!=0` 的第二个 AIV返回，所以只有 32 个 AIV持续工作。
- AIC Cube/MTE wait ratio 接近 100%，绝大多数时间在等待 AIV完成离散 Key gather。
- “铺满 32 个 AIC”不等于“吃满 Cube 算力”。当前属于占核但低利用率。

### 2.3 当前瓶颈

AIV0 对每个 `(N=64,K=64)` tile执行两层标量循环：

```text
读取 64 个离散 Context index
逐元素 gather 64 * 64 个 BF16 Key
将 UB Key tile 搬到 L1
通知 AIC执行 M=1,N=64,K=64 MMAD
```

主要问题有：

1. `M=1` 无法有效利用 Cube。
2. 相同 `(batch, query_token)` 的 K/V被 128 个 heads重复 gather。
3. 第二个 AIV直接退出。
4. softmax max/sum 由每个 SIMT thread 各自遍历完整 `S=2048`。
5. PV 由 SIMT thread 按 Value dimension 分工，每个 dimension 再串行遍历 2048 个 token。

所以当前优化不能只调整 `blockDim`。即使继续保持 32 个 AIC，单 head GEMV、重复
gather 和 SIMT PV 仍会限制性能。

## 3. vLLM-Ascend A5/950PR 方案

本节基于 vLLM-Ascend `main`：
`a5b0ce10d84bf76dd9c5c9e7ab9a5ddeed5af7ca`。

关键源码：

- [`sparse_flash_attention_tiling.cpp`](https://github.com/vllm-project/vllm-ascend/blob/a5b0ce10d84bf76dd9c5c9e7ab9a5ddeed5af7ca/csrc/attention/sparse_flash_attention/op_host/sparse_flash_attention_tiling.cpp)
- [`sparse_flash_attention_kernel_mla.h`](https://github.com/vllm-project/vllm-ascend/blob/a5b0ce10d84bf76dd9c5c9e7ab9a5ddeed5af7ca/csrc/attention/sparse_flash_attention/op_kernel/arch35/sparse_flash_attention_kernel_mla.h)
- [`sparse_flash_attention_service_cube_mla.h`](https://github.com/vllm-project/vllm-ascend/blob/a5b0ce10d84bf76dd9c5c9e7ab9a5ddeed5af7ca/csrc/attention/sparse_flash_attention/op_kernel/arch35/sparse_flash_attention_service_cube_mla.h)
- [`sparse_flash_attention_service_vector_mla.h`](https://github.com/vllm-project/vllm-ascend/blob/a5b0ce10d84bf76dd9c5c9e7ab9a5ddeed5af7ca/csrc/attention/sparse_flash_attention/op_kernel/arch35/sparse_flash_attention_service_vector_mla.h)

### 3.1 按共享 KV group 组织 Head

vLLM-Ascend 定义：

```text
G = query_heads / kv_heads
```

DeepSeek V3.2 中 `G=128`。A5 kernel 的基本块是：

```text
M tile = 64 query heads
S tile = 128 selected tokens
K = 512 NoPE + 64 RoPE = 576
```

当 `G>64` 时启用 `IS_SPLIT_G`。相邻两个 AIC处理同一个
`(batch, query_token)`：

```text
第一个 AIC: heads 0..63
第二个 AIC: heads 64..127
```

当前 case 的基础任务数为：

```text
base tasks = B * Q * ceil(H / 64)
           = 2 * 2 * 2
           = 8 AIC
```

Host tiling 会按物理 AIC/AIV数生成 MIX blockDim，但 kernel 内根据实际
`B*Q` 计算有效核范围。因此这个 shape 中预计只有 8 个 AIC和对应 AIV持续工作，
其余任务提前退出。

### 3.2 四级流水

一个 `M=64,S_tile=128` 的主要流水是：

```text
AIV Vec0: gather K/V tile
AIC BMM1: QK, [64,576] x [576,128]
AIV Vec1: online softmax
AIC BMM2: PV, [64,128] x [128,512]
AIV Vec2: online output update and store
```

与 CannBench 当前实现相比，两个关键变化是：

- QK 的 `M` 从 1 提高到 64。
- PV 从 SIMT 标量累加改为 Cube BMM2。

Kernel 使用多 buffer 将相邻 S tile 的 gather、BMM1、softmax、BMM2 和写出重叠，
不是先生成完整 score 再单独执行 PV。

### 3.3 两个 AIV和跨 Head group 的 KV复用

BMM1 的 Fixpipe 使用 dual-destination，按 M 维将 64 行 score 分给两个 AIV，
每个 AIV处理约 32 个 heads 的 softmax 和输出更新。

在 `G=128` 路径中，两个 AIC对应的四个 AIV还会把一个 128-token KV tile 分成
四份，每个 AIV约 gather 32 个 token。聚合结果写入成对 AIC共享的 GM workspace，
两个 AIC再各自搬到 L1用于计算。这避免从原始 KV cache 为两个 64-head group
完整重复 gather，但引入 workspace 和跨核同步。

### 3.4 当前 A5 路径没有沿 S 跨 AIC拆分

在所调研版本中，host 的 `splitKVFlag_` 默认保持 false，`S=2048` 由每个有效 AIC
按 128-token tile循环 16 次。它优先保证 `M=64` 和 KV复用，而不是为了铺满全部
AIC继续拆 S。

因此这个方案在当前 shape 中具有较高的单核矩阵效率，但预计只使用 8 个有效 AIC。

## 4. DeepSeek FlashMLA 方案

本节基于 DeepSeek FlashMLA `main`：
`9241ae3ef9bac614dd25e45e507e089f888280e0`。

关键资料：

- [官方 FP8 sparse decode deep dive](https://github.com/deepseek-ai/FlashMLA/blob/9241ae3ef9bac614dd25e45e507e089f888280e0/docs/20250929-hopper-fp8-sparse-deep-dive.md)
- [`sm90/decode/sparse_fp8/config.h`](https://github.com/deepseek-ai/FlashMLA/blob/9241ae3ef9bac614dd25e45e507e089f888280e0/csrc/sm90/decode/sparse_fp8/config.h)
- [`sm90/decode/sparse_fp8/splitkv_mla.cuh`](https://github.com/deepseek-ai/FlashMLA/blob/9241ae3ef9bac614dd25e45e507e089f888280e0/csrc/sm90/decode/sparse_fp8/splitkv_mla.cuh)
- [`api/sparse_decode.h`](https://github.com/deepseek-ai/FlashMLA/blob/9241ae3ef9bac614dd25e45e507e089f888280e0/csrc/api/sparse_decode.h)
- [`get_decoding_sched_meta.cu`](https://github.com/deepseek-ai/FlashMLA/blob/9241ae3ef9bac614dd25e45e507e089f888280e0/csrc/smxx/decode/get_decoding_sched_meta/get_decoding_sched_meta.cu)

### 4.1 64 heads 一个 CTA

Hopper sparse decode kernel 定义：

```text
BLOCK_M = 64 heads
TOPK_BLOCK_SIZE = 64 tokens
NUM_M_BLOCKS = H / 64
```

DeepSeek V3.2 的 128 heads 由两个 CTA覆盖。每个 CTA对 64 heads执行 QK、online
softmax 和 PV，QK/PV 都使用 BF16 MMA，累加类型为 FP32。

### 4.2 两 CTA cluster 的 crossover

两个处理同一 Query token 的 CTA组成 cluster：

1. 每个 CTA加载 32 个 quantized KV tokens。
2. 每个 CTA负责反量化自己的一半。
3. 结果写入自己的 shared memory。
4. 同时通过 `st.async` 写入另一个 CTA的 distributed shared memory。
5. 交换结束后，两个 CTA都持有完整 64-token KV tile。

这样每个 KV token 只需反量化一次，却能服务全部 128 query heads。官方报告在
H800 的 `B=128,H=128,Q=2,S=2048` compute-bound 配置中，从没有 crossover 的
约 250 TFLOPS 提升到约 410 TFLOPS。

DeepSeek 使用 FP8 KV cache，而 CannBench 当前 case 使用 BF16 KV，因此反量化细节
不能直接移植；但“将共享一个 KV head 的多个 Query heads 合成 M 维，并复用一次
KV gather”的原则与数据类型无关。

### 4.3 沿 selected tokens 做动态 Split-KV

仅按 64 heads 拆分时，小 `B/Q` 仍不足以占满 GPU。FlashMLA 继续将 selected
tokens 按 64-token block 分配给多个 SM partition：

```text
head_blocks = H / 64
num_sm_parts = max(num_sms / (Q * head_blocks), 1)
grid = (head_blocks, Q, num_sm_parts)
```

Batch 和 S block 范围由单独生成的 scheduler metadata 分配。一个 partition 可以
处理一个请求的一段 S，也可以在负载允许时跨请求继续工作。

如果一个 attention row 被多个 partition 拆开，每个 partition 输出局部：

```text
partial max
partial exp sum
partial weighted output
```

最后由 combine kernel 合并，而不是让主 kernel 中的 CTA直接进行全局同步。

对于 partition `p`，设：

```text
m_p = max(scores_p)
l_p = sum(exp(scores_p - m_p))
o_p = sum(exp(scores_p - m_p) * V_p)
```

最终结果为：

```text
m = max_p(m_p)
l = sum_p(exp(m_p - m) * l_p)
o = sum_p(exp(m_p - m) * o_p) / l
lse = m + log(l)
```

这使 S 方向的任务可以独立调度，代价是 partial workspace 和一个 combine 阶段。

## 5. 三方方案对比

| 项目 | CannBench 当前 v1 | vLLM-Ascend A5 | DeepSeek FlashMLA |
| --- | --- | --- | --- |
| Head 基本块 | 1 | 64 | 64 |
| Selected 基本块 | 64 | 128 | 64 |
| QK | `M=1` Cube | `M=64` Cube | `M=64` Tensor Core |
| PV | SIMT逐维累加 | Cube BMM2 | Tensor Core MMA |
| KV复用 | 每个 head 单独 gather | 64 heads复用，成对 AIC进一步共享 | 64 heads复用，CTA cluster共享 |
| 第二个 AIV/协作单元 | 直接退出 | 两个 AIV均参与 | producer/consumer warp group |
| S 跨计算核 | 否 | 所调研 A5 路径否 | 是，动态 Split-KV |
| 跨 S 分片归约 | 无 | 无 | partial output + combine |
| 当前 shape 基础计算任务 | 32 AIC循环 512 rows | 8 个有效 AIC | Head cluster 乘 S partition |

可以归纳出三个共同或互补的结论：

1. 成熟实现都选择 `head_tile=64`，证明 M 维聚合比单 head GEMV更重要。
2. KV gather 应在共享同一 KV head 的 Query heads 之间复用。
3. 当 `B*Q*ceil(H/64)` 不足以占满设备时，可以像 DeepSeek 一样沿 S 增加并行度，
   但必须增加稳定的 online-softmax combine。

## 6. CannBench 综合候选方案

以下是下一轮实现和 benchmark 的候选，不是已经确定的默认方案。

### 6.1 候选 A：`head_tile=64`，不拆 S

```text
task = (batch, query_token, head_group_64)
task count = B * Q * ceil(H / 64) = 8
```

目标：先验证 vLLM-Ascend 风格的核心收益。

- QK 使用 `M=64`。
- 一个 head group 复用一次 KV gather。
- 两个 AIV都参与 gather/softmax。
- PV 使用 Cube。
- 不需要 partial workspace 或 combine。

该方案实现和验证最简单，但当前 shape 只有 8 个有效 AIC。

### 6.2 候选 B：`head_tile=64`，`S` 拆 4 份

```text
base tasks = B * Q * ceil(H / 64) = 8
selected partitions = 4
task count = 8 * 4 = 32
selected tokens per partition = 2048 / 4 = 512
```

目标：综合 vLLM-Ascend 的 M 维形状和 DeepSeek 的 Split-KV 调度。

每个 task 输出 64 heads 对应的局部 `max/sum/output`，第二阶段按上一节公式合并
4 份 partial。该方案能够构造 32 个独立 AIC任务，同时保持 `M=64`。

代价：

- 需要 partial workspace。
- 增加 combine kernel 或等价归约阶段。
- 每个 partition 的固定开销变大。
- 必须验证 512-token partition 是否足以摊薄 launch、初始化和写回成本。

### 6.3 候选 C：`head_tile=16`，不拆 S

```text
task count = B * Q * ceil(H / 16)
           = 2 * 2 * 8
           = 32
```

该方案无需 Split-KV 即可构造 32 个任务，适合作为对照。但同一 KV token 需要由
8 个 head groups分别 gather，`M=16` 的 Cube 效率也可能低于 `M=64`。结合
vLLM-Ascend 和 DeepSeek 的实现证据，它不应在 benchmark 前被视为主方案。

### 6.4 三组首轮对照

| 方案 | Head tile | S partitions | AIC tasks | 每 task 的 M | 每 task 的 S | 是否 combine |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| A：大 M 基线 | 64 | 1 | 8 | 64 | 2048 | 否 |
| B：大 M + Split-KV | 64 | 4 | 32 | 64 | 512 | 是 |
| C：小 M 满核对照 | 16 | 1 | 32 | 16 | 2048 | 否 |

建议额外测量 B 的 `S partitions=2`，得到 16 个 AIC的中间点，用来判断扩展到
32 AIC的收益是否已经被 combine 成本抵消。

## 7. KV读取与计算量分析

在同一个 `(batch, query_token)` 中，128 heads 共享 indices 和 KV。

当前实现对每个 selected token 的 Key gather 次数近似为：

```text
128 heads / 1 head per task = 128 次
```

`head_tile=64` 且两个 head groups 暂不共享 gather 时：

```text
128 heads / 64 heads per task = 2 次
```

所以即使第一版不实现 vLLM 的成对 AIC共享 workspace，也可以将原始 KV gather
重复次数理论上降低约 64 倍。后续若能让两个 64-head tasks安全共享 gather，才会
进一步从 2 次降到 1 次。

QK 和 PV 的总数学 FLOP 不会因为拆分而减少，变化的是：

- Cube/Tensor Core tile 的有效 M 大小；
- KV从 GM/L2 到片上存储的重复次数；
- AIC/AIV并行度和流水重叠；
- partial output 的 GM流量；
- combine 的额外计算与 launch。

因此不能仅凭“32 个 task”判断方案 B 或 C更快。

## 8. Ascend 实现边界

vLLM-Ascend 和 DeepSeek 的跨计算核复用方式不能原样搬入当前 CannBench 实现：

- vLLM-Ascend 使用跨 AIC/AIV同步和共享 GM staging。
- DeepSeek 使用 Hopper CTA cluster、distributed shared memory 和 cluster barrier。
- CannBench 新设计需要遵守 operator-local 的 `C API + Tensor API + SIMT API`
  边界，不应新增 Basic API 或跨核 flag 依赖。

因此更稳妥的第一阶段是让每个 `(B,Q,head_group,S_partition)` task 独立，使用
partial workspace 和第二阶段 combine 避免主 kernel 跨 AIC同步。AIC pair 之间
共享 gather 应作为后续独立优化，不应成为第一版正确性的前提。

## 9. Host 动态决策候选

固定写死 `head_tile=64,S_partitions=4` 只适合当前 case。后续 host 可以基于 shape
计算候选任务数：

```text
head_groups = ceil(H / head_tile)
base_tasks = B * Q * head_groups
target_partitions = ceil(physical_aic / base_tasks)
selected_partitions = clamp(target_partitions, 1, max_selected_partitions)
```

其中 `max_selected_partitions` 还应受以下条件限制：

- 每个 partition 至少包含足够多的 S tile。
- partial workspace 不超过预算。
- combine 成本不超过扩展 AIC带来的收益。
- ragged selected length、无效 index 和 causal mask 必须保持正确。

`head_tile`、S tile 和 partition 数应该由少量 host 参数控制并复用同一套 kernel
主体，而不是维护三套完全独立的数学实现。

## 10. 第一轮验证指标

每个候选至少记录：

1. 端到端 Sparse Attention 延迟。
2. 主 kernel 和 combine kernel 分阶段延迟。
3. 实际有效 AIC/AIV数，而不只看 host blockDim。
4. Cube、Vector、Scalar、MTE 利用率和 wait ratio。
5. 每核 MMAD 指令数及负载均衡。
6. GM/L2/UB/L1 读写量和 L2 hit rate。
7. partial workspace 大小。
8. output 和 LSE 相对 torch reference 的误差。
9. `B/Q/H/S` 改变后的稳定性，尤其是尾块和非整除 shape。
10. 完整 `dsa_decode` workflow 延迟，避免单算子收益被其他阶段抵消。

## 11. 当前结论

当前可以由源码和 profiler 支持的结论是：

- CannBench 当前 kernel 虽启动 32 个 AIC，但主要瓶颈是 `M=1`、标量离散 gather、
  SIMT PV 和第二个 AIV空闲，而不是 AIC task 数不足。
- vLLM-Ascend 与 DeepSeek 都以 64 query heads 作为基本 M tile，并使用矩阵单元
  同时完成 QK 和 PV。
- DeepSeek 通过沿 selected tokens 做 Split-KV，在小 `B/Q` 时补充并行度。
- 对 CannBench 而言，`head_tile=64 + 可选 S partition + partial combine` 是证据
  最充分的综合方向。
- `head_tile=16` 应保留为不带 combine 的满核对照，而不是提前选为默认方案。

最终默认参数仍需由上述三组对照和目标设备 profiler 决定。

## 12. V3.2 Prefill Head64/P=1 实测结论（2026-07-29）

本节补充 exact V3.2 prefill case 的实现与设备证据：

```text
B=1 Q=4096 H=128 KV_H=1 C=32768 S=2048 Dqk=576 Dv=512
dtype=BF16 phase=prefill family=family_hd576 seed=7
```

该 shape 与本文前半部分的 decode case 不同。它天然包含：

```text
B * Q * ceil(H / 64) = 1 * 4096 * 2 = 8192
```

个 `(batch, query_token, head_group64)` 逻辑任务，因此无需沿 `S` 做 Split-KV
也能让 32 个物理 MIX task 持续取任务。最终实现选择 Head64/P=1：单次 fused
launch 直接写 BF16 output 和 FP32 LSE，不分配 partial，不启动 Combine，也不做
output Cast。只有 exact 默认 shape 自动选择该路径；显式 P=1 可用于缩小 shape
验证，其他默认 shape 与既有 decode P=1/P=2/P=4 路径保持不变。

### 12.1 来源、环境与正确性

通用 baseline 为
`1297d3eb4ac5af62b0113f318136fdfae8ad52ea`，远端目录为
`/tmp/cannbench-sa-v32-prefill-baseline-W1p49H`。最终设备候选为
`fe376f33130c3757d552a95e8337c1e9c024fa18`（base `2c4b7aa`），远端目录为
`/tmp/cannbench-sa-v32-prefill-candidate-rebased-wsnZON`。候选 Head64 device/Host
源码 SHA-256 分别为
`0e61fa35102a088b3f2d6ae482bff0678b524f27f352f7944803a54c2d44842d` 和
`48ee3b1e2bed64a4eb5a697ffae08c24189909c790dd86a0934fbd106c458fda`。

两者都在 `Ascend950PR_9589` 上以 `dav-3510` 构建，CANN 路径为
`/usr/local/Ascend/cann-9.2.0`，clang/bisheng 为 15.0.5、build
`5c68a1cb1231`。最终候选验证结果为：

- reduced prefill P=1：`14 / 14` 通过；新增 `B=1,H=128,Q=4,C=256,S=64`
  causal case 的 row `0/1/2` 分别包含 `2/2/1` 个 in-range future index，row 3
  覆盖 causal end boundary，四行均含 valid-past、negative 和 out-of-range；
  output/LSE 最大绝对误差为 `0.017578125 / 0.020476341247558594`；
- decode 回归：P=1/P=2/P=4 共 `36 / 36` 通过；output/LSE 最大绝对误差为
  `0.0185546875 / 0.01997852325439453`；
- full automatic prefill：seed 7、`atol=rtol=0.05`，检查 query row
  `0,1365,2730,4095` 的全部 128 heads；前三行各注入一个 in-range future，末行
  注入 causal end boundary，四行均含 negative/out-of-range。output `262144` 个、
  LSE `512` 个采样元素 mismatch 都为 0，最大绝对误差为
  `0.0078125 / 0.008250236511230469`。

最终 review 的 causal-boundary 验证副本为
`/tmp/cannbench-sa-v32-prefill-final-review-ACYDoO`，只同步两个 accuracy
runner。该副本再次通过 reduced `14 / 14` 和 full causal 检查，full output/LSE
mismatch 仍均为 0。device/Host/`_C` hash 仍为
`0e61fa35102a088b3f2d6ae482bff0678b524f27f352f7944803a54c2d44842d`、
`48ee3b1e2bed64a4eb5a697ffae08c24189909c790dd86a0934fbd106c458fda`、
`433da93827b198a03eb0c7b8b9d396353f1023e96677bed9d3c95d58fa628a8b`。
benchmark runner hash 仍为
`c41fdcc4defc8a4f5c42859b4683f17df58bdd04c33351429e4fd63e1db850de`，
默认 seed-10 indices 与历史候选逐 byte 相同（SHA-256
`9b2a3011fbe8a05c64c21f856aa469dcbd1857393f4194d7cac1a82b17d8cdfd`），
因此未重跑 wall-time 或 profiler，既有性能证据继续适用。

### 12.2 同输入 Wall Time

三组测量均使用 seed 7、一次 warmup、三个逐次同步样本，并清除两个 tuning
环境变量。baseline 提交早于 benchmark runner，因此执行候选 runner 文件，但
`PYTHONPATH` 和 extension 都只指向 baseline tree；输入构造与计时语义相同。

| 实现 | 三个样本（ms） | 中位数（ms） |
| --- | --- | ---: |
| 通用 baseline `1297d3e` | `272551.605669, 272569.785186, 272568.230243` | `272568.230243` |
| 初始候选 `81b7165` | `358.798329, 359.500686, 358.652544` | `358.798329` |
| 最终 profile 候选 `fe376f3` | `335.586126, 336.215180, 335.327427` | `335.586126` |

最终候选是主对照，因为它是上游 Head64 gather-pipeline rebase 后重新构建、重新
验证并 profile 的源码；初始候选也完整保留，未挑选较快的一组隐藏漂移。主对照
相对 baseline 为 `812.215432x`，中位数降低 `99.876880%`，绝对降低
`272232.644117 ms`。初始候选为 `759.669731x`。

### 12.3 最终候选 Default Profile

最终候选使用 `--aic-metrics=Default --warm-up=5 --launch-count=1` 和 exact glob
`*sparse_attention_head64_fused_kernel*`。600 秒隔离 profiler executable 与
injection library SHA-256 分别为
`9648516a3404b7162c8044d7d9ff9c5bcddf9762d27d68808f00c824fdcf7f0a`、
`abcd89c2651c6c457be10cdc83f61f347d02709455a8b829b64bd7437c387831`。
完整目录为：

`/tmp/msopprof-sa-v32-prefill-head64-candidate-rebased-wsnZON-timeout600/OPPROF_20260729062747_IMRDMKLOXXKVQZVU`

BasicInfo 为 `334959.437500 us`、`Block Dim = 32`、
`Mix Block Dim = 64`。独立于 launch dimension 重新数 CSV 工作行：32/32 AIC
满足 `aic_cube_total_instr_number > 0`，64/64 AIV 满足
`aiv_vec_time(us) > 0`；每个 AIC 都记录 106496 条 Cube 指令。应用 trace 没有
Combine；唯一 Cast 位于 selected kernel 之前，属于确定性输入构造，selected
kernel 之后没有其他 kernel，因此 output Cast 为 0。

有限行上的算术平均值为：

| Cube % | Vector % | Scalar % AIC/AIV | AIC MTE1/MTE2/MTE3 % | AIV MTE2/MTE3 % |
| ---: | ---: | ---: | ---: | ---: |
| `1.674416` | `86.654003` | `0.895138 / 1.324269` | `4.410788 / 0 / 0` | `0 / 1.530755` |

| AIC wait Cube/MTE1/MTE2/MTE3 % | AIV wait Vector/MTE2/MTE3 % |
| ---: | ---: |
| `93.574156 / 93.077944 / 0 / 0` | `99.947698 / 0 / 99.954256` |

GM/L1/L0/UB 流量采用 CSV 原始 `KB` 字段并跨行求和：

| GM read/write | GM-to-L1 | L0C-to-L1 | L0C-to-GM | GM-to-UB | UB-to-GM |
| ---: | ---: | ---: | ---: | ---: | ---: |
| `19878032.375 / 1342591.750` | `12.000` | `37748736.000` | `0` | `90187.750` | `n/a` |

L2 hit rate 由 close/far 的 hit、miss、victim 原始 counter 求和后重算：AIC read
为 `96.875000%`；AIC write 没有事件，因此为 `n/a`；AIV read/write 分别为
`96.271313% / 68.104965%`。L0A read/write、L0B read/write、L0C Cube
read/write 平均带宽依次为 `1.586013 / 1.865898`、
`1.586013 / 3.172026`、`2.985436 / 6.344052 GB/s`。AIV UB Vector
read/write 平均带宽为 `45.321101 / 37.093105 GB/s`，UB write-to-GM 为
`0.004012 GB/s`；UB read-from-GM 和 UB-to-GM traffic 在 schema 中为 `NA`。

Default schema 不提供直接 FLOP/s 或 occupancy；同时 Cube FP instruction 字段为
0、但 total Cube instruction 非 0，因此不从该字段反推 FLOP/s。AIC L2 write
也因零事件而不可用，不能把 CSV 的 0% sentinel 当成命中率。

### 12.4 Baseline Profile 缺口与默认策略

原 100 秒 baseline capture 保留在：

`/tmp/msopprof-sa-v32-prefill-head64-baseline-W1p49H/OPPROF_20260729051202_ECEOIVYVHMWJMVMR`

其应用发生 AI Core timeout；stdout 明确包含
`Get op basic info [Task Duration] failed` 并显示 `0.000000 us`，
`OpBasicInfo.csv` 则写成 `NA`。其余 CSV 只有 timeout 后的 32 行截断数据，不能
作为 baseline 指标。随后 600 秒重试在 warmup 阶段按要求停止，仅在
`/tmp/msopprof-sa-v32-prefill-head64-baseline-W1p49H-timeout600/OPPROF_20260729063339_VNOZAPWFXZNBSNXC`
留下包含 `pc_start_addr.txt` 和 `aicore_binary.o` 两个 setup dump 的局部目录；
没有 BasicInfo、metric CSV、可用 duration 或 exit marker。该局部目录和 baseline
tree 中的 `task-5-artifacts/profile-timeout600.*` 日志均予以保留。

因此，候选与 baseline 的 selected-kernel duration 严格对比没有通过或完成，
这是明确的证据缺口，不能用 wall time 或截断 counter 代替。经批准豁免该项后，
仍保留 exact shape 的自动 Head64/P=1 dispatch：reduced/full accuracy、decode
回归、稳定性、同输入 wall median、候选有效 duration、32/32 Cube、64/64
Vector、零 Combine、零 output Cast 均通过。该决定不引用任何 baseline profiler
counter，也不扩展 P=4 或 ping-pong 方案。
