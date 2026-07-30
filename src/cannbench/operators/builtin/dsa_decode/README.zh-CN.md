# DeepSeek V3.2 DSA Decode 学习指南

本文以 CannBench 中的
`deepseek_v32_flashmla_decode_b2_q2_ctx32768_top2048` realistic case 为例，
解释 DSA decode workflow 的输入形状、Lightning Indexer、Sparse Attention、
MTP/speculative decode，以及当前 Ascend SIMT kernel 的并行方式。

本文首先解释模型和算法语义，再对应到当前实现。阅读本文不要求事先理解
CannBench 的插件代码，但默认读者知道 Transformer 是一种逐 token 生成模型。

## 1. 先明确 CannBench 测量的边界

真实模型的一次 decode 涉及很多步骤：

```text
已确认的 token 序列
    -> Embedding 和前序 Transformer 层
    -> 生成当前层的 Query/Key/Value
    -> DSA Lightning Indexer
    -> Sparse Attention
    -> 当前层剩余计算和后续 Transformer 层
    -> LM Head
    -> logits 和采样
    -> 新 token
```

CannBench 当前的 `dsa_decode` workflow 只覆盖中间的 DSA 核心链路：

```text
Lightning Indexer -> indices -> Sparse Attention -> output/lse
```

workflow 由两个独立组件组成，定义在 [`__init__.py`](./__init__.py)：

1. `lightning_indexer` 产生候选 KV 的 `indices`。
2. `sparse_attention` 读取这些 `indices`，计算正式的稀疏 Attention。

当前 benchmark 不包含：

- 从 token ID 生成 hidden state；
- Query/Key/Value projection；
- MTP 或 draft model 的候选 token 生成；
- RoPE 生成和 KV cache scatter；
- 当前 Attention 之后的其他 Transformer 层；
- LM Head、logits 和 token sampling。

因此，`Q=2` 表示算子收到两个已经准备好的 Query，而不是 CannBench 在运行时
生成了两个候选 token。当前 materializer 按 shape 构造测试张量，用于验证算子
合同和性能。

## 2. V3.2 realistic case 的完整形状

workflow case 位于 [`data/realistic.json`](./data/realistic.json)。两个组件的详细
shape 分别位于：

- [`lightning_indexer/data/realistic_decode.json`](../lightning_indexer/data/realistic_decode.json)
- [`sparse_attention/data/realistic_decode.json`](../sparse_attention/data/realistic_decode.json)

关键参数如下：

| 符号 | 当前值 | 含义 |
| --- | ---: | --- |
| `B` | 2 | batch 中有两个相互独立的序列槽位，服务场景中通常对应两个请求 |
| `Q` | 2 | 每条序列本次提供两个 Query 位置/向量，不是两个 token ID |
| `G` | 64 | Lightning Indexer 的 index head 数 |
| `Di` | 128 | 每个 index head 的向量维度 |
| `C` | 32768 | 每条序列可寻址的 KV context 长度上限 |
| `S` | 2048 | Indexer 最终选择的 TopK token 数 |
| `H` | 128 | 正式 Attention 的 query head 数 |
| `KV_H` | 1 | 所有 query head 共享一组 KV |
| `Dqk` | 576 | 正式 Attention 的 Query/Key 维度 |
| `Dv` | 512 | Value 和 Attention output 维度 |

### 2.1 `batch=2` 是什么

`batch=2` 表示同时处理两个独立的序列槽位，例如序列 A 和序列 B。在在线服务中，
它们通常来自两个请求，但 tensor shape 本身只规定序列彼此隔离。它们分别拥有
自己的 Query、KV context、有效长度和 TopK 结果。A 不会 attention 到 B 的
KV，B 也不会 attention 到 A 的 KV。

```text
序列 A -> A 自己的 Query 和 32768 个 context 位置
序列 B -> B 自己的 Query 和 32768 个 context 位置
```

### 2.2 `query_tokens=2` 是什么

每条序列包含两个 Query 位置，因此共有四个 `(batch, query)` row：

```text
(A, Query 0)
(A, Query 1)
(B, Query 0)
(B, Query 1)
```

Lightning Indexer 的 Query shape 是：

```text
[B, Q, G, Di] = [2, 2, 64, 128]
```

这里的 `64 x 128` 描述每个 query token 内部的检索特征，不表示请求数或
query token 数。

### 2.3 `context_tokens=32768` 是什么

`C=32768` 表示每条序列当前最多有 32768 个可寻址的 KV context 位置。其中通常
主要是已确认 prefix 的缓存，在多 token 验证时也可以包含本轮较早的候选位置，
所以不能把所有位置都严格叫作“此前历史”。这个数来自 FlashMLA V3.2 production
decode 测试形状，不是由 `64 x 128` 计算出来的。
来源说明见
[`docs/datasets/dsa-real-sources.md`](../../../../../docs/datasets/dsa-real-sources.md)。

## 3. 两个 Query 有依赖，为什么仍能并行

普通自回归生成中，后一个未知 token 确实依赖前一个 token，不能凭空提前生成。
但 Attention kernel 处理的是已经存在的 Query 向量，而不是在 kernel 内生成
token。

`Q=2` 可能来自 speculative decode 或 MTP-like decode，下面以此为例：

1. draft/MTP 路径先提出候选 `c1`、`c2`；
2. target model 收到两个已知候选；
3. target model 用一次多 Query 前向验证它们；
4. 验证结果仍按因果顺序接受或拒绝。

在同一 Transformer 层中，两个位置的层输入都已经准备好，因此可以同时投影出
各自的 Q/K/V。因果关系由 mask 和每行的有效 KV 长度表达，不要求 Query 0 的
Attention kernel 完成后才启动 Query 1。

当前 case 声明：

```text
query_start_positions = [32766, 32766]
context_lens          = [32768, 32768]
```

两个数组都按 batch 索引。`query_start=32766` 表示 Query 0 对应的位置从 0 数是
32766。每个 Query 的有效长度按下面的公式计算，其中 `+1` 表示包含当前位置：

```text
valid_context_length = min(context_len, query_start + query_index + 1)
```

所以每条请求中：

```text
Query 0 可见 32767 个 KV token
Query 1 可见 32768 个 KV token
```

Query 0 不能看到后一个位置，Query 1 可以看到前一个位置。两行可以并行计算，
但看到的数据范围不同。实现见
[`lightning_indexer/materialize.py`](../lightning_indexer/materialize.py)。

## 4. Lightning Indexer 做了什么

Lightning Indexer 是一个低成本候选检索器。它为每个 Query 和每个 context
token 计算一个粗排分数，然后保留 TopK 索引。

输入逻辑 shape：

```text
query   [B, Q, G, Di] = [2, 2, 64, 128]
keys    [B, C, Di]    = [2, 32768, 128]
weights [B, Q, G]     = [2, 2, 64]
```

对固定的 `b`、`q` 和 context 位置 `c`，分数公式是：

```text
raw_score[b,q,c] =
    sum_g(weights[b,q,g] * ReLU(dot(query[b,q,g], keys[b,c])))

score[b,q,c] = score_scale * raw_score[b,q,c]
```

V3.2 case 的 `score_scale=1.0`。最后对 `C=32768` 个分数取 Top-2048，输出：

```text
indices [B, Q, S] = [2, 2, 2048]
```

### 4.1 `64` 和 `128` 分别是什么

- `64` 是 index head 数。每个 head 可以学习一种不同的相关性判断方式。
- `128` 是每个 index Query/Key 向量的维度。

对一个 Query 和一个历史 token，Indexer 先做 64 次 128 维点积，再经过 ReLU、
head weight 和跨 head 求和，得到该历史 token 的一个标量分数。

### 4.2 点积后为什么使用 ReLU

ReLU 先把负点积截为 0：

```text
ReLU(x) = max(0, x)
```

例如两个 head 的点积分别为 `8` 和 `-7`：

```text
直接相加:          8 + (-7) = 1
经过 ReLU 后相加:  8 + 0    = 8
```

在当前 benchmark 的 materializer 中，`weights` 从 `[0, 1]` 生成。此时不使用
ReLU，一个 head 的负相关可能抵消另一个 head 的强正相关；使用 ReLU 后，多个
head 更像多个正向相关性探测器。更一般地说，ReLU 只保证被加权前的点积非负；
如果调用者传入负的 head weight，加权后的贡献仍可为负。

因此最根本的答案是：ReLU 是经过训练适配的 V3.2 Indexer 非线性评分函数的一
部分，不是为了避免数值溢出。

### 4.3 为什么 `Q_index` 可以比正式 Query 小很多

`Q_index` 不是从正式的 576 维 Attention Query 中截取 128 维。真实模型从同一个
hidden state 使用两套不同的可训练投影：

```text
q_index[g] = Wq_index[g] * hidden
k_index    = Wk_index * hidden_cache

q_attn[h]  = Wq_attn[h] * hidden
k_attn     = Wk_attn * hidden_cache
```

两套投影承担不同任务：

- `Q_index/K_index` 只需高召回地判断哪些 token 值得进入候选集；
- `Q_attn/K_attn` 需要计算正式 Attention 概率；
- `V` 负责承载最后要读取和混合的信息。

Indexer 不需要重建完整 Attention，也不要求 32768 个 token 的完整排序与正式
Attention 完全一致。它只需尽量让重要 token 落入较宽松的 Top-2048 候选集。
64 个不同的低维 head 和对应权重提供了比单个 128 维检索向量更强的表达能力。

这是训练得到的速度与召回率权衡，并非无损压缩。如果 index 维度或 TopK 太小，
重要 KV 可能被漏掉，后面的 Sparse Attention 无法恢复它。

## 5. Sparse Attention 做了什么

Indexer 只返回 token 位置。Sparse Attention 根据这些位置读取完整 KV，并计算
正式 Attention。

逻辑输入：

```text
Q          [B, H, Q, Dqk] = [2, 128, 2, 576]
shared_kv  [B, 1, C, Dqk] = [2, 1, 32768, 576]
indices    [B, Q, S]       = [2, 2, 2048]
```

V3.2 使用 shared KV：128 个 query head 共享同一组 KV。Key 使用完整的 576 维，
Value 使用 shared KV 的前 512 维：

```text
K = shared_kv
V = shared_kv[..., :512]
```

首先按 `indices` 读取候选：

```text
K_sparse = gather(K, indices)
V_sparse = gather(V, indices)
```

然后计算：

```text
scores = (Q * K_sparse^T) / sqrt(576)
P      = softmax(scores, dim=S)
output = P * V_sparse
```

输出：

```text
output [B, H, Q, Dv] = [2, 128, 2, 512]
lse    [B, H, Q]     = [2, 128, 2]
```

Softmax 沿 `S=2048` 个候选位置归一化。无效或因果上不可见的位置不会参与有效
分数计算。`lse` 是每个 Attention row 的 `logsumexp(scores)`，常用于稳定的
Softmax 实现、分块结果合并或后续校验。

### 5.1 `QK` 是什么

`QK` 点积衡量当前 Query 与每个候选 Key 的正式相关分数。与 Indexer 不同，
这里使用 Attention 自己的完整 Query/Key 投影和 `Dqk=576`。

### 5.2 `P` 和 `PV` 是什么

`P` 是 QK 分数经过 Softmax 后得到的归一化权重：

```text
P = softmax(Q * K_sparse^T / sqrt(Dqk), dim=S)
```

`PV` 是用这些权重对 Value 向量做加权求和。假设选中三个 token：

```text
P = [0.6, 0.3, 0.1]
V = [V0, V1, V2]

P * V = 0.6 * V0 + 0.3 * V1 + 0.1 * V2
```

可以把 QK 理解为“决定从每个历史位置读取多少”，把 PV 理解为“真正取出并
混合这些历史信息”。

## 6. Indexer 已经全量点积，为什么仍能省计算

这里确实存在两阶段相关性计算，但不是用同样的高成本 Attention 重算两遍：

```text
Indexer:          小维度、全 context、只做粗排
Sparse Attention: 大维度、TopK context、执行正式 QK/Softmax/PV
```

下面按单个 Query 位置估算主要乘加次数。MAC 是一次乘法加一次累加
（multiply-accumulate）。省略 batch 和 Query 位置数，因为它们会等比例放大
两种方案。

### 6.1 全量 Dense Attention

```text
QK: 128 * 576 * 32768 = 2,415,919,104
PV: 128 * 512 * 32768 = 2,147,483,648

列出的主 QK/PV 合计 = 4,563,402,752 MACs
```

### 6.2 DSA 两阶段

```text
全量 Indexer:
64 * 128 * 32768 = 268,435,456

Top-2048 Sparse Attention:
QK: 128 * 576 * 2048 = 150,994,944
PV: 128 * 512 * 2048 = 134,217,728

列出的主 Indexer/QK/PV 合计 = 553,648,128 MACs
```

只看这些主要乘加，DSA 大约是 Dense Attention 的 `1/8.2`。原因是：

1. Indexer 每个 context token 的检索计算比正式 QK 小很多；
2. 正式 QK 只处理 `2048/32768 = 1/16` 的 context；
3. 代价很高的正式 PV 也只处理 Top-2048；
4. Indexer 不执行正式 Softmax 和 PV。

这只是算术量估算。真实性能还受 TopK、访存、数据类型、kernel launch、并行度
和硬件利用率影响，因此不能直接把 `8.2` 当作实测加速比。

## 7. MTP/speculative decode 如何生成和验证候选

### 7.1 候选从哪里来

候选通常来自比完整 target model 便宜的路径：

- 小型 draft model 自回归地产生若干候选；或
- MTP 辅助模块复用主模型 hidden state，预测后续 token。

候选生成阶段仍然保留因果依赖。候选 `c2` 通常依赖候选 `c1`，只是这段生成由
较便宜的模型或辅助模块完成。

```text
已确认 prefix
    -> draft/MTP 产生 c1
    -> draft/MTP 基于 c1 产生 c2
    -> target model 一次验证 [c1, c2]
```

### 7.2 小候选模型为什么可能被大模型接受

候选模型不需要完整复制大模型，只要在较多位置提出大模型也认可的 token：

- 许多位置的分布很尖锐，例如固定短语、标点和代码语法；
- draft model 通常经过针对 target model 的蒸馏或对齐；
- MTP 可以复用 target model 已有的高质量 hidden state；
- target model 会检查所有候选，错误不会被无条件接受。

greedy decode 常见的直观判断是 draft 和 target 的最高概率 token 是否一致。
标准随机 speculative sampling 则会依据 target 分布 `p(x)` 和 draft 分布 `q(x)`
执行接受/拒绝，例如候选的接受概率包含：

```text
min(1, p(x) / q(x))
```

拒绝时需要按算法规定的校正分布采样，以保持最终输出分布与直接使用 target
model 一致。直观地说，校正分布会补回 target 比 draft 多出的概率质量，标准
形式与归一化后的 `(p-q)_+` 有关。具体接受规则取决于推理框架，不能把所有
MTP 实现都简化成 argmax 比较。

### 7.3 验证时还能生成 token 吗

可以。target model 的验证前向不只返回“接受/拒绝”，还会产生各位置的 logits。
以两个候选为例，验证逻辑需要下面三组条件分布：

```text
P(c1   | prefix)
P(c2   | prefix, c1)
P(next | prefix, c1, c2)
```

这是算法语义的示意，不表示当前 `Q=2` Attention 算子自己直接输出三组 logits。
实际实现使用 shifted logits：`P(c1 | prefix)` 可以来自 prefix 最后位置已有或
一并计算的 logits，两个候选位置的前向结果依次给出后续分布。Attention 后还要
经过剩余 Transformer 层和 LM Head 才能得到这些 logits。

可能出现：

- `c1` 被拒绝：丢弃 `c2`，由 target 在第一个位置产生校正 token；
- `c1` 接受、`c2` 被拒绝：保留 `c1`，由 target 产生第二个位置的校正 token；
- `c1`、`c2` 都接受：保留两者，并可从最后的 logits 再产生一个 bonus token，
  即由 target 顺带给出的额外 token。

因此标准 speculative decoding 的一次 target 验证通常至少推进一个 token。

### 7.4 接受率低是不是白费

不一定全白费，因为 target 在拒绝位置通常仍会给出校正 token；但候选生成中被
拒绝的部分确实浪费了，接受率过低可能比普通 `Q=1` decode 更慢。真正的收益
条件是：

```text
(候选生成成本 + 多 Query 验证成本) / 平均推进 token 数
    <
普通 target model 单 token decode 成本
```

设第一个候选被接受为事件 `A1`，前两个候选都被接受为事件 `A1∩A2`，并且全部
接受时还能产生一个 bonus token，则一次验证的平均推进量是：

```text
1 + P(A1) + P(A1 ∩ A2)
```

如果 `P(A1)=a` 且条件接受率 `P(A2|A1)=a`，才可简化为 `1+a+a^2`。实际系统
会监控接受率、动态调整候选长度，并在不划算时关闭 speculative decode。

CannBench 当前 case 没有测量候选生成成本或接受率，所以不能仅凭这个 `Q=2`
算子结果断言 MTP 端到端一定更快。

## 8. Sparse Attention 后面还有哪些计算

Attention 只是一个 Transformer layer 的子模块。一个简化的 decoder layer 是：

```text
hidden states
    -> Normalization
    -> Q/K/V projection
    -> Attention
    -> Attention output projection
    -> Residual Add
    -> Normalization
    -> FFN 或 MoE
    -> Residual Add
    -> 下一层 Transformer
```

经过全部 Transformer layer 后，还需要：

```text
final hidden state
    -> Final Norm
    -> LM Head
    -> vocabulary logits
    -> argmax/sampling
    -> token
```

各 head 的输出会先按模型结构拼接或重排，再进入 Attention output projection。
因此，Sparse Attention 的 `output` 是尚未经过 output projection 的逐 head
Attention 中间结果，不是 token。真正的 token 要等所有后续层和 LM Head 完成
后才能产生。

## 9. 当前 Ascend SIMT kernel 为什么是 `blockDim=4/32`，profiler 又显示 2 核

`dsa_decode` workflow 包含两个 kernel 路径，不能用一个 `blockDim` 描述整个
workflow。

### 9.1 Lightning Indexer: `blockDim=4`

当前 Indexer 将一个 `(batch, query)` row 作为一个外层逻辑工作单元：

```text
total_rows = batch_size * query_count
           = 2 * 2
           = 4

used_core_num = min(total_rows, kMaxUsedCoreNum)
              = min(4, 11)
              = 4
```

host launch 因此是：

```text
lightning_indexer_kernel<<<4, dynamic_ub, stream>>>(...)
```

实现见
[`lightning_indexer_fused_family_64x128.asc`](../lightning_indexer/simt/v1/aten_dsa_lightning_indexer/csrc/simt/lightning_indexer_fused_family_64x128.asc)。

这里的 `blockDim=4` 是 launch block 数，不应直接翻译成“4 个物理核”。每个逻辑
工作单元内部继续遍历自己的全部 context：

```text
32768 / 128 = 256 个 context tile
```

当前 `64` 个 index head 和 context tile 都没有扩展为外层独立 block。kernel 使用
`KERNEL_TYPE_MIX_AIC_1_2`，以 `1:2` 配比组合 Cube 侧 AIC 与 Vector 侧 AIV。
因此 profiler 如果展示 AIC/Cube 资源数，`blockDim=4` 的这次 mixed launch 可以
显示为 2 个 AIC；这与源码中的 launch block 数不是同一个统计口径，也不表示
只处理了两个 `(batch, query)` row。内部 `asc_vf_call` 使用 256 个 SIMT thread，
这又是第三种粒度。

可以按下面三层理解：

```text
算法工作量:  B * Q = 2 * 2 = 4 个 row
launch 参数: blockDim = 4 个逻辑 block
profiler:    mixed-kernel 资源口径下可显示 2 个 AIC
```

`kMaxUsedCoreNum=11` 是当前 Indexer 实现的上限，与 mixed kernel 的逐 block
同步 flag 分配有关，不代表整张设备只有 11 个核。

### 9.2 为什么不能直接把 Indexer 改成 32

只把 `kMaxUsedCoreNum` 从 11 改成 32 没有效果：

```text
min(total_rows=4, max=32) = 4
```

强制 launch 32 也不会自动产生更多工作。按当前 row 映射，额外 28 个 block 没有
合法 row；只有在保持正确边界检查时它们才会空跑，否则还可能越界访问。

要让 32 个工作单元都有有效计算，可以将每个 row 的 context 分成 8 份：

```text
4 个 query row * 每行 8 个 context shard = 32 个 task
```

但这需要修改算法数据流，而不只是修改常量：

1. 每个 shard 独立计算约 4096 个 context token；
2. 每个 shard 产生自己的 partial TopK；
3. partial TopK 写入互不冲突的 workspace；
4. 第二阶段合并同一 row 的 8 份候选，得到最终 Top-2048。

如果多个 block 直接写同一份当前 TopK 状态，会产生竞争和错误结果。因此真正的
32-task Indexer 并行需要 context sharding 和跨 shard TopK merge 设计。这里的
32 仍是逻辑 task 数，是否同时占用 32 个物理执行单元还取决于 mixed-kernel
映射和硬件调度。

### 9.3 Sparse Attention: `blockDim=32`

V3.2 BF16 fused Sparse Attention 将 query head 也纳入外层 row：

```text
total_rows = batch_size * query_heads * query_tokens
           = 2 * 128 * 2
           = 512

used_core_num = min(total_rows, kFusedMaxUsedCoreNum)
              = min(512, 32)
              = 32
```

实现见
[`sparse_attention_score_family_hd512.asc`](../sparse_attention/simt/v1/aten_dsa_sparse_attention/csrc/simt/sparse_attention_score_family_hd512.asc)。

`kFusedMaxUsedCoreNum=32` 是该 fused 实现选定的 launch 上限；源码没有把它声明
成设备物理核总数。它和 Indexer 的 11 属于两个 kernel 各自的实现约束。

所以当前源码的主要 launch block 参数是：

```text
Lightning Indexer: blockDim = 4
Sparse Attention:  blockDim = 32
```

## 10. 一页总结

| 容易产生的误解 | 正确理解 |
| --- | --- |
| `B=2` 必然是两个用户请求 | 它严格表示两个隔离的序列槽位，服务中通常对应两个请求 |
| `Q=2` 跳过了自回归依赖 | 候选先按因果关系产生；target 对已知候选做并行张量计算，并用 mask 保持因果性 |
| `64 x 128` 是 8192 个外层任务 | 它是每个 Query 的 Indexer 特征 shape；当前外层 Indexer 只有 `B*Q=4` 个 row |
| Indexer 和 Attention 完全重复 | Indexer 用小投影全量粗排，Attention 用完整 KV 对 TopK 做 QK/Softmax/PV |
| `blockDim=4` 就是使用 4 个物理 AIC | 它是 launch block 数；mixed kernel 的 profiler 可能显示 2 个 AIC |
| MTP 必然加速 | 收益取决于候选成本、多 Query 验证成本和接受率 |
| 此 `Q=2` benchmark 证明接受率很高 | 它不生成候选、不执行 LM Head，也不测接受率，只测两个已准备 Query 的 DSA 链路 |

## 11. 推荐的源码阅读顺序

1. [`data/realistic.json`](./data/realistic.json)：workflow case 入口。
2. [`__init__.py`](./__init__.py)：Indexer 和 Sparse Attention 的 workflow 连接。
3. [`lightning_indexer/data/realistic_decode.json`](../lightning_indexer/data/realistic_decode.json)：Indexer shape。
4. [`lightning_indexer/__init__.py`](../lightning_indexer/__init__.py)：Torch baseline 和 SIMT callable。
5. [`lightning_indexer_fused_family_64x128.asc`](../lightning_indexer/simt/v1/aten_dsa_lightning_indexer/csrc/simt/lightning_indexer_fused_family_64x128.asc)：Indexer 融合 kernel。
6. [`sparse_attention/data/realistic_decode.json`](../sparse_attention/data/realistic_decode.json)：Sparse Attention shape。
7. [`sparse_attention/simt/README.md`](../sparse_attention/simt/README.md)：Sparse Attention 数学过程和 shape。
8. [`sparse_attention_score_family_hd512.asc`](../sparse_attention/simt/v1/aten_dsa_sparse_attention/csrc/simt/sparse_attention_score_family_hd512.asc)：V3.2 wide-head 融合 kernel。
9. [`docs/designs/dsa-v32-three-path-semantics.md`](../../../../../docs/designs/dsa-v32-three-path-semantics.md)：三条实现路径的语义合同。
10. [`docs/designs/dsa-inference-fusion-spec.md`](../../../../../docs/designs/dsa-inference-fusion-spec.md)：DSA workflow 的整体设计背景。
