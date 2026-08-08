# DSA V2 Decode 全流程与 vLLM 风格拆分 Shape 分析

## 1. 文档目的

本文只分析以下 canonical decode case：

```text
deepseek_v32_flashmla_decode_b2_q2_ctx32768_top2048
```

目标是把两层信息分开说明：

1. 不考虑 AIC/AIV 切分时，`dsa_decode` 从 Lightning Indexer 到 Sparse
   Attention 的完整输入、输出、逻辑中间张量和矩阵计算；
2. 采用 vLLM-Ascend MLA 风格后，如何沿 Query head 和 selected token 切成
   Head64、selected128、双 AIC 和四 AIV，以及哪些数据可以跨 Head64 group
   复用。

本文中的 shape 与数据流来自 CannBench case、参考实现和已核对的
vLLM-Ascend `a5b0ce10d84bf76dd9c5c9e7ab9a5ddeed5af7ca` 源码。资源布局中标为
“候选设计”的部分是当前优化方案，不代表已经验证有性能收益。

## 2. 符号和 case 参数

| 符号 | 含义 | 当前值 |
| --- | --- | ---: |
| `B` | batch size | 2 |
| `Tq` | 每个 batch 的 Query token 数 | 2 |
| `R=B*Tq` | 有效 `(batch, query_token)` 行数 | 4 |
| `C` | context token 数 | 32768 |
| `Hi` | Indexer head 数 | 64 |
| `Di` | Indexer head dim | 128 |
| `H` | Attention Query head 数 | 128 |
| `Hkv` | Attention KV head 数 | 1 |
| `S` | TopK 后 selected token 数 | 2048 |
| `Dqk` | QK head dim | 576 |
| `Dv` | Value/output head dim | 512 |
| `Dr=Dqk-Dv` | RoPE dim | 64 |

其他语义：

```text
dtype                  = BF16
causal                 = true
softmax_scale          = 1 / sqrt(576) = 1 / 24
query_lens             = [2, 2]
context_lens           = [32768, 32768]
query_start_positions  = [32766, 32766]
topk_lengths           = [2048, 2048, 2048, 2048]
page_block_size        = 64
```

两个 Query token 在各自 context 中的位置分别为 `32766` 和 `32767`，所以 causal
有效长度为：

```text
valid_length(q=0) = min(32768, 32766 + 0 + 1) = 32767
valid_length(q=1) = min(32768, 32766 + 1 + 1) = 32768

valid_context_lengths [B,Tq]
  = [[32767, 32768],
     [32767, 32768]]
```

序列和分页 metadata 为：

```text
cu_seqlens_q  [B+1] = [0, 2, 4]
cu_seqlens_kv [B+1] = [0, 32768, 65536]

blocks_per_batch = ceil(32768 / 64) = 512
block_tables      [B,512] = [2,512]
```

`block_tables` 第一行对应 page `0..511`，第二行对应 page `512..1023`。

## 3. Workflow 总览

`dsa_decode` 是两个独立 operator plugin 组成的 workflow：

```text
Lightning Indexer
  输入: index_query, index_key, weights, sequence metadata
  输出: indices [2,2,2048]
                    |
                    | workflow 唯一跨阶段绑定
                    v
Sparse Attention
  输入: query, shared_kv, indices, sequence metadata
  输出: out [2,2,128,512], lse [2,2,128]
```

必须注意：

- Indexer 的 `index_query[...,128]` 不是 Attention 的 `query[...,576]`；
- Indexer 的 `index_key[...,128]` 不是 Attention 的 `shared_kv[...,576]`；
- 当前 workflow 不把 Indexer 的 score 或 gathered data 传给 Attention，只传
  `indices`；
- 因而两个阶段可以使用完全不同的特征空间和 kernel 实现。

## 4. 第一阶段：Lightning Indexer

### 4.1 真实输入和输出 shape

| 张量 | Shape | dtype | 元素数 | BF16/实际存储量 |
| --- | --- | --- | ---: | ---: |
| `index_query` | `[B,Tq,Hi,Di] = [2,2,64,128]` | BF16 | 32,768 | 64 KiB |
| `index_key` | `[B,C,Di] = [2,32768,128]` | BF16 | 8,388,608 | 16 MiB |
| `weights` | `[B,Tq,Hi] = [2,2,64]` | BF16 | 256 | 512 B |
| `valid_lengths` | `[B,Tq] = [2,2]` | INT32 | 4 | 16 B |
| `indices` | `[B,Tq,S] = [2,2,2048]` | 实现相关整数类型 | 8,192 | INT64 为 64 KiB，INT32 为 32 KiB |

### 4.2 单个 `(b,q)` 的计算

固定一个 batch `b` 和 Query token `q`：

```text
Q_idx = index_query[b,q,:,:]  [Hi,Di] = [64,128]
K_idx = index_key[b,:,:]      [C,Di]  = [32768,128]
w     = weights[b,q,:]        [Hi]    = [64]
```

先为每个 Indexer head 和 context position 计算点积：

```text
head_scores = Q_idx @ K_idx^T

[64,128] x [128,32768] -> [64,32768]
```

逐元素 ReLU、乘 head weight，再沿 `Hi=64` 求和：

```text
activated[h,c] = ReLU(head_scores[h,c])
weighted[h,c]  = activated[h,c] * w[h]
score[c]       = sum(h=0..63, weighted[h,c])

[64,32768] -> [32768]
```

对 causal 无效位置填 `-inf`，最后沿 context 取 TopK：

```text
indices[b,q,:] = TopKIndices(score, k=2048)

[32768] -> [2048]
```

TopK 按 score 从大到小返回位置。case 的 tie policy 是
`equivalent_score_set`，相同分数只要求返回等价的候选集合。

### 4.3 全 batch 的逻辑中间 shape

| 逻辑张量 | Shape | 说明 |
| --- | --- | --- |
| `head_scores` | `[B,Tq,Hi,C] = [2,2,64,32768]` | 每个 Indexer head 的分数 |
| `activated/weighted` | `[2,2,64,32768]` | 与 `head_scores` 同 shape |
| `index_scores` | `[B,Tq,C] = [2,2,32768]` | 64 heads 聚合后的分数 |
| `indices` | `[B,Tq,S] = [2,2,2048]` | workflow 传给下一阶段 |

`head_scores` 有 `8,388,608` 个元素，若按 FP32 完整物化为 32 MiB；
`index_scores` 有 `131,072` 个元素，若按 FP32 完整物化为 512 KiB。这里描述的是
Torch 参考公式的逻辑 shape，不表示自定义 Indexer kernel 必须在 GM 中写出这些完整
中间张量。自定义实现可以按 context tile 计算、归约并直接进入 TopK。

Indexer 点积总乘加数量为：

```text
B * Tq * Hi * C * Di
= 2 * 2 * 64 * 32768 * 128
= 1,073,741,824 MAC
```

## 5. 第二阶段：Sparse Attention 的完整逻辑计算

### 5.1 真实输入和输出 shape

CannBench canonical 输入布局为 BHTD：

| 张量 | Shape | dtype | 元素数 | 存储量 |
| --- | --- | --- | ---: | ---: |
| `query` | `[B,H,Tq,Dqk] = [2,128,2,576]` | BF16 | 294,912 | 576 KiB |
| `shared_kv` | `[B,Hkv,C,Dqk] = [2,1,32768,576]` | BF16 | 37,748,736 | 72 MiB |
| `indices` | `[B,Tq,S] = [2,2,2048]` | INT64/INT32 | 8,192 | 64/32 KiB |
| `out` | `[B,Tq,H,Dv] = [2,2,128,512]` | BF16 | 262,144 | 512 KiB |
| `lse` | `[B,Tq,H] = [2,2,128]` | FP32 | 512 | 2 KiB |

`shared_kv` 的最后一维有两个重叠用途：

```text
shared_kv[..., 0:512]    -> K_nope，同时也是 V
shared_kv[..., 512:576]  -> K_rope

K = concat(K_nope, K_rope)  [576]
V = shared_kv[..., 0:512]    [512]
```

因此 `V [2,1,32768,512]` 是 `shared_kv` 前 512 维的逻辑 view，包含 64 MiB
有效数据，但不要求额外复制一份 64 MiB buffer。

### 5.2 先去掉 batch 和 Query token 维

固定一个 `(b,q)`，把 Query head 放到矩阵行：

```text
Q = query[b,:,q,:]             [H,Dqk] = [128,576]
I = indices[b,q,:]             [S]     = [2048]
K_selected = shared_kv[b,0,I,:]        = [2048,576]
V_selected = shared_kv[b,0,I,0:512]    = [2048,512]
```

`Hkv=1`，所以同一个 `I`、`K_selected` 和 `V_selected` 被全部 128 个 Query
heads 共享。完整 QK 为：

```text
scores = scale * Q @ K_selected^T

[128,576] x [576,2048] -> [128,2048]
```

应用非法 index、`topk_lengths` 和 causal mask 后，沿 selected 维做 softmax：

```text
P[h,:]   = softmax(scores[h,:])    [2048]
LSE[h]   = logsumexp(scores[h,:])  scalar

scores [128,2048] -> P [128,2048], LSE [128]
```

最后执行 PV：

```text
O = P @ V_selected

[128,2048] x [2048,512] -> [128,512]
```

### 5.3 合并 `B*Tq=4` 后的矩阵形态

把 `(b,q)` 合并为行组 `R=4`：

```text
Q             [R,H,Dqk] = [4,128,576]
K_selected    [R,S,Dqk] = [4,2048,576]
V_selected    [R,S,Dv]  = [4,2048,512]

QK:
[4,128,576] x [4,576,2048] -> scores [4,128,2048]

Softmax:
scores [4,128,2048] -> P [4,128,2048], LSE [4,128]

PV:
[4,128,2048] x [4,2048,512] -> O [4,128,512]
```

QK 和 PV 的总乘加数量分别为：

```text
QK = R * H * S * Dqk
   = 4 * 128 * 2048 * 576
   = 603,979,776 MAC

PV = R * H * S * Dv
   = 4 * 128 * 2048 * 512
   = 536,870,912 MAC
```

### 5.4 参考实现里的 head broadcast

Torch 参考实现为了表达 `Hkv=1` 被 `H=128` 个 Query heads 共享，会先形成以下
逻辑 shape：

```text
expanded K [B,H,C,Dqk] = [2,128,32768,576]
expanded V [B,H,C,Dv]  = [2,128,32768,512]
```

随后 gather 的逻辑结果为：

```text
selected K [B,H,Tq,S,Dqk] = [2,128,2,2048,576]
selected V [B,H,Tq,S,Dv]  = [2,128,2,2048,512]
scores     [B,H,Tq,S]      = [2,128,2,2048]
P          [B,H,Tq,S]      = [2,128,2,2048]
```

若把 broadcast 后的 `selected K` 按 BF16 完整物化，需要 2.25 GiB；`selected V`
需要 1 GiB。它们只是参考实现方便表达逐 head 计算的 shape，不是优化 kernel 的
合理真实存储方式。

优化实现只需要对每个 `(b,q)` gather 一份基础张量：

```text
base selected K [B,Tq,S,Dqk] = [2,2,2048,576]  -> 9 MiB BF16
base selected V [B,Tq,S,Dv]  = [2,2,2048,512]  -> 8 MiB BF16
```

实际 kernel 还会继续按 tile 流式处理，不需要在 GM 中一次性物化这 9 MiB 和
8 MiB 中间结果。

## 6. vLLM-Ascend adapter 的 TND 输入 shape

CannBench 调用 vLLM-Ascend SFA 前会把 canonical BHTD 输入压成 TND，并拆分
NoPE/RoPE：

```text
total_query_tokens   = sum(query_lens)   = 4
total_context_tokens = sum(context_lens) = 65536

query_nope [T,H,Dv] = [4,128,512]
query_rope [T,H,Dr] = [4,128,64]

key_nope  [Tkv,Hkv,Dv] = [65536,1,512]
key_rope  [Tkv,Hkv,Dr] = [65536,1,64]
value     [Tkv,Hkv,Dv] = [65536,1,512]

sparse_indices [T,Hkv,S] = [4,1,2048]
```

这里 `key_nope` 和 `value` 的数值都来自 canonical `shared_kv[...,0:512]`。
语义上可以复用同一份数据；当前 adapter 为满足外部接口分别构造 `key` 和 `value`
参数，不能把 adapter 的物理 buffer 数直接等同于 fused kernel 内部必需的数据量。

长度 metadata 为：

```text
actual_seq_lengths_query [B] = [2,4]
actual_seq_lengths_kv    [B] = [32768,65536]
```

## 7. vLLM 风格的第一层拆分：Head64 和 selected128

### 7.1 任务数

沿 Query head 每 64 个一组：

```text
group 0 -> global heads [0,64)
group 1 -> global heads [64,128)
```

沿 selected token 每 128 行一个 tile：

```text
selected_tile_count = S / 128 = 2048 / 128 = 16
```

不沿 selected 维生成独立 output task 时，基础 AIC 任务数为：

```text
AIC tasks = B * Tq * ceil(H / 64)
          = 2 * 2 * 2
          = 8
```

每个 AIC task 顺序处理 16 个 selected128 tile。

### 7.2 单个 Head64、selected128 tile 的 shape

固定 `(b,q,head_group,tile)`：

```text
Q_g       [64,576]
indices_t [128]
K_t       [128,576]
V_t       [128,512]
```

QK：

```text
[64,576] x [576,128] -> Score_t [64,128]
```

Softmax tile：

```text
Score_t [64,128] -> P_t [64,128]
```

PV：

```text
[64,128] x [128,512] -> O_t [64,512]
```

单 tile 的 QK 和 PV 乘加量为：

```text
QK_tile = 64 * 128 * 576 = 4,718,592 MAC
PV_tile = 64 * 128 * 512 = 4,194,304 MAC
```

### 7.3 为什么 tile softmax 不能各自独立归一化

16 个 selected128 tile 共同组成长度 2048 的同一行 softmax。每个 tile 必须更新
running max、running sum 和 running output。概念上的稳定递推为：

```text
m_t     = max(m_{t-1}, rowmax(Score_t))
alpha_t = exp(m_{t-1} - m_t)
E_t     = exp(Score_t - m_t)
l_t     = alpha_t * l_{t-1} + rowsum(E_t)
o_t     = alpha_t * o_{t-1} + E_t @ V_t

final_output = o_15 / l_15
final_lse    = m_15 + log(l_15)
```

每个 Head64 task 的持久状态 shape 为：

```text
running_max [64]
running_sum [64]
running_out [64,512]
```

它们属于当前 Head64 group，不能与另一个 group 共享；`indices_t/K_t/V_t` 则可以
共享。

## 8. 第二层拆分：双 AIC、四 AIV

### 8.1 两个 AIC 负责不同 Query heads

对同一个 `(b,q)`：

| AIC | Query head 范围 | Query shape | Score shape | Output shape |
| --- | --- | --- | --- | --- |
| `AIC0` | `[0,64)` | `[64,576]` | `[64,128]` | `[64,512]` |
| `AIC1` | `[64,128)` | `[64,576]` | `[64,128]` | `[64,512]` |

两个 AIC 的 Query、Score、P、running state 和 output 不同，但读取完全相同的：

```text
indices_t [128]
K_t       [128,576]
V_t       [128,512]
```

所以共享边界应该放在 KV gather，而不是 QK、softmax 或 PV 结果上。

### 8.2 四个 AIV 的 gather quarter

两个相邻 Head64 AIC 各有两个 AIV subblock。vLLM 风格的共享 gather 把
selected128 分成四个 32-row quarter：

| Quarter | 物理执行者 | selected 行范围 | Gather shape |
| ---: | --- | --- | --- |
| 0 | Head64-0 AIV0 | `[0,32)` | `[32,576]` |
| 1 | Head64-0 AIV1 | `[32,64)` | `[32,576]` |
| 2 | Head64-1 AIV0 | `[64,96)` | `[32,576]` |
| 3 | Head64-1 AIV1 | `[96,128)` | `[32,576]` |

每个 quarter 为：

```text
32 * 576 BF16 = 18,432 elements = 36 KiB
```

四个 quarter 拼成唯一的一份：

```text
KV_t [128,576]
= 128 * 576 BF16
= 73,728 elements
= 147,456 B
= 144 KiB
```

两个 AIC 随后分别把这同一份连续 KV tile 搬入各自私有 L1。L1 不能跨 AIC
直接共享，所以“共享 gather”消除的是对原始离散 KV 的重复读取，不会消除两个
AIC 各自的连续 GM-to-L1 搬运。

### 8.3 每个 AIV 消费的 Query head half

Cube 侧仍按一个 Head64 矩阵计算，L0C 结果沿 M 维双发给直属两个 AIV：

| AIV subblock | Query head 范围 | Q half | Score half | P half | Output half |
| --- | --- | --- | --- | --- | --- |
| 0 | group 内 `[0,32)` | `[32,576]` | `[32,128]` | `[32,128]` | `[32,512]` |
| 1 | group 内 `[32,64)` | `[32,576]` | `[32,128]` | `[32,128]` | `[32,512]` |

需要区分两种互相独立的“32”：

- gather quarter 的 `32` 是 selected rows；
- Score/P/output half 的 `32` 是 Query heads。

同一个 AIV 在 gather 阶段可能生产某 32 个 selected rows，但在 softmax/output
阶段消费的是自己所属 Head64 group 中的 32 个 Query heads。二者不能因为数值相同
而混成同一维。

### 8.4 AIV 局部中间数据量

| 张量 | 单 AIV shape | dtype | 大小 |
| --- | --- | --- | ---: |
| Query half | `[32,576]` | BF16 | 36 KiB |
| Gather quarter | `[32,576]` | BF16 | 36 KiB |
| Score half | `[32,128]` | FP32 | 16 KiB |
| Probability half | `[32,128]` | BF16 | 8 KiB |
| Running output half | `[32,512]` | FP32 | 64 KiB |
| Running max | `[32]` | FP32 | 128 B |
| Running sum | `[32]` | FP32 | 128 B |

这些大小只是 capacity worksheet 的基础项。真实动态 UB 还必须计入多槽、搬运
staging、offset、对齐、临时归约区和编译器保留空间，不能把表中数字直接相加后
当成最终 launch UB 大小。

## 9. QK 与 PV 如何复用同一份 KV tile

`K_t [128,576]` 可拆成：

```text
K_nope/V [128,512] -> 65,536 BF16 -> 128 KiB
K_rope   [128,64]  ->  8,192 BF16 ->  16 KiB
```

QK 使用完整 576 维：

```text
Q_nope [64,512] x K_nope^T [512,128]
+ Q_rope [64,64] x K_rope^T [64,128]
-> Score [64,128]
```

PV 只使用前 512 维作为 Value：

```text
P [64,128] x V [128,512] -> O [64,512]
```

QK 完成后，`K_rope [128,64]` 不再存活；恰好：

```text
K_rope = 128 * 64 BF16 = 8192 BF16 = 16 KiB
P      =  64 * 128 BF16 = 8192 BF16 = 16 KiB
```

因此同一个 L1 slot 可以做生命周期复用：

```text
QK 阶段:
[ V/K_nope: 128 x 512 ][ K_rope: 128 x 64 ]

PV 阶段:
[ V:        128 x 512 ][ P:       64 x 128 ]
```

这里只是字节容量相等，`K_rope` 和 `P` 的逻辑二维布局不同；搬运和 Cube 输入必须
分别按对应的物理格式解释该尾部地址。

当前 CannBench rolling 候选的 PV 仍沿输出 dim 拆成两次：

```text
[64,128] x [128,256] -> [64,256]  # Dv [0,256)
[64,128] x [128,256] -> [64,256]  # Dv [256,512)
```

vLLM-Ascend 的逻辑矩阵是一次完整：

```text
[64,128] x [128,512] -> [64,512]
```

这是一项仍需结合 Tensor API 实际 tile 和 timeline 验证的实现差异，不能仅由逻辑
shape 断言哪种更快。

## 10. 三槽 rolling pipeline 的 shape 和生命周期

一个 Head64 task 有 16 个 selected128 tile。三个 rolling slot 交错承载：

```text
tile t:     gather KV + QK
tile t-1:   online softmax + PV
tile t-2:   running output update
```

填充、稳态和排空共 18 个 round：

| Round | Vector/MTE | Cube |
| ---: | --- | --- |
| 0 | gather tile 0 | QK tile 0 |
| 1 | gather 1, softmax 0 | QK 1, PV 0 |
| 2 | gather 2, softmax 1, update 0 | QK 2, PV 1 |
| `...` | 三阶段滚动 | QK/PV 滚动 |
| 15 | gather 15, softmax 14, update 13 | QK 15, PV 14 |
| 16 | softmax 15, update 14 | PV 15 |
| 17 | update 15, final store | drain |

若每个 L1 slot 暂存完整 `KV_t [128,576]`，候选三槽的主要 L1 shape 为：

```text
Query       [64,576]      =  73,728 B
KV slot 0   [128,576]     = 147,456 B
KV slot 1   [128,576]     = 147,456 B
KV slot 2   [128,576]     = 147,456 B
-------------------------------------
payload                       516,096 B
```

在 512 KiB L1 下只剩 `8,192 B`，所以三槽方案依赖上一节的尾部复用，不能再为
完整 Score 或 P 单独申请 L1 区域。这是当前候选设计的 capacity 约束，不应外推成
所有 vLLM-Ascend 版本的固定物理布局。

## 11. 跨 Head64 group 共享 gather 的总量

### 11.1 当前不共享时

每个 `(b,q)` 有两个 Head64 group，每个 group 都完整 gather 16 个 selected128
tile：

```text
完整 KV tile gather 次数
= R * head_groups * selected_tile_count
= 4 * 2 * 16
= 128
```

按有效数据量估算，对原始 `shared_kv` 的离散读取为：

```text
128 tiles * 128 rows/tile * 576 dims * 2 B
= 18,874,368 B
= 18 MiB
```

若 Value 又按独立路径重新 gather 前 512 维，则额外：

```text
128 * 128 * 512 * 2 B = 16 MiB
```

合计为 34 MiB 原始离散 KV 读取。这个 34 MiB 描述的是当前 Key/Value 分别 gather
的实现边界，不是 Sparse Attention 数学语义要求。

### 11.2 跨 group 共享后

两个 Head64 group 共用一份 tile：

```text
唯一完整 KV tile gather 次数
= R * selected_tile_count
= 4 * 16
= 64
```

原始离散 `576` 维读取为：

```text
64 * 128 * 576 * 2 B = 9 MiB
```

但共享方案还会增加连续 workspace 流量：

```text
四个 AIV quarter -> 一次共享 GM slot 写入
共享 GM slot -> 两个 Head64 AIC 的私有 L1，各读取一次
```

所以它减少的是最昂贵的重复随机 gather，不是把总 GM 字节数简单减半。是否有净
收益还取决于连续写读、同步、地址生成和流水重叠，必须用完整 operator profile
验证。

### 11.3 三槽共享 GM workspace shape

对于 `R=4` 个 `(b,q)` pair，每个 pair 三个 144 KiB slot：

```text
counter area = 512 B
data area    = 4 pairs * 3 slots * 128 * 576 * 2 B
             = 1,769,472 B
total        = 1,769,984 B
```

候选布局为：

```text
[0, 512)             ready/consumed counters
[512, 147968)        pair 0, slot 0, [128,576] BF16
[147968, 295424)     pair 0, slot 1, [128,576] BF16
[295424, 442880)     pair 0, slot 2, [128,576] BF16
...                  pair 1..3
```

这是候选共享协议的物理 workspace，不是 Sparse Attention 对外 contract 的输出。

## 12. 当前 CannBench 与 vLLM 风格拆分的关键差异

| 维度 | 当前 CannBench P=1 rolling 候选 | vLLM 风格目标 |
| --- | --- | --- |
| Query head tile | 64 | 64 |
| selected tile | 128 | 128 |
| 每 `(b,q)` AIC 数 | 2 | 2 |
| 两个 AIV 的计算职责 | 各消费 32 Query heads | 各消费 32 Query heads |
| 原始 KV gather | 每个 Head64 group 各 gather `[128,576]` | 两个 group 合作 gather一份 `[128,576]` |
| Gather producer | 每组两个 AIV 各 64 selected rows | 四个 AIV 各 32 selected rows |
| QK | `[64,576]x[576,128]` | 相同 |
| PV | 两次 `[64,128]x[128,256]` | 逻辑上 `[64,128]x[128,512]` |
| selected 分区 | P=1，单 task 顺序处理 16 tiles | P=1，单 task 顺序处理 16 tiles |
| 最终结果 | 直接写 output/LSE | 直接写 output/LSE |

当前 main 的 P=4 路径则是另一种任务拆分：

```text
tasks = B * Tq * head_groups * selected_partitions
      = 2 * 2 * 2 * 4
      = 32
```

每个 partition 只处理 `2048/4=512` selected rows，但必须写 partial output/LSE，
最后由 Combine 合并。其 FP32 partial output shape 可写为：

```text
[B,Tq,head_groups,P,64,Dv]
= [2,2,2,4,64,512]
= 1,048,576 FP32
= 4 MiB
```

P=1 rolling/vLLM 风格通过在线 softmax 在同一 task 内顺序合并 16 个 tile，避免这块
4 MiB partial output 和 Combine；与此同时 AIC task 数从 32 降为 8。前者减少 GM
和 launch，后者降低并行度，因此不能只看其中一项预测最终性能。

## 13. 一条数据从输入到输出的完整路径

固定 `b=0, q=0`，完整路径可以压缩为：

```text
1. Indexer
   index_query[0,0,:,:] [64,128]
     x index_key[0,:,:]^T [128,32768]
     -> head_scores [64,32768]
     -> ReLU * weights[0,0,:] -> reduce heads
     -> index_scores [32768]
     -> TopK -> indices[0,0,:] [2048]

2. Attention，未拆分
   indices[0,0,:] gather shared_kv[0,0,:,:]
     -> K [2048,576], V view [2048,512]
   query[0,:,0,:] [128,576] x K^T [576,2048]
     -> scores [128,2048]
     -> softmax/LSE
     -> P [128,2048], LSE [128]
   P [128,2048] x V [2048,512]
     -> output [128,512]

3. Attention，Head64/selected128 拆分
   heads [0,64) 和 [64,128) -> 2 个 AIC
   indices[0,0,:] -> 16 个 [128] tile
   每轮四个 AIV各 gather [32,576]
     -> 合成共享 KV [128,576]
   两个 AIC分别执行：
     QK [64,576] x [576,128] -> [64,128]
     online softmax -> P [64,128]
     PV [64,128] x [128,512] -> [64,512]
   16 轮后各自完成 64 heads 的 running output/LSE
   拼回 out[0,0,:,:] [128,512] 和 lse[0,0,:] [128]
```

## 14. 结论与后续判断边界

从 shape 和计算依赖可以确定：

1. Workflow 两阶段唯一可直接复用的当前 contract 是 `indices [2,2,2048]`；
   Indexer 的 128 维特征不能直接当成 Attention 的 576 维 KV。
2. 同一个 `(b,q)` 的两个 Head64 group 使用完全相同的 selected indices 和 KV，
   原始 KV gather 在数学上只需一次。
3. Query、Score、softmax state、P 和 output 都依赖 Query head，不能跨两个
   Head64 group 共享。
4. selected128 tile 只是在线 softmax 的一个分块，16 个 tile 必须通过 running
   max/sum/output 合并，不能做 16 个互不相关的 softmax。
5. 四 AIV 共享 gather 把 `[128,576]` 切成四个 `[32,576]` producer quarter；
   每个 AIV 后续消费的 `[32,128]` Score half 则沿 Query head 切分，两种 32
   属于不同维度。
6. QK 后 `K_rope [128,64]` 与 `P [64,128]` 字节数相等，允许做 L1 生命周期
   复用；这不代表两者布局相同。
7. 共享 gather 和三槽流水目前仍是候选优化边界。随机读取减少多少可以由 shape
   算出，最终延迟收益必须把 workspace 连续读写、同步和完整 operator 时间一起
   测量。

新实现仍应遵守仓库的 operator API 边界：Vector 算术使用 SIMT API，Cube 算术
使用 Tensor API，搬运使用允许的 C API/Tensor API；不得因为 vLLM-Ascend 上游
存在 Basic API 协议，就在新设计中引入新的 C++ Basic API 或跨核 flag 依赖。

## 15. 依据

CannBench：

- `src/cannbench/operators/builtin/dsa_decode/__init__.py`
- `src/cannbench/operators/builtin/lightning_indexer/data/realistic_decode.json`
- `src/cannbench/operators/builtin/lightning_indexer/materialize.py`
- `src/cannbench/operators/builtin/lightning_indexer/__init__.py`
- `src/cannbench/operators/builtin/sparse_attention/data/realistic_decode.json`
- `src/cannbench/operators/builtin/sparse_attention/materialize.py`
- `src/cannbench/operators/builtin/sparse_attention/simt/v2/aten_dsa_sparse_attention_v2/ops.py`
- `src/cannbench/operators/builtin/sparse_attention/external.py`
- `docs/optimization/dsa-v2-sparse-attention-vllm-ascend-analysis.zh-CN.md`
- `docs/optimization/dsa-v2-sparse-attention-vllm-rolling-pipeline-design.zh-CN.md`

vLLM-Ascend：

- commit `a5b0ce10d84bf76dd9c5c9e7ab9a5ddeed5af7ca`
- `csrc/attention/sparse_flash_attention/op_kernel/arch35/sparse_flash_attention_kernel_mla.h`
- `csrc/attention/sparse_flash_attention/op_kernel/arch35/sparse_flash_attention_service_cube_mla.h`
- `csrc/attention/sparse_flash_attention/op_kernel/arch35/sparse_flash_attention_service_vector_mla.h`
