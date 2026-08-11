# DSA V2 Prefill Full-Flow Shape Analysis

## 1. Scope And Evidence

本文面向需要逐步核对 DeepSeek V3.2 prefill 矩阵流的 operator 开发者，只分析
canonical case：

```text
deepseek_v32_flashmla_prefill_q4096_ctx32768_top2048
```

文中的 case 参数来自 `lightning_indexer` 和 `sparse_attention` 的
`realistic_prefill` component case；workflow 绑定、逻辑 shape 和 device 状态来自
`dsa_prefill` plugin 及其 production shape trace。所有 shape 都是
`shape_scope=rank_local`、`TP=DP=CP=1`、shared-KV `replicated` 的单卡算子口径，
不能外推为未知 serving 并行配置下的 local shape。

本文只证明算法级 shape、收缩轴和 workflow 数据绑定，不声明 prefill 的 tile、core
或 launch 分解。

### Related shape resources

- [Decode full/split shape analysis](./dsa-v2-decode-full-and-split-shape-analysis.zh-CN.md)
- [Shape Explorer design](../designs/dsa-v32-shape-explorer/design.md)
- [Saved visual baseline / static mockup](../designs/dsa-v32-shape-explorer/shape-explorer-mockup.html)

## 2. Symbols And Canonical Case Parameters

| 符号 | 含义 | canonical 值 |
| --- | --- | ---: |
| `B` | batch size | 1 |
| `Q` | 每个 request 的 Query token 数 | 4096 |
| `R=B*Q` | 展平后的 Query row 数 | 4096 |
| `C` | context token 数 | 32768 |
| `Hi` | Indexer head 数 | 64 |
| `Di` | Indexer feature dim | 128 |
| `H` | Attention Query head 数 | 128 |
| `Hkv` | shared KV head 数 | 1 |
| `S` | TopK/selected token 数 | 2048 |
| `Dqk` | Q/K head dim | 576 |
| `Dv` | V/output head dim | 512 |
| `Dr=Dqk-Dv` | shared-KV 尾部维度 | 64 |

由 component case 直接得到：

```text
causal                 = true
query_lens             = [4096]
context_lens           = [32768]
query_start_positions  = [28672]
page_block_size        = 64
blocks_per_request     = ceil(32768 / 64) = 512
block_tables           [B,512] = [1,512]
score_scale            = 1.0
softmax_scale          = 0.041666666666666664 = 1 / sqrt(576) = 1 / 24
topk_lengths           [B,Q] = [1,4096]，每项均为 2048
```

`cu_seqlens_q=[0,4096]`，`cu_seqlens_kv=[0,32768]`。由于 `B=1`，
`block_tables` 的唯一一行是 page `0..511`。

## 3. Workflow Overview

`dsa_prefill` 是独立 workflow plugin，按以下顺序展开：

```text
Lightning Indexer
  输入: index_query, index_key, weights, sequence metadata
  输出: indices [1,4096,2048]
                    |
                    | workflow 唯一跨 component 绑定
                    v
Sparse Attention
  输入: query, shared_kv, indices, sequence metadata
  输出: out [1,4096,128,512], lse [1,4096,128]
```

Indexer 的 `index_query[...,128]` 和 `index_key[...,128]` 属于 retrieval feature
space；Attention 的 `query[...,576]` 和 `shared_kv[...,576]` 属于 QK feature
space。workflow 只传 `indices`，不传 Indexer score，也不把 128 维 Indexer 特征
解释成 576 维 Attention 特征。

production shape trace 用相同的八个算法 stage 表达该流程：Index inputs、Indexer
matmul、head reduction、TopK、shared-KV gather、QK、Softmax/LSE、PV/output。

## 4. Lightning Indexer Inputs

canonical materializer 的输入和输出布局如下：

| 张量 | 物理/contract shape | 作用 |
| --- | --- | --- |
| `index_query` | `[B,Q,Hi,Di] = [1,4096,64,128]` | 每个 Query row 的 64 个 retrieval head |
| `index_key` | `[B,C,Di] = [1,32768,128]` | 当前 request 的 context retrieval key |
| `weights` | `[B,Q,Hi] = [1,4096,64]` | 每个 Query row 的 head 权重 |
| `valid_context_lengths` | `[B,Q] = [1,4096]` | 每行 causal 上界 |
| `indices` | `[B,Q,S] = [1,4096,2048]` | TopK 输出及 workflow 绑定 |

固定一个展平 Query row `r=(b,q)` 时：

```text
Q_idx = index_query[b,q,:,:]  [Hi,Di] = [64,128]
K_idx = index_key[b,:,:]      [C,Di]  = [32768,128]
w     = weights[b,q,:]        [Hi]    = [64]
```

`B=1`，所以 4096 个 row 都读取同一个 request 的 `index_key`；这表示共享输入，
不表示把 key 复制 4096 份。

## 5. Per-Query Indexer Matrix Calculation

对固定的 `(b,q)`，先转置 `K_idx` 并沿 `Di=128` 做点积：

```text
head_scores = Q_idx @ K_idx^T

[64,128] x [128,32768] -> head_scores [64,32768]
```

然后逐元素执行 ReLU、广播乘以 `w[h]`，再沿 `Hi=64` 求和：

```text
activated[h,c]    = ReLU(head_scores[h,c])
weighted[h,c]     = activated[h,c] * w[h]
index_scores[c]   = sum(h=0..63, weighted[h,c]) * score_scale

[64,32768] -> index_scores [32768]
score_scale = 1.0
```

对 `c >= valid_length(q)` 的位置填 `-inf`，之后才执行 TopK。单个 row 的点积
MAC 数可直接核对为：

```text
Hi * C * Di = 64 * 32768 * 128 = 268,435,456 MAC
```

## 6. Aggregate R=4096 Indexer Shapes

把 `(B,Q)` 展平为 `R=B*Q=1*4096=4096` 后，production trace 中的 aggregate
shape 是：

| 张量 | aggregate shape | 状态 |
| --- | --- | --- |
| `index_query_all` | `[R,Hi,Di] = [4096,64,128]` | 输入 view |
| `head_scores_all` | `[R,Hi,C] = [4096,64,32768]` | `logical_only=True` |
| `index_scores_all` | `[R,C] = [4096,32768]` | `logical_only=True` |
| `indices_all` | `[R,S] = [4096,2048]` | workflow 输出 |

每个 row 独立执行第 5 节的矩阵计算，因此总 MAC 数为：

```text
R * Hi * C * Di
= 4096 * 64 * 32768 * 128
= 1,099,511,627,776 MAC
```

`head_scores_all` 和 `index_scores_all` 用于核对完整数学域；production trace 将
二者标为 `logical_only=True`，不要求在 global memory 中一次性物化。

## 7. TopK And Workflow Binding

每个 row 的 TopK 输入和输出严格为：

```text
[64,32768] -> index_scores [32768] -> TopK -> indices [2048]
```

更完整地写：

```text
indices[b,q,:] = TopKIndices(masked_index_scores, k=2048)
[32768] -> [2048]
```

TopK 取最大 score 并按 score 降序返回。case 的 `tie_policy` 是
`equivalent_score_set`，相同 score 只要求输出等价候选集合。最短 causal 有效长度
是 28673，大于 `S=2048`，所以 4096 个有效 Query row 都能选择完整的 2048 个位置。

`lightning_indexer` step 声明 `produces=("indices",)`；`sparse_attention` step
声明 `consumes=("indices",)`。因此 `[1,4096,2048]` 的 `indices` 是两个
component 之间唯一的 workflow 数据绑定。

## 8. Sparse Attention Inputs And Shared KV

Sparse Attention 的 canonical contract 布局如下：

| 张量 | 物理/contract shape | 作用 |
| --- | --- | --- |
| `query` | `[B,H,Q,Dqk] = [1,128,4096,576]` | BHTD Query |
| `shared_kv` | `[B,Hkv,C,Dqk] = [1,1,32768,576]` | 单 shared KV head |
| `indices` | `[B,Q,S] = [1,4096,2048]` | Indexer 绑定结果 |
| `topk_lengths` | `[B,Q] = [1,4096]` | 每项均为 2048 |
| `out` | `[B,Q,H,Dv] = [1,4096,128,512]` | BTHD 输出 |
| `lse` | `[B,Q,H] = [1,4096,128]` | 每个 Query head 的 log-sum-exp |

`Hkv=1` 表示同一 `(b,q)` 的 128 个 Query heads 共享同一组 selected indices、K
和 V。shared-KV 最后一维的语义是：

```text
K = shared_kv[..., 0:576]       [576]
V = shared_kv[..., 0:512]       [512]
Dr = 576 - 512 = 64
```

所以 V 是 shared-KV 前 512 维的逻辑 view；数学 contract 不要求另复制一份完整
Value buffer。

## 9. Per-Query QK, Softmax/LSE, And PV

固定一个 `(b,q)`，先由同一组 `indices[b,q,:]` gather：

```text
Q          = query[b,:,q,:]               [H,Dqk] = [128,576]
I          = indices[b,q,:]               [S]     = [2048]
K_selected = shared_kv[b,0,I,:]                     [2048,576]
V_selected = shared_kv[b,0,I,0:512]                 [2048,512]
```

QK 沿 `Dqk=576` 收缩，并乘 `1/24`：

```text
scores = (1 / 24) * Q @ K_selected^T

[128,576] x [576,2048] -> scores [128,2048]
```

非法 index、`topk_lengths` 和 causal 位置先被 mask；随后沿 `S=2048` 做
Softmax 和 log-sum-exp：

```text
scores [128,2048] -> P [128,2048], LSE [128]
```

最后沿同一个 `S` 收缩：

```text
[128,2048] x [2048,512] -> output [128,512]
```

单个 Query row 的 QK 和 PV 分别是
`128*2048*576=150,994,944 MAC` 与
`128*2048*512=134,217,728 MAC`。

## 10. Aggregate R=4096 Attention Shapes

为与 Indexer 统一比较，把 contract `query [B,H,Q,Dqk]` 逻辑换轴并展平为
`[R,H,Dqk]`。production trace 的 aggregate shape 是：

| 张量 | aggregate shape | 状态 |
| --- | --- | --- |
| `query_all` | `[R,H,Dqk] = [4096,128,576]` | 输入 view |
| `selected_k_all` | `[R,S,Dqk] = [4096,2048,576]` | `logical_only=True` |
| `selected_v_all` | `[R,S,Dv] = [4096,2048,512]` | `logical_only=True` |
| `scores_all` | `[R,H,S] = [4096,128,2048]` | `logical_only=True` |
| `output_all` | `[R,H,Dv] = [4096,128,512]` | 输出 view |
| `lse_all` | `[R,H] = [4096,128]` | 输出 view |

Softmax stage 复用同一个 `scores_all` aggregate trace tensor 表达 probability
view，不创建独立的 probability aggregate tensor ID。下文公式中的 `P` 是数学结果
名称，不是另一个 production trace ID。

对应的 batched 数学关系为：

```text
QK:
[4096,128,576] x [4096,576,2048]
  -> scores [4096,128,2048]

Softmax/LSE:
scores [4096,128,2048]
  -> P [4096,128,2048], LSE [4096,128]

PV:
[4096,128,2048] x [4096,2048,512]
  -> output [4096,128,512]
```

这里的第一个 `4096` 是相互独立的 batch 维，不参与矩阵收缩。总计算量可核对为：

```text
QK = R * H * S * Dqk
   = 4096 * 128 * 2048 * 576
   = 618,475,290,624 MAC

PV = R * H * S * Dv
   = 4096 * 128 * 2048 * 512
   = 549,755,813,888 MAC
```

## 11. Causal Valid-Length Evolution

Query chunk 右对齐在长度 32768 的 context 尾部，起点是
`32768-4096=28672`。对 `q in [0,4095]`，Query 的绝对位置和允许读取的长度为：

```text
position(q)     = 28672 + q
valid_length(q) = min(32768, 28672 + q + 1)
```

因此：

```text
q=0:     position=28672, valid_length=28673
q=1:     position=28673, valid_length=28674
...
q=4094:  position=32766, valid_length=32767
q=4095:  position=32767, valid_length=32768
```

`valid_length(q)` 恰好从 28673 单调增加到 32768，共 4096 个值。Indexer 将
`context_position >= valid_length(q)` 的 score 填为 `-inf`，所以 TopK 结果必须
满足 `0 <= index < valid_length(q)`。Sparse Attention 还会把
`index > position(q)` 的项 mask 掉；因为 `position(q)=valid_length(q)-1`，两者
是相同 causal 边界的两种写法。

## 12. Logical Views Versus Materialized Buffers

必须区分算法 shape 与 buffer contract：

| 类别 | 张量 | 结论 |
| --- | --- | --- |
| component 输入 | `index_query`、`index_key`、`weights`、`query`、`shared_kv` | materializer 按第 4、8 节 contract shape 创建 |
| workflow 绑定 | `indices [1,4096,2048]` | Indexer 输出并作为 Attention 输入 |
| component 输出 | `out [1,4096,128,512]`、`lse [1,4096,128]` | 对外返回的物理结果 |
| production trace 逻辑中间量 | 下列五个 aggregate tensor ID | `logical_only=True`，不是 GM 全量物化要求 |
| 非 trace aggregate ID 的 contract view | `V=shared_kv[...,0:512]`、展平后的 `R` 视图 | 不凭 shape 推断额外复制 |

production trace 中 `logical_only=True` 的 aggregate tensor ID 恰好是：

```text
head_scores_all
index_scores_all
selected_k_all
selected_v_all
scores_all
```

其中 `scores_all` 在 QK stage 表示 score，在 Softmax stage 复用为 probability
view；production trace 不定义独立的 probability aggregate tensor ID。Torch 参考
公式可能通过 head broadcast、gather 或 layout 变换表达这些中间量；这只说明参考
计算的张量语义。Shape Explorer 的 `logical_only=True` 标记不能反过来证明某个未
优化 prefill device kernel 的真实缓存、workspace 或搬运策略。

作为数量级校验，若把 `head_scores_all` 全量物化，它包含
`4096*64*32768=8,589,934,592` 个元素；若把 `selected_k_all` 全量物化，它包含
`4096*2048*576=4,831,838,208` 个元素。这些数字说明为什么文档必须把完整数学域
与实际 buffer 分开，而不是建议这样的物化方案。

## 13. Device Decomposition Boundary

Prefill has not yet been optimized. 因此本文有意不记录 prefill 的 device tile、
core allocation、task count、context shard、launch geometry、local tensor 或
pipeline 数据；production trace 的 `device_execution.status` 是 `unavailable`，
`version` 为空，`kernels` 为空。

production Shape Explorer 的 Device execution view 逐字渲染 operator payload
message：

```text
Prefill is not optimized yet. This view intentionally shows only the
algorithm-level matrix flow.
```

尤其不得从 [current decode shape information](./dsa-v2-decode-full-and-split-shape-analysis.zh-CN.md)
推断或复用 decode v2 的 Head64、selected128、AIC/AIV 数、context shard 或 tile
布局。prefill 和 decode 的逐 row 数学公式相同，不代表它们具有相同的 device
decomposition。

## 14. End-To-End Shape Summary

| 流程 | 单个 Query row | 全部 `R=4096` row | contract/逻辑状态 |
| --- | --- | --- | --- |
| Indexer 输入 | `Q_idx [64,128]`、`K_idx [32768,128]` | `index_query [4096,64,128]`，key 由同一 request 共享 | 输入 |
| Indexer matmul | `[64,128]x[128,32768] -> [64,32768]` | `head_scores_all [4096,64,32768]` | `logical_only=True` |
| Head reduction | `[64,32768] -> [32768]` | `index_scores_all [4096,32768]` | `logical_only=True` |
| Causal TopK | `[32768] -> [2048]` | `indices [4096,2048]` | workflow 绑定 |
| Shared-KV gather | `K [2048,576]`、`V [2048,512]` | `selected_k_all [4096,2048,576]`、`selected_v_all [4096,2048,512]` | `logical_only=True` |
| QK | `[128,576]x[576,2048] -> [128,2048]` | `scores_all [4096,128,2048]` | `logical_only=True` |
| Softmax/LSE | `[128,2048] -> [128,2048], [128]` | `scores_all` 复用为 probability view；`lse_all [4096,128]` | `scores_all`: `logical_only=True`；LSE 输出 |
| PV/output | `[128,2048]x[2048,512] -> [128,512]` | `output [4096,128,512]` | 输出 |

完整 contract 路径可压缩为：

```text
Indexer:
  index_query [1,4096,64,128] + index_key [1,32768,128]
    -> indices [1,4096,2048]

Sparse Attention:
  query [1,128,4096,576]
    + shared_kv [1,1,32768,576]
    + indices [1,4096,2048]
    -> out [1,4096,128,512] + lse [1,4096,128]
```

## 15. Evidence

Production implementation and cases：

- [`dsa_prefill/shape_trace.py`](../../src/cannbench/operators/builtin/dsa_prefill/shape_trace.py)
- [`dsa_prefill/__init__.py`](../../src/cannbench/operators/builtin/dsa_prefill/__init__.py)
- [`lightning_indexer/realistic_prefill.json`](../../src/cannbench/operators/builtin/lightning_indexer/data/realistic_prefill.json)
- [`lightning_indexer/materialize.py`](../../src/cannbench/operators/builtin/lightning_indexer/materialize.py)
- [`lightning_indexer/__init__.py`](../../src/cannbench/operators/builtin/lightning_indexer/__init__.py)
- [`sparse_attention/realistic_prefill.json`](../../src/cannbench/operators/builtin/sparse_attention/data/realistic_prefill.json)
- [`sparse_attention/materialize.py`](../../src/cannbench/operators/builtin/sparse_attention/materialize.py)
- [`sparse_attention/__init__.py`](../../src/cannbench/operators/builtin/sparse_attention/__init__.py)
- [`dsa_prefill/test/test_shape_trace.py`](../../src/cannbench/operators/builtin/dsa_prefill/test/test_shape_trace.py)
- [`DeviceExecutionView.tsx`](../../web/src/components/DeviceExecutionView.tsx)
- [DSA realistic case sources](../datasets/dsa-real-sources.md#deepseek-v32-flashmla-prefill)
