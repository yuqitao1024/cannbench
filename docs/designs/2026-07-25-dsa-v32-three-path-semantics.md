# DeepSeek V3.2 DSA 三路径语义对齐说明

## 结论

CannBench 当前的 Ascend SIMT、vLLM-Ascend 和 CUDA 三条路径采用了相同的
DSA 两阶段抽象和主要 shape，但还不能认为对外输入输出语义已经完全一致。

- 三条路径都执行 `Lightning Indexer -> indices -> Sparse Attention`。
- 三条路径的目标是从同一套逻辑输入生成各自后端需要的物理布局；当前
  shared-KV 公开接口和跨后端 conformance 还没有完全收敛。
- 底层布局、存储精度和 kernel 数量可以不同，这些属于实现差异。
- 当前仍存在 shared-KV 公开合同、Indexer scaling/tie 行为和 Attention 附加
  输出不一致。
- 在这些差异修复前，性能测试只能作为初步数据，不能视为严格的同语义对标。

本文只讨论 DeepSeek V3.2。V4/V4 Pro、attention sink、SWA 以及完整 Attention
Layer 的其他前后处理不在当前对齐范围内。

## DSA 背景

普通 Attention 需要让每个 Query 与全部 `C` 个历史 Token 计算分数，计算量
近似为：

```text
O(B * Q * H * C * D)
```

DeepSeek Sparse Attention 使用一个较轻量的 Lightning Indexer，先从 `C` 个
KV 位置中选择 `TopK` 个候选位置，再只对这些位置执行正式 Attention：

```text
Lightning Indexer -> indices -> Sparse Attention
```

V3.2 当前保留的真实 case 使用以下主要参数：

```text
Lightning Indexer:
  index_heads G = 64
  index_dim   Di = 128
  top_k       S = 2048

Sparse Attention:
  query_heads H = 128
  qk_head_dim   = 576
  value_dim     = 512
  kv_heads      = 1 (shared-KV)

Decode:
  B=2, Q=2, C=32768

Prefill:
  B=1, Q=4096, C=32768
```

Indexer Query 与正式 Attention Query 不是同一个张量：

- `Q_index` 是 Indexer 使用的低成本索引 Query，逻辑维度为 `64 x 128`。
- `Q_attn` 是正式 Attention Query，逻辑维度为 `128 x 576`。
- 真实模型中二者来自同一层输入的不同投影。
- CannBench 当前分别生成 Indexer 和 Attention 输入，只将 `indices` 在设备侧串联。

因此，当前 workflow 测试的是 DSA 核心链路，而不是包含 projection、RoPE
生成、cache scatter 等操作的完整 Transformer Attention Layer。

## 目标逻辑接口

### Lightning Indexer

```text
输入:
  query   [B, Q, G, Di]  BF16
  keys    [B, C, Di]     BF16
  weights [B, Q, G]      BF16

计算:
  score[b,q,c] =
      sum_g(weights[b,q,g] * relu(dot(query[b,q,g], keys[b,c])))

输出:
  indices [B, Q, S]      INT32
```

`indices` 应按分数从高到低排列，并且只能包含当前 Query 可见的 KV 位置。

### Sparse Attention

```text
输入:
  query     [B, H, Q, 576] BF16
  shared_kv [B, 1, C, 576] BF16
  indices   [B, Q, S]      INT32

计算:
  keys = shared_kv
  values = shared_kv[..., :512]
  selected_kv = gather(keys, values, indices)
  scores = query @ selected_keys^T / sqrt(576)
  probabilities = softmax(scores)
  out = probabilities @ selected_values

输出:
  out [B, H, Q, 512] BF16
  lse [B, H, Q]      FP32
```

Workflow 由两个独立算子组成，只把 Indexer 产生的 `indices` 直接绑定给
Sparse Attention。当前目标不是把整个 workflow 合并为一个 kernel。

因为这是两个 kernel，`indices` 必须作为 device tensor 保存在设备全局内存中；
要求是它不能回到 host，也不能由第二个算子重新随机生成或重新物化。

## 三条实现路径

### Ascend SIMT

Lightning Indexer 是一个自定义 CV 融合 kernel：Cube 计算点积，结果从 L0C
搬运到 UB，SIMT VF 在 UB 上完成 ReLU、权重乘法、归约和 TopK。中间 logits
不写入 GM。

Sparse Attention 也是一个自定义 CV 融合 kernel：在一个 kernel 内完成选中
KV 的读取、Cube QK、UB 上的 Softmax/LSE，以及与 V 的乘加。两个组件之间只
通过 GM 中的最终 `indices` 连接。

该路径的公开逻辑输入为 BF16；具体的 BQHD/BHTD 布局由自定义算子 wrapper
约定。当前 wrapper 仍接收独立的 `keys` 和 `values`，尚未收敛到单一
`shared_kv` 参数。

### vLLM-Ascend

Lightning Indexer 调用 `torch_npu.npu_lightning_indexer`，使用 TND 布局、
累计序列长度和 `sparse_mode=3`。原生接口可能返回 indices 和 values，CannBench
当前只保留 indices。

V3.2 Sparse Attention 调用 `npu_sparse_flash_attention`。Adapter 将逻辑上的
576 维 Q/K 拆成 512 维 NoPE 部分和 64 维 RoPE 部分，并将 KV 转换为分页布局。

这些 TND、分页 KV 和 NoPE/RoPE 拆分属于物理接口适配，不应改变逻辑算子
语义。该路径目前最接近 V3.2 的 right-aligned causal 语义。

### CUDA

Lightning Indexer 先将统一 BF16 输入量化为 DeepGEMM 所需的 FP8 表示，然后
由 DeepGEMM MQA logits kernel 计算分数，再调用 `torch.topk` 产生 indices。
这一路径至少包含两个 launch，并且 FP32 logits 会经过 GM，因此目前不是
score 与 TopK 的单 kernel 融合实现。

Sparse Attention 的 prefill 使用 `flash_mla_sparse_fwd`，decode 使用
`flash_mla_with_kvcache`。Decode adapter 会把 KV 转换为 FlashMLA V3.2 使用的
FP8 cache 格式。FlashMLA 假定传入的 indices 已经满足有效范围，不再为稀疏
候选补做完整的因果过滤。

## 逻辑差异与物理差异

以下物理差异是允许的：

- SIMT 使用 Ascend Cube、UB 和 SIMT VF。
- vLLM-Ascend 使用 TND、分页 KV 和 CANN 原生算子布局。
- CUDA Indexer 和 Decode KV 内部使用 FP8，FlashMLA 使用自己的 cache 布局。
- 三条路径的 kernel 数量和融合程度不同。

这些差异不应改变对外张量代表的数学对象。Adapter 应负责布局和存储格式的
转换。

以下是当前需要修复的逻辑差异。

### 1. Sparse Attention 返回值

SIMT 约定返回 `(out, lse)`。FlashMLA 的不同入口还可能返回 `max_logits`
等统计量；vLLM-Ascend 原生算子返回 softmax 统计值，但当前 adapter 只保留
`out`。目标公开契约应统一为：

```text
(out: BF16 [B,H,Q,512], lse: FP32 [B,H,Q])
```

各 adapter 可丢弃其他后端私有统计量，或从原生统计量规范化出 LSE。

### 2. 精度与结果比较

虽然三条路径可以使用同一份 BF16 逻辑输入，CUDA 路径内部的 FP8 量化仍可能
改变临界位置的 TopK 排序。因此验证不能只要求 indices 逐项完全相同：

- 单独验证 Indexer 时，应比较 TopK recall、重叠率和选中分数误差。
- 单独验证 Sparse Attention 时，应给三条路径传入同一份合法 indices。
- 验证完整 workflow 时，应比较最终 output/LSE，并记录 Indexer recall。

## V3.2 最终契约

当前建议将 V3.2 固定为以下契约：

```text
Lightning Indexer:
  BF16 canonical inputs
  right-aligned causal valid length
  INT32 [B,Q,TopK] sorted indices

Sparse Attention:
  BF16 canonical query/shared_kv
  K = shared_kv
  V = shared_kv[..., :512]
  INT32 indices
  normalized (BF16 out, FP32 lse)

Workflow:
  indices 在设备侧从 Indexer 传给 Sparse Attention
  不包含 attention sink
  不包含 projection、RoPE 生成、cache scatter 和 SWA
```

在完成 shared-KV、Indexer scaling/tie 行为和返回值规范化之后，三条路径才可以
称为“相同逻辑输入、相同逻辑输出、不同硬件实现”，并用于严格的 V3.2 性能对标。

## V3.2 剩余工作

本节只记录尚未闭环的事项，不保留已完成事项或历史提交清单。

### 1. 收敛 shared-KV 公开合同

Materializer 和算子 ABI 仍暴露独立的 `keys` 和 `values`。需要收敛为单一
`shared_kv [B,1,C,576]` canonical 输入，由各后端内部使用完整 576 维作为 K，
并使用前 512 维作为 V。

### 2. 明确 Indexer scaling 和 tie 行为

明确 Indexer scaling，并定义跨后端同分数时的 tie 行为，或者明确允许比较
等价 Top-K 集合。

### 3. 统一 Sparse Attention 数学合同

- 将 softmax scale 写入 case/schema，并从 V3.2 模型合同验证其数值，而不是
  仅由 adapter 默认使用 `d_qk^-0.5`。
- 统一 invalid indices 和每行有效 `topk_length`。
- 统一自然对数 LSE 的定义和特殊值处理。
- 将三后端结果规范化为 `out [B,Q,H,Dv]` 和 `lse [B,Q,H]`；vLLM adapter
  不能继续只返回 `out`。

### 4. 增加真实序列元数据

Case schema 需要表达每条请求的 `query_len`、`context_len`、绝对 Query
position、`cu_seqlens` 和 block table，并支持 ragged batch。Prefill 不能默认
Query 从位置 0 开始。

### 5. 明确生产量化的计时边界

需要区分静态 KV cache packing/metadata 准备与每步动态 Q 量化。静态准备应放在
计时外；动态量化是否计时应以真实推理调用边界为准，并在三条路径中保持可比。

### 6. 建立 rank-local shape 合同

Case 需要记录 TP/DP/CP、local heads、local batch 和 KV shard。否则无法证明
vLLM 多卡路径与单卡 FlashMLA/SIMT 路径处理了相同的数据量。

### 7. 建立三后端 Conformance 门禁

- Indexer 在相同输入下比较 Top-K recall、集合重叠和分数误差。
- Sparse Attention 使用同一份合法 indices 比较 output 和自然对数 LSE。
- 完整 workflow 比较最终输出，同时记录 Indexer recall。
- 每个 case 通过 conformance 后才能进入正式性能对比。
