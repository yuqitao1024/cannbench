# DeepSeek V4/V4 Pro DSA 对齐待办

## 文档范围

本文归档 DeepSeek V4-Flash 和 V4-Pro 的 DSA 对齐事项。它们不属于当前
V3.2 三路径对齐范围，不能把当前简化的单 sparse-KV workflow 当作完整的
V4/V4 Pro Attention 实现。

本文只记录尚未完成的工作，不保留已完成事项或历史 case 清理记录。

## 剩余工作

### 1. V4 Attention 数据模型

当前 Sparse Attention 只有一组稀疏 K/V，仍是简化模型。完整 V4 路径需要
明确表达并实现：

- original/SWA KV；
- compressed KV；
- `cmp_ratio`；
- 两部分 logits/softmax 的联合归一化；
- 两部分输出的合并规则。

在这部分完成前，V4 case 只能用于现有简化算子的 shape 压测，不能声称等价于
真实 V4 Attention。

### 2. Attention Sink

V4 路径需要确认并接入模型实际使用的 attention sink：包括 sink 的输入 shape、
参与 softmax 的位置、是否按 head 独立，以及 LSE 中是否包含 sink。当前实现没有
attention sink。该需求不得回灌到 V3.2 合同。

### 3. Shared-KV 与量化格式

V4/V4-Pro 不能直接复用 V3.2 的 `K=shared_kv, V=shared_kv[...,:512]` 结论。
需要先按官方实现确定 original KV、compressed KV 和 value 的真实关系，再定义
canonical 输入。

后端生产格式也尚未对齐：

- DeepGEMM Indexer 的 FP8 输入；
- FlashMLA 对应 V4 配置的 KV cache 格式；
- Atlas A5 production quantized indexer/shared-KV；
- 静态 KV packing 与动态 Q 量化各自是否计入单步延迟。

若三条路径都强制使用纯 BF16，只能作为算法基线，不能代表 DeepSeek.io 或
Atlas A5 的真实生产路径。

### 4. 真实序列元数据

当前 case 主要记录总 batch、Query 数和 context 上限，尚未记录：

- 每条请求的 `query_len` 和 `context_len`；
- Query 的绝对位置；
- `cu_seqlens` 和 block table 的真实内容；
- MTP Token 的逐行因果关系；
- ragged batch。

Prefill 不能默认 Query 从位置 0 开始，Decode 的多 MTP Token 也不能共享同一个
完整 context length。

### 5. Rank-local Shape

V4 的现有 case 来自模型配置、算子模板或多卡部署参数的推导，尚未显式记录
TP/DP/CP 配置、每 rank 的 local heads、local batch 和 KV shard。特别是
V4-Pro 的 `batch=60, context=131072` 不能在缺少并行配置时直接视为每张卡的
实际数据量。

在 rank-local 合同补齐前，vLLM 多卡结果不能与单卡 FlashMLA/SIMT 直接做
等数据量性能比较。

### 6. 数学与输出合同

V4 仍需统一以下语义：

- Indexer scaling、mask、有效长度、排序和 tie-breaking；
- Sparse Attention softmax scale；
- invalid indices 和每行 `topk_length`；
- causal/query position；
- 包含 sink 和双 KV 路径后的自然对数 LSE；
- 统一的 `[B,Q,H,Dv]` 输出布局。

### 7. 三后端 Conformance

V4/V4-Pro 尚无同 input 的 SIMT、vLLM-Ascend、CUDA 三方 conformance。正式
性能测试前至少需要：

1. Indexer 比较 Top-K set/recall，而不是量化路径下要求 indices 完全相同。
2. Sparse Attention 在相同合法 indices、序列元数据和 sink 下比较 output/LSE。
3. 完整 workflow 比较最终输出，并记录 Indexer recall。
4. 明确量化、cache packing、metadata 和 Top-K 的计时边界。

## 后续入口条件

V4 工作恢复时，应先从官方模型与推理实现冻结完整数学合同，再修改 case 和算子。
不能仅凭 `512/512` shape 直接复用当前 V3.2 风格的单 sparse-KV workflow。
