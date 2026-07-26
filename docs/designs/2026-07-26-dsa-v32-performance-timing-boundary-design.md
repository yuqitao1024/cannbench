# DeepSeek V3.2 DSA 三后端性能计时边界设计

## 目标

本文定义 Ascend SIMT、vLLM-Ascend 和 CUDA 三条 DeepSeek V3.2 DSA 路径的
统一性能计时边界。目标不是让三端执行相同的物理 kernel，而是让三端从相同的
canonical BF16 逻辑输入出发，统计各自完成同一 DSA 核心任务所必需的动态设备
工作。

DSA workflow 保持两个顺序阶段：

```text
Lightning Indexer -> device indices -> Sparse Attention
```

最终 workflow latency 定义为两个阶段计时段的设备时间之和。两个阶段之间存在
数据依赖，不能重叠执行。

## 统一原则

### 计时内

- 每个推理 step 都必须重新执行的 Query lowering。
- CUDA Indexer 的 BF16 `Q_index` 到 FP8 的动态 cast。当前 DeepGEMM legacy
  MQA 接口的 Q 不携带显式 scale；KV 的静态量化仍包含 scale 生成。
- 每个推理 step 的动态 Indexer weights lowering；CUDA 的 BF16 到 FP32 转换
  也在计时内。
- vLLM-Ascend 对动态 Query 执行的 TND packing、NoPE/RoPE split 或 contiguous
  copy；纯 view/reshape 没有设备耗时。
- Lightning Indexer 的 dot、ReLU、head weight、reduce 和 Top-K。
- Indexer 最终 indices 写入设备内存，以及 Sparse Attention 对该 tensor 的读取。
- 由本步 indices 派生后端 cache indices 的动态设备转换。
- Sparse Attention 的 gather、QK、Softmax/LSE 和 PV。

### 计时外

- case 解析、随机或确定性输入生成。
- Host 到 Device 输入上传和输出 Device 到 Host 回读。
- Tensor 分配以及 profiler 初始化。
- Index KV cache 和 shared-KV cache 的 blocking、TND/分页布局、NoPE/RoPE
  split、FP8 packing 及静态 scale 生成。
- block table、`cu_seqlens`、context length 和固定调度 metadata 的构造。
- warmup、精度参考计算和结果比较。
- Python/C++ Host launch 开销；正式指标保持为 profiler 观测到的设备时间。

静态 cache 准备必须每个 case 只做一次，不得留在被重复调用的 operator closure
中。动态 Query 和 indices 相关工作必须保留在 closure 内。

## 统一时间轴

图例：`OUT` 表示计时外，`IN` 表示计时内，箭头表示设备侧数据依赖。

```text
时间 -------------------------------------------------------------------------->

公共输入
  OUT [case/materialize] [H2D] [tensor alloc] [static metadata] [warmup]
   IN                                      | Indexer | -> indices -> | Attention |
  OUT                                                                  [D2H/verify]

Ascend SIMT
  OUT [BF16 Index-KV ready] [BF16 shared-KV ready]
   IN                          [Q_index BF16 fused CV Indexer]
                                  -> [INT32 indices in GM]
                                  -> [Q_attn BF16 fused CV Sparse Attention]

vLLM-Ascend
  OUT [Index-KV TND pack] [shared-KV TND + NoPE/RoPE pack]
   IN                          [dynamic Q_index/weights TND pack]
                                  -> [npu_lightning_indexer + Top-K]
                                  -> [dynamic Q_attn TND + NoPE/RoPE lowering]
                                  -> [npu_sparse_flash_attention]

CUDA
  OUT [Index-KV FP8 cache + scale] [FlashMLA KV cache FP8/BF16 pack]
   IN                          [dynamic Q_index BF16 -> FP8 cast]
                                  -> [dynamic weights BF16 -> FP32]
                                  -> [DeepGEMM logits]
                                  -> [torch.topk]
                                  -> [dynamic indices -> cache indices]
                                  -> [dynamic Q_attn lowering]
                                  -> [FlashMLA Sparse Attention]
```

`indices` 不经过 Host。GM 或 HBM 中的最终 indices 是两个独立算子之间必要的
设备接口，其写入和读取成本分别包含在对应 kernel 的设备时间中，不另加人工
拷贝时间。

## 分阶段边界

### Lightning Indexer

统一逻辑输入：

```text
Q_index BF16 [B,Q,64,128]
K_index BF16 [B,C,128]
weights BF16 [B,Q,64]
sequence metadata
```

静态 `K_index` 可以在计时前转换成后端生产格式。动态 `Q_index` 必须从 BF16
canonical tensor 开始计时；某后端不需要转换时没有额外成本，需要 FP8 或 TND
lowering 时按实际设备执行计入。weights 同样是每步动态输入，其布局转换和 CUDA
BF16 到 FP32 转换也必须计入。

Indexer latency 包含 Top-K。CUDA 的 DeepGEMM 只输出 logits，因此
`torch.topk` 是 CUDA Indexer 合同的一部分，不能从 profile 中排除。

### Sparse Attention

统一逻辑输入：

```text
Q_attn BF16 [B,H,Q,576]
shared_kv BF16 [B,1,C,576]
indices INT32 [B,Q,TopK]
sequence metadata
```

`shared_kv` 的 TND/分页布局、分块、FP8 cache 和静态 scale 都在计时前准备。`Q_attn`
仍从 canonical BF16 tensor 开始计时；动态 TND packing、NoPE/RoPE split 或
contiguous copy 计入对应后端。CUDA decode 根据本步 Top-K indices 生成
FlashMLA cache indices 的转换也属于动态 Attention 路径。

Sparse Attention latency 包含 output 和自然对数 LSE 的生产路径，但不包含
输出回读和精度比较。

### Workflow

```text
workflow_device_latency = indexer_device_latency + attention_device_latency
```

该指标使用同一 case、seed 和 canonical inputs 的两个组件 profile。不能把
Sparse Attention 为准备 bound input 而额外执行的一次 Indexer 重复计入；该次
执行只负责建立设备输入依赖，应发生在测量范围外或由已测 Indexer 输出复用。

## 后端落地要求

### Ascend SIMT

- 保持两个独立融合 kernel 的目标 kernel 计时。
- 输入 tensor、metadata 和静态 cache 在 kernel launch 前准备。
- profile 只汇总每个组件自己的融合 kernel。

### vLLM-Ascend

- 静态 Index-KV、shared-KV TND 布局在 callable 构造阶段准备一次。
- 动态 Query lowering 移入被测 callable；若 lowering 只是 view，不产生额外
  device latency。
- profile 汇总动态 lowering kernel 和目标 CANN 算子，但不包含静态 cache
  packing。
- 当前统一 output/LSE 合同使用一次 TND KV 调用；CANN 不支持
  `PA_BSND + return_softmax_lse=True`，不得用 paged output 和 TND LSE 两次
  Attention 调用拼接性能结果。

### CUDA

- 将 `_blocked_kv_cache`、KV FP8 packing、静态 KV scale 和固定 schedule metadata
  移出重复调用路径。
- 保留每步 Q FP8 量化、DeepGEMM、Top-K、动态 index lowering 和 FlashMLA 在
  被测路径内。
- NCU 必须覆盖该组件完整动态 kernel 序列。远程执行必须遵守 operator plugin
  声明的 kernel selection 与 launch count，不能固定只抓第一个 launch。
- CUDA 两个组件各自用 operator-local NVTX range 包住动态 closure。NCU 通过
  plugin 声明的 range 采集范围内全部 kernel，不再用固定 launch-count 截断；
  静态 cache preparation 位于 range 外。

## 框架边界

实现优先通过 `lightning_indexer` 和 `sparse_attention` operator package 中的
callable 构造、adapter 和 profile selection 完成。不得在 CLI、backend 或 core
中增加具体 DSA/operator 名称分支。

如果远程 profiler 需要修正 launch-count 传播，只能实现为对所有 operator plugin
生效的通用机制：读取现有 `ProfileKernelSelection`，不能识别具体算子名称。

公开 benchmark record 继续保留 `metrics.latency_ms` 合同。workflow record 仍由
两个 component latency 相加生成，不修改已发布数据 schema。

## 验证标准

1. 单元测试证明静态 KV/cache preparation 每个 callable 只执行一次，重复调用只
   执行动态路径。
2. 单元测试证明 CUDA Indexer profile 同时覆盖 Q 量化、DeepGEMM 和 Top-K。
3. 单元测试证明 workflow 聚合不会重复统计用于 bound input 的 Indexer。
4. 定向搜索确认公共 CLI/core/backend 没有新增具体 DSA 算子分支。
5. Ascend 远端至少选择一个 V3.2 prefill 和一个 decode case，检查 profile 中只
   包含预期动态路径，并确认精度不回退。
6. CUDA 节点恢复后，对相同 case 执行同样的 kernel trace 审计；在此之前不能
   宣称 CUDA 计时边界已实机闭环。

## 完成定义

- 三端都从 canonical BF16 Query 进入计时段。
- 所有静态 KV/cache 转换都在计时外且每 case 只执行一次。
- 所有动态 Query lowering、Indexer Top-K 和 Attention 动态 index lowering 都在
  计时内。
- component latency 与 workflow latency 使用一致、可解释的设备时间合同。
- 当前 published data schema 和 plugin architecture 保持不变。
