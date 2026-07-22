# DSA 真实场景 Case 来源

本文档记录 realistic DSA case 所使用的公开来源、固定版本以及具体推导过程。
只有同时满足以下条件的 case 才会被收录：模型维度来自官方模型配置，运行时
维度来自官方算子测试或公开的 vLLM-Ascend 端到端测试/部署指南。

## 固定来源

- DeepSeek-V3.2 模型配置，版本
  `a7e62ac04ecb2c0a54d736dc46601c5606cf10a6`：
  <https://huggingface.co/deepseek-ai/DeepSeek-V3.2/blob/a7e62ac04ecb2c0a54d736dc46601c5606cf10a6/config.json>
- DeepSeek-V4-Flash 模型配置，版本
  `60d8d70770c6776ff598c94bb586a859a38244f1`：
  <https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/60d8d70770c6776ff598c94bb586a859a38244f1/config.json>
- DeepSeek-V4-Pro 模型配置，版本
  `b5968e9190ef611bbf34a7229255be88a0e937c1`：
  <https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/b5968e9190ef611bbf34a7229255be88a0e937c1/config.json>
- GLM-5.2 模型配置，版本
  `b4734de4facf877f85769a911abafc5283eab3d9`：
  <https://huggingface.co/zai-org/GLM-5.2/blob/b4734de4facf877f85769a911abafc5283eab3d9/config.json>
- FlashMLA 稀疏 decode 和 prefill 测试，版本
  `9241ae3ef9bac614dd25e45e507e089f888280e0`：
  <https://github.com/deepseek-ai/FlashMLA/blob/9241ae3ef9bac614dd25e45e507e089f888280e0/tests/test_flash_mla_sparse_decoding.py>
  以及
  <https://github.com/deepseek-ai/FlashMLA/blob/9241ae3ef9bac614dd25e45e507e089f888280e0/tests/test_flash_mla_sparse_prefill.py>
- vLLM-Ascend 端到端测试，版本
  `b269feeed211b1de089c9ad23f8b1a94ed981c58`：
  <https://github.com/vllm-project/vllm-ascend/blob/b269feeed211b1de089c9ad23f8b1a94ed981c58/tests/e2e/weekly/multi_node/external_dp/config/DeepSeek-V4-flash-w8a8-PD.yaml>
  以及
  <https://github.com/vllm-project/vllm-ascend/blob/b269feeed211b1de089c9ad23f8b1a94ed981c58/tests/e2e/nightly/multi_node/internal_dp/config/GLM5_2-W8A8-A3-dual-nodes.yaml>
- vLLM-Ascend DeepSeek-V4-Pro 官方部署指南，版本
  `e4c88fb0b070c0c0100ce6fecb5f84b05a4afc03`：
  <https://github.com/vllm-project/vllm-ascend/blob/e4c88fb0b070c0c0100ce6fecb5f84b05a4afc03/docs/source/tutorials/models/DeepSeek-V4-Pro.md>

<a id="deepseek-v32-flashmla-decode"></a>

## DeepSeek V3.2 FlashMLA Decode

Case：`deepseek_v32_flashmla_decode_b2_q2_ctx32768_top2048`。

官方模型配置提供 `index_n_heads=64`、`index_head_dim=128` 和
`index_topk=2048`。FlashMLA 将其 production decode 模板标记为 V3.2，并
给出 `h_q=128`、`s_q=2`、`s_k=32768`、`topk=2048`、`d_qk=576`，
测试 batch size 中包含 `2`。CannBench 当前使用统一的 sparse-attention
`head_dim` 字段，因此将 FlashMLA 的 `d_qk` 映射到该字段。

<a id="deepseek-v32-flashmla-prefill"></a>

## DeepSeek V3.2 FlashMLA Prefill

Case：`deepseek_v32_flashmla_prefill_q4096_ctx32768_top2048`。

模型维度来自同一份官方配置。FlashMLA 的 V3.2 performance 模板提供
`d_qk=576`、`h_q=128`、`topk=2048`、`s_q=4096`，测试的 KV 长度中
包含 `32768`。

<a id="deepseek-v4-flash-vllm-decode"></a>

## DeepSeek V4 Flash vLLM Decode

Case：`deepseek_v4_flash_vllm_decode_b16_q1_ctx32768_top512`。

官方 V4-Flash 配置提供 `index_n_heads=64`、`index_head_dim=128`、
`index_topk=512`、`num_attention_heads=64`、`num_key_value_heads=1` 和
`head_dim=512`。vLLM-Ascend performance 测试使用真实的
`GSM8K_prefix0_in32768` 数据集，`batch_size=16`。单个 query token 表示
常规 decode 步骤；同一部署还启用了一个 MTP 推测 token。该 case 是从真实
端到端 workload 推导出的算子级形状，并不是直接捕获的底层 kernel tensor。

<a id="deepseek-v4-flash-flashmla-prefill"></a>

## DeepSeek V4 Flash FlashMLA Prefill

Case：`deepseek_v4_flash_flashmla_prefill_q4096_ctx32768_top512`。

模型维度来自官方 V4-Flash 配置。FlashMLA 的 `MODEL1 CONFIG1`
performance 模板独立提供了与之匹配的 `d_qk=512`、`h_q=64`、
`topk=512`、`s_q=4096` 和 `s_kv=32768` 算子形状。

<a id="deepseek-v4-pro-vllm-decode"></a>

## DeepSeek V4 Pro vLLM-Ascend Decode

Case：`deepseek_v4_pro_vllm_decode_b60_q1_ctx131072_top1024`。

官方 V4-Pro 配置提供 `index_n_heads=64`、`index_head_dim=128`、
`index_topk=1024`、`num_attention_heads=128`、`num_key_value_heads=1` 和
`head_dim=512`。vLLM-Ascend 官方 PD 部署指南的 A3 decode 节点配置给出
`max-model-len=131072`、`max-num-seqs=60` 和
`max-num-batched-tokens=120`。CannBench 将其映射为 `batch=60`、
`query_tokens=1` 的常规 decode 步骤；同一配置里也启用了
`num_speculative_tokens=1`，但为了与现有 decode realistic case 的口径一致，
不将该 MTP 推测 token 折算进 `query_tokens`。

<a id="deepseek-v4-pro-vllm-prefill"></a>

## DeepSeek V4 Pro vLLM-Ascend Prefill

Case：`deepseek_v4_pro_vllm_prefill_q4096_ctx131072_top1024`。

模型维度来自同一份官方 V4-Pro 配置。vLLM-Ascend 官方 PD 部署指南的 A3
prefill 节点配置给出 `max-model-len=131072`、`max-num-batched-tokens=4096`
和 `max-num-seqs=16`。CannBench 将其映射为单条序列的算子级 prefill chunk，
因此记录为 `batch=1`、`query_tokens=4096`、`context_tokens=131072`、
`topk=1024` 的 derived realistic case。

<a id="glm-52-vllm-ascend-decode"></a>

## GLM 5.2 vLLM-Ascend Decode

Case：`glm52_vllm_ascend_decode_b3_q3_ctx131072_top2048`。

官方 GLM-5.2 配置提供 `index_n_heads=32`、`index_head_dim=128`、
`index_topk=2048`、`num_attention_heads=64`，以及相同的
`qk_head_dim=256` 和 `v_head_dim=256`。vLLM-Ascend performance 测试使用
`GSM8K_prefix90_in131072`、`batch_size=3` 和三个 MTP 推测 token。

<a id="glm-52-vllm-ascend-prefill"></a>

## GLM 5.2 vLLM-Ascend Prefill

Case：`glm52_vllm_ascend_prefill_q4096_ctx131072_top2048`。

模型维度和 context 长度来自同一份官方配置与 vLLM-Ascend 测试。
`query_tokens=4096` 取自测试部署的 `max-num-batched-tokens`；`batch=1`
用于在算子级 prefill chunk 中隔离单条序列。因此，manifest 将该 case 标记为
`derived_official_e2e_test`，而不是将其描述为直接捕获的 kernel 形状。

## Contract 说明

CannBench 当前使用同一个 `head_dim` 表示 Q、K 和 V 的维度，而 FlashMLA
公开的是更细化的压缩 MLA contract。V3.2 case 保留 FlashMLA 实测的
`d_qk=576`；V4、V4-Pro 和 GLM-5.2 case 使用各自公开测试、部署指南或模型
配置中的匹配维度。对来自 vLLM-Ascend 官方部署指南的 case，manifest 继续
沿用现有的 `derived_official_e2e_test` 标签，以保持已发布 dataset contract
稳定。这些新增 family 当前首先用于扩展数据集覆盖；在实现对应 kernel family
之前，具体实现可以将其报告为不支持。
