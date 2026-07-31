# DSA 跨后端一致性与 Profiling 记录

本目录提供 canonical DSA workflow 的跨后端一致性检查。DSA workflow 保持相同
的组件合同：

```text
lightning_indexer -> indices -> sparse_attention -> output/lse
```

不同实现必须保持输入输出语义一致，但设备侧 kernel 图可以不同。本文件记录用于
审计该差异的远端 profiler 结果。

## V3.2 远端 Kernel 筛选审计

以下数据采集于 2026-07-30，设备为 Ascend 950PR，软件环境为 CANN 9.2.0，
数据类型为 BF16，`seed=0`。Prefill case 为
`deepseek_v32_flashmla_prefill_q4096_ctx32768_top2048`，decode case 为
`deepseek_v32_flashmla_decode_b2_q2_ctx32768_top2048`。

数据来源：

- SIMT prefill、vLLM-Ascend prefill/decode：`778f02b`
- SIMT decode 自动 Head64 dispatch rerun：`b301f4e`
- profiler 原始字段：`Op Name`、`Op Type`、`Task Duration(us)`

表中的“计入”表示 kernel 名命中对应 operator plugin 的
`ProfileKernelSelection`，其耗时进入 `profile-summary.json`；“排除”表示 raw
CSV 中存在该 kernel，但不进入组件 latency。

### Raw Kernel 明细

为便于阅读，SIMT C++ 符号在表中使用去除参数签名后的 kernel 名；原始 CSV 中
保留完整 mangled symbol。同名 kernel 的多次 launch 分行记录。

| 实现 | 阶段 | 被 profile 组件 | Raw kernel | 耗时（us） | 状态 | 说明 |
| --- | --- | --- | --- | ---: | --- | --- |
| SIMT v1 | Prefill | Indexer | `Fill_INT64_FLOAT32_high_performance_101` | 10.601000 | 排除 | Indexer Top-K score 初始化 |
| SIMT v1 | Prefill | Indexer | `ZerosLike_4_BYTES_high_performance_1` | 9.178000 | 排除 | Indexer Top-K index 初始化 |
| SIMT v1 | Prefill | Indexer | `lightning_indexer_fused_family_64x128_kernel` | 3134926.000000 | 计入 | Indexer 主 kernel |
| SIMT v1 | Prefill | Attention | `Fill_INT64_FLOAT32_high_performance_101` | 11.553000 | 排除 | bound Indexer 辅助 kernel |
| SIMT v1 | Prefill | Attention | `ZerosLike_2_BYTES_high_performance_1` | 328.335022 | 排除 | 未命中 Attention 白名单 |
| SIMT v1 | Prefill | Attention | `ZerosLike_2_BYTES_high_performance_1` | 10.795000 | 排除 | 未命中 Attention 白名单 |
| SIMT v1 | Prefill | Attention | `ZerosLike_4_BYTES_high_performance_1` | 10.406000 | 排除 | bound Indexer 辅助 kernel |
| SIMT v1 | Prefill | Attention | `sparse_attention_head64_fused_kernel` | 332552.312500 | 计入 | Attention 主 kernel |
| SIMT v1 | Prefill | Attention | `lightning_indexer_fused_family_64x128_kernel` | 3135268.000000 | 排除 | 为 bound indices 重跑的 Indexer |
| SIMT v1 | Decode | Indexer | `lightning_indexer_context_sharded_family_64x128_kernel` | 1986.091064 | 计入 | Indexer 主 kernel |
| SIMT v1 | Decode | Attention | `Cast_9f288b80370c6545ffe8cef142d37f5c_high_performance_0` | 3.459000 | 排除 | 未命中 Attention 白名单 |
| SIMT v1 | Decode | Attention | `sparse_attention_head64_fused_kernel` | 369.535004 | 计入 | Attention fused kernel |
| SIMT v1 | Decode | Attention | `sparse_attention_head64_combine_kernel` | 36.964001 | 计入 | Attention Combine kernel |
| SIMT v1 | Decode | Attention | `lightning_indexer_context_sharded_family_64x128_kernel` | 1986.458008 | 排除 | 为 bound indices 重跑的 Indexer |
| vLLM-Ascend | Prefill | Indexer | `LightningIndexer_2517851d7f53b28f971ff94bfa4b7037_570628891_mix_aic` | 5929.356934 | 计入 | Indexer 主 kernel |
| vLLM-Ascend | Prefill | Attention | `Add_FLOAT32_high_performance_8` | 6.488000 | 计入 | Attention 动态路径 |
| vLLM-Ascend | Prefill | Attention | `AsStrided_float16_int64_high_performance_102` | 723.628967 | 计入 | Attention 动态 lowering |
| vLLM-Ascend | Prefill | Attention | `AsStrided_float16_int64_high_performance_102` | 93.774002 | 计入 | Attention 动态 lowering |
| vLLM-Ascend | Prefill | Attention | `LightningIndexer_2517851d7f53b28f971ff94bfa4b7037_570628891_mix_aic` | 5935.953125 | 排除 | 为 bound indices 重跑的 Indexer |
| vLLM-Ascend | Prefill | Attention | `Log_fb42ea349cd1f1779279250bf66d5d90_high_performance_3` | 3.880000 | 计入 | LSE 路径 |
| vLLM-Ascend | Prefill | Attention | `SparseFlashAttention_9ec0bacf213f2b2dce3ba70149a4903b_17476_mix_aic` | 11838.093750 | 计入 | Attention 主 kernel |
| vLLM-Ascend | Prefill | Attention | `ZerosLike_2_BYTES_high_performance_1` | 327.502014 | 排除 | 未命中 Attention 白名单 |
| vLLM-Ascend | Prefill | Attention | `ZerosLike_2_BYTES_high_performance_1` | 10.125000 | 排除 | 未命中 Attention 白名单 |
| vLLM-Ascend | Prefill | Attention | `ZerosLike_2_BYTES_high_performance_1` | 3.482000 | 排除 | 未命中 Attention 白名单 |
| vLLM-Ascend | Prefill | Attention | `ZerosLike_2_BYTES_high_performance_1` | 9.224000 | 排除 | 未命中 Attention 白名单 |
| vLLM-Ascend | Decode | Indexer | `LightningIndexer_2517851d7f53b28f971ff94bfa4b7037_570628891_mix_aic` | 97.700996 | 计入 | Indexer 主 kernel |
| vLLM-Ascend | Decode | Attention | `Add_FLOAT32_high_performance_8` | 2.079000 | 计入 | Attention 动态路径 |
| vLLM-Ascend | Decode | Attention | `LightningIndexer_2517851d7f53b28f971ff94bfa4b7037_570628891_mix_aic` | 97.123001 | 排除 | 为 bound indices 重跑的 Indexer |
| vLLM-Ascend | Decode | Attention | `Log_fb42ea349cd1f1779279250bf66d5d90_high_performance_3` | 2.055000 | 计入 | LSE 路径 |
| vLLM-Ascend | Decode | Attention | `Slice_ee29dafe4fe21e12bb8d56ae91626cba_high_performance_150` | 3.586000 | 计入 | Attention 动态 lowering |
| vLLM-Ascend | Decode | Attention | `Slice_ee29dafe4fe21e12bb8d56ae91626cba_high_performance_400` | 2.224000 | 计入 | Attention 动态 lowering |
| vLLM-Ascend | Decode | Attention | `SparseFlashAttention_9ec0bacf213f2b2dce3ba70149a4903b_17476_mix_aic` | 56.924000 | 计入 | Attention 主 kernel |
| vLLM-Ascend | Decode | Attention | `Transpose_float16_int64_high_performance_10001` | 5.228000 | 计入 | Attention 动态 lowering |

### 汇总

“非上游辅助排除”只汇总当前组件 raw trace 中未被选择的辅助 kernel；“bound
Indexer 排除”是 Attention 为生成 bound indices 而额外执行的完整 Indexer。后者
已由 workflow 的 Indexer 组件单独计时，必须排除以避免重复计费。

| 实现/阶段/组件 | 当前计入 | 非上游辅助排除 | bound Indexer 排除 |
| --- | ---: | ---: | ---: |
| SIMT prefill Indexer | 3134.926000 ms | 0.019779 ms | - |
| SIMT prefill Attention | 332.552313 ms | 0.361089 ms | 3135.268000 ms |
| SIMT decode Indexer | 1.986091 ms | 0 ms | - |
| SIMT decode Attention | 0.406499 ms | 0.003459 ms | 1.986458 ms |
| vLLM-Ascend prefill Indexer | 5.929357 ms | 0 ms | - |
| vLLM-Ascend prefill Attention | 12.665865 ms | 0.350333 ms | 5.935953 ms |
| vLLM-Ascend decode Indexer | 0.097701 ms | 0 ms | - |
| vLLM-Ascend decode Attention | 0.072096 ms | 0 ms | 0.097123 ms |

## 结论

- SIMT 与 vLLM-Ascend 保持相同的两组件数据流，但设备 kernel 图不同。
- vLLM-Ascend 的独立 Indexer profile 没有外部 Fill/ZerosLike；其 prefill
  Attention raw trace 包含四个 `ZerosLike`，合计 `0.350333 ms`，当前被排除。
- SIMT prefill Indexer 的 Fill/ZerosLike 合计 `0.019779 ms`，当前也被排除。
- Attention profile 中重跑的 bound Indexer 只负责生成输入，不属于 Attention
  latency；workflow 总耗时由两个组件各自的 `profile-summary.json` 相加得到。
