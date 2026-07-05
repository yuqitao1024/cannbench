# CannBench Operator Benchmark

| field | value |
| --- | --- |
| backend | ascend |
| device_name | Ascend950PR_9589 |
| op | sparse_attention |
| dtype | bfloat16 |
| case_id | deepseek_a5_prefill_b1_q256_ctx512_top512 |
| family | prefill_sparse_attention |
| payload | batch=1, causal=True, context_tokens=512, head_dim=512, kv_heads=1, phase=prefill, query_heads=64, query_tokens=256, selected_tokens=512 |
| source_model | DeepSeek-A5-compatible |
| warmup | 2 |
| iterations | 2 |
