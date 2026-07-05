# CannBench Operator Benchmark

| field | value |
| --- | --- |
| backend | ascend |
| device_name | Ascend950PR_9589 |
| op | lightning_indexer |
| dtype | bfloat16 |
| case_id | deepseek_a5_prefill_b2_q128_ctx512_top512 |
| family | prefill_indexing |
| payload | batch=2, context_tokens=512, index_dim=128, index_heads=64, query_tokens=128, top_k=512 |
| source_model | DeepSeek-A5-compatible |
| warmup | 2 |
| iterations | 2 |
