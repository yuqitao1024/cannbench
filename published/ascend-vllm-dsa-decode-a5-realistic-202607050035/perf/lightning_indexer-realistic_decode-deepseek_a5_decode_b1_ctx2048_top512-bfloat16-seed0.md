# CannBench Operator Benchmark

| field | value |
| --- | --- |
| backend | ascend |
| device_name | Ascend950PR_9589 |
| op | lightning_indexer |
| dtype | bfloat16 |
| case_id | deepseek_a5_decode_b1_ctx2048_top512 |
| family | decode_indexing |
| payload | batch=1, context_tokens=2048, index_dim=128, index_heads=64, query_tokens=1, top_k=512 |
| source_model | DeepSeek-A5-compatible |
| warmup | 2 |
| iterations | 2 |
