# CannBench Operator Benchmark

| field | value |
| --- | --- |
| backend | ascend |
| device_name | Ascend950PR_9589 |
| op | lightning_indexer |
| dtype | bfloat16 |
| case_id | deepseek_a5_prefill_b1_q512_ctx1024_top1024 |
| family | prefill_indexing |
| payload | batch=1, context_tokens=1024, index_dim=128, index_heads=64, query_tokens=512, top_k=1024 |
| source_model | DeepSeek-A5-compatible |
| warmup | 2 |
| iterations | 2 |
