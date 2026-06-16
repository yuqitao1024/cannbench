# Scatter Add Dataset Manifest

This directory contains the built-in `torch.scatter_add` benchmark datasets used by CannBench.

## Dataset Design

- `smoke`: small rank-2 and rank-3 accumulation fixtures for validation.
- `realistic`: curated model-shaped accumulation cases with TritonBench-style source metadata.
- `stress`: operator-specific boundary cases for long context, wide vocabularies, and large batches.

## Case Tables

### Smoke

| case_id | family | input_shape | index_shape | src_shape | dim | source_model |
| --- | --- | --- | --- | --- | --- | --- |
| `tiny_rank2_scatter_add` | `token_accumulation` | `[32, 64]` | `[32, 64]` | `[32, 64]` | `1` | `scatter_add_smoke_fixture` |
| `tiny_rank3_scatter_add` | `batched_accumulation` | `[4, 16, 32]` | `[4, 16, 32]` | `[4, 16, 32]` | `-1` | `scatter_add_rank3_fixture` |
| `tiny_sequence_scatter_add` | `sequence_accumulation` | `[2, 32, 64]` | `[2, 32, 64]` | `[2, 32, 64]` | `1` | `scatter_add_sequence_fixture` |

### Realistic

| case_id | family | input_shape | index_shape | src_shape | dim | source_model | source_file |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `bert_token_scatter_add` | `token_accumulation` | `[16, 128, 30522]` | `[16, 128, 30522]` | `[16, 128, 30522]` | `-1` | `BERT_pytorch` | `torchbench_train/BERT_pytorch_train.json` |
| `t5_attention_scatter_add` | `attention_accumulation` | `[4, 8, 1024, 1024]` | `[4, 8, 1024, 1024]` | `[4, 8, 1024, 1024]` | `-1` | `T5Small` | `hf_train/T5Small_train.json` |
| `opt_vocab_scatter_add` | `lm_logits_accumulation` | `[8, 256, 50272]` | `[8, 256, 50272]` | `[8, 256, 50272]` | `-1` | `OPTForCausalLM` | `hf_train/OPTForCausalLM_train.json` |

### Stress

| case_id | family | input_shape | index_shape | src_shape | dim | source_model |
| --- | --- | --- | --- | --- | --- | --- |
| `long_context_scatter_add` | `long_context_accumulation` | `[2, 16, 4096, 4096]` | `[2, 16, 4096, 4096]` | `[2, 16, 4096, 4096]` | `-1` | `scatter_add_long_context_boundary` |
| `wide_vocab_scatter_add` | `wide_vocab_accumulation` | `[4, 512, 128000]` | `[4, 512, 128000]` | `[4, 512, 128000]` | `-1` | `scatter_add_wide_vocab_boundary` |
| `large_batch_scatter_add` | `large_batch_accumulation` | `[64, 2048, 512]` | `[64, 2048, 512]` | `[64, 2048, 512]` | `1` | `scatter_add_large_batch_boundary` |
