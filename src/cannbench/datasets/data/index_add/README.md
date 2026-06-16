# Index Add Dataset Manifest

This directory contains the built-in `torch.index_add` benchmark datasets used by CannBench.

## Dataset Design

- `smoke`: small rank-2 and rank-3 accumulation fixtures for validation.
- `realistic`: curated model-shaped accumulation cases with TritonBench-style source metadata.
- `stress`: operator-specific boundary cases for long context, wide vocabularies, and large batches.

## Case Tables

### Smoke

| case_id | family | input_shape | index_shape | src_shape | dim | source_model |
| --- | --- | --- | --- | --- | --- | --- |
| `tiny_rank2_index_add` | `token_accumulation` | `[32, 64]` | `[16]` | `[32, 16]` | `1` | `index_add_smoke_fixture` |
| `tiny_rank3_index_add` | `batched_accumulation` | `[4, 16, 32]` | `[8]` | `[4, 8, 32]` | `1` | `index_add_rank3_fixture` |
| `tiny_sequence_index_add` | `sequence_accumulation` | `[2, 32, 64]` | `[12]` | `[2, 12, 64]` | `1` | `index_add_sequence_fixture` |

### Realistic

| case_id | family | input_shape | index_shape | src_shape | dim | source_model | source_file |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `bert_hidden_index_add` | `hidden_accumulation` | `[16, 128, 768]` | `[128]` | `[16, 128, 768]` | `1` | `BERT_pytorch` | `torchbench_train/BERT_pytorch_train.json` |
| `t5_hidden_index_add` | `hidden_accumulation` | `[4, 8, 1024]` | `[1024]` | `[4, 1024, 1024]` | `1` | `T5Small` | `hf_train/T5Small_train.json` |
| `opt_hidden_index_add` | `hidden_accumulation` | `[8, 256, 2048]` | `[256]` | `[8, 256, 2048]` | `1` | `OPTForCausalLM` | `hf_train/OPTForCausalLM_train.json` |

### Stress

| case_id | family | input_shape | index_shape | src_shape | dim | source_model |
| --- | --- | --- | --- | --- | --- | --- |
| `long_context_index_add` | `long_context_accumulation` | `[2, 16, 4096, 512]` | `[4096]` | `[2, 4096, 4096, 512]` | `1` | `index_add_long_context_boundary` |
| `wide_vocab_index_add` | `wide_vocab_accumulation` | `[4, 512, 128000]` | `[512]` | `[4, 512, 128000]` | `1` | `index_add_wide_vocab_boundary` |
| `large_batch_index_add` | `large_batch_accumulation` | `[64, 2048, 512]` | `[2048]` | `[64, 2048, 512]` | `1` | `index_add_large_batch_boundary` |
