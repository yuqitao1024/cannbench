# Take Along Dim Dataset Manifest

This directory contains the built-in `torch.take_along_dim` benchmark datasets used by CannBench.

## Dataset Design

- `smoke`: tiny rank-2 and rank-3 fixtures that validate indexed selection along different axes.
- `realistic`: curated model-shaped top-k and token-score selection cases with source metadata derived from TritonBench-style model workloads.
- `stress`: operator-specific boundary cases for long-context attention, wide-vocabulary top-k, and large sequence reordering.

The `realistic` split uses TritonBench-derived model names and source files as of the dataset curation pass used for this repository. Cases are filtered to keep tensor sizes meaningful for single-card operator benchmarking while avoiding duplicate shapes and unrelated indexing semantics.

## Case Tables

### Smoke

| case_id | family | input_shape | index_shape | dim | source_model |
| --- | --- | --- | --- | --- | --- |
| `tiny_rank2_take_along_dim` | `topk_axis` | `[32, 64]` | `[32, 16]` | `1` | `take_along_dim_rank2_fixture` |
| `tiny_rank3_take_along_dim` | `batched_topk` | `[4, 16, 32]` | `[4, 16, 8]` | `-1` | `take_along_dim_rank3_fixture` |
| `tiny_sequence_reorder` | `sequence_reorder` | `[2, 32, 64]` | `[2, 12, 64]` | `1` | `take_along_dim_sequence_fixture` |

### Realistic

| case_id | family | input_shape | index_shape | dim | source_model | source_file |
| --- | --- | --- | --- | --- | --- | --- |
| `t5_attention_topk_values` | `attention_topk` | `[4, 8, 1024, 1024]` | `[4, 8, 1024, 64]` | `-1` | `T5Small` | `hf_train/T5Small_train.json` |
| `bert_token_score_select` | `token_scores` | `[16, 128, 30522]` | `[16, 128, 64]` | `-1` | `BERT_pytorch` | `torchbench_train/BERT_pytorch_train.json` |
| `opt_vocab_topk_values` | `lm_logits_topk` | `[8, 256, 50272]` | `[8, 256, 50]` | `-1` | `OPTForCausalLM` | `hf_train/OPTForCausalLM_train.json` |

### Stress

| case_id | family | input_shape | index_shape | dim | source_model |
| --- | --- | --- | --- | --- | --- |
| `long_context_attention_topk` | `long_context_topk` | `[2, 16, 4096, 4096]` | `[2, 16, 4096, 128]` | `-1` | `take_along_dim_long_context_boundary` |
| `wide_vocab_topk` | `wide_vocab_topk` | `[4, 512, 128000]` | `[4, 512, 100]` | `-1` | `take_along_dim_wide_vocab_boundary` |
| `large_batch_sequence_reorder` | `sequence_reorder` | `[64, 2048, 512]` | `[64, 512, 512]` | `1` | `take_along_dim_large_batch_boundary` |
