# Embedding Dataset Manifest

This directory contains the built-in `torch.nn.Embedding` benchmark datasets used by CannBench.

## Dataset Design

- `smoke`: tiny token lookup fixtures for CLI and backend validation
- `realistic`: curated real-model token embedding shapes with source metadata
- `stress`: embedding-specific boundary cases for table width, sequence length, and hidden width

## Case Tables

### Smoke

| case_id | family | num_embeddings | embedding_dim | index_shape | source_model |
| --- | --- | --- | --- | --- | --- |
| `tiny_token_lookup` | `token_lookup` | `128` | `64` | `[32]` | `embedding_smoke_fixture` |
| `tiny_batched_lookup` | `batched_lookup` | `256` | `32` | `[4, 16]` | `embedding_batch_fixture` |
| `tiny_prompt_lookup` | `prompt_tokens` | `1024` | `128` | `[2, 64]` | `embedding_prompt_fixture` |

### Realistic

| case_id | family | num_embeddings | embedding_dim | index_shape | source_model | source_file |
| --- | --- | --- | --- | --- | --- | --- |
| `bert_word_embeddings` | `encoder_tokens` | `30522` | `768` | `[16, 128]` | `BERT_pytorch` | `torchbench_train/BERT_pytorch_train.json` |
| `t5_token_embeddings` | `seq2seq_tokens` | `32128` | `512` | `[8, 512]` | `T5Small` | `hf_train/T5Small_train.json` |
| `gptj_token_embeddings` | `decoder_tokens` | `50400` | `4096` | `[4, 128]` | `GPTJForCausalLM` | `hf_train/GPTJForCausalLM_train.json` |
| `mobilebert_token_embeddings` | `mobile_encoder_tokens` | `30522` | `512` | `[32, 128]` | `MobileBertForMaskedLM` | `hf_train/MobileBertForMaskedLM_train.json` |
| `opt_token_embeddings` | `llm_tokens` | `50272` | `2048` | `[8, 256]` | `OPTForCausalLM` | `hf_train/OPTForCausalLM_train.json` |

### Stress

| case_id | family | num_embeddings | embedding_dim | index_shape | source_model |
| --- | --- | --- | --- | --- | --- |
| `wide_table_lookup` | `wide_embedding_table` | `262144` | `4096` | `[8, 128]` | `embedding_wide_table_boundary` |
| `long_sequence_lookup` | `long_sequence_tokens` | `65536` | `1024` | `[4, 4096]` | `embedding_long_sequence_boundary` |
| `micro_batch_large_hidden` | `micro_batch_hidden_width` | `50000` | `8192` | `[1, 64]` | `embedding_hidden_width_boundary` |
