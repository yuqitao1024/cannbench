# Cross Entropy Dataset Manifest

This directory contains the built-in `torch.nn.functional.cross_entropy` benchmark datasets used by CannBench.

## Dataset Design

- `smoke`: small token and sequence classification losses for CLI and backend validation.
- `realistic`: curated model-shaped losses derived from TritonBench-style workloads.
- `stress`: boundary cases for long context, wide vocabularies, and large batches.

## Case Tables

### Smoke

| case_id | family | logits_shape | target_shape | num_classes | source_model |
| --- | --- | --- | --- | --- | --- |
| `tiny_token_classification_loss` | `token_classification` | `[32, 128, 64]` | `[32, 128]` | `64` | `cross_entropy_smoke_fixture` |
| `tiny_sequence_classification_loss` | `sequence_classification` | `[16, 32, 8]` | `[16, 32]` | `8` | `cross_entropy_sequence_fixture` |
| `tiny_language_model_loss` | `language_model` | `[2, 64, 256]` | `[2, 64]` | `256` | `cross_entropy_lm_fixture` |

### Realistic

| case_id | family | logits_shape | target_shape | num_classes | source_model | source_file |
| --- | --- | --- | --- | --- | --- | --- |
| `bert_token_classification_loss` | `token_classification` | `[16, 128, 30522]` | `[16, 128]` | `30522` | `BERT_pytorch` | `torchbench_train/BERT_pytorch_train.json` |
| `t5_decoder_language_loss` | `seq2seq_language_model` | `[4, 512, 32128]` | `[4, 512]` | `32128` | `T5Small` | `hf_train/T5Small_train.json` |
| `opt_next_token_loss` | `next_token` | `[8, 256, 50272]` | `[8, 256]` | `50272` | `OPTForCausalLM` | `hf_train/OPTForCausalLM_train.json` |

### Stress

| case_id | family | logits_shape | target_shape | num_classes | source_model |
| --- | --- | --- | --- | --- | --- |
| `long_context_token_loss` | `long_context` | `[2, 4096, 128000]` | `[2, 4096]` | `128000` | `cross_entropy_long_context_boundary` |
| `wide_vocab_token_loss` | `wide_vocab` | `[4, 1024, 262144]` | `[4, 1024]` | `262144` | `cross_entropy_wide_vocab_boundary` |
| `large_batch_token_loss` | `large_batch` | `[64, 2048, 8192]` | `[64, 2048]` | `8192` | `cross_entropy_large_batch_boundary` |
