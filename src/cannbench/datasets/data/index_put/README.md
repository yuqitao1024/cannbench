# Index Put Dataset Manifest

This directory contains the built-in `torch.index_put` benchmark datasets used by CannBench.

## Dataset Design

- `smoke`: small rank-2 and rank-3 advanced-indexing update fixtures for validation.
- `realistic`: curated model-shaped update cases with TritonBench-style source metadata.
- `stress`: operator-specific boundary cases for long context, wide vocabularies, and large batches.

## Shape Semantics

The manifest follows the native `torch.index_put(input, indices, values, accumulate=False)` interface. `index_shapes` records the shape of each tensor in the `indices` tuple, and the first N input axes are indexed by N index tensors. All index tensors currently share one broadcast shape. `values_shape` equals that broadcast shape plus the unindexed trailing input axes. The initial dataset keeps `accumulate=false` to isolate overwrite semantics; future accumulate cases can be added without changing the schema.

## Realistic Source Policy

The realistic split follows the same curation policy used for the other indexing/update operators: retain TritonBench-style source metadata, deduplicate by full shape and semantic family, and keep shapes that exercise production model dimensions without copying the full upstream benchmark corpus. Refreshes should record the TritonBench version or commit used to regenerate the manifest.

## Case Tables

### Smoke

| case_id | family | input_shape | index_shapes | values_shape | accumulate | source_model |
| --- | --- | --- | --- | --- | --- | --- |
| `tiny_rank2_index_put` | `token_update` | `[32, 64]` | `[[16], [16]]` | `[16]` | `false` | `index_put_smoke_fixture` |
| `tiny_rank3_index_put` | `batched_update` | `[4, 16, 32]` | `[[4, 8], [4, 8]]` | `[4, 8, 32]` | `false` | `index_put_rank3_fixture` |
| `tiny_sequence_index_put` | `sequence_update` | `[2, 32, 64]` | `[[2, 12], [2, 12]]` | `[2, 12, 64]` | `false` | `index_put_sequence_fixture` |

### Realistic

| case_id | family | input_shape | index_shapes | values_shape | accumulate | source_model | source_file |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `bert_hidden_index_put` | `hidden_update` | `[16, 128, 768]` | `[[16, 128], [16, 128]]` | `[16, 128, 768]` | `false` | `BERT_pytorch` | `torchbench_train/BERT_pytorch_train.json` |
| `t5_hidden_index_put` | `hidden_update` | `[4, 8, 1024]` | `[[4, 8], [4, 8]]` | `[4, 8, 1024]` | `false` | `T5Small` | `hf_train/T5Small_train.json` |
| `opt_hidden_index_put` | `hidden_update` | `[8, 256, 2048]` | `[[8, 256], [8, 256]]` | `[8, 256, 2048]` | `false` | `OPTForCausalLM` | `hf_train/OPTForCausalLM_train.json` |

### Stress

| case_id | family | input_shape | index_shapes | values_shape | accumulate | source_model |
| --- | --- | --- | --- | --- | --- | --- |
| `long_context_index_put` | `long_context_update` | `[2, 16, 4096, 512]` | `[[2, 4096], [2, 4096]]` | `[2, 4096, 4096, 512]` | `false` | `index_put_long_context_boundary` |
| `wide_vocab_index_put` | `wide_vocab_update` | `[4, 512, 128000]` | `[[4, 512], [4, 512]]` | `[4, 512, 128000]` | `false` | `index_put_wide_vocab_boundary` |
| `large_batch_index_put` | `large_batch_update` | `[64, 2048, 512]` | `[[64, 2048], [64, 2048]]` | `[64, 2048, 512]` | `false` | `index_put_large_batch_boundary` |
