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
| `basic_gnn_gin_neighbor_index_add` | `gnn_neighbor_accumulation` | `[10000, 64]` | `[200000]` | `[200000, 64]` | `-2` | `basic_gnn_gin` | `torchbench_train/basic_gnn_gin_train.json` |
| `bigbird_block_index_add` | `block_sparse_accumulation` | `[384, 64, 64]` | `[1008]` | `[1008, 64, 64]` | `0` | `hf_BigBird` | `torchbench_train/hf_BigBird_train.json` |
| `xlnet_memory_index_add` | `memory_accumulation` | `[8, 16, 512, 1023]` | `[512]` | `[8, 16, 512, 512]` | `3` | `XLNetLMHeadModel` | `hf_train/XLNetLMHeadModel_train.json` |
| `allenai_longformer_large_index_add_inplace` | `longformer_global_accumulation` | `[4718592]` | `[9437184]` | `[9437184]` | `0` | `AllenaiLongformerBase` | `hf_train/AllenaiLongformerBase_train.json` |
| `gpt_decoder_hidden_index_add` | `hidden_accumulation` | `[8, 512, 768]` | `[256]` | `[8, 256, 768]` | `1` | `GPTDecoder` | `curated_transformer_decoder_shape` |
| `llama_moe_token_combine_index_add` | `moe_token_combine` | `[2048, 1024]` | `[4096]` | `[4096, 1024]` | `0` | `LlamaMoE` | `curated_moe_token_combine_shape` |
| `embedding_backward_vocab_grad_index_add` | `embedding_gradient_accumulation` | `[8000, 512]` | `[4096]` | `[4096, 512]` | `0` | `TransformerEmbeddingBackward` | `curated_embedding_backward_shape` |
| `packed_sequence_restore_index_add` | `packed_sequence_restore` | `[2048, 512]` | `[2048]` | `[2048, 512]` | `0` | `PackedSequenceTransformer` | `curated_packed_sequence_restore_shape` |
| `llama_hidden_large_batch_index_add` | `hidden_accumulation` | `[16, 512, 512]` | `[256]` | `[16, 256, 512]` | `1` | `LlamaDecoder` | `curated_large_batch_hidden_shape` |
| `block_sparse_mid_index_add` | `block_sparse_accumulation` | `[128, 64, 64]` | `[256]` | `[256, 64, 64]` | `0` | `BlockSparseTransformer` | `curated_block_sparse_mid_shape` |
| `last_dim_memory_medium_index_add` | `memory_accumulation` | `[2, 8, 128, 2048]` | `[512]` | `[2, 8, 128, 512]` | `3` | `SegmentMemoryTransformer` | `curated_last_dim_memory_shape` |
| `longformer_medium_1d_index_add_inplace` | `longformer_global_accumulation` | `[1048576]` | `[2097152]` | `[2097152]` | `0` | `LongformerMedium` | `curated_global_token_1d_shape` |

The additional curated realistic shapes cover model-relevant accumulation
patterns that appear around embedding-gradient accumulation, routed-token
combination in MoE-style models, packed/ragged sequence restoration,
block-sparse/global-token accumulation, and memory-style last-dimension
updates. Dense Transformer inference usually does not make `index_add` a main
hot operator; these cases focus on sparse, routed, ragged, or backward-style
paths where duplicate indices and scatter-add semantics are more representative.

### Stress

| case_id | family | input_shape | index_shape | src_shape | dim | source_model |
| --- | --- | --- | --- | --- | --- | --- |
| `long_context_index_add` | `long_context_accumulation` | `[2, 16, 4096, 512]` | `[4096]` | `[2, 4096, 4096, 512]` | `1` | `index_add_long_context_boundary` |
| `wide_vocab_index_add` | `wide_vocab_accumulation` | `[4, 512, 128000]` | `[512]` | `[4, 512, 128000]` | `1` | `index_add_wide_vocab_boundary` |
| `large_batch_index_add` | `large_batch_accumulation` | `[64, 2048, 512]` | `[2048]` | `[64, 2048, 512]` | `1` | `index_add_large_batch_boundary` |
