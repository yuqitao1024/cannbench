# Softmax Dataset Manifest

This directory contains the built-in `softmax` benchmark datasets used by CannBench.

## Dataset Design

CannBench splits `softmax` inputs into three datasets so functionality checks, realistic coverage, and stress coverage do not get mixed together:

- `smoke`: minimal synthetic cases for CLI, schema, output, and backend wiring checks.
- `realistic`: curated real-model shapes derived from TritonBench traces, with source metadata preserved for each case.
- `stress`: synthetic operator-specific boundary cases that target known `softmax` pressure points and should not be reused as generic cases for other operators.

`smoke` and `stress` are maintained as generated manifests. Their source-of-truth construction logic lives in:

- [src/cannbench/operators/builtin/softmax/tools/generate_synthetic_datasets.py](/root/aiagent/cannbench/src/cannbench/operators/builtin/softmax/tools/generate_synthetic_datasets.py)

## Realistic Dataset Provenance

The current `realistic` dataset was curated from:

- Project: `https://github.com/meta-pytorch/tritonbench`
- Baseline commit: `daebb02b0728140203f88dbc56272e2af9f89b9d`
- Source area: `tritonbench/data/input_configs/`
- Included ops: `aten._softmax.default`, `aten._log_softmax.default`

`realistic` is intentionally not a full TritonBench dump. It is a filtered representative set for single-operator benchmarking.

### Realistic Selection Logic

The current curation rules are:

1. Start from TritonBench traces that contain `softmax` or `log_softmax` input configs.
2. Keep real-model shapes from multiple workload families:
   - encoder / decoder attention
   - local attention
   - vision attention
   - speech / torchbench attention
   - language-model logits
3. Deduplicate cases by semantic workload signature:
   - current practical key: `family + shape + dim`
   - if multiple source traces collapse to the same signature, keep one representative source record
4. Filter out low-value repetitive traces:
   - repeated classification heads with nearly identical `[N, 1000]` style shapes
   - generate-time trace fragments that mostly differ by tiny token-step artifacts
   - near-duplicate models that do not expand workload coverage
5. Preserve provenance fields in the manifest:
   - `source_project`
   - `source_model`
   - `source_file`
   - `source_op`

### Realistic Refresh Workflow

When updating `realistic` in the future:

1. Pin the TritonBench commit hash in this file.
2. Re-extract `softmax` and `log_softmax` configs from `input_configs`.
3. Re-apply deduplication by `family + shape + dim`.
4. Re-apply the filtering rules above.
5. Update the case tables below and keep provenance fields aligned with the selected source rows.

## Case Tables

### Smoke

Design intent: smallest possible functional fixture set for wiring checks.

Current coverage target:

- tiny logits softmax
- tiny attention score softmax
- tiny channel-axis softmax

| case_id | family | shape | dim | source_kind | source_model | source_file | source_op |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `tiny_logits` | `lm_logits` | `[32, 128]` | `-1` | `synthetic_smoke` | `smoke_fixture` | `built-in` | `softmax` |

### Realistic

Design intent: curated representative shapes from real traced models. All rows in this table use `source_kind=real_model` and `source_project=TritonBench`.

| case_id | family | shape | dim | source_model | source_file | source_op |
| --- | --- | --- | --- | --- | --- | --- |
| `t5_attention` | `attention` | `[4, 8, 1024, 1024]` | `-1` | `T5Small` | `hf_train/T5Small_train.json` | `aten._softmax.default` |
| `xcit_attention` | `attention` | `[4, 16, 48, 48]` | `-1` | `xcit_large_24_p8_224` | `timm_train/xcit_large_24_p8_224_train.json` | `aten._softmax.default` |
| `convbert_attention` | `attention` | `[16, 6, 512, 512]` | `-1` | `YituTechConvBert` | `hf_train/YituTechConvBert_train.json` | `aten._softmax.default` |
| `convbert_local_kernel` | `local_attention` | `[49152, 9, 1]` | `1` | `YituTechConvBert` | `hf_train/YituTechConvBert_train.json` | `aten._softmax.default` |
| `deberta_attention` | `attention` | `[2, 24, 512, 512]` | `-1` | `DebertaV2ForMaskedLM` | `hf_train/DebertaV2ForMaskedLM_train.json` | `aten._softmax.default` |
| `electra_attention` | `attention` | `[32, 4, 512, 512]` | `-1` | `ElectraForCausalLM` | `hf_train/ElectraForCausalLM_train.json` | `aten._softmax.default` |
| `gptneo_attention` | `attention` | `[32, 16, 128, 128]` | `-1` | `GPTNeoForCausalLM` | `hf_train/GPTNeoForCausalLM_train.json` | `aten._softmax.default` |
| `gptj_attention` | `attention` | `[1, 16, 128, 128]` | `-1` | `GPTJForCausalLM` | `hf_train/GPTJForCausalLM_train.json` | `aten._softmax.default` |
| `layoutlm_attention` | `attention` | `[16, 12, 512, 512]` | `-1` | `LayoutLMForMaskedLM` | `hf_train/LayoutLMForMaskedLM_train.json` | `aten._softmax.default` |
| `mobilebert_attention` | `attention` | `[128, 4, 128, 128]` | `-1` | `MobileBertForMaskedLM` | `hf_train/MobileBertForMaskedLM_train.json` | `aten._softmax.default` |
| `xlnet_attention` | `attention` | `[8, 16, 512, 512]` | `3` | `XLNetLMHeadModel` | `hf_train/XLNetLMHeadModel_train.json` | `aten._softmax.default` |
| `plbart_attention` | `attention` | `[96, 1024, 1024]` | `-1` | `PLBartForCausalLM` | `hf_train/PLBartForCausalLM_train.json` | `aten._softmax.default` |
| `pegasus_attention` | `attention` | `[512, 128, 128]` | `-1` | `PegasusForConditionalGeneration` | `hf_train/PegasusForConditionalGeneration_train.json` | `aten._softmax.default` |
| `trocr_attention` | `attention` | `[512, 256, 256]` | `-1` | `TrOCRForCausalLM` | `hf_train/TrOCRForCausalLM_train.json` | `aten._softmax.default` |
| `levit_global_attention` | `vision_attention` | `[1024, 4, 196, 196]` | `-1` | `levit_128` | `timm_train/levit_128_train.json` | `aten._softmax.default` |
| `levit_mixed_attention` | `vision_attention` | `[1024, 8, 49, 196]` | `-1` | `levit_128` | `timm_train/levit_128_train.json` | `aten._softmax.default` |
| `swin_window_attention` | `vision_attention` | `[256, 16, 49, 49]` | `-1` | `swin_base_patch4_window7_224` | `timm_train/swin_base_patch4_window7_224_train.json` | `aten._softmax.default` |
| `crossvit_cls_attention` | `vision_attention` | `[256, 4, 1, 197]` | `-1` | `crossvit_9_240` | `timm_train/crossvit_9_240_train.json` | `aten._softmax.default` |
| `halonet_window_attention` | `vision_attention` | `[1024, 4, 64, 144]` | `-1` | `eca_halonext26ts` | `timm_train/eca_halonext26ts_train.json` | `aten._softmax.default` |
| `speech_transformer_attention` | `attention` | `[80, 204, 204]` | `2` | `speech_transformer` | `torchbench_train/speech_transformer_train.json` | `aten._softmax.default` |
| `bert_pytorch_attention` | `attention` | `[16, 12, 128, 128]` | `-1` | `BERT_pytorch` | `torchbench_train/BERT_pytorch_train.json` | `aten._softmax.default` |
| `t5_logits` | `lm_logits` | `[4096, 32128]` | `1` | `T5Small` | `hf_train/T5Small_train.json` | `aten._log_softmax.default` |
| `convbert_logits` | `lm_logits` | `[8192, 30522]` | `1` | `YituTechConvBert` | `hf_train/YituTechConvBert_train.json` | `aten._log_softmax.default` |
| `longformer_logits` | `lm_logits` | `[4096, 50265]` | `1` | `AllenaiLongformerBase` | `hf_train/AllenaiLongformerBase_train.json` | `aten._log_softmax.default` |
| `plbart_logits` | `lm_logits` | `[8192, 50005]` | `1` | `PLBartForCausalLM` | `hf_train/PLBartForCausalLM_train.json` | `aten._log_softmax.default` |
| `camembert_logits` | `lm_logits` | `[8192, 32005]` | `1` | `CamemBert` | `hf_train/CamemBert_train.json` | `aten._log_softmax.default` |
| `m2m100_logits` | `lm_logits` | `[2048, 128112]` | `1` | `M2M100ForConditionalGeneration` | `hf_train/M2M100ForConditionalGeneration_train.json` | `aten._log_softmax.default` |
| `mt5_logits` | `lm_logits` | `[2048, 250112]` | `1` | `MT5ForConditionalGeneration` | `hf_train/MT5ForConditionalGeneration_train.json` | `aten._log_softmax.default` |
| `xglm_logits` | `lm_logits` | `[1024, 256008]` | `1` | `XGLMForCausalLM` | `hf_train/XGLMForCausalLM_train.json` | `aten._log_softmax.default` |
| `opt_logits` | `lm_logits` | `[4094, 50272]` | `1` | `OPTForCausalLM` | `hf_train/OPTForCausalLM_train.json` | `aten._log_softmax.default` |

### Ascend SIMT Softmax Realistic Performance Snapshot

Date: 2026-06-22

Purpose: record the first broad Ascend comparison between the CANN ops library
baseline softmax and the built-in SIMT `aten_softmax` implementation
for the `realistic` split.

Measurement settings:

- Backend: Ascend NPU
- Profiler: `msprof op`
- Dtype: `float32`
- Seed: `7`
- Warmup: `0`
- Iterations: `1`
- Samples per row: `1`
- Reported value: parsed device-side duration from profiler CSV output

Run status:

- Total realistic cases: `30`
- Successful profile rows: `55 / 60`
- Fully comparable cases: `27 / 30`
- Failed profile rows: `5 / 60`
- Failure reason for incomplete rows: NPU runtime failed to open device context
  after the long run (`LazySetDevice`, error code `507033`,
  `rtSetDevice execution failed`, `context is a null pointer`).

Interpretation: completed rows show the SIMT implementation is broadly in the
same performance range as the CANN ops library baseline implementation.
Because each row has one profiler sample, treat this as a checkpoint record, not
a stable benchmark summary.

| case_id | cann_ops_library_ms | simt_v1_ms | simt_v1/cann_ops_library | status |
| --- | ---: | ---: | ---: | --- |
| `t5_attention` | 0.135135 | 0.138428 | 1.024x | ok |
| `xcit_attention` | 0.003714 | 0.003626 | 0.976x | ok |
| `convbert_attention` | 0.093750 | 0.093697 | 0.999x | ok |
| `convbert_local_kernel` | 0.007134 | 0.007406 | 1.038x | ok |
| `deberta_attention` | 0.039373 | 0.038644 | 0.981x | ok |
| `electra_attention` | 0.137837 | 0.137303 | 0.996x | ok |
| `gptneo_attention` | 0.026213 | 0.026223 | 1.000x | ok |
| `gptj_attention` | 0.003928 | 0.003615 | 0.920x | ok |
| `layoutlm_attention` | 0.230790 | 0.226357 | 0.981x | ok |
| `mobilebert_attention` | 0.026779 | 0.025852 | 0.965x | ok |
| `xlnet_attention` | 0.138712 | 0.138616 | 0.999x | ok |
| `plbart_attention` | 0.505655 | 0.506869 | 1.002x | ok |
| `pegasus_attention` | 0.026647 | 0.025515 | 0.958x | ok |
| `trocr_attention` | 0.139692 | 0.139721 | 1.000x | ok |
| `levit_global_attention` | 0.842046 | 0.836390 | 0.993x | ok |
| `levit_mixed_attention` | 0.404607 | 0.396170 | 0.979x | ok |
| `swin_window_attention` | 0.036603 | 0.036397 | 0.994x | ok |
| `crossvit_cls_attention` | 0.004265 | 0.004483 | 1.051x | ok |
| `halonet_window_attention` | 0.170329 | 0.170361 | 1.000x | ok |
| `speech_transformer_attention` | 0.014321 | 0.014734 | 1.029x | ok |
| `bert_pytorch_attention` | 0.012228 | 0.012452 | 1.018x | ok |
| `t5_logits` | 0.707640 | 0.706114 | 0.998x | ok |
| `convbert_logits` | 1.353036 | 1.348364 | 0.997x | ok |
| `longformer_logits` | 1.115980 | 1.119064 | 1.003x | ok |
| `plbart_logits` | 2.252802 | 2.256534 | 1.002x | ok |
| `camembert_logits` | 1.426481 | 1.426887 | 1.000x | ok |
| `m2m100_logits` | 1.530845 | 1.526289 | 0.997x | ok |
| `mt5_logits` | 3.136916 |  |  | incomplete |
| `xglm_logits` |  |  |  | incomplete |
| `opt_logits` |  |  |  | incomplete |

### Stress

Design intent: synthetic `softmax` boundary cases. These are specific to `softmax` workload semantics and should not be mechanically reused for other operators.

Current coverage target:

- long-context attention reduction
- wide-vocabulary logits reduction
- MoE router score normalization
- very small reduction axis
- batched vision-window attention
- channel-axis activation map normalization
- beam-search token score normalization

| case_id | family | shape | dim | source_kind | source_model | source_file | source_op |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `long_context_attention` | `attention` | `[1, 32, 4096, 4096]` | `-1` | `synthetic_boundary` | `llm_attention_boundary` | `generated` | `softmax` |
| `wide_vocab_lm_logits` | `lm_logits` | `[8192, 128256]` | `1` | `synthetic_boundary` | `llm_logits_boundary` | `generated` | `softmax` |
| `moe_router_scores` | `router_scores` | `[4096, 128]` | `-1` | `synthetic_boundary` | `moe_router_boundary` | `generated` | `softmax` |
| `small_reduction_axis` | `reduction_edge` | `[16384, 2]` | `-1` | `synthetic_boundary` | `softmax_small_axis_boundary` | `generated` | `softmax` |
| `vision_window_batch` | `vision_attention` | `[2048, 16, 49, 49]` | `-1` | `synthetic_boundary` | `vision_window_batch_boundary` | `generated` | `softmax` |
| `channelwise_activation_map` | `channel_activation` | `[64, 2048, 7, 7]` | `1` | `synthetic_boundary` | `channel_activation_boundary` | `generated` | `softmax` |
| `beam_search_token_scores` | `decode_logits` | `[512, 4, 64000]` | `-1` | `synthetic_boundary` | `beam_search_token_boundary` | `generated` | `softmax` |
