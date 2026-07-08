import pytest
from cannbench.operators.builtin.cross_entropy import (
    get_cross_entropy_case,
    get_cross_entropy_dataset,
)
from cannbench.operators.builtin.cross_entropy.materialize import materialize_cross_entropy_inputs
from cannbench.operators.builtin.embedding import get_embedding_case, get_embedding_dataset
from cannbench.operators.builtin.embedding.materialize import materialize_embedding_inputs
from cannbench.operators.builtin.gather import get_gather_case, get_gather_dataset
from cannbench.operators.builtin.gather.materialize import materialize_gather_inputs
from cannbench.operators.builtin.index_add import get_index_add_case, get_index_add_dataset
from cannbench.operators.builtin.index_add.materialize import materialize_index_add_inputs
from cannbench.operators.builtin.index_put import get_index_put_case, get_index_put_dataset
from cannbench.operators.builtin.index_put.materialize import materialize_index_put_inputs
from cannbench.operators.builtin.index_select import (
    get_index_select_case,
    get_index_select_dataset,
)
from cannbench.operators.builtin.index_select.materialize import materialize_index_select_inputs
from cannbench.operators.builtin.lightning_indexer import (
    get_lightning_indexer_case,
    get_lightning_indexer_dataset,
)
from cannbench.operators.builtin.lightning_indexer.materialize import materialize_lightning_indexer_inputs
from cannbench.operators.builtin.masked_select import (
    get_masked_select_case,
    get_masked_select_dataset,
)
from cannbench.operators.builtin.masked_select.materialize import materialize_masked_select_inputs
from cannbench.operators.builtin.scatter import get_scatter_case, get_scatter_dataset
from cannbench.operators.builtin.scatter.materialize import materialize_scatter_inputs
from cannbench.operators.builtin.scatter_add import (
    get_scatter_add_case,
    get_scatter_add_dataset,
)
from cannbench.operators.builtin.scatter_add.materialize import materialize_scatter_add_inputs
from cannbench.operators.builtin.softmax import get_softmax_case, get_softmax_dataset
from cannbench.operators.builtin.softmax.materialize import materialize_softmax_inputs
from cannbench.operators.builtin.sparse_attention import (
    get_sparse_attention_case,
    get_sparse_attention_dataset,
)
from cannbench.operators.builtin.sparse_attention.materialize import materialize_sparse_attention_inputs
from cannbench.operators.builtin.take_along_dim import (
    get_take_along_dim_case,
    get_take_along_dim_dataset,
)
from cannbench.operators.builtin.take_along_dim.materialize import materialize_take_along_dim_inputs
from cannbench.operators.builtin.topk import get_topk_case, get_topk_dataset
from cannbench.operators.builtin.topk.materialize import materialize_topk_inputs


def test_get_softmax_dataset_loads_builtin_datasets():
    smoke = get_softmax_dataset("smoke")
    realistic = get_softmax_dataset("realistic")
    stress = get_softmax_dataset("stress")

    assert smoke.name == "smoke"
    assert realistic.name == "realistic"
    assert stress.name == "stress"
    assert len(smoke.cases) >= 3
    assert realistic.cases
    assert len(stress.cases) >= 7


def test_get_softmax_case_preserves_realistic_source_metadata():
    case = get_softmax_case("realistic", "t5_attention")

    assert case.case_id == "t5_attention"
    assert case.family == "attention"
    assert case.shape == (4, 8, 1024, 1024)
    assert case.dim == -1
    assert case.source_kind == "real_model"
    assert case.source_project == "TritonBench"
    assert case.source_model == "T5Small"
    assert case.source_file
    assert case.source_op == "aten._softmax.default"


def test_realistic_dataset_includes_required_tritonbench_cases():
    dataset = get_softmax_dataset("realistic")
    cases = {case.case_id: case for case in dataset.cases}

    assert len(cases) >= 20
    assert len(cases) == len(dataset.cases)
    assert cases["t5_attention"].shape == (4, 8, 1024, 1024)
    assert cases["t5_attention"].source_model == "T5Small"
    assert cases["xcit_attention"].shape == (4, 16, 48, 48)
    assert cases["xcit_attention"].source_model == "xcit_large_24_p8_224"
    assert cases["speech_transformer_attention"].dim == 2
    assert cases["xlnet_attention"].dim == 3
    assert cases["opt_logits"].shape == (4094, 50272)

    lm_logits_cases = [
        case for case in dataset.cases if case.family == "lm_logits"
    ]
    assert lm_logits_cases
    assert all(case.source_project == "TritonBench" for case in lm_logits_cases)
    assert all(case.source_model for case in lm_logits_cases)
    assert "cm3leon_generate" not in {case.source_model for case in dataset.cases}


def test_stress_dataset_uses_operator_specific_case_ids():
    dataset = get_softmax_dataset("stress")

    assert {
        "long_context_attention",
        "wide_vocab_lm_logits",
        "moe_router_scores",
    }.issubset({case.case_id for case in dataset.cases})


def test_synthetic_datasets_use_consistent_builtin_metadata():
    smoke = get_softmax_dataset("smoke")
    stress = get_softmax_dataset("stress")

    assert {case.source_project for case in smoke.cases} == {"cannbench"}
    assert {case.source_file for case in smoke.cases} == {"built-in"}
    assert {case.source_op for case in smoke.cases} == {"softmax"}

    assert {case.source_project for case in stress.cases} == {"cannbench"}
    assert {case.source_file for case in stress.cases} == {"generated"}
    assert {case.source_op for case in stress.cases} == {"softmax"}


def test_smoke_dataset_covers_multiple_softmax_usage_patterns():
    dataset = get_softmax_dataset("smoke")

    assert {
        "tiny_logits",
        "tiny_attention_scores",
        "tiny_channel_softmax",
    }.issubset({case.case_id for case in dataset.cases})


def test_stress_dataset_covers_multiple_softmax_pressure_patterns():
    dataset = get_softmax_dataset("stress")

    assert {
        "long_context_attention",
        "wide_vocab_lm_logits",
        "moe_router_scores",
        "small_reduction_axis",
        "vision_window_batch",
        "channelwise_activation_map",
        "beam_search_token_scores",
    }.issubset({case.case_id for case in dataset.cases})


def test_get_softmax_dataset_rejects_unknown_name():
    with pytest.raises(ValueError, match="Unknown softmax dataset"):
        get_softmax_dataset("unknown")


def test_get_softmax_case_rejects_unknown_case_id():
    with pytest.raises(ValueError, match="Unknown softmax case"):
        get_softmax_case("smoke", "missing")


def test_materialized_softmax_inputs_are_deterministic_for_same_seed():
    case = get_softmax_case("smoke", "tiny_logits")

    left = materialize_softmax_inputs(case, dtype="float16", seed=123)
    right = materialize_softmax_inputs(case, dtype="float16", seed=123)

    assert left["dim"] == right["dim"] == -1
    assert left["shape"] == right["shape"] == (32, 128)
    assert left["dtype"] == right["dtype"] == "float16"
    assert left["values"] == right["values"]


def test_materialized_softmax_inputs_change_with_different_seed():
    case = get_softmax_case("smoke", "tiny_logits")

    left = materialize_softmax_inputs(case, dtype="float16", seed=123)
    right = materialize_softmax_inputs(case, dtype="float16", seed=456)

    assert left["values"] != right["values"]


def test_get_embedding_case_preserves_realistic_source_metadata():
    case = get_embedding_case("realistic", "t5_token_embeddings")

    assert case.case_id == "t5_token_embeddings"
    assert case.embedding_dim == 512
    assert case.index_shape == (8, 512)
    assert case.source_project == "TritonBench"
    assert case.source_model == "T5Small"


def test_get_embedding_dataset_loads_builtin_splits():
    smoke = get_embedding_dataset("smoke")
    realistic = get_embedding_dataset("realistic")
    stress = get_embedding_dataset("stress")

    assert smoke.name == "smoke"
    assert realistic.cases
    assert len(stress.cases) == 3


def test_materialized_embedding_inputs_are_deterministic_for_same_seed():
    case = get_embedding_case("smoke", "tiny_token_lookup")

    left = materialize_embedding_inputs(case, dtype="float16", seed=123)
    right = materialize_embedding_inputs(case, dtype="float16", seed=123)

    assert left["dtype"] == right["dtype"] == "float16"
    assert left["index_shape"] == right["index_shape"] == (32,)
    assert left["indices"] == right["indices"]
    assert left["weights"] == right["weights"]


def test_materialized_embedding_inputs_change_with_different_seed():
    case = get_embedding_case("smoke", "tiny_token_lookup")

    left = materialize_embedding_inputs(case, dtype="float16", seed=123)
    right = materialize_embedding_inputs(case, dtype="float16", seed=456)

    assert left["indices"] != right["indices"] or left["weights"] != right["weights"]


def test_get_topk_case_preserves_tritonbench_source_metadata():
    case = get_topk_case("realistic", "vision_maskrcnn_rpn_topk_182400")

    assert case.case_id == "vision_maskrcnn_rpn_topk_182400"
    assert case.input_shape == (1, 182400)
    assert case.k == 2000
    assert case.dim == 1
    assert case.source_project == "TritonBench"
    assert case.source_model == "vision_maskrcnn"
    assert case.source_file == "torchbench_train/vision_maskrcnn_train.json"
    assert case.source_op == "aten.topk.default"


def test_topk_dataset_loads_builtin_splits():
    smoke = get_topk_dataset("smoke")
    realistic = get_topk_dataset("realistic")
    stress = get_topk_dataset("stress")

    assert smoke.name == "smoke"
    assert len(smoke.cases) == 3
    assert len(realistic.cases) >= 6
    assert len(stress.cases) >= 3


def test_materialized_topk_inputs_are_deterministic_for_same_seed():
    case = get_topk_case("smoke", "tiny_scores_top4")

    left = materialize_topk_inputs(case, dtype="float16", seed=123)
    right = materialize_topk_inputs(case, dtype="float16", seed=123)

    assert left["input_shape"] == right["input_shape"] == (2, 16)
    assert left["k"] == right["k"] == 4
    assert left["dim"] == right["dim"] == 1
    assert left["largest"] is right["largest"] is True
    assert left["sorted"] is right["sorted"] is True
    assert left["values"] == right["values"]


def test_get_lightning_indexer_case_preserves_realistic_source_metadata():
    case = get_lightning_indexer_case("realistic", "opt_prefill_2048_top512")

    assert case.case_id == "opt_prefill_2048_top512"
    assert case.batch == 2
    assert case.query_tokens == 2048
    assert case.context_tokens == 2048
    assert case.index_heads == 4
    assert case.index_dim == 64
    assert case.top_k == 512
    assert case.source_project == "TritonBench"
    assert case.source_model == "OPTForCausalLM"
    assert case.source_file == "hf_train/OPTForCausalLM_train.json"


def test_lightning_indexer_dataset_loads_builtin_splits():
    smoke = get_lightning_indexer_dataset("smoke")
    realistic = get_lightning_indexer_dataset("realistic")
    stress = get_lightning_indexer_dataset("stress")

    assert smoke.name == "smoke"
    assert len(smoke.cases) >= 5
    assert {case.source_project for case in realistic.cases} >= {"TritonBench"}
    assert len(stress.cases) >= 3


def test_materialized_lightning_indexer_inputs_are_deterministic_for_same_seed():
    case = get_lightning_indexer_case("smoke", "tiny_decode_top4")

    left = materialize_lightning_indexer_inputs(case, dtype="float16", seed=123)
    right = materialize_lightning_indexer_inputs(case, dtype="float16", seed=123)

    assert left["query_shape"] == right["query_shape"] == (2, 1, 2, 16)
    assert left["key_shape"] == right["key_shape"] == (2, 32, 16)
    assert left["weight_shape"] == right["weight_shape"] == (2, 1, 2)
    assert left["top_k"] == right["top_k"] == 4
    assert left["query"] == right["query"]
    assert left["keys"] == right["keys"]
    assert left["weights"] == right["weights"]


def test_lightning_indexer_smoke_includes_vllm_ascend_a5_case():
    case = get_lightning_indexer_case(
        "smoke", "vllm_ascend_a5_decode_b1_ctx512_top512"
    )

    assert case.batch == 1
    assert case.query_tokens == 1
    assert case.context_tokens == 512
    assert case.index_heads == 64
    assert case.index_dim == 128
    assert case.top_k == 512
    assert case.source_project == "vllm-ascend"
    assert case.source_op == "npu_vllm_quant_lightning_indexer"


def test_lightning_indexer_smoke_includes_vllm_ascend_a5_prefill_case():
    case = get_lightning_indexer_case(
        "smoke", "vllm_ascend_a5_prefill_b1_q512_ctx512_top512"
    )

    assert case.batch == 1
    assert case.query_tokens == 512
    assert case.context_tokens == 512
    assert case.index_heads == 64
    assert case.index_dim == 128
    assert case.top_k == 512
    assert case.source_project == "vllm-ascend"
    assert case.source_op == "npu_vllm_quant_lightning_indexer"


def test_lightning_indexer_realistic_splits_include_a5_cases():
    decode_case = get_lightning_indexer_case(
        "realistic_decode", "deepseek_a5_mtp3_b16_ctx1024_top1024"
    )
    prefill_case = get_lightning_indexer_case(
        "realistic_prefill", "deepseek_a5_prefill_b1_q512_ctx512_top512"
    )

    assert decode_case.family == "decode_indexing"
    assert decode_case.batch == 16
    assert decode_case.query_tokens == 3
    assert decode_case.context_tokens == 1024
    assert decode_case.index_heads == 64
    assert decode_case.index_dim == 128
    assert decode_case.top_k == 1024
    assert prefill_case.family == "prefill_indexing"
    assert prefill_case.query_tokens == 512
    assert prefill_case.index_heads == 64
    assert prefill_case.index_dim == 128
    assert prefill_case.top_k == 512


def test_lightning_indexer_realistic_splits_are_a5_fused_contract_compatible():
    for split in ("realistic_decode", "realistic_prefill"):
        dataset = get_lightning_indexer_dataset(split)

        expected_len = 20 if split == "realistic_decode" else 20
        assert len(dataset.cases) == expected_len
        if split == "realistic_decode":
            assert {case.source_kind for case in dataset.cases} == {
                "library_compatible_realistic",
                "paper_shape",
            }
            assert {case.source_project for case in dataset.cases} == {
                "vllm-ascend",
                "cannbench",
                "DeepSeek",
            }
            assert {case.source_model for case in dataset.cases} == {
                "DeepSeek-V4-compatible",
                "DeepSeek-A5-compatible",
                "DeepSeek-V3.2",
            }
            assert {case.source_op for case in dataset.cases} == {
                "npu_vllm_quant_lightning_indexer",
                "lightning_indexer",
            }
        else:
            assert {case.source_kind for case in dataset.cases} == {
                "library_compatible_realistic",
                "paper_shape",
            }
            assert {case.source_project for case in dataset.cases} == {
                "vllm-ascend",
                "DeepSeek",
            }
            assert {case.source_model for case in dataset.cases} == {
                "DeepSeek-V4-compatible",
                "DeepSeek-A5-compatible",
                "DeepSeek-V4-Pro-like",
                "DeepSeek-V3.2",
            }
            assert {case.source_op for case in dataset.cases} == {
                "npu_vllm_quant_lightning_indexer",
                "lightning_indexer",
            }
        expected_index_heads = {64, 4}
        expected_index_dim = {128, 64}
        assert all(case.index_heads in expected_index_heads for case in dataset.cases)
        assert all(case.index_dim in expected_index_dim for case in dataset.cases)
        expected_topk = {512, 1024, 2048} if split == "realistic_decode" else {512, 1024, 2048}
        assert all(case.top_k in expected_topk for case in dataset.cases)
        assert all(case.context_tokens % 128 == 0 for case in dataset.cases)


def test_lightning_indexer_realistic_prefill_dataset_contains_v4pro_and_v32_cases():
    dataset = get_lightning_indexer_dataset("realistic_prefill")
    case_ids = {case.case_id for case in dataset.cases}

    assert "deepseek_v4pro_prefill_b1_q512_ctx4096_top1024" in case_ids
    assert "deepseek_v32_prefill_b1_q128_ctx16384_top2048" in case_ids


def test_get_sparse_attention_case_preserves_realistic_source_metadata():
    case = get_sparse_attention_case("realistic", "nanogpt_prefill_64_top32")

    assert case.case_id == "nanogpt_prefill_64_top32"
    assert case.batch == 1
    assert case.query_heads == 12
    assert case.kv_heads == 12
    assert case.query_tokens == 64
    assert case.context_tokens == 64
    assert case.selected_tokens == 32
    assert case.head_dim == 64
    assert case.causal is True
    assert case.phase == "prefill"
    assert case.source_project == "TritonBench"
    assert case.source_model == "nanogpt"
    assert case.source_file == "torchbench_train/nanogpt_train.json"


def test_sparse_attention_dataset_loads_builtin_splits():
    smoke = get_sparse_attention_dataset("smoke")
    realistic = get_sparse_attention_dataset("realistic")
    stress = get_sparse_attention_dataset("stress")

    assert smoke.name == "smoke"
    assert len(smoke.cases) >= 4
    assert len(realistic.cases) >= 4
    assert len(stress.cases) >= 3


def test_sparse_attention_smoke_includes_vllm_ascend_sharedkv_case():
    case = get_sparse_attention_case("smoke", "vllm_ascend_decode_sharedkv_top64")

    assert case.batch == 1
    assert case.query_heads == 64
    assert case.kv_heads == 1
    assert case.query_tokens == 1
    assert case.context_tokens == 1024
    assert case.selected_tokens == 64
    assert case.head_dim == 512
    assert case.phase == "decode"
    assert case.source_project == "vllm-ascend"
    assert case.source_op == "npu_sparse_attn_sharedkv"


def test_sparse_attention_smoke_includes_vllm_ascend_a5_case():
    case = get_sparse_attention_case(
        "smoke", "vllm_ascend_a5_decode_b1_ctx512_top512"
    )

    assert case.batch == 1
    assert case.query_heads == 64
    assert case.kv_heads == 1
    assert case.query_tokens == 1
    assert case.context_tokens == 512
    assert case.selected_tokens == 512
    assert case.head_dim == 512
    assert case.phase == "decode"
    assert case.source_project == "vllm-ascend"
    assert case.source_op == "npu_kv_quant_sparse_attn_sharedkv"


def test_sparse_attention_smoke_includes_vllm_ascend_a5_prefill_case():
    case = get_sparse_attention_case(
        "smoke", "vllm_ascend_a5_prefill_b1_q512_ctx512_top512"
    )

    assert case.batch == 1
    assert case.query_heads == 64
    assert case.kv_heads == 1
    assert case.query_tokens == 512
    assert case.context_tokens == 512
    assert case.selected_tokens == 512
    assert case.head_dim == 512
    assert case.phase == "prefill"
    assert case.source_project == "vllm-ascend"
    assert case.source_op == "npu_kv_quant_sparse_attn_sharedkv"


def test_sparse_attention_realistic_splits_include_a5_cases():
    decode_case = get_sparse_attention_case(
        "realistic_decode", "deepseek_a5_mtp3_b16_ctx1024_top1024"
    )
    prefill_case = get_sparse_attention_case(
        "realistic_prefill", "deepseek_a5_prefill_b1_q512_ctx512_top512"
    )

    assert decode_case.phase == "decode"
    assert decode_case.batch == 16
    assert decode_case.query_tokens == 3
    assert decode_case.context_tokens == 1024
    assert decode_case.query_heads == 64
    assert decode_case.kv_heads == 1
    assert decode_case.selected_tokens == 1024
    assert decode_case.head_dim == 512
    assert prefill_case.phase == "prefill"
    assert prefill_case.query_tokens == 512
    assert prefill_case.query_heads == 64
    assert prefill_case.kv_heads == 1
    assert prefill_case.selected_tokens == 512
    assert prefill_case.head_dim == 512


def test_sparse_attention_realistic_splits_are_a5_fused_contract_compatible():
    for split, phase in (
        ("realistic_decode", "decode"),
        ("realistic_prefill", "prefill"),
    ):
        dataset = get_sparse_attention_dataset(split)

        expected_len = 20 if split == "realistic_decode" else 20
        assert len(dataset.cases) == expected_len
        if split == "realistic_decode":
            assert {case.source_kind for case in dataset.cases} == {
                "library_compatible_realistic",
                "paper_shape",
            }
            assert {case.source_project for case in dataset.cases} == {
                "vllm-ascend",
                "cannbench",
                "DeepSeek",
            }
            assert {case.source_model for case in dataset.cases} == {
                "DeepSeek-V4-compatible",
                "DeepSeek-A5-compatible",
                "DeepSeek-V3.2",
            }
            assert {case.source_op for case in dataset.cases} == {
                "npu_kv_quant_sparse_attn_sharedkv",
                "sparse_attention",
            }
        else:
            assert {case.source_kind for case in dataset.cases} == {
                "library_compatible_realistic",
                "paper_shape",
            }
            assert {case.source_project for case in dataset.cases} == {
                "vllm-ascend",
                "DeepSeek",
            }
            assert {case.source_model for case in dataset.cases} == {
                "DeepSeek-V4-compatible",
                "DeepSeek-A5-compatible",
                "DeepSeek-V4-Pro-like",
                "DeepSeek-V3.2",
            }
            assert {case.source_op for case in dataset.cases} == {
                "npu_kv_quant_sparse_attn_sharedkv",
                "sparse_attention",
            }
        assert all(case.phase == phase for case in dataset.cases)
        expected_query_heads = {64, 128} if split == "realistic_decode" else {64, 128}
        expected_kv_heads = {1}
        expected_head_dim = {512, 128} if split == "realistic_decode" else {512, 128}
        assert all(case.query_heads in expected_query_heads for case in dataset.cases)
        assert all(case.kv_heads in expected_kv_heads for case in dataset.cases)
        assert all(case.head_dim in expected_head_dim for case in dataset.cases)
        expected_selected = {512, 1024, 2048} if split == "realistic_decode" else {512, 1024, 2048}
        assert all(case.selected_tokens in expected_selected for case in dataset.cases)
        assert all(case.context_tokens % 128 == 0 for case in dataset.cases)


def test_sparse_attention_realistic_prefill_dataset_contains_v4pro_and_v32_cases():
    dataset = get_sparse_attention_dataset("realistic_prefill")
    case_ids = {case.case_id for case in dataset.cases}

    assert "deepseek_v4pro_prefill_b1_q512_ctx4096_top1024" in case_ids
    assert "deepseek_v32_prefill_b1_q128_ctx16384_top2048" in case_ids


def test_materialized_sparse_attention_inputs_are_deterministic_for_same_seed():
    case = get_sparse_attention_case("smoke", "tiny_decode_top4")

    left = materialize_sparse_attention_inputs(case, dtype="float16", seed=123)
    right = materialize_sparse_attention_inputs(case, dtype="float16", seed=123)

    assert left["query_shape"] == right["query_shape"] == (2, 2, 1, 16)
    assert left["key_shape"] == right["key_shape"] == (2, 2, 32, 16)
    assert left["value_shape"] == right["value_shape"] == (2, 2, 32, 16)
    assert left["indices_shape"] == right["indices_shape"] == (2, 1, 4)
    assert left["query"] == right["query"]
    assert left["keys"] == right["keys"]
    assert left["values"] == right["values"]
    assert left["indices"] == right["indices"]


def test_get_gather_case_preserves_source_metadata():
    case = get_gather_case("realistic", "t5_attention_probs")

    assert case.case_id == "t5_attention_probs"
    assert case.input_shape == (4, 8, 1024, 1024)
    assert case.index_shape == (4, 8, 1024, 1024)
    assert case.dim == -1
    assert case.source_model == "T5Small"


def test_get_gather_dataset_loads_builtin_splits():
    smoke = get_gather_dataset("smoke")
    realistic = get_gather_dataset("realistic")
    stress = get_gather_dataset("stress")

    assert smoke.name == "smoke"
    assert realistic.cases
    assert len(stress.cases) == 2


def test_materialized_gather_inputs_are_deterministic_for_same_seed():
    case = get_gather_case("smoke", "tiny_rank2_gather")

    left = materialize_gather_inputs(case, dtype="float16", seed=123)
    right = materialize_gather_inputs(case, dtype="float16", seed=123)

    assert left["input_shape"] == right["input_shape"] == (32, 64)
    assert left["index_shape"] == right["index_shape"] == (32, 32)
    assert left["indices"] == right["indices"]
    assert left["values"] == right["values"]


def test_materialized_gather_inputs_change_with_different_seed():
    case = get_gather_case("smoke", "tiny_rank2_gather")

    left = materialize_gather_inputs(case, dtype="float16", seed=123)
    right = materialize_gather_inputs(case, dtype="float16", seed=456)

    assert left["indices"] != right["indices"] or left["values"] != right["values"]


def test_get_index_select_case_preserves_source_metadata():
    case = get_index_select_case("realistic", "bert_hidden_token_select")

    assert case.case_id == "bert_hidden_token_select"
    assert case.input_shape == (16, 128, 768)
    assert case.index_shape == (128,)
    assert case.dim == 1
    assert case.source_model == "BERT_pytorch"


def test_get_index_select_dataset_loads_builtin_splits():
    smoke = get_index_select_dataset("smoke")
    realistic = get_index_select_dataset("realistic")
    stress = get_index_select_dataset("stress")

    assert smoke.name == "smoke"
    assert realistic.cases
    assert stress.cases


def test_materialized_index_select_inputs_are_deterministic_for_same_seed():
    case = get_index_select_case("smoke", "tiny_rank2_index_select")

    left = materialize_index_select_inputs(case, dtype="float16", seed=123)
    right = materialize_index_select_inputs(case, dtype="float16", seed=123)

    assert left["input_shape"] == right["input_shape"] == (32, 64)
    assert left["index_shape"] == right["index_shape"] == (16,)
    assert left["indices"] == right["indices"]
    assert left["values"] == right["values"]


def test_materialized_index_select_inputs_change_with_different_seed():
    case = get_index_select_case("smoke", "tiny_rank2_index_select")

    left = materialize_index_select_inputs(case, dtype="float16", seed=123)
    right = materialize_index_select_inputs(case, dtype="float16", seed=456)

    assert left["indices"] != right["indices"] or left["values"] != right["values"]


def test_get_take_along_dim_case_preserves_source_metadata():
    case = get_take_along_dim_case("realistic", "t5_attention_topk_values")

    assert case.case_id == "t5_attention_topk_values"
    assert case.input_shape == (4, 8, 1024, 1024)
    assert case.index_shape == (4, 8, 1024, 64)
    assert case.dim == -1
    assert case.source_project == "TritonBench"
    assert case.source_model == "T5Small"
    assert case.source_op == "torch.take_along_dim"


def test_get_take_along_dim_dataset_loads_builtin_splits():
    smoke = get_take_along_dim_dataset("smoke")
    realistic = get_take_along_dim_dataset("realistic")
    stress = get_take_along_dim_dataset("stress")

    assert smoke.name == "smoke"
    assert len(smoke.cases) == 3
    assert len(realistic.cases) >= 3
    assert len(stress.cases) >= 3


def test_materialized_take_along_dim_inputs_are_deterministic_for_same_seed():
    case = get_take_along_dim_case("smoke", "tiny_rank2_take_along_dim")

    left = materialize_take_along_dim_inputs(case, dtype="float16", seed=123)
    right = materialize_take_along_dim_inputs(case, dtype="float16", seed=123)

    assert left["input_shape"] == right["input_shape"] == (32, 64)
    assert left["index_shape"] == right["index_shape"] == (32, 16)
    assert left["indices"] == right["indices"]
    assert left["values"] == right["values"]


def test_materialized_take_along_dim_inputs_change_with_different_seed():
    case = get_take_along_dim_case("smoke", "tiny_rank2_take_along_dim")

    left = materialize_take_along_dim_inputs(case, dtype="float16", seed=123)
    right = materialize_take_along_dim_inputs(case, dtype="float16", seed=456)

    assert left["indices"] != right["indices"] or left["values"] != right["values"]


def test_get_masked_select_case_preserves_source_metadata():
    case = get_masked_select_case("realistic", "bert_attention_masked_scores")

    assert case.case_id == "bert_attention_masked_scores"
    assert case.input_shape == (16, 12, 128, 128)
    assert case.mask_shape == (16, 12, 128, 128)
    assert case.mask_density == 0.75
    assert case.source_project == "TritonBench"
    assert case.source_model == "BERT_pytorch"
    assert case.source_op == "torch.masked_select"


def test_get_masked_select_dataset_loads_builtin_splits():
    smoke = get_masked_select_dataset("smoke")
    realistic = get_masked_select_dataset("realistic")
    stress = get_masked_select_dataset("stress")

    assert smoke.name == "smoke"
    assert len(smoke.cases) == 3
    assert len(realistic.cases) >= 3
    assert len(stress.cases) >= 3


def test_materialized_masked_select_inputs_are_deterministic_for_same_seed():
    case = get_masked_select_case("smoke", "tiny_rank2_masked_select")

    left = materialize_masked_select_inputs(case, dtype="float16", seed=123)
    right = materialize_masked_select_inputs(case, dtype="float16", seed=123)

    assert left["input_shape"] == right["input_shape"] == (32, 64)
    assert left["mask_shape"] == right["mask_shape"] == (32, 64)
    assert left["mask"] == right["mask"]
    assert left["values"] == right["values"]


def test_materialized_masked_select_inputs_change_with_different_seed():
    case = get_masked_select_case("smoke", "tiny_rank2_masked_select")

    left = materialize_masked_select_inputs(case, dtype="float16", seed=123)
    right = materialize_masked_select_inputs(case, dtype="float16", seed=456)

    assert left["mask"] != right["mask"] or left["values"] != right["values"]


def test_get_cross_entropy_case_preserves_source_metadata():
    case = get_cross_entropy_case("realistic", "bert_token_classification_loss")

    assert case.case_id == "bert_token_classification_loss"
    assert case.logits_shape == (16, 128, 30522)
    assert case.target_shape == (16, 128)
    assert case.num_classes == 30522
    assert case.source_project == "TritonBench"
    assert case.source_model == "BERT_pytorch"
    assert case.source_op == "torch.nn.functional.cross_entropy"


def test_get_cross_entropy_dataset_loads_builtin_splits():
    smoke = get_cross_entropy_dataset("smoke")
    realistic = get_cross_entropy_dataset("realistic")
    stress = get_cross_entropy_dataset("stress")

    assert smoke.name == "smoke"
    assert len(smoke.cases) == 3
    assert len(realistic.cases) >= 3
    assert len(stress.cases) >= 3


def test_materialized_cross_entropy_inputs_are_deterministic_for_same_seed():
    case = get_cross_entropy_case("smoke", "tiny_token_classification_loss")

    left = materialize_cross_entropy_inputs(case, dtype="float16", seed=123)
    right = materialize_cross_entropy_inputs(case, dtype="float16", seed=123)

    assert left["logits_shape"] == right["logits_shape"] == (32, 128, 64)
    assert left["target_shape"] == right["target_shape"] == (32, 128)
    assert left["targets"] == right["targets"]
    assert left["logits"] == right["logits"]


def test_materialized_cross_entropy_inputs_change_with_different_seed():
    case = get_cross_entropy_case("smoke", "tiny_token_classification_loss")

    left = materialize_cross_entropy_inputs(case, dtype="float16", seed=123)
    right = materialize_cross_entropy_inputs(case, dtype="float16", seed=456)

    assert left["targets"] != right["targets"] or left["logits"] != right["logits"]


def test_get_scatter_add_case_preserves_source_metadata():
    case = get_scatter_add_case("realistic", "bert_token_scatter_add")

    assert case.case_id == "bert_token_scatter_add"
    assert case.input_shape == (16, 128, 30522)
    assert case.index_shape == (16, 128, 30522)
    assert case.src_shape == (16, 128, 30522)
    assert case.dim == -1
    assert case.source_project == "TritonBench"
    assert case.source_model == "BERT_pytorch"
    assert case.source_op == "torch.scatter_add"


def test_get_scatter_add_dataset_loads_builtin_splits():
    smoke = get_scatter_add_dataset("smoke")
    realistic = get_scatter_add_dataset("realistic")
    stress = get_scatter_add_dataset("stress")

    assert smoke.name == "smoke"
    assert len(smoke.cases) == 3
    assert len(realistic.cases) >= 3
    assert len(stress.cases) >= 3


def test_materialized_scatter_add_inputs_are_deterministic_for_same_seed():
    case = get_scatter_add_case("smoke", "tiny_rank2_scatter_add")

    left = materialize_scatter_add_inputs(case, dtype="float16", seed=123)
    right = materialize_scatter_add_inputs(case, dtype="float16", seed=123)

    assert left["input_shape"] == right["input_shape"] == (32, 64)
    assert left["index_shape"] == right["index_shape"] == (32, 64)
    assert left["src_shape"] == right["src_shape"] == (32, 64)
    assert left["indices"] == right["indices"]
    assert left["values"] == right["values"]
    assert left["src"] == right["src"]


def test_materialized_scatter_add_inputs_change_with_different_seed():
    case = get_scatter_add_case("smoke", "tiny_rank2_scatter_add")

    left = materialize_scatter_add_inputs(case, dtype="float16", seed=123)
    right = materialize_scatter_add_inputs(case, dtype="float16", seed=456)

    assert left["indices"] != right["indices"] or left["src"] != right["src"] or left["values"] != right["values"]


def test_get_index_add_case_preserves_source_metadata():
    case = get_index_add_case("realistic", "bert_hidden_index_add")

    assert case.case_id == "bert_hidden_index_add"
    assert case.input_shape == (16, 128, 768)
    assert case.index_shape == (128,)
    assert case.src_shape == (16, 128, 768)
    assert case.dim == 1
    assert case.source_project == "TritonBench"
    assert case.source_model == "BERT_pytorch"
    assert case.source_op == "torch.index_add"


def test_get_index_add_dataset_loads_builtin_splits():
    smoke = get_index_add_dataset("smoke")
    realistic = get_index_add_dataset("realistic")
    stress = get_index_add_dataset("stress")

    assert smoke.name == "smoke"
    assert len(smoke.cases) == 3
    assert len(realistic.cases) >= 3
    assert len(stress.cases) >= 3


def test_materialized_index_add_inputs_are_deterministic_for_same_seed():
    case = get_index_add_case("smoke", "tiny_rank2_index_add")

    left = materialize_index_add_inputs(case, dtype="float16", seed=123)
    right = materialize_index_add_inputs(case, dtype="float16", seed=123)

    assert left["input_shape"] == right["input_shape"] == (32, 64)
    assert left["index_shape"] == right["index_shape"] == (16,)
    assert left["src_shape"] == right["src_shape"] == (32, 16)
    assert left["indices"] == right["indices"]
    assert left["values"] == right["values"]
    assert left["src"] == right["src"]


def test_materialized_index_add_inputs_change_with_different_seed():
    case = get_index_add_case("smoke", "tiny_rank2_index_add")

    left = materialize_index_add_inputs(case, dtype="float16", seed=123)
    right = materialize_index_add_inputs(case, dtype="float16", seed=456)

    assert left["indices"] != right["indices"] or left["src"] != right["src"] or left["values"] != right["values"]


def test_get_scatter_case_preserves_source_metadata():
    case = get_scatter_case("realistic", "bert_token_scatter")

    assert case.case_id == "bert_token_scatter"
    assert case.input_shape == (16, 128, 30522)
    assert case.index_shape == (16, 128, 30522)
    assert case.src_shape == (16, 128, 30522)
    assert case.dim == -1
    assert case.source_project == "TritonBench"
    assert case.source_model == "BERT_pytorch"
    assert case.source_op == "torch.scatter"


def test_get_scatter_dataset_loads_builtin_splits():
    smoke = get_scatter_dataset("smoke")
    realistic = get_scatter_dataset("realistic")
    stress = get_scatter_dataset("stress")

    assert smoke.name == "smoke"
    assert len(smoke.cases) == 3
    assert len(realistic.cases) >= 3
    assert len(stress.cases) >= 3


def test_materialized_scatter_inputs_are_deterministic_for_same_seed():
    case = get_scatter_case("smoke", "tiny_rank2_scatter")

    left = materialize_scatter_inputs(case, dtype="float16", seed=123)
    right = materialize_scatter_inputs(case, dtype="float16", seed=123)

    assert left["input_shape"] == right["input_shape"] == (32, 64)
    assert left["index_shape"] == right["index_shape"] == (32, 64)
    assert left["src_shape"] == right["src_shape"] == (32, 64)
    assert left["indices"] == right["indices"]
    assert left["values"] == right["values"]
    assert left["src"] == right["src"]


def test_materialized_scatter_inputs_change_with_different_seed():
    case = get_scatter_case("smoke", "tiny_rank2_scatter")

    left = materialize_scatter_inputs(case, dtype="float16", seed=123)
    right = materialize_scatter_inputs(case, dtype="float16", seed=456)

    assert left["indices"] != right["indices"] or left["src"] != right["src"] or left["values"] != right["values"]


def test_get_index_put_case_preserves_source_metadata():
    case = get_index_put_case("realistic", "bert_hidden_index_put")

    assert case.case_id == "bert_hidden_index_put"
    assert case.input_shape == (16, 128, 768)
    assert case.index_shapes == ((16, 128), (16, 128))
    assert case.values_shape == (16, 128, 768)
    assert case.accumulate is False
    assert case.source_project == "TritonBench"
    assert case.source_model == "BERT_pytorch"
    assert case.source_op == "torch.index_put"


def test_get_index_put_dataset_loads_builtin_splits():
    smoke = get_index_put_dataset("smoke")
    realistic = get_index_put_dataset("realistic")
    stress = get_index_put_dataset("stress")

    assert smoke.name == "smoke"
    assert len(smoke.cases) == 3
    assert len(realistic.cases) >= 3
    assert len(stress.cases) >= 3


def test_materialized_index_put_inputs_are_deterministic_for_same_seed():
    case = get_index_put_case("smoke", "tiny_rank2_index_put")

    left = materialize_index_put_inputs(case, dtype="float16", seed=123)
    right = materialize_index_put_inputs(case, dtype="float16", seed=123)

    assert left["input_shape"] == right["input_shape"] == (32, 64)
    assert left["index_shapes"] == right["index_shapes"] == ((16,), (16,))
    assert left["values_shape"] == right["values_shape"] == (16,)
    assert left["accumulate"] is right["accumulate"] is False
    assert left["values"] == right["values"]
    assert left["indices"] == right["indices"]
    assert left["put_values"] == right["put_values"]


def test_materialized_index_put_inputs_change_with_different_seed():
    case = get_index_put_case("smoke", "tiny_rank2_index_put")

    left = materialize_index_put_inputs(case, dtype="float16", seed=123)
    right = materialize_index_put_inputs(case, dtype="float16", seed=456)

    assert (
        left["indices"] != right["indices"]
        or left["put_values"] != right["put_values"]
        or left["values"] != right["values"]
    )
