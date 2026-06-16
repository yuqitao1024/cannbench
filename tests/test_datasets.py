import pytest

from cannbench.datasets import (
    get_embedding_case,
    get_embedding_dataset,
    get_gather_case,
    get_gather_dataset,
    get_index_select_case,
    get_index_select_dataset,
    get_softmax_case,
    get_softmax_dataset,
    get_take_along_dim_case,
    get_take_along_dim_dataset,
)
from cannbench.datasets.materialize import (
    materialize_embedding_inputs,
    materialize_gather_inputs,
    materialize_index_select_inputs,
    materialize_softmax_inputs,
    materialize_take_along_dim_inputs,
)
from cannbench.datasets.synthetic import (
    build_softmax_smoke_case,
    build_softmax_stress_case,
)


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


def test_build_softmax_smoke_case_applies_shared_metadata():
    case = build_softmax_smoke_case(
        case_id="tiny_attention_scores",
        family="attention",
        shape=(2, 4, 8, 8),
        dim=-1,
        source_model="smoke_attention_fixture",
    )

    assert case.case_id == "tiny_attention_scores"
    assert case.shape == (2, 4, 8, 8)
    assert case.source_kind == "synthetic_smoke"
    assert case.source_project == "cannbench"
    assert case.source_file == "built-in"
    assert case.source_op == "softmax"


def test_build_softmax_stress_case_applies_shared_metadata():
    case = build_softmax_stress_case(
        case_id="small_reduction_axis",
        family="reduction_edge",
        shape=(16384, 2),
        dim=-1,
        source_model="softmax_small_axis_boundary",
    )

    assert case.case_id == "small_reduction_axis"
    assert case.shape == (16384, 2)
    assert case.source_kind == "synthetic_boundary"
    assert case.source_project == "cannbench"
    assert case.source_file == "generated"
    assert case.source_op == "softmax"


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
