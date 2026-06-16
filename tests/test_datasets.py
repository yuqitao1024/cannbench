import pytest

from cannbench.datasets import get_softmax_case, get_softmax_dataset
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
