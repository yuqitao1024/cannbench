import pytest

from cannbench.datasets.dsa_workflow import (
    build_dsa_inference_workflow,
    get_dsa_inference_workflow_dataset,
    list_dsa_inference_workflows,
)


def test_build_decode_workflow_uses_indexer_then_sparse_decode():
    workflow = build_dsa_inference_workflow(
        dataset="smoke",
        case_id="tiny_decode_top4",
        dtype="float16",
        seed=7,
    )

    assert workflow.phase == "decode"
    assert workflow.dataset == "smoke"
    assert workflow.case_id == "tiny_decode_top4"
    assert [step.contract for step in workflow.steps] == [
        "dsa_index_select",
        "sparse_mla_decode",
    ]
    assert [step.op for step in workflow.steps] == [
        "lightning_indexer",
        "sparse_attention",
    ]
    assert workflow.steps[0].produces == ("indices",)
    assert workflow.steps[1].consumes == ("indices",)
    assert workflow.steps[0].prepared.op == "lightning_indexer"
    assert workflow.steps[1].prepared.op == "sparse_attention"
    assert workflow.steps[1].prepared.case.payload["phase"] == "decode"


def test_build_prefill_workflow_uses_indexer_then_sparse_prefill():
    workflow = build_dsa_inference_workflow(
        dataset="smoke",
        case_id="tiny_prefill_top8",
        dtype="bfloat16",
        seed=11,
    )

    assert workflow.phase == "prefill"
    assert [step.contract for step in workflow.steps] == [
        "dsa_index_select",
        "sparse_mla_prefill",
    ]
    assert all(step.prepared.dtype == "bfloat16" for step in workflow.steps)
    assert all(step.prepared.seed == 11 for step in workflow.steps)
    assert workflow.steps[1].prepared.case.payload["phase"] == "prefill"


def test_list_dsa_workflows_filters_to_cases_with_matching_component_cases():
    decode_workflows = list_dsa_inference_workflows("smoke", phase="decode")
    prefill_workflows = list_dsa_inference_workflows("smoke", phase="prefill")

    assert [workflow.case_id for workflow in decode_workflows] == ["tiny_decode_top4"]
    assert [workflow.case_id for workflow in prefill_workflows] == ["tiny_prefill_top8"]


def test_dsa_workflow_dataset_is_the_case_selection_source():
    dataset = get_dsa_inference_workflow_dataset("smoke")

    assert dataset.name == "smoke"
    assert [case.case_id for case in dataset.cases] == [
        "tiny_decode_top4",
        "tiny_prefill_top8",
    ]
    assert [case.workflow for case in dataset.cases] == [
        "dsa_decode",
        "dsa_prefill",
    ]


def test_realistic_workflow_datasets_are_split_by_inference_phase():
    decode_workflows = list_dsa_inference_workflows("realistic_decode", phase="decode")
    prefill_workflows = list_dsa_inference_workflows("realistic_prefill", phase="prefill")

    decode_case_ids = [workflow.case_id for workflow in decode_workflows]
    prefill_case_ids = [workflow.case_id for workflow in prefill_workflows]

    assert len(decode_case_ids) == 16
    assert len(prefill_case_ids) == 15
    assert {
        "deepseek_decode_b1_ctx4096_top512",
        "deepseek_decode_b8_ctx32768_top2048",
        "deepseek_decode_b1_ctx65536_top2048",
        "llama4_decode_32760_top2048",
    }.issubset(decode_case_ids)
    assert {
        "clip_text_prefill_50_top12",
        "distilbert_prefill_128_top64",
        "gpt2_prefill_512_top128",
        "opt_prefill_2048_top512",
        "longformer_prefill_4096_top512",
    }.issubset(prefill_case_ids)
    assert list_dsa_inference_workflows("realistic_decode", phase="prefill") == ()
    assert list_dsa_inference_workflows("realistic_prefill", phase="decode") == ()


def test_build_workflow_rejects_case_outside_workflow_manifest():
    with pytest.raises(ValueError, match="Unknown DSA inference workflow case"):
        build_dsa_inference_workflow(
            dataset="smoke",
            case_id="tiny_mqa_decode_top8",
            dtype="float16",
            seed=0,
        )


def test_list_dsa_workflows_rejects_unknown_phase():
    with pytest.raises(ValueError, match="phase must be decode or prefill"):
        list_dsa_inference_workflows("smoke", phase="training")


def test_dsa_workflow_api_is_exported_from_datasets_package():
    from cannbench.datasets import build_dsa_inference_workflow as exported_builder
    from cannbench.datasets import (
        get_dsa_inference_workflow_dataset as exported_dataset_loader,
    )

    workflow = exported_builder(
        dataset="smoke",
        case_id="tiny_decode_top4",
        dtype="float16",
        seed=0,
    )

    assert workflow.steps[0].contract == "dsa_index_select"
    assert exported_dataset_loader("smoke").cases[0].workflow == "dsa_decode"
