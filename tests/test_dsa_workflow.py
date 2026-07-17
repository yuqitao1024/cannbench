import pytest

from cannbench.operators.builtin.dsa_decode import (
    build_dsa_decode_workflow,
    get_dsa_decode_dataset,
    list_dsa_decode_workflows,
)
from cannbench.operators.builtin.dsa_prefill import (
    build_dsa_prefill_workflow,
    get_dsa_prefill_dataset,
    list_dsa_prefill_workflows,
)


def test_build_decode_workflow_uses_indexer_then_sparse_decode():
    workflow = build_dsa_decode_workflow(
        dataset="stress",
        case_id="vllm_ascend_a5_decode_b1_ctx512_top512",
        dtype="bfloat16",
        seed=7,
    )

    assert workflow.phase == "decode"
    assert workflow.dataset == "stress"
    assert workflow.case_id == "vllm_ascend_a5_decode_b1_ctx512_top512"
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
    assert workflow.steps[0].prepared.case.payload["phase"] == "decode"
    assert workflow.steps[1].prepared.case.payload["phase"] == "decode"


def test_build_prefill_workflow_uses_indexer_then_sparse_prefill():
    workflow = build_dsa_prefill_workflow(
        dataset="smoke",
        case_id="vllm_ascend_a5_prefill_b1_q512_ctx512_top512",
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
    assert workflow.steps[0].prepared.case.payload["phase"] == "prefill"
    assert workflow.steps[1].prepared.case.payload["phase"] == "prefill"


def test_dsa_prefill_components_support_simt_ready_shapes():
    workflow = build_dsa_prefill_workflow(
        dataset="stress",
        case_id="deepseek_v4pro_prefill_b1_q512_ctx4096_top1024",
        dtype="float16",
        seed=0,
    )

    assert tuple(step.op for step in workflow.steps) == (
        "lightning_indexer",
        "sparse_attention",
    )


def test_dsa_decode_components_support_simt_ready_shapes():
    workflow = build_dsa_decode_workflow(
        dataset="realistic",
        case_id="deepseek_128k_decode_top2048",
        dtype="float16",
        seed=0,
    )

    assert tuple(step.op for step in workflow.steps) == (
        "lightning_indexer",
        "sparse_attention",
    )


def test_list_dsa_workflows_filters_to_cases_with_matching_component_cases():
    decode_workflows = list_dsa_decode_workflows("realistic")
    prefill_workflows = list_dsa_prefill_workflows("smoke")

    assert [workflow.case_id for workflow in decode_workflows] == [
        "deepseek_128k_decode_top2048",
        "deepseek_v32_flashmla_decode_b2_q2_ctx32768_top2048",
        "deepseek_v4_flash_vllm_decode_b16_q1_ctx32768_top512",
        "glm52_vllm_ascend_decode_b3_q3_ctx131072_top2048",
    ]
    assert [workflow.case_id for workflow in prefill_workflows] == [
        "vllm_ascend_a5_prefill_b1_q512_ctx512_top512"
    ]


def test_dsa_fused_operator_datasets_are_phase_specific_case_selection_sources():
    decode_dataset = get_dsa_decode_dataset("realistic")
    prefill_dataset = get_dsa_prefill_dataset("smoke")

    assert decode_dataset.name == "realistic"
    assert prefill_dataset.name == "smoke"
    assert [case.case_id for case in decode_dataset.cases] == [
        "deepseek_128k_decode_top2048",
        "deepseek_v32_flashmla_decode_b2_q2_ctx32768_top2048",
        "deepseek_v4_flash_vllm_decode_b16_q1_ctx32768_top512",
        "glm52_vllm_ascend_decode_b3_q3_ctx131072_top2048",
    ]
    assert [case.case_id for case in prefill_dataset.cases] == [
        "vllm_ascend_a5_prefill_b1_q512_ctx512_top512",
    ]
    assert all(case.workflow == "dsa_decode" for case in decode_dataset.cases)
    assert [case.workflow for case in prefill_dataset.cases] == ["dsa_prefill"]


def test_realistic_workflow_datasets_are_split_by_fused_operator():
    decode_workflows = list_dsa_decode_workflows("realistic")
    prefill_workflows = list_dsa_prefill_workflows("realistic")

    decode_case_ids = [workflow.case_id for workflow in decode_workflows]
    prefill_case_ids = [workflow.case_id for workflow in prefill_workflows]

    assert len(decode_case_ids) == 4
    assert len(prefill_case_ids) == 9
    assert decode_case_ids == [
        "deepseek_128k_decode_top2048",
        "deepseek_v32_flashmla_decode_b2_q2_ctx32768_top2048",
        "deepseek_v4_flash_vllm_decode_b16_q1_ctx32768_top512",
        "glm52_vllm_ascend_decode_b3_q3_ctx131072_top2048",
    ]
    assert prefill_case_ids == [
        "deepseek_v32_prefill_b1_q128_ctx16384_top2048",
        "deepseek_v32_prefill_b1_q128_ctx32768_top2048",
        "deepseek_v32_prefill_b1_q128_ctx65536_top2048",
        "deepseek_v32_prefill_b1_q128_ctx131072_top2048",
        "deepseek_v32_prefill_b2_q128_ctx65536_top2048",
        "deepseek_128k_prefill_microbatch_top2048",
        "deepseek_v32_flashmla_prefill_q4096_ctx32768_top2048",
        "deepseek_v4_flash_flashmla_prefill_q4096_ctx32768_top512",
        "glm52_vllm_ascend_prefill_q4096_ctx131072_top2048",
    ]
    assert all(workflow.phase == "decode" for workflow in decode_workflows)
    assert all(workflow.phase == "prefill" for workflow in prefill_workflows)


def test_build_workflow_rejects_case_outside_workflow_manifest():
    with pytest.raises(ValueError, match="Unknown DSA decode case"):
        build_dsa_decode_workflow(
            dataset="realistic",
            case_id="tiny_mqa_decode_top8",
            dtype="float16",
            seed=0,
        )


def test_decode_smoke_is_empty_and_stress_contains_moved_cases():
    assert list_dsa_decode_workflows("smoke") == ()
    workflows = list_dsa_decode_workflows("stress")

    assert len(workflows) == 19
    assert workflows[0].case_id == "vllm_ascend_a5_decode_b1_ctx512_top512"
    assert workflows[-1].case_id == "deepseek_a5_mtp3_b64_ctx262144_top1024"


def test_prefill_stress_dataset_contains_moved_cases():
    workflows = list_dsa_prefill_workflows("stress")

    case_ids = [workflow.case_id for workflow in workflows]

    assert len(case_ids) == 15
    assert "deepseek_128k_prefill_microbatch_top2048" in case_ids
    assert "deepseek_a5_prefill_b1_q512_ctx512_top512" in case_ids
    assert "deepseek_v4pro_prefill_b8_q128_ctx16384_top1024" in case_ids


def test_dsa_workflow_api_is_exported_from_operator_packages():
    from cannbench.operators.builtin.dsa_decode import (
        build_dsa_decode_workflow as exported_decode_builder,
    )
    from cannbench.operators.builtin.dsa_prefill import (
        get_dsa_prefill_dataset as exported_prefill_dataset_loader,
    )

    workflow = exported_decode_builder(
        dataset="stress",
        case_id="vllm_ascend_a5_decode_b1_ctx512_top512",
        dtype="bfloat16",
        seed=0,
    )

    assert workflow.steps[0].contract == "dsa_index_select"
    assert exported_prefill_dataset_loader("smoke").cases[0].workflow == "dsa_prefill"
