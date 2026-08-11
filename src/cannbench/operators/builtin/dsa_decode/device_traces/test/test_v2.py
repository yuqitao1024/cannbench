from pathlib import Path

from cannbench.operators.builtin.dsa_decode.device_traces.v2 import (
    build_v2_device_trace,
)
from cannbench.operators.builtin.lightning_indexer.cases import (
    get_lightning_indexer_case,
)
from cannbench.operators.builtin.sparse_attention.cases import (
    get_sparse_attention_case,
)

CASE_ID = "deepseek_v32_flashmla_decode_b2_q2_ctx32768_top2048"


def test_v2_device_trace_matches_current_task_and_tile_layout():
    indexer = get_lightning_indexer_case("realistic_decode", CASE_ID)
    sparse = get_sparse_attention_case("realistic_decode", CASE_ID)
    trace = build_v2_device_trace(indexer, sparse)
    assert trace.status == "available"
    assert trace.version == "v2"
    index_kernel, attention_kernel = trace.kernels
    assert (index_kernel.task_count, index_kernel.used_core_count) == (32, 32)
    assert "2 x 1 x 16" in index_kernel.task_formula
    assert [axis.value for axis in index_kernel.tile_tensors[1].axes] == [32, 128]
    assert (attention_kernel.task_count, attention_kernel.used_core_count) == (8, 8)
    assert attention_kernel.summary == "Head64, P=1 direct output, no Combine"
    assert [
        [axis.value for axis in tensor.axes]
        for tensor in attention_kernel.tile_tensors
    ] == [
        [64, 256],
        [256, 128],
        [64, 128],
    ]


def test_v2_descriptor_assumptions_are_present_in_current_sources():
    builtin_root = Path(__file__).resolve().parents[3]
    index_source = (
        builtin_root
        / "lightning_indexer/simt/v2/aten_dsa_lightning_indexer_v2/csrc/lightning_indexer.asc"
    ).read_text()
    sparse_source = (
        builtin_root
        / "sparse_attention/simt/v2/aten_dsa_sparse_attention_v2/csrc/sparse_attention.asc"
    ).read_text()
    assert "const int32_t shard_counts[] = {16, 8, 4, 2, 1};" in index_source
    assert "auto_head64_decode ? 1" in sparse_source

    plan_builder = sparse_source.split(
        "SparseAttentionHead64Plan make_sparse_attention_head64_plan(", 1
    )[1].split('extern "C" void', 1)[0]
    assert "query.size(0) * query.size(2) * head_group_count * selected_partitions" in plan_builder
    assert "kHead64SelectedTile" in plan_builder
    assert "selected_partitions == 1\n      ? kHead64OutputDirectBfloat16" in plan_builder

    plan_source = (
        builtin_root
        / "sparse_attention/simt/v2/aten_dsa_sparse_attention_v2/csrc/simt/sparse_attention_head64_plan.h"
    ).read_text()
    assert "kHead64Tile = 64" in plan_source
    assert "kHead64SelectedTile = 64" in plan_source

    direct_output = sparse_source.split(
        "sparse_attention_forward_family_hd576_head64(", 1
    )[1].split("if (plan.selected_partitions == 4)", 1)[0]
    assert "plan.output_mode == kHead64OutputDirectBfloat16" in direct_output
    assert "run_sparse_attention_head64_fused_hd576_bf16(" in direct_output
    assert "nullptr,\n        nullptr,\n        &output,\n        &lse," in direct_output

    fused_launch = sparse_source.split(
        "void run_sparse_attention_head64_fused_hd576_bf16(", 1
    )[1].split("void run_sparse_attention_head64_combine_hd576_bf16(", 1)[0]
    assert (
        "launch_sparse_attention_head64_fused_hd576_bf16_rolling_restored_v2("
        in fused_launch
    )

    fused_source = (
        builtin_root
        / "sparse_attention/simt/v2/aten_dsa_sparse_attention_v2/csrc/simt/sparse_attention_head64_fused_hd576.asc"
    ).read_text()
    rolling_gate = fused_source.split(
        "head64_fused_is_vllm_rolling_decode(", 1
    )[1].split("}\n", 1)[0]
    assert " ".join(rolling_gate.split()) == (
        "const SparseAttentionHead64Plan& plan, int32_t causal) { return "
        "causal != 0 && plan.used_core_num == 8 && plan.task_count == 8 && "
        "plan.batch_size == 2 && plan.query_heads == 128 && "
        "plan.query_tokens == 2 && plan.context_tokens == 32768 && "
        "plan.selected_tokens == 2048 && plan.qk_head_dim == 576 && "
        "plan.value_head_dim == 512 && plan.head_tile == 64 && "
        "plan.head_group_count == 2 && plan.selected_tile == 64 && "
        "plan.selected_partitions == 1 && "
        "plan.selected_partition_tile_capacity == 32 && "
        "plan.output_mode == kHead64OutputDirectBfloat16;"
    )

    restored_kernel = fused_source.split(
        "sparse_attention_head64_fused_mix12_restored_kernel(", 1
    )[1].split("extern \"C\" __global__ __mix__(1, 2) void", 1)[0]
    assert (
        "head64_fused_is_vllm_rolling_decode(dispatch_plan, causal)"
        in restored_kernel
    )
    assert "sparse_attention_head64_fused_vllm_aic(" in restored_kernel
    assert "sparse_attention_head64_fused_vllm_aiv(" in restored_kernel

    assert "kHead64FusedVllmSelectedTile = 128" in fused_source
    assert "kHead64FusedVllmQkTile = 256" in fused_source
    assert "kHead64FusedVllmValueTile = 256" in fused_source

    vllm_aic_start = "sparse_attention_head64_fused_vllm_aic("
    vllm_aic_end = "sparse_attention_head64_fused_aiv("
    assert vllm_aic_start in fused_source
    assert vllm_aic_end in fused_source
    vllm_aic = fused_source.split(vllm_aic_start, 1)[1].split(vllm_aic_end, 1)[0]
    assert "k_start += kHead64FusedVllmQkTile" in vllm_aic
    assert "576 - k_start < kHead64FusedVllmQkTile" in vllm_aic
    assert "MakeShape(64, current_k)" in vllm_aic
    assert "MakeFrameLayout<ZNLayoutPtn, bfloat16_t>(current_k, 128)" in vllm_aic
    assert "MakeFrameLayout<NZLayoutPtn>(64, 128)" in vllm_aic
    assert "params.m = 64;" in vllm_aic
    assert "params.n = 128;" in vllm_aic
    assert "params.k = current_k;" in vllm_aic
    assert "Mmad(qk_mm.with(params), l0_scores, l0_query, l0_keys);" in vllm_aic
