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
        [64, 128],
        [128, 64],
        [64, 64],
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
    plan_source = (
        builtin_root
        / "sparse_attention/simt/v2/aten_dsa_sparse_attention_v2/csrc/simt/sparse_attention_head64_plan.h"
    ).read_text()
    assert "kHead64Tile = 64" in plan_source
    assert "kHead64SelectedTile = 64" in plan_source
    assert "kHead64QkTile = 128" in plan_source
    assert "kHead64ValueTile = 128" in plan_source

    head64_source = (
        builtin_root
        / "sparse_attention/simt/v2/aten_dsa_sparse_attention_v2/csrc/simt/sparse_attention_head64_hd576.asc"
    ).read_text()
    assert "MakeShape(kHead64Tile, current_k)" in head64_source
    assert "current_k, current_selected" in head64_source
    assert "MakeFrameLayout<NZLayoutPtn>(kHead64Tile, current_selected)" in head64_source
    assert "params.m = 64;" in head64_source
    assert "params.n = current_selected;" in head64_source
    assert "params.k = current_k;" in head64_source

    fused_source = (
        builtin_root
        / "sparse_attention/simt/v2/aten_dsa_sparse_attention_v2/csrc/simt/sparse_attention_head64_fused_hd576.asc"
    ).read_text()
    canonical_gate = fused_source.split(
        "head64_fused_is_canonical_v32_decode", 1
    )[1].split("}\n", 1)[0]
    assert "plan.selected_partitions == 4" in canonical_gate
    assert "plan.output_mode == kHead64OutputPartialFloat" in canonical_gate
