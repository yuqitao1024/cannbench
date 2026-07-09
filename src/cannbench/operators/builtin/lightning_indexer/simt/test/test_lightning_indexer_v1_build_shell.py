from pathlib import Path


def test_lightning_indexer_simt_v1_setup_uses_bisheng_toolchain():
    setup_py = Path(
        "src/cannbench/operators/builtin/lightning_indexer/simt/v1/setup.py"
    ).read_text(encoding="utf-8")

    assert "bisheng" in setup_py
    assert "--enable-simt" in setup_py
    assert 'library_name = "aten_dsa_lightning_indexer"' in setup_py


def test_lightning_indexer_simt_v1_register_has_python_module_entry():
    source = Path(
        "src/cannbench/operators/builtin/lightning_indexer/simt/v1/"
        "aten_dsa_lightning_indexer/csrc/register.asc"
    ).read_text(encoding="utf-8")

    assert "PyInit__C" in source
    assert "PyModuleDef_HEAD_INIT" in source


def test_lightning_indexer_prefill_family_4x64_bridge_uses_bmm_for_score_body():
    source = Path(
        "src/cannbench/operators/builtin/lightning_indexer/simt/v1/"
        "aten_dsa_lightning_indexer/csrc/lightning_indexer.asc"
    ).read_text(encoding="utf-8")

    assert "at::bmm(" in source or "at::matmul(" in source
    assert (
        "launch_lightning_indexer_prefill_family_4x64_postprocess_float" in source
    )


def test_lightning_indexer_prefill_family_4x64_bridge_tiles_context_scores():
    source = Path(
        "src/cannbench/operators/builtin/lightning_indexer/simt/v1/"
        "aten_dsa_lightning_indexer/csrc/lightning_indexer.asc"
    ).read_text(encoding="utf-8")

    assert "for (int64_t context_start = 0; context_start < context_count;" in source
    assert "best_scores = at::full(" in source
    assert "best_indices = at::zeros(" in source
    assert (
        "auto scores = at::bmm(query_2d, key_bmm).reshape("
        not in source
    )


def test_lightning_indexer_prefill_family_4x64_bridge_avoids_key_repeat():
    source = Path(
        "src/cannbench/operators/builtin/lightning_indexer/simt/v1/"
        "aten_dsa_lightning_indexer/csrc/lightning_indexer.asc"
    ).read_text(encoding="utf-8")

    assert ".repeat({query_count, 1, 1})" not in source


def test_lightning_indexer_prefill_family_4x64_bridge_tiles_queries_too():
    source = Path(
        "src/cannbench/operators/builtin/lightning_indexer/simt/v1/"
        "aten_dsa_lightning_indexer/csrc/lightning_indexer.asc"
    ).read_text(encoding="utf-8")

    assert "for (int64_t query_start = 0; query_start < query_count;" in source
    assert "query.narrow(1, query_start, current_query)" in source


def test_lightning_indexer_prefill_family_4x64_kernel_is_postprocess_only():
    source = Path(
        "src/cannbench/operators/builtin/lightning_indexer/simt/v1/"
        "aten_dsa_lightning_indexer/csrc/simt/"
        "lightning_indexer_prefill_family_4x64.asc"
    ).read_text(encoding="utf-8")

    assert (
        "for (int32_t dim_index = 0; dim_index < kFamily4x64HeadDim; ++dim_index)"
        not in source
    )


def test_lightning_indexer_prefill_family_4x64_kernel_updates_existing_topk_state():
    source = Path(
        "src/cannbench/operators/builtin/lightning_indexer/simt/v1/"
        "aten_dsa_lightning_indexer/csrc/simt/"
        "lightning_indexer_prefill_family_4x64.asc"
    ).read_text(encoding="utf-8")

    assert "best_scores" not in source
    assert "best_indices" not in source
    assert "context_start" in source
