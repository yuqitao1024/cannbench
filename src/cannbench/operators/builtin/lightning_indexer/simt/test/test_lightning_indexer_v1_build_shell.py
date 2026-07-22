from pathlib import Path


def test_lightning_indexer_simt_v1_setup_uses_bisheng_toolchain():
    setup_py = Path(
        "src/cannbench/operators/builtin/lightning_indexer/simt/v1/setup.py"
    ).read_text(encoding="utf-8")

    assert "bisheng" in setup_py
    assert "--enable-simt" not in setup_py
    assert 'library_name = "aten_dsa_lightning_indexer"' in setup_py


def test_lightning_indexer_simt_v1_register_has_python_module_entry():
    source = Path(
        "src/cannbench/operators/builtin/lightning_indexer/simt/v1/"
        "aten_dsa_lightning_indexer/csrc/register.asc"
    ).read_text(encoding="utf-8")

    assert "PyInit__C" in source
    assert "PyModuleDef_HEAD_INIT" in source


def test_lightning_indexer_package_imports_torch_before_loading_cpp_extension():
    source = Path(
        "src/cannbench/operators/builtin/lightning_indexer/simt/v1/"
        "aten_dsa_lightning_indexer/__init__.py"
    ).read_text(encoding="utf-8")

    assert 'import_module("torch")' in source
    assert source.index('import_module("torch")') < source.index(
        'import_module(f"{__name__}._C")'
    )


def test_lightning_indexer_bridge_uses_fused_family_launchers():
    source = Path(
        "src/cannbench/operators/builtin/lightning_indexer/simt/v1/"
        "aten_dsa_lightning_indexer/csrc/lightning_indexer.asc"
    ).read_text(encoding="utf-8")

    assert "launch_lightning_indexer_fused_family_4x64_float" in source
    assert "launch_lightning_indexer_fused_family_64x128_float" in source
    assert "launch_lightning_indexer_score_4x64_float" not in source
    assert "launch_lightning_indexer_score_64x128_float" not in source
    assert "launch_lightning_indexer_prefill_family_4x64_postprocess_float" not in source
    assert (
        "launch_lightning_indexer_prefill_family_64x128_postprocess_float"
        not in source
    )


def test_lightning_indexer_prefill_family_4x64_bridge_tiles_context_scores():
    source = Path(
        "src/cannbench/operators/builtin/lightning_indexer/simt/v1/"
        "aten_dsa_lightning_indexer/csrc/lightning_indexer.asc"
    ).read_text(encoding="utf-8")

    assert "for (int64_t context_start = 0; context_start < context_count;" in source
    assert "best_scores = at::full(" in source
    assert "best_indices = at::zeros(" in source
    assert "at::matmul(" not in source
    assert "at::bmm(" not in source


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


def test_lightning_indexer_prefill_family_4x64_bridge_uses_named_tile_constants():
    source = Path(
        "src/cannbench/operators/builtin/lightning_indexer/simt/v1/"
        "aten_dsa_lightning_indexer/csrc/lightning_indexer.asc"
    ).read_text(encoding="utf-8")

    assert "constexpr int64_t kFamily4x64QueryTile" in source
    assert "constexpr int64_t kFamily4x64ContextTile" in source


def test_lightning_indexer_prefill_family_4x64_bridge_extracts_tile_postprocess_helper():
    source = Path(
        "src/cannbench/operators/builtin/lightning_indexer/simt/v1/"
        "aten_dsa_lightning_indexer/csrc/lightning_indexer.asc"
    ).read_text(encoding="utf-8")

    assert "run_lightning_indexer_family_4x64_tile(" in source


def test_lightning_indexer_split_family_4x64_sources_are_removed():
    base = Path(
        "src/cannbench/operators/builtin/lightning_indexer/simt/v1/"
        "aten_dsa_lightning_indexer/csrc/simt"
    )

    assert not (base / "lightning_indexer_score_family_4x64.asc").exists()
    assert not (base / "lightning_indexer_prefill_family_4x64.asc").exists()


def test_lightning_indexer_prefill_family_64x128_bridge_uses_named_tile_constants():
    source = Path(
        "src/cannbench/operators/builtin/lightning_indexer/simt/v1/"
        "aten_dsa_lightning_indexer/csrc/lightning_indexer.asc"
    ).read_text(encoding="utf-8")

    assert "constexpr int64_t kFamily64x128QueryTile" in source
    assert "constexpr int64_t kFamily64x128ContextTile = 128;" in source


def test_lightning_indexer_prefill_family_64x128_bridge_extracts_tile_helper():
    source = Path(
        "src/cannbench/operators/builtin/lightning_indexer/simt/v1/"
        "aten_dsa_lightning_indexer/csrc/lightning_indexer.asc"
    ).read_text(encoding="utf-8")

    assert "run_lightning_indexer_family_64x128_tile(" in source


def test_lightning_indexer_prefill_family_64x128_uses_decode_fused_path():
    source = Path(
        "src/cannbench/operators/builtin/lightning_indexer/simt/v1/"
        "aten_dsa_lightning_indexer/csrc/lightning_indexer.asc"
    ).read_text(encoding="utf-8")

    assert "launch_lightning_indexer_fused_family_64x128_float" in source
    assert "launch_lightning_indexer_prefill_family_64x128_postprocess_float" not in source


def test_lightning_indexer_prefill_family_64x128_fp16_avoids_split_score_then_postprocess():
    source = Path(
        "src/cannbench/operators/builtin/lightning_indexer/simt/v1/"
        "aten_dsa_lightning_indexer/csrc/lightning_indexer.asc"
    ).read_text(encoding="utf-8")
    body = source.split(
        "at::Tensor lightning_indexer_forward_prefill_family_64x128_float(",
        1,
    )[1].split(
        "at::Tensor lightning_indexer_forward_decode_family_4x64_float(",
        1,
    )[0]

    assert "return lightning_indexer_forward_decode_family_64x128_float(" in body
    assert "launch_lightning_indexer_prefill_family_64x128_postprocess_float" not in body
    assert "run_lightning_indexer_family_64x128_tile(" not in body


def test_lightning_indexer_decode_family_64x128_reuses_fused_helper():
    source = Path(
        "src/cannbench/operators/builtin/lightning_indexer/simt/v1/"
        "aten_dsa_lightning_indexer/csrc/lightning_indexer.asc"
    ).read_text(encoding="utf-8")
    body = source.split(
        "at::Tensor lightning_indexer_forward_decode_family_64x128_float(",
        1,
    )[1].split(
        "at::Tensor lightning_indexer_forward_prefill_family_64x128_float(",
        1,
    )[0]

    assert "run_lightning_indexer_family_64x128_tile(" in body
    assert "record_tensor_on_stream(best_scores_tile, npu_stream);" in body
    assert "record_tensor_on_stream(best_indices_tile, npu_stream);" in body


def test_lightning_indexer_bridge_declares_only_fused_launchers_with_c_linkage():
    source = Path(
        "src/cannbench/operators/builtin/lightning_indexer/simt/v1/"
        "aten_dsa_lightning_indexer/csrc/lightning_indexer.asc"
    ).read_text(encoding="utf-8")

    assert 'extern "C" void launch_lightning_indexer_fused_family_4x64_float' in source
    assert 'extern "C" void launch_lightning_indexer_fused_family_64x128_float' in source
    assert (
        'extern "C" void launch_lightning_indexer_prefill_family_4x64_postprocess_float'
        not in source
    )
    assert (
        'extern "C" void launch_lightning_indexer_prefill_family_64x128_postprocess_float'
        not in source
    )
    assert 'extern "C" void launch_lightning_indexer_score_4x64_float' not in source
    assert 'extern "C" void launch_lightning_indexer_score_64x128_float' not in source


def test_lightning_indexer_split_family_64x128_sources_are_removed():
    base = Path(
        "src/cannbench/operators/builtin/lightning_indexer/simt/v1/"
        "aten_dsa_lightning_indexer/csrc/simt"
    )

    assert not (base / "lightning_indexer_score_family_64x128.asc").exists()
    assert not (base / "lightning_indexer_postprocess_family_64x128.asc").exists()


def test_lightning_indexer_family_4x64_fp16_fused_path_does_not_materialize_score_tile():
    source = Path(
        "src/cannbench/operators/builtin/lightning_indexer/simt/v1/"
        "aten_dsa_lightning_indexer/csrc/lightning_indexer.asc"
    ).read_text(encoding="utf-8")
    body = source.split(
        "at::Tensor lightning_indexer_forward_family_4x64_score_tiled_float(",
        1,
    )[1].split(
        "at::Tensor lightning_indexer_forward_prefill_family_4x64_float(",
        1,
    )[0]

    assert "auto score_tile = at::empty(" not in body


def test_lightning_indexer_family_64x128_fp16_fused_path_does_not_materialize_score_tile():
    source = Path(
        "src/cannbench/operators/builtin/lightning_indexer/simt/v1/"
        "aten_dsa_lightning_indexer/csrc/lightning_indexer.asc"
    ).read_text(encoding="utf-8")
    body = source.split(
        "at::Tensor lightning_indexer_forward_decode_family_64x128_float(",
        1,
    )[1].split(
        "at::Tensor lightning_indexer_forward_prefill_family_64x128_float(",
        1,
    )[0]

    assert "auto score_tile = at::empty(" not in body
