from pathlib import Path


def test_sparse_attention_simt_v1_setup_uses_bisheng_toolchain():
    setup_py = Path(
        "src/cannbench/operators/builtin/sparse_attention/simt/v1/setup.py"
    ).read_text(encoding="utf-8")

    assert "bisheng" in setup_py
    assert 'library_name = "aten_dsa_sparse_attention"' in setup_py
    assert "\"-shared\"" in setup_py


def test_sparse_attention_simt_v1_setup_does_not_force_enable_simt():
    setup_py = Path(
        "src/cannbench/operators/builtin/sparse_attention/simt/v1/setup.py"
    ).read_text(encoding="utf-8")

    assert "--enable-simt" not in setup_py


def test_sparse_attention_hd512_postprocess_uses_mixed_vector_wrapper():
    source = Path(
        "src/cannbench/operators/builtin/sparse_attention/simt/v1/"
        "aten_dsa_sparse_attention/csrc/simt/"
        "sparse_attention_postprocess_family_hd512.asc"
    ).read_text(encoding="utf-8")

    assert "__simt_vf__" in source
    assert "asc_vf_call<" in source


def test_sparse_attention_hd128_postprocess_uses_mixed_vector_wrapper():
    source = Path(
        "src/cannbench/operators/builtin/sparse_attention/simt/v1/"
        "aten_dsa_sparse_attention/csrc/simt/"
        "sparse_attention_postprocess_family_hd128.asc"
    ).read_text(encoding="utf-8")

    assert "__simt_vf__" in source
    assert "asc_vf_call<" in source


def test_sparse_attention_simt_v1_register_has_python_module_entry():
    source = Path(
        "src/cannbench/operators/builtin/sparse_attention/simt/v1/"
        "aten_dsa_sparse_attention/csrc/register.asc"
    ).read_text(encoding="utf-8")

    assert "PyInit__C" in source
    assert "PyModuleDef_HEAD_INIT" in source


def test_sparse_attention_hd128_bridge_uses_hybrid_score_body():
    source = Path(
        "src/cannbench/operators/builtin/sparse_attention/simt/v1/"
        "aten_dsa_sparse_attention/csrc/sparse_attention.asc"
    ).read_text(encoding="utf-8")

    assert "launch_sparse_attention_score_" in source
    assert "launch_sparse_attention_hd128_postprocess_float" in source


def test_sparse_attention_hd128_bridge_uses_named_tile_constants():
    source = Path(
        "src/cannbench/operators/builtin/sparse_attention/simt/v1/"
        "aten_dsa_sparse_attention/csrc/sparse_attention.asc"
    ).read_text(encoding="utf-8")

    assert "constexpr int64_t kFamilyHd128QueryTile" in source
    assert "constexpr int64_t kFamilyHd128SelectedTile" in source


def test_sparse_attention_hd512_bridge_uses_hybrid_score_body():
    source = Path(
        "src/cannbench/operators/builtin/sparse_attention/simt/v1/"
        "aten_dsa_sparse_attention/csrc/sparse_attention.asc"
    ).read_text(encoding="utf-8")

    assert "launch_sparse_attention_score_" in source
    assert "launch_sparse_attention_keys_gather_pack_hd512_float" in source
    assert "launch_sparse_attention_hd512_postprocess_float" in source
    assert "sparse_attention_forward_family_hd512_hybrid(" in source


def test_sparse_attention_hd512_has_custom_gather_pack_source():
    source = Path(
        "src/cannbench/operators/builtin/sparse_attention/simt/v1/"
        "aten_dsa_sparse_attention/csrc/simt/"
        "sparse_attention_keys_gather_pack_hd512.asc"
    ).read_text(encoding="utf-8")

    assert "selected_keys" in source
    assert "indices" in source
    assert "launch_sparse_attention_keys_gather_pack_hd512_float" in source


def test_sparse_attention_hd512_score_source_uses_tensor_api():
    source = Path(
        "src/cannbench/operators/builtin/sparse_attention/simt/v1/"
        "aten_dsa_sparse_attention/csrc/simt/"
        "sparse_attention_score_family_hd512.asc"
    ).read_text(encoding="utf-8")

    assert "tensor_api/tensor.h" in source
    assert "MakeMmad(" in source
    assert "__global__ __cube__" in source
    assert "matmul_intf.h" not in source


def test_sparse_attention_hd128_score_source_uses_tensor_api():
    source = Path(
        "src/cannbench/operators/builtin/sparse_attention/simt/v1/"
        "aten_dsa_sparse_attention/csrc/simt/"
        "sparse_attention_score_family_hd128.asc"
    ).read_text(encoding="utf-8")

    assert "tensor_api/tensor.h" in source
    assert "MakeMmad(" in source
    assert "__global__ __cube__" in source
    assert "simt_api/asc_simt.h" not in source


def test_sparse_attention_score_helper_avoids_reshape_bmm_path():
    source = Path(
        "src/cannbench/operators/builtin/sparse_attention/simt/v1/"
        "aten_dsa_sparse_attention/csrc/sparse_attention.asc"
    ).read_text(encoding="utf-8")

    assert "at::bmm(" not in source
    assert "at::matmul(" not in source
    assert "run_sparse_attention_score_family_hd512_tile(" in source
    assert "run_sparse_attention_score_family_hd128_tile(" in source


def test_sparse_attention_hd512_bridge_uses_named_tile_constants():
    source = Path(
        "src/cannbench/operators/builtin/sparse_attention/simt/v1/"
        "aten_dsa_sparse_attention/csrc/sparse_attention.asc"
    ).read_text(encoding="utf-8")

    assert "constexpr int64_t kFamilyHd512QueryTile" in source
    assert "constexpr int64_t kFamilyHd512SelectedTile" in source


def test_sparse_attention_hd512_bridge_extracts_tile_helper():
    source = Path(
        "src/cannbench/operators/builtin/sparse_attention/simt/v1/"
        "aten_dsa_sparse_attention/csrc/sparse_attention.asc"
    ).read_text(encoding="utf-8")

    assert "run_sparse_attention_family_hd512_tile(" in source
    assert "run_sparse_attention_score_family_hd512_tile(" in source


def test_sparse_attention_hd128_bridge_extracts_tile_helper():
    source = Path(
        "src/cannbench/operators/builtin/sparse_attention/simt/v1/"
        "aten_dsa_sparse_attention/csrc/sparse_attention.asc"
    ).read_text(encoding="utf-8")

    assert "run_sparse_attention_family_hd128_tile(" in source
    assert "run_sparse_attention_score_family_hd128_tile(" in source


def test_sparse_attention_hd128_kernel_is_postprocess_only():
    source = Path(
        "src/cannbench/operators/builtin/sparse_attention/simt/v1/"
        "aten_dsa_sparse_attention/csrc/simt/"
        "sparse_attention_postprocess_family_hd128.asc"
    ).read_text(encoding="utf-8")

    assert "dot += query[" not in source


def test_sparse_attention_hd512_kernel_is_postprocess_only():
    source = Path(
        "src/cannbench/operators/builtin/sparse_attention/simt/v1/"
        "aten_dsa_sparse_attention/csrc/simt/"
        "sparse_attention_postprocess_family_hd512.asc"
    ).read_text(encoding="utf-8")

    assert "dot += query[" not in source


def test_sparse_attention_hd128_postprocess_source_uses_postprocess_symbol_names():
    source = Path(
        "src/cannbench/operators/builtin/sparse_attention/simt/v1/"
        "aten_dsa_sparse_attention/csrc/simt/"
        "sparse_attention_postprocess_family_hd128.asc"
    ).read_text(encoding="utf-8")

    assert "sparse_attention_postprocess_family_hd128_kernel" in source


def test_sparse_attention_hd512_postprocess_source_uses_postprocess_symbol_names():
    source = Path(
        "src/cannbench/operators/builtin/sparse_attention/simt/v1/"
        "aten_dsa_sparse_attention/csrc/simt/"
        "sparse_attention_postprocess_family_hd512.asc"
    ).read_text(encoding="utf-8")

    assert "sparse_attention_postprocess_family_hd512_kernel" in source


def test_sparse_attention_bridge_does_not_keep_debug_zero_output_path():
    source = Path(
        "src/cannbench/operators/builtin/sparse_attention/simt/v1/"
        "aten_dsa_sparse_attention/csrc/sparse_attention.asc"
    ).read_text(encoding="utf-8")

    assert "output_tile.zero_();" not in source
