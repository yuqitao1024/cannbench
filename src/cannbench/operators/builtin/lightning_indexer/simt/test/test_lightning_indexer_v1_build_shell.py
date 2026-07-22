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


def test_lightning_indexer_prefill_family_4x64_bridge_uses_tensor_api_score_body():
    source = Path(
        "src/cannbench/operators/builtin/lightning_indexer/simt/v1/"
        "aten_dsa_lightning_indexer/csrc/lightning_indexer.asc"
    ).read_text(encoding="utf-8")

    assert "launch_lightning_indexer_score_4x64_float" in source
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
    assert "__simt_vf__" in source
    assert "asc_vf_call<" in source


def test_lightning_indexer_prefill_family_4x64_kernel_updates_existing_topk_state():
    source = Path(
        "src/cannbench/operators/builtin/lightning_indexer/simt/v1/"
        "aten_dsa_lightning_indexer/csrc/simt/"
        "lightning_indexer_prefill_family_4x64.asc"
    ).read_text(encoding="utf-8")

    assert "best_scores" not in source
    assert "best_indices" not in source
    assert "context_start" in source


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


def test_lightning_indexer_prefill_family_64x128_uses_postprocess_kernel():
    source = Path(
        "src/cannbench/operators/builtin/lightning_indexer/simt/v1/"
        "aten_dsa_lightning_indexer/csrc/lightning_indexer.asc"
    ).read_text(encoding="utf-8")

    assert "launch_lightning_indexer_prefill_family_64x128_postprocess_float" in source


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


def test_lightning_indexer_family_64x128_postprocess_dispatches_via_asc_vf_call():
    source = Path(
        "src/cannbench/operators/builtin/lightning_indexer/simt/v1/"
        "aten_dsa_lightning_indexer/csrc/simt/"
        "lightning_indexer_postprocess_family_64x128.asc"
    ).read_text(encoding="utf-8")

    assert "__simt_vf__" in source
    assert "asc_vf_call<" in source
    assert "launch_lightning_indexer_prefill_family_64x128_postprocess_float" in source


def test_lightning_indexer_bridge_declares_postprocess_launchers_with_c_linkage():
    source = Path(
        "src/cannbench/operators/builtin/lightning_indexer/simt/v1/"
        "aten_dsa_lightning_indexer/csrc/lightning_indexer.asc"
    ).read_text(encoding="utf-8")

    assert (
        'extern "C" void launch_lightning_indexer_prefill_family_4x64_postprocess_float'
        in source
    )
    assert (
        'extern "C" void launch_lightning_indexer_prefill_family_64x128_postprocess_float'
        in source
    )


def test_lightning_indexer_family_64x128_kernel_is_postprocess_only():
    source = Path(
        "src/cannbench/operators/builtin/lightning_indexer/simt/v1/"
        "aten_dsa_lightning_indexer/csrc/simt/"
        "lightning_indexer_postprocess_family_64x128.asc"
    ).read_text(encoding="utf-8")

    assert (
        "for (int32_t dim_index = 0; dim_index < kFamily64x128HeadDim; ++dim_index)"
        not in source
    )
    assert "__simt_vf__" in source
    assert "asc_vf_call<" in source


def test_lightning_indexer_family_64x128_postprocess_kernel_has_dedicated_filename():
    source_path = Path(
        "src/cannbench/operators/builtin/lightning_indexer/simt/v1/setup.py"
    )

    assert source_path.exists()
    assert Path(
        "src/cannbench/operators/builtin/lightning_indexer/simt/v1/"
        "aten_dsa_lightning_indexer/csrc/simt/"
        "lightning_indexer_postprocess_family_64x128.asc"
    ).exists()


def test_lightning_indexer_family_64x128_postprocess_source_uses_postprocess_symbol_names():
    source = Path(
        "src/cannbench/operators/builtin/lightning_indexer/simt/v1/"
        "aten_dsa_lightning_indexer/csrc/simt/"
        "lightning_indexer_postprocess_family_64x128.asc"
    ).read_text(encoding="utf-8")

    assert "lightning_indexer_postprocess_family_64x128_kernel" in source


def test_lightning_indexer_score_sources_use_tensor_api():
    source_4x64 = Path(
        "src/cannbench/operators/builtin/lightning_indexer/simt/v1/"
        "aten_dsa_lightning_indexer/csrc/simt/"
        "lightning_indexer_score_family_4x64.asc"
    ).read_text(encoding="utf-8")
    source_64x128 = Path(
        "src/cannbench/operators/builtin/lightning_indexer/simt/v1/"
        "aten_dsa_lightning_indexer/csrc/simt/"
        "lightning_indexer_score_family_64x128.asc"
    ).read_text(encoding="utf-8")

    for source in (source_4x64, source_64x128):
        assert "tensor_api/tensor.h" in source
        assert "MakeMmad(" in source
        assert "__global__ __cube__" in source


def test_lightning_indexer_family_64x128_fixpipe_uses_msize_aligned_src_stride():
    source_64x128 = Path(
        "src/cannbench/operators/builtin/lightning_indexer/simt/v1/"
        "aten_dsa_lightning_indexer/csrc/simt/"
        "lightning_indexer_score_family_64x128.asc"
    ).read_text(encoding="utf-8")

    assert "FixpipeParamsC310<AscendC::CO2Layout::ROW_MAJOR> fixpipe_params;" in source_64x128
    assert "fixpipe_params.mSize = static_cast<uint16_t>(shape.m);" in source_64x128
    assert "fixpipe_params.srcStride = align_u16(fixpipe_params.mSize, 16);" in source_64x128


def test_lightning_indexer_score_source_basic_api_usage_is_isolated_to_64x128():
    source_4x64 = Path(
        "src/cannbench/operators/builtin/lightning_indexer/simt/v1/"
        "aten_dsa_lightning_indexer/csrc/simt/"
        "lightning_indexer_score_family_4x64.asc"
    ).read_text(encoding="utf-8")
    source_64x128 = Path(
        "src/cannbench/operators/builtin/lightning_indexer/simt/v1/"
        "aten_dsa_lightning_indexer/csrc/simt/"
        "lightning_indexer_score_family_64x128.asc"
    ).read_text(encoding="utf-8")

    assert "basic_api/" not in source_4x64
    assert "SetFlag" not in source_4x64
    assert "WaitFlag" not in source_4x64
    assert "PipeBarrier" not in source_4x64

    assert '#include "basic_api/kernel_basic_intf.h"' in source_64x128
    assert '#include "basic_api/kernel_operator_fixpipe_intf.h"' in source_64x128
    assert "AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>" in source_64x128
    assert "AscendC::WaitFlag<AscendC::HardEvent::FIX_M>" in source_64x128


def test_lightning_indexer_family_64x128_score_uses_per_head_fallback_launch():
    source = Path(
        "src/cannbench/operators/builtin/lightning_indexer/simt/v1/"
        "aten_dsa_lightning_indexer/csrc/simt/"
        "lightning_indexer_score_family_64x128.asc"
    ).read_text(encoding="utf-8")

    assert "for (int64_t head_index = 0; head_index < kHeadCount; ++head_index)" in source
    assert "TODO: Restore the original m=64 score launch" in source
