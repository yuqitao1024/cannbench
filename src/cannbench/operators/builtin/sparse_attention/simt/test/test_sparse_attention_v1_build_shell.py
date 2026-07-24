from pathlib import Path


def _score_source(head_dim: int) -> str:
    return Path(
        "src/cannbench/operators/builtin/sparse_attention/simt/v1/"
        "aten_dsa_sparse_attention/csrc/simt/"
        f"sparse_attention_score_family_hd{head_dim}.asc"
    ).read_text(encoding="utf-8")


def _function_body(source: str, start_marker: str, end_marker: str) -> str:
    return source.split(start_marker, 1)[1].split(end_marker, 1)[0]


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


def test_sparse_attention_package_imports_torch_before_loading_cpp_extension():
    source = Path(
        "src/cannbench/operators/builtin/sparse_attention/simt/v1/"
        "aten_dsa_sparse_attention/__init__.py"
    ).read_text(encoding="utf-8")

    assert 'import_module("torch")' in source
    assert source.index('import_module("torch")') < source.index(
        'import_module(f"{__name__}._C")'
    )


def test_sparse_attention_hd128_bridge_uses_hybrid_score_body():
    source = Path(
        "src/cannbench/operators/builtin/sparse_attention/simt/v1/"
        "aten_dsa_sparse_attention/csrc/sparse_attention.asc"
    ).read_text(encoding="utf-8")

    assert "launch_sparse_attention_score_" in source
    assert "launch_sparse_attention_hd128_postprocess_decode_direct_float" in source
    assert "launch_sparse_attention_hd128_postprocess_float" not in source


def test_sparse_attention_hd128_bridge_uses_named_tile_constants():
    source = Path(
        "src/cannbench/operators/builtin/sparse_attention/simt/v1/"
        "aten_dsa_sparse_attention/csrc/sparse_attention.asc"
    ).read_text(encoding="utf-8")

    assert "constexpr int64_t kFamilyHd128QueryTile" in source
    assert "constexpr int64_t kFamilyHd128SelectedTile" in source


def test_sparse_attention_hd128_prefill_fp16_has_fused_helper():
    source = Path(
        "src/cannbench/operators/builtin/sparse_attention/simt/v1/"
        "aten_dsa_sparse_attention/csrc/sparse_attention.asc"
    ).read_text(encoding="utf-8")

    assert "run_sparse_attention_family_hd128_prefill_fused_tile(" in source
    assert "launch_sparse_attention_hd128_prefill_fused_float(" in source


def test_sparse_attention_hd128_prefill_bf16_avoids_gm_scores_and_old_postprocess():
    source = Path(
        "src/cannbench/operators/builtin/sparse_attention/simt/v1/"
        "aten_dsa_sparse_attention/csrc/sparse_attention.asc"
    ).read_text(encoding="utf-8")
    body = _function_body(
        source,
        "std::tuple<at::Tensor, at::Tensor> sparse_attention_forward_family_hd128_hybrid(",
        "std::tuple<at::Tensor, at::Tensor> sparse_attention_forward_family_hd128_decode_fused(",
    )

    bf16_body = body.split("if (query.scalar_type() == at::ScalarType::BFloat16) {", 1)[1]
    bf16_body = bf16_body.split("} else {", 1)[0]

    assert "run_sparse_attention_family_hd128_prefill_fused_tile(" in bf16_body
    assert "auto scores = at::empty(" not in bf16_body
    assert "run_sparse_attention_score_gather_family_hd128_tile(" not in bf16_body
    assert "run_sparse_attention_family_hd128_decode_direct_tile(" not in bf16_body


def test_sparse_attention_hd128_prefill_fused_score_ready_uses_vector_fixpipe_handshake():
    source = _score_source(128)
    body = _function_body(
        source,
        "__aicore__ inline void sparse_attention_prefill_fused_family_hd128_aic(",
        "__global__ __aicore__ void sparse_attention_prefill_fused_family_hd128_kernel(",
    )
    aic_body, aiv_body = body.split(
        "__aicore__ inline void sparse_attention_prefill_fused_family_hd128_aiv(",
        1,
    )

    assert (
        "AscendC::CrossCoreWaitFlag<kCrossCoreSyncMode, PIPE_FIX>(score_ready_flag);"
        in aic_body
    )
    assert (
        "AscendC::CrossCoreSetFlag<kCrossCoreSyncMode, PIPE_FIX>(score_ready_flag);"
        in aic_body
    )
    assert (
        "AscendC::CrossCoreSetFlag<kCrossCoreSyncMode, PIPE_V>(score_ready_flag);"
        in aiv_body
    )
    assert (
        "AscendC::CrossCoreWaitFlag<kCrossCoreSyncMode, PIPE_V>(score_ready_flag);"
        in aiv_body
    )


def test_sparse_attention_hd128_prefill_fused_postprocess_dispatches_via_asc_vf_call():
    source = _score_source(128)
    body = _function_body(
        source,
        "__aicore__ inline void sparse_attention_prefill_fused_family_hd128_aic(",
        "__global__ __aicore__ void sparse_attention_prefill_fused_family_hd128_kernel(",
    )
    _, aiv_body = body.split(
        "__aicore__ inline void sparse_attention_prefill_fused_family_hd128_aiv(",
        1,
    )

    assert "asc_vf_call<" in aiv_body
    assert "__simt_vf__" in source


def test_sparse_attention_hd128_decode_bf16_reuses_fused_helper():
    source = Path(
        "src/cannbench/operators/builtin/sparse_attention/simt/v1/"
        "aten_dsa_sparse_attention/csrc/sparse_attention.asc"
    ).read_text(encoding="utf-8")
    body = _function_body(
        source,
        "std::tuple<at::Tensor, at::Tensor> sparse_attention_forward_family_hd128_decode_fused(",
        "std::tuple<at::Tensor, at::Tensor> sparse_attention_forward(",
    )

    bf16_body = body.split("if (query.scalar_type() == at::ScalarType::BFloat16) {", 1)[1]
    bf16_body = bf16_body.split("} else {", 1)[0]

    assert "run_sparse_attention_family_hd128_prefill_fused_tile(" in bf16_body
    assert "auto scores = at::empty(" not in bf16_body
    assert "run_sparse_attention_score_gather_family_hd128_tile(" not in bf16_body
    assert "run_sparse_attention_family_hd128_decode_direct_tile(" not in bf16_body


def test_sparse_attention_hd512_bridge_uses_hybrid_score_body():
    source = Path(
        "src/cannbench/operators/builtin/sparse_attention/simt/v1/"
        "aten_dsa_sparse_attention/csrc/sparse_attention.asc"
    ).read_text(encoding="utf-8")

    assert "launch_sparse_attention_score_gather_hd512_float" in source
    assert "launch_sparse_attention_keys_gather_pack_hd512_float" not in source
    assert "launch_sparse_attention_hd512_postprocess_decode_direct_float" in source
    assert "launch_sparse_attention_hd512_postprocess_float" not in source
    assert "sparse_attention_forward_family_hd512_hybrid(" in source


def test_sparse_attention_hd512_fp16_has_fused_helper():
    source = Path(
        "src/cannbench/operators/builtin/sparse_attention/simt/v1/"
        "aten_dsa_sparse_attention/csrc/sparse_attention.asc"
    ).read_text(encoding="utf-8")

    assert "run_sparse_attention_family_hd512_fused_tile(" in source
    assert "launch_sparse_attention_hd512_fused_float(" in source


def test_sparse_attention_hd512_prefill_bf16_avoids_gm_scores_and_old_postprocess():
    source = Path(
        "src/cannbench/operators/builtin/sparse_attention/simt/v1/"
        "aten_dsa_sparse_attention/csrc/sparse_attention.asc"
    ).read_text(encoding="utf-8")
    body = _function_body(
        source,
        "std::tuple<at::Tensor, at::Tensor> sparse_attention_forward_family_hd512_hybrid(",
        "std::tuple<at::Tensor, at::Tensor> sparse_attention_forward_family_hd128_hybrid(",
    )

    bf16_body = body.split("if (query.scalar_type() == at::ScalarType::BFloat16) {", 1)[1]
    bf16_body = bf16_body.split("} else {", 1)[0]

    assert "run_sparse_attention_family_hd512_fused_tile(" in bf16_body
    assert "auto scores = at::empty(" not in bf16_body
    assert "run_sparse_attention_score_gather_family_hd512_tile(" not in bf16_body
    assert "run_sparse_attention_family_hd512_decode_direct_tile(" not in bf16_body


def test_sparse_attention_hd512_decode_bf16_uses_fused_helper():
    source = Path(
        "src/cannbench/operators/builtin/sparse_attention/simt/v1/"
        "aten_dsa_sparse_attention/csrc/sparse_attention.asc"
    ).read_text(encoding="utf-8")
    body = _function_body(
        source,
        "std::tuple<at::Tensor, at::Tensor> sparse_attention_forward_family_hd512_decode_fused(",
        "std::tuple<at::Tensor, at::Tensor> sparse_attention_forward_family_hd128_hybrid(",
    )

    bf16_body = body.split("if (query.scalar_type() == at::ScalarType::BFloat16) {", 1)[1]
    bf16_body = bf16_body.split("} else {", 1)[0]

    assert "run_sparse_attention_family_hd512_fused_tile(" in bf16_body
    assert "auto scores = at::empty(" not in bf16_body
    assert "run_sparse_attention_score_gather_family_hd512_tile(" not in bf16_body
    assert "run_sparse_attention_family_hd512_decode_direct_tile(" not in bf16_body


def test_sparse_attention_hd512_fused_score_ready_uses_vector_fixpipe_handshake():
    source = _score_source(512)
    body = _function_body(
        source,
        "__aicore__ inline void sparse_attention_fused_family_hd512_aic(",
        "__global__ __aicore__ void sparse_attention_fused_family_hd512_kernel(",
    )
    aic_body, aiv_body = body.split(
        "__aicore__ inline void sparse_attention_fused_family_hd512_aiv(",
        1,
    )

    assert (
        "AscendC::CrossCoreWaitFlag<kCrossCoreSyncMode, PIPE_FIX>(score_ready_flag);"
        in aic_body
    )
    assert (
        "AscendC::CrossCoreSetFlag<kCrossCoreSyncMode, PIPE_FIX>(score_ready_flag);"
        in aic_body
    )
    assert (
        "AscendC::CrossCoreSetFlag<kCrossCoreSyncMode, PIPE_V>(score_ready_flag);"
        in aiv_body
    )
    assert (
        "AscendC::CrossCoreWaitFlag<kCrossCoreSyncMode, PIPE_V>(score_ready_flag);"
        in aiv_body
    )


def test_sparse_attention_hd512_fused_postprocess_dispatches_via_asc_vf_call():
    source = _score_source(512)
    body = _function_body(
        source,
        "__aicore__ inline void sparse_attention_fused_family_hd512_aic(",
        "__global__ __aicore__ void sparse_attention_fused_family_hd512_kernel(",
    )
    _, aiv_body = body.split(
        "__aicore__ inline void sparse_attention_fused_family_hd512_aiv(",
        1,
    )

    assert "asc_vf_call<" in aiv_body
    assert "__simt_vf__" in source


def test_sparse_attention_wide_fused_kernel_uses_runtime_head_dim():
    source = _score_source(512)
    fused = _function_body(
        source,
        "sparse_attention_fused_family_hd512_postprocess_vf(",
        'extern "C" void launch_sparse_attention_score_gather_hd512_float(',
    )

    assert "const int32_t head_dim = shape.k;" in fused
    assert "int32_t head_dim" in fused
    assert "dim_index < head_dim" in fused
    assert "kHeadDim" not in fused


def test_sparse_attention_bridge_routes_all_wide_bf16_families_through_fused_helper():
    source = Path(
        "src/cannbench/operators/builtin/sparse_attention/simt/v1/"
        "aten_dsa_sparse_attention/csrc/sparse_attention.asc"
    ).read_text(encoding="utf-8")

    assert 'family == "family_hd256"' in source
    assert 'family == "family_hd512"' in source
    assert 'family == "family_hd576"' in source
    assert "run_sparse_attention_family_hd512_fused_tile(" in source
    assert "indices.size(2) <= 2048" in source
    assert "wide-head custom op requires selected_tokens <= 2048" in source


def test_sparse_attention_dead_gather_pack_sources_are_removed():
    hd128_gather_pack = Path(
        "src/cannbench/operators/builtin/sparse_attention/simt/v1/"
        "aten_dsa_sparse_attention/csrc/simt/"
        "sparse_attention_keys_gather_pack_hd128.asc"
    )
    hd512_gather_pack = Path(
        "src/cannbench/operators/builtin/sparse_attention/simt/v1/"
        "aten_dsa_sparse_attention/csrc/simt/"
        "sparse_attention_keys_gather_pack_hd512.asc"
    )

    assert not hd128_gather_pack.exists()
    assert not hd512_gather_pack.exists()


def test_sparse_attention_hd512_score_source_uses_tensor_api():
    source = Path(
        "src/cannbench/operators/builtin/sparse_attention/simt/v1/"
        "aten_dsa_sparse_attention/csrc/simt/"
        "sparse_attention_score_family_hd512.asc"
    ).read_text(encoding="utf-8")

    assert "tensor_api/tensor.h" in source
    assert "MakeMmad(" in source
    assert "__global__ __aicore__" in source
    assert "KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2)" in source
    assert "ASCEND_IS_AIC" in source
    assert "kernel_operator.h" in source
    assert "TPipe" not in source
    assert "matmul_intf.h" not in source


def test_sparse_attention_score_sources_do_not_use_basic_api_or_crosscore_flags():
    sources = (
        Path(
            "src/cannbench/operators/builtin/sparse_attention/simt/v1/"
            "aten_dsa_sparse_attention/csrc/simt/"
            "sparse_attention_score_family_hd128.asc"
        ).read_text(encoding="utf-8"),
        Path(
            "src/cannbench/operators/builtin/sparse_attention/simt/v1/"
            "aten_dsa_sparse_attention/csrc/simt/"
            "sparse_attention_score_family_hd512.asc"
        ).read_text(encoding="utf-8"),
    )

    for source in sources:
        basic_api_includes = [
            line.strip()
            for line in source.splitlines()
            if line.strip().startswith('#include "basic_api/')
        ]
        if source == sources[0]:
            assert basic_api_includes == [
                '#include "basic_api/kernel_basic_intf.h"',
                '#include "basic_api/kernel_operator_block_sync_intf.h"',
            ]
            assert '#include "kernel_common.h"' not in source
            assert "kernel_operator.h" in source
        else:
            assert basic_api_includes == [
                '#include "basic_api/kernel_basic_intf.h"',
                '#include "basic_api/kernel_operator_block_sync_intf.h"',
            ]
            assert '#include "kernel_common.h"' not in source
            assert "kernel_operator.h" in source
        assert "AscendC::LocalTensor<bfloat16_t>" in source
        assert "AscendC::SetFlag<" in source
        assert "AscendC::WaitFlag<" in source
        assert "CrossCoreSetFlag" in source
        assert "CrossCoreWaitFlag" in source
        assert "PipeBarrier" in source


def test_sparse_attention_hd128_score_source_uses_tensor_and_mixed_kernel_api():
    source = Path(
        "src/cannbench/operators/builtin/sparse_attention/simt/v1/"
        "aten_dsa_sparse_attention/csrc/simt/"
        "sparse_attention_score_family_hd128.asc"
    ).read_text(encoding="utf-8")

    assert "tensor_api/tensor.h" in source
    assert "kernel_operator.h" in source
    assert "MakeMmad(" in source
    assert "__global__ __aicore__" in source


def test_sparse_attention_hd128_score_source_uses_cross_core_sync_flags():
    source = _score_source(128)

    assert "AscendC::InitSocState();" in source
    assert "CrossCoreSetFlag<" in source
    assert "CrossCoreWaitFlag" in source


def test_sparse_attention_hd128_score_source_restores_mixed_aic_aiv_handshake():
    source = _score_source(128)

    assert "constexpr uint32_t kGatherKeysL1Offset = 64 * 1024;" in source
    assert "constexpr int32_t kMaxUsedCoreNum = 11;" in source
    assert "constexpr uint8_t kGatherKeysReadyFlagBase = 0;" in source
    assert "constexpr uint8_t kCrossCoreSyncMode = 4;" in source
    assert "KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);" in source
    assert "if (n_loop > 1) {" in source
    assert "const uint16_t ready_flag =" in source
    assert "AscendC::CrossCoreSetFlag<kCrossCoreSyncMode, PIPE_MTE1>(ready_flag);" in source
    assert "AscendC::CrossCoreWaitFlag<kCrossCoreSyncMode, PIPE_MTE1>(ready_flag);" in source
    assert "AscendC::CrossCoreWaitFlag<kCrossCoreSyncMode, PIPE_MTE3>(ready_flag);" in source
    assert "AscendC::CrossCoreSetFlag<kCrossCoreSyncMode, PIPE_MTE3>(ready_flag);" in source
    assert "if (AscendC::GetSubBlockIdx() != 0) {" in source
    assert "AscendC::GetBlockIdx() / AscendC::GetTaskRatio()" in source
    assert "if (block_idx >= static_cast<uint32_t>(shape.used_core_num)) {" in source


def test_sparse_attention_score_helper_avoids_reshape_bmm_path():
    source = Path(
        "src/cannbench/operators/builtin/sparse_attention/simt/v1/"
        "aten_dsa_sparse_attention/csrc/sparse_attention.asc"
    ).read_text(encoding="utf-8")

    assert "at::bmm(" not in source
    assert "at::matmul(" not in source
    assert "run_sparse_attention_score_gather_family_hd512_tile(" in source
    assert "run_sparse_attention_score_gather_family_hd128_tile(" in source
    assert "run_sparse_attention_score_family_hd512_tile(" not in source
    assert "run_sparse_attention_score_family_hd128_tile(" not in source


def test_sparse_attention_score_tiles_write_directly_to_full_scores():
    bridge_source = Path(
        "src/cannbench/operators/builtin/sparse_attention/simt/v1/"
        "aten_dsa_sparse_attention/csrc/sparse_attention.asc"
    ).read_text(encoding="utf-8")
    hd128_score_source = Path(
        "src/cannbench/operators/builtin/sparse_attention/simt/v1/"
        "aten_dsa_sparse_attention/csrc/simt/"
        "sparse_attention_score_family_hd128.asc"
    ).read_text(encoding="utf-8")
    hd512_score_source = Path(
        "src/cannbench/operators/builtin/sparse_attention/simt/v1/"
        "aten_dsa_sparse_attention/csrc/simt/"
        "sparse_attention_score_family_hd512.asc"
    ).read_text(encoding="utf-8")
    hd128_query_pack_source = Path(
        "src/cannbench/operators/builtin/sparse_attention/simt/v1/"
        "aten_dsa_sparse_attention/csrc/simt/"
        "sparse_attention_query_pack_hd128.asc"
    ).read_text(encoding="utf-8")
    hd512_query_pack_source = Path(
        "src/cannbench/operators/builtin/sparse_attention/simt/v1/"
        "aten_dsa_sparse_attention/csrc/simt/"
        "sparse_attention_query_pack_hd512.asc"
    ).read_text(encoding="utf-8")

    assert "scores_chunk" not in bridge_source
    assert "score_stride" in bridge_source
    assert "score_offset" in bridge_source
    assert "score_stride" in hd128_score_source
    assert "score_offset" in hd128_score_source
    assert "score_stride" in hd512_score_source
    assert "score_offset" in hd512_score_source


def test_sparse_attention_postprocess_writes_directly_to_full_output():
    bridge_source = Path(
        "src/cannbench/operators/builtin/sparse_attention/simt/v1/"
        "aten_dsa_sparse_attention/csrc/sparse_attention.asc"
    ).read_text(encoding="utf-8")
    hd128_postprocess_source = Path(
        "src/cannbench/operators/builtin/sparse_attention/simt/v1/"
        "aten_dsa_sparse_attention/csrc/simt/"
        "sparse_attention_postprocess_family_hd128.asc"
    ).read_text(encoding="utf-8")
    hd512_postprocess_source = Path(
        "src/cannbench/operators/builtin/sparse_attention/simt/v1/"
        "aten_dsa_sparse_attention/csrc/simt/"
        "sparse_attention_postprocess_family_hd512.asc"
    ).read_text(encoding="utf-8")

    assert "output_tile" not in bridge_source
    assert "lse_tile" not in bridge_source
    assert ".copy_(output" not in bridge_source
    assert ".copy_(lse" not in bridge_source
    assert "output_query_stride" in bridge_source
    assert "output_query_offset" in bridge_source
    assert "output_query_stride" in hd128_postprocess_source
    assert "output_query_offset" in hd128_postprocess_source
    assert "output_query_stride" in hd512_postprocess_source
    assert "output_query_offset" in hd512_postprocess_source


def test_sparse_attention_key_gather_reads_full_indices_directly():
    bridge_source = Path(
        "src/cannbench/operators/builtin/sparse_attention/simt/v1/"
        "aten_dsa_sparse_attention/csrc/sparse_attention.asc"
    ).read_text(encoding="utf-8")
    hd128_gather_source = Path(
        "src/cannbench/operators/builtin/sparse_attention/simt/v1/"
        "aten_dsa_sparse_attention/csrc/simt/"
        "sparse_attention_score_family_hd128.asc"
    ).read_text(encoding="utf-8")
    hd512_gather_source = Path(
        "src/cannbench/operators/builtin/sparse_attention/simt/v1/"
        "aten_dsa_sparse_attention/csrc/simt/"
        "sparse_attention_score_family_hd512.asc"
    ).read_text(encoding="utf-8")

    assert "indices_chunk" not in bridge_source
    assert ".narrow(2, selected_start" not in bridge_source
    assert "indices_query_stride" in bridge_source
    assert "indices_query_offset" in bridge_source
    assert "indices_selected_stride" in bridge_source
    assert "indices_selected_offset" in bridge_source
    assert "indices_query_stride" in hd128_gather_source
    assert "indices_selected_offset" in hd128_gather_source
    assert "indices_query_stride" in hd512_gather_source
    assert "indices_selected_offset" in hd512_gather_source


def test_sparse_attention_postprocess_reads_full_indices_directly():
    bridge_source = Path(
        "src/cannbench/operators/builtin/sparse_attention/simt/v1/"
        "aten_dsa_sparse_attention/csrc/sparse_attention.asc"
    ).read_text(encoding="utf-8")
    hd128_postprocess_source = Path(
        "src/cannbench/operators/builtin/sparse_attention/simt/v1/"
        "aten_dsa_sparse_attention/csrc/simt/"
        "sparse_attention_postprocess_family_hd128.asc"
    ).read_text(encoding="utf-8")
    hd512_postprocess_source = Path(
        "src/cannbench/operators/builtin/sparse_attention/simt/v1/"
        "aten_dsa_sparse_attention/csrc/simt/"
        "sparse_attention_postprocess_family_hd512.asc"
    ).read_text(encoding="utf-8")

    assert "indices_tile" not in bridge_source
    assert "indices.narrow(" not in bridge_source
    assert "indices_query_stride" in hd128_postprocess_source
    assert "indices_query_offset" in hd128_postprocess_source
    assert "indices_selected_stride" in hd128_postprocess_source
    assert "indices_selected_offset" in hd128_postprocess_source
    assert "indices_query_stride" in hd512_postprocess_source
    assert "indices_query_offset" in hd512_postprocess_source
    assert "indices_selected_stride" in hd512_postprocess_source
    assert "indices_selected_offset" in hd512_postprocess_source


def test_sparse_attention_query_pack_replaces_aten_query_tile_materialization():
    bridge_source = Path(
        "src/cannbench/operators/builtin/sparse_attention/simt/v1/"
        "aten_dsa_sparse_attention/csrc/sparse_attention.asc"
    ).read_text(encoding="utf-8")
    hd128_score_source = Path(
        "src/cannbench/operators/builtin/sparse_attention/simt/v1/"
        "aten_dsa_sparse_attention/csrc/simt/"
        "sparse_attention_score_family_hd128.asc"
    ).read_text(encoding="utf-8")
    hd512_score_source = Path(
        "src/cannbench/operators/builtin/sparse_attention/simt/v1/"
        "aten_dsa_sparse_attention/csrc/simt/"
        "sparse_attention_score_family_hd512.asc"
    ).read_text(encoding="utf-8")
    hd128_query_pack_source = Path(
        "src/cannbench/operators/builtin/sparse_attention/simt/v1/"
        "aten_dsa_sparse_attention/csrc/simt/"
        "sparse_attention_query_pack_hd128.asc"
    ).read_text(encoding="utf-8")
    hd512_query_pack_source = Path(
        "src/cannbench/operators/builtin/sparse_attention/simt/v1/"
        "aten_dsa_sparse_attention/csrc/simt/"
        "sparse_attention_query_pack_hd512.asc"
    ).read_text(encoding="utf-8")
    hd128_postprocess_source = Path(
        "src/cannbench/operators/builtin/sparse_attention/simt/v1/"
        "aten_dsa_sparse_attention/csrc/simt/"
        "sparse_attention_postprocess_family_hd128.asc"
    ).read_text(encoding="utf-8")
    hd512_postprocess_source = Path(
        "src/cannbench/operators/builtin/sparse_attention/simt/v1/"
        "aten_dsa_sparse_attention/csrc/simt/"
        "sparse_attention_postprocess_family_hd512.asc"
    ).read_text(encoding="utf-8")

    assert "query.narrow(2, query_start, current_query)" not in bridge_source
    assert ".mul(scale)" not in bridge_source
    assert ".to(at::kHalf)" not in bridge_source
    assert ".to(at::kBFloat16)" not in bridge_source
    assert "run_sparse_attention_query_pack_hd512_tile(" in bridge_source
    assert "run_sparse_attention_query_pack_hd128_tile(" in bridge_source
    assert "query_query_stride" in bridge_source
    assert "query_query_offset" in bridge_source
    assert "query_query_stride" in hd128_score_source
    assert "query_query_offset" in hd128_score_source
    assert "query_query_stride" in hd512_score_source
    assert "query_query_offset" in hd512_score_source
    assert "query_query_stride" in hd128_query_pack_source
    assert "query_query_offset" in hd128_query_pack_source
    assert "query_query_stride" in hd512_query_pack_source
    assert "query_query_offset" in hd512_query_pack_source
    assert "score_scale" in bridge_source
    assert "score_scale" in hd128_postprocess_source
    assert "score_scale" in hd512_postprocess_source


def test_sparse_attention_bfloat16_query_fast_path_skips_query_pack():
    bridge_source = Path(
        "src/cannbench/operators/builtin/sparse_attention/simt/v1/"
        "aten_dsa_sparse_attention/csrc/sparse_attention.asc"
    ).read_text(encoding="utf-8")

    assert "query_bfloat" in bridge_source
    assert "query.scalar_type() == at::ScalarType::BFloat16" in bridge_source
    assert "query.to(c10::kBFloat16).contiguous()" in bridge_source
    assert bridge_source.count("query_bfloat,") >= 4
    assert "sparse_attention_forward_family_hd512_decode_fused(" in bridge_source
    assert "sparse_attention_forward_family_hd512_hybrid(" in bridge_source
    assert "sparse_attention_forward_family_hd128_decode_fused(" in bridge_source
    assert "sparse_attention_forward_family_hd128_hybrid(" in bridge_source
    assert "score_query_stride = query_tokens" in bridge_source
    assert "score_query_stride = current_query" in bridge_source
    assert "run_sparse_attention_query_pack_hd512_tile(" in bridge_source
    assert "run_sparse_attention_query_pack_hd128_tile(" in bridge_source


def test_sparse_attention_bridge_uses_bfloat16_query_tiles():
    source = Path(
        "src/cannbench/operators/builtin/sparse_attention/simt/v1/"
        "aten_dsa_sparse_attention/csrc/sparse_attention.asc"
    ).read_text(encoding="utf-8")

    assert "const at::BFloat16* query" in source
    assert "at::BFloat16* query_tile" in source
    assert "const_data_ptr<at::BFloat16>()" in source
    assert "mutable_data_ptr<at::BFloat16>()" in source
    assert "dtype(c10::kBFloat16)" in source


def test_sparse_attention_query_pack_sources_use_bfloat16_storage():
    for head_dim in (128, 512):
        source = Path(
            "src/cannbench/operators/builtin/sparse_attention/simt/v1/"
            "aten_dsa_sparse_attention/csrc/simt/"
            f"sparse_attention_query_pack_hd{head_dim}.asc"
        ).read_text(encoding="utf-8")

        assert '#include "simt_api/asc_bf16.h"' in source
        assert "__gm__ bfloat16_t* query_tile" in source
        assert "static_cast<bfloat16_t>(query[source_offset])" in source


def test_sparse_attention_bridge_does_not_use_aten_gather_path():
    source = Path(
        "src/cannbench/operators/builtin/sparse_attention/simt/v1/"
        "aten_dsa_sparse_attention/csrc/sparse_attention.asc"
    ).read_text(encoding="utf-8")

    assert "at::gather(" not in source
    assert "launch_sparse_attention_keys_gather_pack_hd128_float" not in source
    assert "launch_sparse_attention_keys_gather_pack_hd512_float" not in source
    assert "run_sparse_attention_keys_gather_pack_hd128_tile(" not in source
    assert "run_sparse_attention_keys_gather_pack_hd512_tile(" not in source


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

    assert "run_sparse_attention_family_hd512_tile(" not in source
    assert "run_sparse_attention_family_hd512_fused_tile(" in source
    assert "run_sparse_attention_score_gather_family_hd512_tile(" in source


def test_sparse_attention_hd128_bridge_extracts_tile_helper():
    source = Path(
        "src/cannbench/operators/builtin/sparse_attention/simt/v1/"
        "aten_dsa_sparse_attention/csrc/sparse_attention.asc"
    ).read_text(encoding="utf-8")

    assert "run_sparse_attention_family_hd128_tile(" not in source
    assert "run_sparse_attention_family_hd128_prefill_fused_tile(" in source
    assert "run_sparse_attention_score_gather_family_hd128_tile(" in source
    assert "run_sparse_attention_family_hd128_decode_direct_tile(" in source


def test_sparse_attention_hd128_prefill_does_not_materialize_selected_values():
    source = Path(
        "src/cannbench/operators/builtin/sparse_attention/simt/v1/"
        "aten_dsa_sparse_attention/csrc/sparse_attention.asc"
    ).read_text(encoding="utf-8")
    prefill_source = source.split(
        "std::tuple<at::Tensor, at::Tensor> sparse_attention_forward_family_hd128_hybrid("
    )[1].split(
        "std::tuple<at::Tensor, at::Tensor> sparse_attention_forward_family_hd128_decode_fused("
    )[0]

    assert "selected_values" not in prefill_source
    assert "run_sparse_attention_values_gather_hd128_tile(" not in prefill_source
    assert "run_sparse_attention_family_hd128_decode_direct_tile(" in prefill_source


def test_sparse_attention_key_gather_score_boundary_is_fused_for_all_primary_paths():
    source = Path(
        "src/cannbench/operators/builtin/sparse_attention/simt/v1/"
        "aten_dsa_sparse_attention/csrc/sparse_attention.asc"
    ).read_text(encoding="utf-8")
    path_specs = [
        (
            "hd512_prefill",
            "sparse_attention_forward_family_hd512_hybrid(",
            "sparse_attention_forward_family_hd512_decode_fused(",
            "run_sparse_attention_score_gather_family_hd512_tile(",
        ),
        (
            "hd512_decode",
            "sparse_attention_forward_family_hd512_decode_fused(",
            "sparse_attention_forward_family_hd128_hybrid(",
            "run_sparse_attention_score_gather_family_hd512_tile(",
        ),
        (
            "hd128_prefill",
            "sparse_attention_forward_family_hd128_hybrid(",
            "sparse_attention_forward_family_hd128_decode_fused(",
            "run_sparse_attention_score_gather_family_hd128_tile(",
        ),
        (
            "hd128_decode",
            "sparse_attention_forward_family_hd128_decode_fused(",
            "std::tuple<at::Tensor, at::Tensor> sparse_attention_forward(",
            "run_sparse_attention_score_gather_family_hd128_tile(",
        ),
    ]

    for _name, start_marker, end_marker, fused_call in path_specs:
        path_source = source.split(start_marker)[1].split(end_marker)[0]
        assert fused_call in path_source
        assert "selected_keys_chunk" not in path_source
        assert "run_sparse_attention_keys_gather_pack_hd" not in path_source


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

    assert "sparse_attention_postprocess_family_hd128_decode_direct_kernel" in source
    assert "sparse_attention_postprocess_family_hd128_kernel" not in source


def test_sparse_attention_hd512_postprocess_source_uses_postprocess_symbol_names():
    source = Path(
        "src/cannbench/operators/builtin/sparse_attention/simt/v1/"
        "aten_dsa_sparse_attention/csrc/simt/"
        "sparse_attention_postprocess_family_hd512.asc"
    ).read_text(encoding="utf-8")

    assert "sparse_attention_postprocess_family_hd512_decode_direct_kernel" in source
    assert "sparse_attention_postprocess_family_hd512_kernel" not in source


def test_sparse_attention_hd128_decode_score_fuses_key_gather_but_not_postprocess():
    score_source = Path(
        "src/cannbench/operators/builtin/sparse_attention/simt/v1/"
        "aten_dsa_sparse_attention/csrc/simt/"
        "sparse_attention_score_family_hd128.asc"
    ).read_text(encoding="utf-8")
    postprocess_source = Path(
        "src/cannbench/operators/builtin/sparse_attention/simt/v1/"
        "aten_dsa_sparse_attention/csrc/simt/"
        "sparse_attention_postprocess_family_hd128.asc"
    ).read_text(encoding="utf-8")

    assert "launch_sparse_attention_score_hd128_decode_direct_float" not in score_source
    assert "launch_sparse_attention_score_gather_hd128_float" in score_source
    assert "sparse_attention_score_gather_family_hd128_aiv(" in score_source
    assert "CopyUB2L1{}" in score_source
    assert "AscendC::CrossCoreSetFlag<kCrossCoreSyncMode, PIPE_MTE3>(" in score_source
    assert "launch_sparse_attention_hd128_postprocess_decode_direct_float" in postprocess_source


def test_sparse_attention_hd128_score_gather_keeps_gather_on_aiv_side():
    source = _score_source(128)
    aic_source = _function_body(
        source,
        "__aicore__ inline void sparse_attention_score_family_hd128_aic(",
        "__aicore__ inline void sparse_attention_score_gather_family_hd128_aiv(",
    )
    gather_source = _function_body(
        source,
        "__aicore__ inline void sparse_attention_score_gather_family_hd128_aiv(",
        "__global__ __aicore__ void sparse_attention_score_family_hd128_kernel(",
    )

    assert "AscendC::LocalTensor<bfloat16_t>" in source
    assert "keys[" in gather_source
    assert "indices[" in gather_source
    assert "keys[" not in aic_source
    assert "indices[" not in aic_source


def test_sparse_attention_hd512_score_gather_keeps_gather_on_aiv_side():
    source = _score_source(512)
    aic_source = _function_body(
        source,
        "__aicore__ inline void sparse_attention_score_gather_family_hd512_aic(",
        "__aicore__ inline void sparse_attention_score_gather_family_hd512_aiv(",
    )
    gather_source = _function_body(
        source,
        "__aicore__ inline void sparse_attention_score_gather_family_hd512_aiv(",
        "__global__ __aicore__ void sparse_attention_score_gather_family_hd512_kernel(",
    )

    assert "AscendC::LocalTensor<bfloat16_t>" in source
    assert "CrossCoreSetFlag<" in source
    assert "CrossCoreWaitFlag<" in source
    assert "keys[" in gather_source
    assert "indices[" in gather_source
    assert "keys[" not in aic_source
    assert "indices[" not in aic_source


def test_sparse_attention_score_gather_uses_single_mixed_kernel_launch():
    for head_dim in (128, 512):
        source = _score_source(head_dim)
        launcher_source = source.split(
            f'extern "C" void launch_sparse_attention_score_gather_hd{head_dim}_float(',
            1,
        )[1]
        score_launch = (
            (
                f"sparse_attention_score_gather_family_hd{head_dim}_kernel"
                "<<<shape.used_core_num, 0, stream>>>"
            )
            if head_dim == 512
            else (
                f"sparse_attention_score_family_hd{head_dim}_kernel"
                "<<<shape.used_core_num, 0, stream>>>"
            )
        )

        assert score_launch in launcher_source
        assert (
            f"sparse_attention_gather_pack_family_hd{head_dim}_kernel<<<"
            not in launcher_source
        )


def test_sparse_attention_hd512_score_gather_offsets_keys_by_batch_only():
    source = _score_source(512)
    gather_source = _function_body(
        source,
        "__aicore__ inline void sparse_attention_score_gather_family_hd512_aiv(",
        "__global__ __aicore__ void sparse_attention_score_gather_family_hd512_kernel(",
    )

    assert "const int64_t head_index = head_row % query_heads;" not in gather_source
    assert "batch_index * context_tokens * kHeadDim" in gather_source


def test_sparse_attention_score_aic_orders_copy_mmad_and_global_store():
    for head_dim in (128, 512):
        source = _score_source(head_dim)
        start_marker = (
            f"__aicore__ inline void sparse_attention_score_family_hd{head_dim}_aic("
            if head_dim == 128
            else f"__aicore__ inline void sparse_attention_score_gather_family_hd{head_dim}_aic("
        )
        end_marker = (
            f"__global__ __aicore__ void sparse_attention_score_family_hd{head_dim}_kernel("
            if head_dim == 128
            else f"__aicore__ inline void sparse_attention_score_gather_family_hd{head_dim}_aiv("
        )
        body = _function_body(
            source,
            start_marker,
            end_marker,
        )

        assert "Copy(copy_gm_to_l1" in body
        assert "Copy(copy_l1_to_l0" in body
        assert "Mmad(" in body
        assert "Copy(copy_l0c_to_gm" in body
        assert body.rindex("Copy(copy_l0c_to_gm") > body.rindex("Mmad(")


def test_sparse_attention_score_source_does_not_use_gm_tile_scratch_layout():
    for head_dim in (128, 512):
        source = _score_source(head_dim)

        assert "SparseAttentionScratchLayout" not in source
        assert "make_hd" not in source
        assert "scratch.packed_keys" not in source


def test_sparse_attention_hd512_decode_score_fuses_key_gather_but_not_postprocess():
    score_source = Path(
        "src/cannbench/operators/builtin/sparse_attention/simt/v1/"
        "aten_dsa_sparse_attention/csrc/simt/"
        "sparse_attention_score_family_hd512.asc"
    ).read_text(encoding="utf-8")
    postprocess_source = Path(
        "src/cannbench/operators/builtin/sparse_attention/simt/v1/"
        "aten_dsa_sparse_attention/csrc/simt/"
        "sparse_attention_postprocess_family_hd512.asc"
    ).read_text(encoding="utf-8")

    assert "launch_sparse_attention_score_hd512_decode_direct_float" not in score_source
    assert "launch_sparse_attention_score_gather_hd512_float" in score_source
    assert "sparse_attention_score_gather_family_hd512_aiv" in score_source
    assert "sparse_attention_score_gather_family_hd512_kernel" in score_source
    assert "constexpr int32_t kMaxUsedCoreNum = 11;" in score_source
    assert "constexpr uint8_t kCrossCoreSyncMode = 4;" in score_source
    assert "if (AscendC::GetSubBlockIdx() != 0) {" in score_source
    assert "AscendC::GetBlockIdx() / AscendC::GetTaskRatio()" in score_source
    assert "CrossCoreSetFlag<kCrossCoreSyncMode, PIPE_MTE3>(ready_flag);" in score_source
    assert "launch_sparse_attention_hd512_postprocess_decode_direct_float" in postprocess_source


def test_sparse_attention_bridge_does_not_keep_debug_zero_output_path():
    source = Path(
        "src/cannbench/operators/builtin/sparse_attention/simt/v1/"
        "aten_dsa_sparse_attention/csrc/sparse_attention.asc"
    ).read_text(encoding="utf-8")

    assert "output_tile.zero_();" not in source


def test_sparse_attention_bridge_does_not_use_aten_softmax_path():
    source = Path(
        "src/cannbench/operators/builtin/sparse_attention/simt/v1/"
        "aten_dsa_sparse_attention/csrc/sparse_attention.asc"
    ).read_text(encoding="utf-8")

    assert "at::softmax(" not in source
    assert "at::logsumexp(" not in source
