from pathlib import Path


def _score_source(head_dim: int) -> str:
    return Path(
        "src/cannbench/operators/builtin/sparse_attention/simt/v1/"
        "aten_dsa_sparse_attention/csrc/simt/"
        f"sparse_attention_score_family_hd{head_dim}.asc"
    ).read_text(encoding="utf-8")


def _bridge_source() -> str:
    return Path(
        "src/cannbench/operators/builtin/sparse_attention/simt/v1/"
        "aten_dsa_sparse_attention/csrc/sparse_attention.asc"
    ).read_text(encoding="utf-8")


def _head64_source() -> str:
    path = (
        Path(__file__).parents[1]
        / "v1/aten_dsa_sparse_attention/csrc/simt"
        / "sparse_attention_head64_hd576.asc"
    )
    assert path.is_file(), f"missing Head64 device source: {path}"
    return path.read_text(encoding="utf-8")


def _head64_plan_source() -> str:
    path = (
        Path(__file__).parents[1]
        / "v1/aten_dsa_sparse_attention/csrc/simt"
        / "sparse_attention_head64_plan.h"
    )
    assert path.is_file(), f"missing Head64 plan header: {path}"
    return path.read_text(encoding="utf-8")


def _function_body(source: str, start_marker: str, end_marker: str) -> str:
    return source.split(start_marker, 1)[1].split(end_marker, 1)[0]


def test_sparse_attention_custom_op_schema_keeps_legacy_tuning_defaults():
    source = _bridge_source()

    assert "int64_t head_tile" in source
    assert "int64_t selected_partitions" in source
    assert "int head_tile=1, int selected_partitions=1" in source


def test_sparse_attention_head64_sources_are_registered():
    project = Path(__file__).parents[1] / "v1" / "aten_dsa_sparse_attention"
    setup_source = (Path(__file__).parents[1] / "v1" / "setup.py").read_text()

    assert (project / "csrc/simt/sparse_attention_head64_plan.h").is_file()
    assert (project / "csrc/simt/sparse_attention_head64_hd576.asc").is_file()
    assert "libsparse_attention_head64_hd576_kernel.so" in setup_source
    assert "sparse_attention_head64_hd576.asc" in setup_source


def test_sparse_attention_head64_plan_keeps_dynamic_task_mapping():
    plan = _head64_plan_source()
    bridge = _bridge_source()
    device = _head64_source()

    assert "struct SparseAttentionHead64Plan" in plan
    assert "constexpr int32_t kHead64PhysicalAicLimit = 32;" in plan
    assert "make_sparse_attention_head64_plan(" in bridge
    assert (
        "query.size(0) * query.size(2) * head_group_count * "
        "selected_partitions"
    ) in bridge
    assert "std::min<int64_t>(task_count, kHead64PhysicalAicLimit)" in bridge
    assert "task_id % plan.selected_partitions" in device
    assert "task_id /= plan.selected_partitions" in device
    assert "task_id % plan.head_group_count" in device
    assert "task_id /= plan.head_group_count" in device
    assert "task_id % plan.query_tokens" in device
    assert "task_id / plan.query_tokens" in device


def test_sparse_attention_head64_host_supports_split_kv_route():
    bridge = _bridge_source()
    plan = _head64_plan_source()

    assert "is_supported_head64_partitions" in bridge
    assert "selected_partitions == 2" in bridge
    assert "selected_partitions == 4" in bridge
    assert "selected_partition_tile_capacity" in plan
    assert "selected_tile_count" in bridge
    assert "partition_tile_capacity" in bridge


def test_sparse_attention_head64_task_mapping_keeps_partition_innermost():
    source = _head64_source()
    assert "int32_t partition;" in source
    assert "task_id % plan.selected_partitions" in source
    assert "task_id /= plan.selected_partitions" in source
    assert "head64_partition_begin" in source
    assert "head64_partition_end" in source


def test_sparse_attention_head64_qk_uses_partition_local_scores():
    source = _head64_source()
    bridge = _bridge_source()
    assert "plan.selected_partition_tile_capacity * plan.selected_tile" in bridge
    assert "partition_token_capacity" in source
    assert "partition_begin + local_selected_start" in source
    assert "logical_task) * kHead64Tile * partition_token_capacity" in " ".join(
        source.split()
    )


def test_sparse_attention_head64_source_uses_only_allowed_basic_api():
    source = _head64_source()
    basic_headers = {
        line.strip()
        for line in source.splitlines()
        if line.strip().startswith('#include "basic_api/')
    }

    assert basic_headers == {
        '#include "basic_api/kernel_common.h"',
        '#include "basic_api/kernel_operator_block_sync_intf.h"',
    }
    assert '#include "kernel_operator.h"' not in source
    assert "AscendC::SetFlag" not in source
    assert "AscendC::WaitFlag" not in source
    assert "AscendC::PipeBarrier" not in source
    assert "AscendC::SyncAll" not in source
    assert "AscendC::InitSocState" not in source


def test_sparse_attention_head64_uses_1024_thread_dual_aiv_contract():
    source = _head64_source()

    assert "__launch_bounds__(1024)" in source
    assert "dim3(1024, 1, 1)" in source
    assert "GetSubBlockIdx()" in source
    assert "GetSubBlockIdx() != 0" not in source
    assert "threadIdx.x / 32" in source
    assert "threadIdx.x % 32" in source
    assert "constexpr uint8_t kAivToAicReady = 8;" in source
    assert "constexpr uint8_t kAicToAivReady = 9;" in source
    assert "CrossCoreSetFlag<2" in source
    assert "CrossCoreWaitFlag<2" in source


def test_sparse_attention_head64_reads_subblock_outside_simt_vf():
    source = _head64_source()
    vf_body = _function_body(
        source,
        "head64_probe_vf(",
        "__global__ __aicore__ void sparse_attention_head64_probe_kernel(",
    )

    assert "uint32_t subblock_index" in vf_body
    assert "GetSubBlockIdx()" not in vf_body
    assert "const uint32_t subblock_index = AscendC::GetSubBlockIdx();" in source
    assert "probe, task_id, subblock_index" in source


def test_sparse_attention_head64_qk_uses_m64_tensor_api():
    source = _head64_source()

    assert "launch_sparse_attention_head64_qk_hd576_bf16" in source
    assert "sparse_attention_head64_qk_kernel" in source
    assert "MmadParams" in source
    assert "kHead64Tile" in source
    assert "params.m = 64" in source
    assert "params.n = current_selected" in source
    assert "params.k = current_k" in source
    assert "MakeMmad(" in source


def test_sparse_attention_head64_qk_synchronizes_mmad_and_fixpipe():
    source = _head64_source()
    aic = _function_body(
        source,
        "sparse_attention_head64_qk_aic(",
        "__global__ __aicore__ void sparse_attention_head64_qk_kernel(",
    )

    assert "asc_sync_notify(PIPE_FIX, PIPE_M, EVENT_ID0);" in aic
    assert "asc_sync_wait(PIPE_FIX, PIPE_M, EVENT_ID0);" in aic
    assert "asc_sync_notify(PIPE_M, PIPE_FIX, EVENT_ID0);" in aic
    assert "asc_sync_wait(PIPE_M, PIPE_FIX, EVENT_ID0);" in aic

    mm_to_fix = aic.index("asc_sync_notify(PIPE_M, PIPE_FIX, EVENT_ID0);")
    fix_wait = aic.index("asc_sync_wait(PIPE_M, PIPE_FIX, EVENT_ID0);")
    fix_copy = aic.index("Copy(copy_l0c_to_gm, gm_score_tile, l0_scores);")
    assert mm_to_fix < fix_wait < fix_copy


def test_sparse_attention_head64_qk_synchronizes_l0_inputs_and_mmad():
    source = _head64_source()
    aic = _function_body(
        source,
        "sparse_attention_head64_qk_aic(",
        "__global__ __aicore__ void sparse_attention_head64_qk_kernel(",
    )

    assert "asc_sync_notify(PIPE_M, PIPE_MTE1, EVENT_ID1);" in aic
    assert "asc_sync_wait(PIPE_M, PIPE_MTE1, EVENT_ID1);" in aic
    assert "asc_sync_notify(PIPE_MTE1, PIPE_M, EVENT_ID1);" in aic
    assert "asc_sync_wait(PIPE_MTE1, PIPE_M, EVENT_ID1);" in aic

    release_wait = aic.index("asc_sync_wait(PIPE_M, PIPE_MTE1, EVENT_ID1);")
    copy_a = aic.index("Copy(copy_l1_to_l0a, l0_query, l1_query_tile);")
    copy_b = aic.index("Copy(copy_l1_to_l0b, l0_keys, l1_keys);")
    ready_notify = aic.index("asc_sync_notify(PIPE_MTE1, PIPE_M, EVENT_ID1);")
    ready_wait = aic.index("asc_sync_wait(PIPE_MTE1, PIPE_M, EVENT_ID1);")
    mm = aic.index("Mmad(mm.with(params), l0_scores, l0_query, l0_keys);")
    assert release_wait < copy_a < copy_b < ready_notify < ready_wait < mm


def test_sparse_attention_head64_qk_kernel_avoids_const_bf16_launch_abi():
    source = _head64_source()
    kernel = _function_body(
        source,
        "sparse_attention_head64_qk_kernel(",
        "}  // namespace",
    )

    assert "__gm__ bfloat16_t* query" in kernel
    assert "__gm__ bfloat16_t* keys" in kernel
    assert "const_cast<bfloat16_t*>(query)" in source
    assert "const_cast<bfloat16_t*>(keys)" in source


def test_sparse_attention_hd512_postprocess_avoids_const_bf16_launch_abi():
    source = Path(
        "src/cannbench/operators/builtin/sparse_attention/simt/v1/"
        "aten_dsa_sparse_attention/csrc/simt/"
        "sparse_attention_postprocess_family_hd512.asc"
    ).read_text(encoding="utf-8")
    kernel = _function_body(
        source,
        "sparse_attention_postprocess_family_hd512_decode_direct_kernel(",
        "} // namespace",
    )

    assert "__gm__ bfloat16_t* values" in kernel
    assert "const_cast<bfloat16_t*>(values)" in source


def test_sparse_attention_head64_qk_maps_dynamic_tasks():
    source = _head64_source()
    normalized = " ".join(source.split())

    assert "logical_task += plan.used_core_num" in source
    assert "plan.task_count" in source
    assert "plan.head_group_count" in source
    assert "plan.query_tokens" in source
    assert "plan.selected_tokens" in source
    assert (
        "static_cast<int64_t>(logical_task) * kHead64Tile * "
        "partition_token_capacity"
    ) in normalized


def test_sparse_attention_head64_qk_uses_both_aiv_for_query_and_key_pack():
    source = _head64_source()

    assert "head64_query_pack_vf" in source
    assert "head64_key_pack_vf" in source
    assert "const uint32_t sub_block = AscendC::GetSubBlockIdx();" in source
    assert "const uint32_t head_begin = sub_block * 32;" in source
    assert "const uint32_t selected_begin = sub_block * 32;" in source
    assert "CrossCoreSetFlag<2" in source
    assert "CrossCoreWaitFlag<2" in source


def test_sparse_attention_head64_qk_has_staged_host_route():
    source = _bridge_source()
    normalized = " ".join(source.split())

    assert "sparse_attention_forward_family_hd576_head64(" in source
    assert "launch_sparse_attention_head64_qk_hd576_bf16(" in source
    assert (
        "{plan.task_count, kHead64Tile, "
        "plan.selected_partition_tile_capacity * plan.selected_tile}"
    ) in normalized


def test_sparse_attention_head64_routes_task_scores_to_staged_pv():
    source = _bridge_source()
    head64 = _function_body(
        source,
        "sparse_attention_forward_family_hd576_head64(",
        "std::tuple<at::Tensor, at::Tensor> sparse_attention_forward_privateuse1(",
    )

    assert "run_sparse_attention_head64_qk_hd576_bf16(" in head64
    assert "run_sparse_attention_head64_pv_hd576_bf16(" in head64
    assert "task_probabilities" in head64
    assert "task_output" in head64
    assert "legacy_scores" not in head64
    assert "run_sparse_attention_family_hd512_decode_direct_tile(" not in head64


def test_sparse_attention_head64_softmax_uses_1024_threads():
    source = _head64_source()

    assert "head64_online_softmax_vf" in source
    assert "__launch_bounds__(1024)" in source
    assert "const uint32_t local_head = threadIdx.x / 32;" in source
    assert "const uint32_t lane = threadIdx.x % 32;" in source
    assert "selected_index += 32" in source
    assert "dim_index += 32" in source


def test_sparse_attention_head64_pv_uses_cube_m64_n128():
    source = _head64_source()

    assert "launch_sparse_attention_head64_pv_hd576_bf16" in source
    assert "sparse_attention_head64_pv_kernel" in source
    assert "pv_params.m = 64" in source
    assert "pv_params.n = current_value" in source
    assert "pv_params.k = current_selected" in source
    assert "kHead64ValueTile" in source


def test_sparse_attention_head64_pv_keeps_bf16_gm_tensor_non_const():
    source = _head64_source()
    aic = _function_body(
        source,
        "sparse_attention_head64_pv_aic(",
        "__global__ __aicore__ void sparse_attention_head64_pv_kernel(",
    )

    assert "__gm__ bfloat16_t* probabilities" in aic
    assert "sqrtf(static_cast<float>(plan.qk_head_dim))" not in source


def test_sparse_attention_head64_pv_uses_int32_causal_device_abi():
    source = _head64_source()
    softmax = source.split("head64_online_softmax_vf(", 1)[1].split(") {", 1)[0]
    kernel = source.split("sparse_attention_head64_pv_kernel(", 1)[1].split(
        ") {", 1
    )[0]

    assert "int32_t causal" in softmax
    assert "bool causal" not in softmax
    assert "int32_t causal" in kernel
    assert "bool causal" not in kernel
    assert "static_cast<int32_t>(causal)" in source


def test_sparse_attention_head64_softmax_does_not_read_invalid_scores():
    source = _head64_source()
    softmax = _function_body(
        source,
        "head64_probability_pack_vf(",
        "__simt_vf__ __aicore__ __launch_bounds__(1024) inline void\nhead64_value_pack_vf(",
    )

    invalid_branch = softmax.index("if (!valid || running_sum <= 0.0F) {")
    score_read = softmax.index(
        "scores[row_offset + selected_index] * score_scale - running_max",
        invalid_branch,
    )
    assert "continue;" in softmax[invalid_branch:score_read]


def test_sparse_attention_head64_probability_checkpoint_uses_mte3_before_ready():
    source = _head64_source()
    aiv = _function_body(
        source,
        "sparse_attention_head64_pv_aiv(",
        "__aicore__ inline void sparse_attention_head64_pv_aic(",
    )

    pack = aiv.index("asc_vf_call<head64_probability_pack_vf>(")
    copy = aiv.index("asc_copy_ub2gm_align(", pack)
    ready = aiv.index(
        "CrossCoreSetFlag<4, PIPE_MTE3>(", copy
    )
    assert pack < copy < ready


def test_sparse_attention_head64_crosscore_handshakes_are_paired():
    source = _head64_source()

    assert "constexpr uint8_t kAivToAicReady = 8;" in source
    assert "constexpr uint8_t kAicToAivReady = 9;" in source
    assert source.count("CrossCoreSetFlag<2") >= 2
    assert "CrossCoreWaitFlag<2" in source
    assert "CrossCoreSetFlag<0" not in source
    assert "CrossCoreSetFlag<1" not in source


def test_sparse_attention_head64_pv_waits_for_both_aiv_subblocks():
    source = _head64_source()
    aiv = _function_body(
        source,
        "sparse_attention_head64_pv_aiv(",
        "__aicore__ inline void sparse_attention_head64_pv_aic(",
    )
    aic = _function_body(
        source,
        "sparse_attention_head64_pv_aic(",
        "__global__ __aicore__ void sparse_attention_head64_pv_kernel(",
    )

    assert "CrossCoreSetFlag<4, PIPE_MTE3>(" in aiv
    assert "kAivToAicReady + sub_block * 16" in aiv
    assert "CrossCoreWaitFlag<4, PIPE_MTE1>(kAivToAicReady);" in aic
    assert "kAivToAicReady + 16" in aic
    assert "CrossCoreSetFlag<2, PIPE_MTE1>(kAicToAivReady);" in aic
    assert "CrossCoreWaitFlag<2, PIPE_V>(kAicToAivReady);" in aiv


def test_sparse_attention_head64_recomputes_first_pv_value_tile_with_cube():
    source = _head64_source()
    aiv = _function_body(
        source,
        "sparse_attention_head64_pv_aiv(",
        "__aicore__ inline void sparse_attention_head64_pv_aic(",
    )
    aic = _function_body(
        source,
        "sparse_attention_head64_pv_aic(",
        "__global__ __aicore__ void sparse_attention_head64_pv_kernel(",
    )
    value_tile_count = (
        "const int32_t value_tile_count =\n"
        "        (plan.value_head_dim + kHead64ValueTile - 1) /\n"
        "        kHead64ValueTile;"
    )

    assert "head64_pv_value_tile_index" in source
    assert "return value_iteration == 0 ? 0 : value_iteration - 1;" in source
    assert "TODO(cann): remove the duplicated first PV tile" in source
    for body in (aiv, aic):
        assert value_tile_count in body
        assert "for (int32_t value_iteration = 0;" in body
        assert "value_iteration <= value_tile_count;" in body
        assert (
            "head64_pv_value_tile_index(value_iteration) *\n"
            "          kHead64ValueTile" in body
        )

    assert "head64_first_value_tile_repair_vf" not in source
    assert "kAicToAivPvDone" not in source


def test_sparse_attention_head64_reduced_accuracy_covers_boundaries():
    path = Path(__file__).with_name("head64_reduced_accuracy.py")

    assert path.is_file()
    source = path.read_text(encoding="utf-8")
    for case_name in (
        "valid_s64",
        "tail_s70",
        "invalid_causal_s70",
        "all_invalid_s17",
    ):
        assert case_name in source
    assert "atol=0.05" in source
    assert "rtol=0.05" in source
    assert "torch.count_nonzero(actual_output)" in source
    assert "torch.isneginf(actual_lse).all()" in source


def test_sparse_attention_host_uses_right_aligned_absolute_query_start():
    source = Path(
        "src/cannbench/operators/builtin/sparse_attention/simt/v1/"
        "aten_dsa_sparse_attention/csrc/sparse_attention.asc"
    ).read_text(encoding="utf-8")

    assert source.count(
        "const auto absolute_query_start = context_tokens - query_tokens + query_start;"
    ) == 4
    assert source.count(
        "scale,\n        context_tokens - query_tokens,\n        causal"
    ) == 4
    assert source.count("scale,\n        absolute_query_start,\n        causal") == 4


def test_sparse_attention_host_applies_causal_mask_to_nextn_decode():
    source = Path(
        "src/cannbench/operators/builtin/sparse_attention/simt/v1/"
        "aten_dsa_sparse_attention/csrc/sparse_attention.asc"
    ).read_text(encoding="utf-8")

    assert "const bool apply_causal_mask = causal;" in source
    assert 'causal && phase == "prefill"' not in source


def test_sparse_attention_simt_v1_setup_uses_bisheng_toolchain():
    setup_py = Path(
        "src/cannbench/operators/builtin/sparse_attention/simt/v1/setup.py"
    ).read_text(encoding="utf-8")

    assert "bisheng" in setup_py
    assert 'library_name = "aten_dsa_sparse_attention"' in setup_py
    assert "\"-shared\"" in setup_py


def test_sparse_attention_build_isolates_each_device_source_library():
    setup_py = Path(
        "src/cannbench/operators/builtin/sparse_attention/simt/v1/setup.py"
    ).read_text(encoding="utf-8")

    expected_libraries = (
        "libsparse_attention_postprocess_family_hd128_kernel.so",
        "libsparse_attention_postprocess_family_hd512_kernel.so",
        "libsparse_attention_query_pack_hd128_kernel.so",
        "libsparse_attention_query_pack_hd512_kernel.so",
        "libsparse_attention_score_family_hd128_kernel.so",
        "libsparse_attention_score_family_hd512_kernel.so",
    )
    for library in expected_libraries:
        assert f'"{library}"' in setup_py

    assert "for kernel_library, kernel_source in KERNEL_LIBRARIES.items()" in setup_py
    assert "self._build_kernel_library(" in setup_py
    assert '"-Wl,-rpath,$ORIGIN"' in setup_py
    assert "outputs.extend(self._kernel_outputs)" in setup_py
    assert "def copy_extensions_to_source(self):" in setup_py
    assert "def get_output_mapping(self):" in setup_py
    assert "sources=HOST_SOURCES" in setup_py


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
    bf16_body = bf16_body.split("}\n\n  for (int64_t query_start", 1)[0]

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
    bf16_body = bf16_body.split("}\n\n  for (int64_t query_start", 1)[0]

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
    bf16_body = bf16_body.split("}\n\n  for (int64_t query_start", 1)[0]

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
    bf16_body = bf16_body.split("}\n\n  for (int64_t query_start", 1)[0]

    assert "run_sparse_attention_family_hd512_fused_tile(" in bf16_body
    assert "auto scores = at::empty(" not in bf16_body
    assert "run_sparse_attention_score_gather_family_hd512_tile(" not in bf16_body
    assert "run_sparse_attention_family_hd512_decode_direct_tile(" not in bf16_body


def test_sparse_attention_bf16_fused_paths_submit_before_host_query_loop():
    source = Path(
        "src/cannbench/operators/builtin/sparse_attention/simt/v1/"
        "aten_dsa_sparse_attention/csrc/sparse_attention.asc"
    ).read_text(encoding="utf-8")
    functions = (
        "sparse_attention_forward_family_hd512_hybrid",
        "sparse_attention_forward_family_hd512_decode_fused",
        "sparse_attention_forward_family_hd128_hybrid",
        "sparse_attention_forward_family_hd128_decode_fused",
    )

    for index, function in enumerate(functions):
        end_marker = (
            f"std::tuple<at::Tensor, at::Tensor> {functions[index + 1]}("
            if index + 1 < len(functions)
            else "std::tuple<at::Tensor, at::Tensor> sparse_attention_forward("
        )
        body = _function_body(
            source,
            f"std::tuple<at::Tensor, at::Tensor> {function}(",
            end_marker,
        )
        bf16_branch = body.split(
            "if (query.scalar_type() == at::ScalarType::BFloat16) {",
            1,
        )[1].split("for (int64_t query_start", 1)[0]

        assert "return {output, lse};" in bf16_branch
        assert body.index("if (query.scalar_type() == at::ScalarType::BFloat16)") < body.index(
            "for (int64_t query_start"
        )


def test_sparse_attention_fused_launchers_submit_one_persistent_kernel():
    for head_dim, launcher_name, kernel_name in (
        (
            128,
            "launch_sparse_attention_hd128_prefill_fused_float",
            "sparse_attention_prefill_fused_family_hd128_kernel",
        ),
        (
            512,
            "launch_sparse_attention_hd512_fused_float",
            "sparse_attention_fused_family_hd512_kernel",
        ),
    ):
        source = _score_source(head_dim)
        launcher = source.split(f'extern "C" void {launcher_name}(', 1)[1]

        assert launcher.count(f"{kernel_name}<<<") == 1
        assert "for (int64_t row_start" not in launcher
        assert "row_index += shape.used_core_num" in source


def test_sparse_attention_hd512_fused_kernel_uses_all_aicore_pairs():
    source = _score_source(512)
    fused = _function_body(
        source,
        "__aicore__ inline void sparse_attention_fused_family_hd512_aic(",
        "__global__ __aicore__ void sparse_attention_fused_family_hd512_kernel(",
    )

    assert "constexpr int32_t kFusedMaxUsedCoreNum = 32;" in source
    assert "constexpr uint8_t kScoreTileReadyFlag = 1;" in source
    assert "kGatherKeysReadyFlagBase + block_idx" not in fused
    assert "kScoreTileReadyFlagBase + block_idx" not in fused


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
    assert "context_tokens * head_dim" in fused
    assert "* head_dim +" in fused
    assert "kHeadDim" not in fused


def test_sparse_attention_wide_fused_kernel_uses_distinct_value_head_dim():
    source = _score_source(512)
    wrapper = (
        Path(__file__).parents[1]
        / "v1"
        / "aten_dsa_sparse_attention"
        / "csrc"
        / "sparse_attention.asc"
    ).read_text()

    assert "int32_t value_head_dim" in source
    assert "dim_index < value_head_dim" in source
    assert "const int64_t value_offset" in source
    assert "* value_token_stride +" in source
    assert "batch_index * context_tokens * value_token_stride" in source
    assert "Tensor shared_kv, Tensor indices, int value_head_dim" in wrapper
    assert "value_head_dim > 0 && value_head_dim <= shared_kv.size(3)" in wrapper
    assert "query_tokens, value_head_dim}" in wrapper
    assert "values.size(3) == expected_head_dim" not in wrapper


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
