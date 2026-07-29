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


def _head64_fused_source() -> str:
    path = (
        Path(__file__).parents[1]
        / "v1/aten_dsa_sparse_attention/csrc/simt"
        / "sparse_attention_head64_fused_hd576.asc"
    )
    assert path.is_file(), f"missing fused Head64 device source: {path}"
    return path.read_text(encoding="utf-8")


def _head64_common_source() -> str:
    path = (
        Path(__file__).parents[1]
        / "v1/aten_dsa_sparse_attention/csrc/simt"
        / "sparse_attention_head64_common.h"
    )
    assert path.is_file(), f"missing Head64 common header: {path}"
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


def _function_definition(
    source: str, start_marker: str, declaration_marker: str | None = None
) -> str:
    start = source.index(start_marker)
    if declaration_marker is not None:
        start = source.rindex(declaration_marker, 0, start)
    while True:
        body_start = source.index("{", start)
        declaration_end = source.find(";", start, body_start)
        if declaration_end == -1:
            break
        start = source.index(start_marker, start + len(start_marker))
    depth = 0
    for index in range(body_start, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return source[start : index + 1]
    raise AssertionError(f"unterminated function definition: {start_marker}")


def _normalized_whitespace(source: str) -> str:
    return " ".join(source.split())


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


def test_sparse_attention_head64_fused_source_is_registered_separately():
    setup_source = (Path(__file__).parents[1] / "v1/setup.py").read_text()

    assert _head64_common_source()
    assert _head64_fused_source()
    assert "libsparse_attention_head64_fused_hd576_kernel.so" in setup_source
    assert "sparse_attention_head64_fused_hd576.asc" in setup_source


def test_sparse_attention_head64_fused_source_uses_required_cross_core_flags():
    source = _head64_fused_source()
    basic_headers = {
        line.strip()
        for line in source.splitlines()
        if line.strip().startswith('#include "basic_api/')
    }

    assert '#include "c_api/asc_simd.h"' in source
    assert basic_headers == {
        '#include "basic_api/kernel_common.h"',
        '#include "basic_api/kernel_operator_block_sync_intf.h"',
    }
    assert '#include "kernel_operator.h"' not in source
    assert "CrossCoreSetFlag<2" in source
    assert "CrossCoreWaitFlag<2" in source
    assert "AscendC::SetFlag" not in source
    assert "AscendC::WaitFlag" not in source


def test_sparse_attention_head64_fused_keeps_dual_aiv_1024_contract():
    source = _head64_fused_source()

    assert "KERNEL_TYPE_MIX_AIC_1_2" in source
    assert "__launch_bounds__(1024)" in source
    assert "kHead64FusedLaunchThreads = 1024" in source
    assert "dim3(kHead64FusedLaunchThreads, 1, 1)" in source
    assert "GetSubBlockIdx()" in source
    assert "GetSubBlockIdx() != 0" not in source
    assert "threadIdx.x / 32" in source
    assert "threadIdx.x % 32" in source


def test_sparse_attention_head64_fused_qk_is_m64_tensor_api():
    source = _head64_fused_source()

    assert "Head64QkMmadTrait" in source
    assert "params.m = 64" in source
    assert "params.n = current_selected" in source
    assert "params.k = current_k" in source
    assert "Mmad(" in source


def test_sparse_attention_head64_fused_scores_stay_tile_local():
    source = _head64_fused_source()
    launcher = _function_definition(
        source, "launch_sparse_attention_head64_fused_hd576_bf16("
    )

    assert "float* scores" not in launcher
    assert "bfloat16_t* probabilities" not in launcher
    assert "l1_scores_buf" in source
    assert "ub_scores_buf" in source
    assert "ub_probabilities_buf" in source
    assert "running_max" in source
    assert "running_sum" in source


def test_sparse_attention_head64_fused_uses_one_explicit_l1_layout():
    source = _head64_fused_source()
    aiv = _function_definition(source, "sparse_attention_head64_fused_aiv(")
    aic = _function_definition(source, "sparse_attention_head64_fused_aic(")

    declaration = "__cbuf__ uint8_t l1_workspace[kHead64FusedL1Bytes];"
    assert aiv.count(declaration) == 1
    assert aic.count(declaration) == 1
    for body in (aiv, aic):
        assert "__cbuf__ bfloat16_t l1_query_buf[" not in body
        assert "__cbuf__ bfloat16_t l1_keys_buf[" not in body
        assert "__cbuf__ float l1_scores_buf[" not in body
        assert "__cbuf__ bfloat16_t l1_probabilities_buf[" not in body
        assert "__cbuf__ bfloat16_t l1_values_buf[" not in body
        assert "__cbuf__ float l1_pv_buf[" not in body
        assert "kHead64FusedL1QueryOffset" in body
        assert "kHead64FusedL1KeysOffset" in body
        assert "kHead64FusedL1ScoresOffset" in body
        assert "kHead64FusedL1ProbabilitiesOffset" in body
        assert "kHead64FusedL1ValuesOffset" in body
        assert "kHead64FusedL1PvOffset" in body


def test_sparse_attention_head64_fused_key_gather_uses_two_l1_slots():
    source = _head64_fused_source()

    assert "kHead64FusedKeyGatherSlots = 2" in source
    assert "kHead64FusedL1KeySlotBytes" in source
    assert "slot * kHead64FusedL1KeySlotBytes" in source


def test_sparse_attention_head64_fused_gather_slots_use_mode2_flags():
    source = _head64_fused_source()

    assert "kAicToAivGatherSlot0 = 0" in source
    assert "kAicToAivGatherSlot1 = 1" in source
    assert "kAivToAicGatherSlot0 = 2" in source
    assert "kAivToAicGatherSlot1 = 3" in source
    for helper in (
        "head64_aic_request_gather_slot(",
        "head64_aiv_wait_gather_slot(",
        "head64_aiv_publish_gather_slot(",
        "head64_aic_wait_gather_slot(",
    ):
        body = _function_definition(source, helper)
        assert "CrossCore" in body
        assert "<2," in body


def test_sparse_attention_head64_fused_prefetches_key_before_current_qk_mmad():
    source = _head64_fused_source()
    aic = _function_definition(source, "sparse_attention_head64_fused_aic(")

    qk_loop = aic.index("for (int32_t k_start")
    qk_request = aic.index("head64_aic_request_gather_slot(next_slot)", qk_loop)
    qk_mmad = aic.index("Mmad(mm.with(params)", qk_request)

    assert qk_request < qk_mmad


def test_sparse_attention_head64_fused_pv_is_m64_tensor_api():
    source = _head64_fused_source()

    assert "Head64PvMmadTrait" in source
    assert "pv_params.m = 64" in source
    assert "pv_params.n = current_value" in source
    assert "pv_params.k = current_selected" in source
    assert "running_output" in source
    assert "old_scale" in source


def test_sparse_attention_head64_fused_writes_combine_compatible_partials():
    source = _head64_fused_source()

    assert "task_output" in source
    assert "partial_lse" in source
    assert "running_max + logf(running_sum)" in source
    assert "running_output_values[local_head * 512 + dim] / running_sum" in source
    assert "-std::numeric_limits<float>::infinity()" in source


def test_sparse_attention_head64_fused_keeps_fixpipe_tiles_in_nz_layout():
    source = _head64_fused_source()
    helper = _normalized_whitespace(
        _function_definition(source, "head64_copy_l0c_to_l1_nz(")
    )

    assert source.count("head64_copy_l0c_to_l1_nz(") == 3
    assert "asc_copy_l0c2l1_sync(" in helper
    assert "const uint16_t aligned_m" in helper
    assert "const uint32_t dst_stride" in helper
    assert "dst_stride, aligned_m" in helper
    assert (
        "0, 0, 0, 0, 0, false, false, 0, 0, false, 0, false, false, false, "
        "false);"
    ) in helper
    assert "head64_fused_nz_offset(" in source


def test_sparse_attention_head64_fused_scores_are_delivered_from_aic_to_both_aivs():
    source = _head64_fused_source()
    aiv = _function_definition(source, "sparse_attention_head64_fused_aiv(")
    aic = _function_definition(source, "sparse_attention_head64_fused_aic(")

    declaration = "__ubuf__ uint8_t ub_workspace[kHead64FusedUbBytes];"
    assert aiv.count(declaration) == 1
    assert aic.count(declaration) == 1
    for body in (aiv, aic):
        assert "kHead64FusedUbScoresOffset" in body

    assert "asc_copy_l12ub(" not in aiv
    assert "Copy(copy_l1_to_ub, ub_scores" not in aiv
    assert aic.count("asc_copy_l12ub_sync(") >= 2
    assert "l1_scores_buf, false, 4, 64, 64, 0" in (
        _normalized_whitespace(aic)
    )
    assert "l1_scores_buf + 32 * 16, true, 4, 64, 64, 0" in (
        _normalized_whitespace(aic)
    )
    assert "CrossCoreSetFlag<2, PIPE_MTE2>(kAicToAivReady)" in aic
    assert "CrossCoreWaitFlag<2, PIPE_V>(kAicToAivReady)" in aiv


def test_sparse_attention_head64_fused_1024_launch_covers_32_rows_with_28_warps():
    source = _head64_fused_source()

    assert "kHead64FusedLaunchThreads = 1024" in source
    assert "kHead64FusedActiveThreads = 896" in source
    assert "kHead64FusedActiveWarps = 28" in source
    assert "dim3(kHead64FusedLaunchThreads, 1, 1)" in source
    assert source.count("local_head += kHead64FusedActiveWarps") >= 5
    assert "warp + kHead64FusedActiveWarps" in source
    assert source.count("local_selected += kHead64FusedActiveWarps") >= 2
    assert "index += kHead64FusedActiveThreads" in source


def test_sparse_attention_head64_fused_pv_is_delivered_from_aic_to_both_aivs():
    source = _head64_fused_source()
    aiv = _function_definition(source, "sparse_attention_head64_fused_aiv(")
    aic = _function_definition(source, "sparse_attention_head64_fused_aic(")

    assert "Copy(copy_l1_to_ub, ub_pv" not in aiv
    assert "l1_pv_buf, false, 8, 64, 64, 0" in _normalized_whitespace(aic)
    assert "l1_pv_buf + 32 * 16, true, 8, 64, 64, 0" in (
        _normalized_whitespace(aic)
    )
    assert aic.count("CrossCoreSetFlag<2, PIPE_MTE2>(kAicToAivReady)") >= 2


def test_sparse_attention_head64_fused_has_no_s64_score_probe_shortcut():
    source = _head64_fused_source()

    assert "head64_fused_score_probe_vf" not in source
    assert "plan.selected_tokens == 64" not in source


def test_sparse_attention_head64_fused_pv_reuses_mte1_event_in_order():
    source = _head64_fused_source()
    aic = _function_definition(source, "sparse_attention_head64_fused_aic(")
    probability_copy = aic.index("Copy(copy_l1_to_l0a, l0_probability")
    value_loop = aic.index("for (int32_t value_start", probability_copy)
    first_value_wait = aic.index("if (value_start != 0)", value_loop)
    value_ready = aic.index(
        "CrossCoreSetFlag<2, PIPE_MTE1>(kAicToAivReady)", first_value_wait
    )
    value_copy = aic.index("Copy(copy_l1_to_l0b, l0_values", value_ready)
    mte1_ready = aic.index(
        "asc_sync_notify(PIPE_MTE1, PIPE_M, EVENT_ID1)", value_copy
    )
    pv_mmad = aic.index("Mmad(pv_mm.with(pv_params)", mte1_ready)

    assert "asc_sync_notify(PIPE_MTE1, PIPE_M, EVENT_ID1)" not in aic[
        probability_copy:value_loop
    ]
    assert (
        probability_copy
        < value_loop
        < first_value_wait
        < value_ready
        < value_copy
        < mte1_ready
        < pv_mmad
    )


def test_sparse_attention_head64_plan_keeps_dynamic_task_mapping():
    plan = _head64_plan_source()
    bridge = _bridge_source()
    device = _head64_source() + _head64_common_source()

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


def test_sparse_attention_head64_host_requires_matching_shared_kv_width():
    bridge = _normalized_whitespace(_bridge_source())

    assert (
        'shared_kv.size(3) == 576, "head64 requires shared_kv_head_dim=576"'
        in bridge
    )


def test_sparse_attention_empty_inputs_return_initialized_attention_identity():
    bridge = _bridge_source()
    forward = _function_definition(
        bridge,
        "sparse_attention_forward_privateuse1(",
        declaration_marker="std::tuple<at::Tensor, at::Tensor>",
    )
    empty_branch = forward.split(
        "if (query.numel() == 0 || shared_kv.numel() == 0 ||", 1
    )[1].split("auto query_bfloat", 1)[0]

    assert "at::zeros(" in empty_branch
    assert "at::full(" in empty_branch
    assert "-std::numeric_limits<float>::infinity()" in empty_branch
    assert "at::empty(" not in empty_branch


def test_sparse_attention_int64_indices_are_clamped_before_int32_narrowing():
    bridge = _bridge_source()
    forward = _function_definition(
        bridge,
        "sparse_attention_forward_privateuse1(",
        declaration_marker="std::tuple<at::Tensor, at::Tensor>",
    )
    conversion = forward.split("auto indices_int =", 1)[1].split(
        "if (use_head64)", 1
    )[0]

    clamp = conversion.index(".clamp(")
    narrow = conversion.index(".to(at::kInt)")
    assert "std::numeric_limits<int32_t>::min()" in conversion
    assert "std::numeric_limits<int32_t>::max()" in conversion
    assert clamp < narrow


def test_sparse_attention_head64_task_mapping_keeps_partition_innermost():
    source = _head64_common_source()
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


def test_sparse_attention_head64_pv_uses_partition_bounds_and_local_stride():
    source = _head64_source()
    assert "partition_begin" in _function_body(
        source,
        "sparse_attention_head64_pv_aiv(",
        "__aicore__ inline void sparse_attention_head64_pv_aic(",
    )
    assert "partition_length" in source
    assert "partition_token_capacity" in source
    assert "partial_lse" in source
    assert "task.partition" in _head64_common_source()


def test_sparse_attention_head64_score_pipeline_uses_one_padded_row_stride():
    source = _head64_source()
    qk = _function_body(
        source,
        "sparse_attention_head64_qk_aic(",
        "__global__ __aicore__ void sparse_attention_head64_qk_kernel(",
    )
    softmax = _function_body(
        source,
        "head64_online_softmax_vf(",
        "head64_probability_pack_vf(",
    )
    probability_pack = _function_body(
        source,
        "head64_probability_pack_vf(",
        "head64_value_pack_vf(",
    )
    pv = _function_body(
        source,
        "sparse_attention_head64_pv_aiv(",
        "sparse_attention_head64_pv_aic(",
    )

    assert "head64_score_row_stride" in source
    assert (
        "plan.selected_partition_tile_capacity * plan.selected_tile"
        in _head64_common_source()
    )
    assert (
        "const int32_t partition_token_capacity = "
        "head64_score_row_stride(plan);"
    ) in " ".join(qk.split())
    assert "kHead64Tile * partition_token_capacity" in " ".join(qk.split())
    assert "int32_t partition_token_capacity" in softmax
    assert "task_head) * partition_token_capacity" in " ".join(softmax.split())
    assert "int32_t partition_token_capacity" in probability_pack
    assert "task_head) * partition_token_capacity" in " ".join(
        probability_pack.split()
    )
    assert (
        "const int32_t partition_token_capacity = "
        "head64_score_row_stride(plan);"
    ) in " ".join(pv.split())


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


def test_sparse_attention_head64_p4_routes_fused_then_combine():
    bridge = _bridge_source()
    body = _function_definition(
        bridge, "sparse_attention_forward_family_hd576_head64("
    )

    assert "if (plan.selected_partitions == 4)" in body
    fused = body.index("run_sparse_attention_head64_fused_hd576_bf16(")
    combine = body.index("run_sparse_attention_head64_combine_hd576_bf16(")
    assert fused < combine


def test_sparse_attention_head64_p4_has_no_full_score_probability_workspace():
    bridge = _bridge_source()
    body = _function_definition(
        bridge, "sparse_attention_forward_family_hd576_head64("
    )

    p4 = body.split("if (plan.selected_partitions == 4)", 1)[1].split(
        "auto task_scores", 1
    )[0]
    assert "task_scores" not in p4
    assert "task_probabilities" not in p4
    assert "run_sparse_attention_head64_qk_hd576_bf16" not in p4
    assert "run_sparse_attention_head64_pv_hd576_bf16" not in p4


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
        "scores[score_row + local_selected] * score_scale - running_max",
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


def test_sparse_attention_head64_qk_copies_query_before_releasing_aiv():
    source = _head64_source()
    aic = _function_body(
        source,
        "sparse_attention_head64_qk_aic(",
        "__global__ __aicore__ void sparse_attention_head64_qk_kernel(",
    )

    task_query_ready = aic.index(
        "CrossCoreWaitFlag<2, PIPE_MTE1>(kAivToAicReady)"
    )
    query_copy = aic.index("Copy(copy_l1_to_l0a", task_query_ready)
    release_aiv = aic.index(
        "CrossCoreSetFlag<2, PIPE_MTE1>(kAicToAivReady)", task_query_ready
    )
    key_ready = aic.index(
        "CrossCoreWaitFlag<2, PIPE_MTE1>(kAivToAicReady)",
        task_query_ready + 1,
    )
    key_copy = aic.index("Copy(copy_l1_to_l0b", key_ready)

    assert task_query_ready < query_copy < release_aiv < key_ready < key_copy


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


def test_sparse_attention_head64_combine_uses_1024_thread_dual_aiv():
    source = _head64_source()
    combine_vf = _function_definition(
        source,
        "head64_combine_vf(",
        declaration_marker="__simt_vf__",
    )
    combine_kernel = _function_definition(
        source,
        "__global__ __aicore__ void sparse_attention_head64_combine_kernel(",
    )
    launcher = _function_definition(
        source,
        "extern \"C\" void launch_sparse_attention_head64_combine_hd576_bf16(",
    )
    combine_scope = combine_vf + combine_kernel + launcher
    combine_vf_normalized = _normalized_whitespace(combine_vf)
    combine_kernel_normalized = _normalized_whitespace(combine_kernel)

    assert "__launch_bounds__(1024)" in combine_vf
    assert "partial_lse" in combine_vf
    assert "partial_output" in combine_vf
    assert "float global_max = -std::numeric_limits<float>::infinity();" in combine_vf_normalized
    assert "global_max = global_max < value ? value : global_max;" in combine_vf_normalized
    assert "float global_sum = 0.0F;" in combine_vf_normalized
    assert "global_sum += __expf(value - global_max);" in combine_vf_normalized
    assert "const float global_lse = global_max + logf(global_sum);" in combine_vf_normalized
    assert "__expf(partial_lse_value - global_lse) *" in combine_vf_normalized
    assert "partial_output[partial_row * value_head_dim + dim]" in combine_vf_normalized
    assert "if (!isfinite(global_max))" in combine_vf_normalized
    assert "lse[lse_offset] = -std::numeric_limits<float>::infinity();" in combine_vf_normalized
    assert "output[output_row + dim] = 0.0F;" in combine_vf_normalized
    assert (
        "int32_t remainder = base_task; const int32_t head_group = remainder % "
        "head_group_count; remainder /= head_group_count; const int32_t "
        "query_token = remainder % query_tokens; const int32_t batch_index = "
        "remainder / query_tokens;"
    ) in combine_vf_normalized
    assert (
        "const int32_t task_head = sub_block_index * 32 + "
        "static_cast<int32_t>(local_head);"
    ) in combine_vf_normalized
    assert "constexpr int32_t kHead64Tile = 64;" in _head64_plan_source()
    assert (
        "(static_cast<int64_t>(base_task) * selected_partitions + p) * "
        "kHead64Tile + task_head;"
    ) in combine_vf_normalized
    assert (
        "const int64_t lse_offset = (static_cast<int64_t>(batch_index) * "
        "query_heads + global_head) * query_tokens + query_token;"
    ) in combine_vf_normalized
    assert "const int64_t output_row = lse_offset * value_head_dim;" in combine_vf_normalized

    assert "KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);" in combine_kernel_normalized
    assert "if ASCEND_IS_AIC { return; } else if ASCEND_IS_AIV {" in combine_kernel_normalized
    assert "AscendC::GetBlockIdx() / AscendC::GetTaskRatio()" in combine_kernel_normalized
    assert "const uint32_t sub_block_index = AscendC::GetSubBlockIdx();" in combine_kernel_normalized
    assert "asc_vf_call<head64_combine_vf>( dim3(1024, 1, 1)," in combine_kernel_normalized
    assert "base_task, sub_block_index);" in combine_kernel_normalized
    assert "CrossCore" not in combine_scope

    assert "extern \"C\" void launch_sparse_attention_head64_combine_hd576_bf16(" in launcher
    assert "plan->task_count / plan->selected_partitions" in launcher
    assert "sparse_attention_head64_combine_kernel" in launcher


def test_sparse_attention_head64_host_skips_combine_for_p1():
    head64_host = _function_definition(
        _bridge_source(),
        "sparse_attention_forward_family_hd576_head64(",
        declaration_marker="std::tuple<at::Tensor, at::Tensor>",
    )
    p1 = head64_host.index("if (plan.selected_partitions == 1)")
    p1_return = head64_host.index("return {output, task_lse};", p1)
    split_output = head64_host.index("auto output = at::empty(", p1)
    split_lse = head64_host.index("auto lse = at::empty(", p1)
    combine = head64_host.index(
        "run_sparse_attention_head64_combine_hd576_bf16(", split_lse
    )

    assert p1 < p1_return < split_output < split_lse < combine


def test_sparse_attention_head64_reduced_accuracy_covers_boundaries():
    path = Path(__file__).with_name("head64_reduced_accuracy.py")

    assert path.is_file()
    source = path.read_text(encoding="utf-8")
    for case_name in (
        "valid_s64",
        "tail_s70",
        "invalid_causal_s70",
        "all_invalid_s17",
        "int64_overflow_s17",
    ):
        assert case_name in source
    assert "atol=0.05" in source
    assert "rtol=0.05" in source
    assert "torch.count_nonzero(actual_output)" in source
    assert "torch.isneginf(actual_lse).all()" in source
    assert "empty_selected_s0" in source
    assert "empty_context_c0" in source
    assert "_run_width_rejection(torch, ops, 512)" in source
    assert "_run_width_rejection(torch, ops, 640)" in source


def test_sparse_attention_head64_reduced_accuracy_covers_split_kv():
    source = Path(__file__).with_name("head64_reduced_accuracy.py").read_text(
        encoding="utf-8"
    )

    assert "PARTITIONS = (1, 2, 4)" in source
    assert '"valid_s128"' in source
    assert '"valid_s2048"' in source
    assert '"multi_task_b2_q9_s70"' in source
    assert '"batch": 2' in source
    assert '"query_tokens": 9' in source
    assert "partition_results.extend(" in source
    assert "results.extend(partition_results)" in source
    assert 'result["selected_partitions"]' in source
    assert "str(selected_partitions)" in source


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
