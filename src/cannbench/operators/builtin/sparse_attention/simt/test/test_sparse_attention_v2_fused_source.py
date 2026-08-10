from pathlib import Path


def _v2_fused_source() -> str:
    path = (
        Path(__file__).parents[1]
        / "v2/aten_dsa_sparse_attention_v2/csrc/simt"
        / "sparse_attention_head64_fused_hd576.asc"
    )
    assert path.is_file(), f"missing V2 fused Head64 device source: {path}"
    return path.read_text(encoding="utf-8")


def _v2_head64_source() -> str:
    path = (
        Path(__file__).parents[1]
        / "v2/aten_dsa_sparse_attention_v2/csrc/simt"
        / "sparse_attention_head64_hd576.asc"
    )
    assert path.is_file(), f"missing V2 Head64 device source: {path}"
    return path.read_text(encoding="utf-8")


def _v2_host_source() -> str:
    path = (
        Path(__file__).parents[1]
        / "v2/aten_dsa_sparse_attention_v2/csrc/sparse_attention.asc"
    )
    assert path.is_file(), f"missing V2 sparse attention host source: {path}"
    return path.read_text(encoding="utf-8")


def _function_definition(source: str, start_marker: str) -> str:
    start = source.index(start_marker)
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


def test_v2_fused_uses_all_1024_threads_and_32_warps():
    source = _v2_fused_source()

    assert "kHead64FusedLaunchThreads = 1024" in source
    assert "kHead64FusedActiveThreads = 1024" in source
    assert "kHead64FusedActiveWarps = 32" in source
    assert "warp + kHead64FusedActiveWarps" not in source


def test_v2_fused_waits_for_output_update_before_pv_release():
    aiv = _function_definition(
        _v2_fused_source(), "sparse_attention_head64_fused_aiv("
    )

    update = aiv.index("asc_vf_call<head64_fused_output_update_vf>")
    done = aiv.index("asc_sync_wait(PIPE_V, PIPE_MTE3, EVENT_ID1);", update)
    release = aiv.index(
        "AscendC::CrossCoreSetFlag<2, PIPE_MTE3>(kAivToAicReady);", done
    )

    assert update < done < release


def test_v2_fused_canonical_decode_reuses_kv_row_offsets():
    source = _v2_fused_source()

    assert "kHead64FusedUbKvRowOffsetsOffset" in source
    assert (
        "kHead64FusedUbKvRowOffsetsBytes =\n"
        "    kHead64FusedMaxSelectedOwnerRows * sizeof(int32_t)"
    ) in source
    assert "__ubuf__ int32_t* kv_row_offsets" in source
    assert "kHead64InvalidKvRowOffset" in source
    assert "head64_fused_key_pack_prepare_vf" in source
    assert "head64_fused_key_pack_fast_vf" in source
    assert "head64_fused_value_pack_fast_vf" in source


def test_v2_fused_fast_pack_is_isolated_to_exact_canonical_decode():
    source = _v2_fused_source()
    predicate = _function_definition(
        source, "head64_fused_is_canonical_v32_decode("
    )
    aiv = _function_definition(source, "sparse_attention_head64_fused_aiv(")

    for contract in (
        "plan.batch_size == 2",
        "plan.query_heads == 128",
        "plan.query_tokens == 2",
        "plan.context_tokens == 32768",
        "plan.selected_tokens == 2048",
        "plan.qk_head_dim == 576",
        "plan.value_head_dim == 512",
        "plan.selected_partitions == 4",
        "plan.output_mode == kHead64OutputPartialFloat",
    ):
        assert contract in predicate
    assert "head64_fused_is_canonical_v32_decode(plan, causal)" in aiv
    assert "head64_fused_key_pack_vf" in aiv
    assert "head64_fused_key_pack_prepare_vf" in aiv
    assert "head64_fused_key_pack_fast_vf" in aiv


def test_v2_fused_first_key_pack_prepares_offsets_for_all_value_tiles():
    source = _v2_fused_source()
    aiv = _function_definition(source, "sparse_attention_head64_fused_aiv(")
    prepare = aiv.index("asc_vf_call<head64_fused_key_pack_prepare_vf>")
    later_key = aiv.index("asc_vf_call<head64_fused_key_pack_fast_vf>", prepare)
    first_value = aiv.index("head64_fused_gather_value_fast_slot(", later_key)
    next_value = aiv.index("head64_fused_gather_value_fast_slot(", first_value + 1)

    assert "k_tile_index == 0" in aiv[:prepare]
    assert prepare < later_key < first_value < next_value


def test_v2_fused_canonical_value_pack_transposes_while_loading_l0b():
    source = _v2_fused_source()
    fast_vf = _function_definition(source, "head64_fused_value_pack_fast_vf(")
    fast_slot = _function_definition(
        source, "head64_fused_gather_value_fast_slot("
    )
    aic = _function_definition(source, "sparse_attention_head64_fused_aic(")

    assert "kHead64FusedUbValueStagingOffset" in source
    assert (
        "kHead64FusedUbValueStagingBytes =\n"
        "    kHead64FusedMaxSelectedSubtileHalf * kHead64FusedMaxValueTile *\n"
        "    sizeof(bfloat16_t)"
    ) in source
    assert "__ubuf__ bfloat16_t* staged_values" in fast_vf
    assert "tile_index * 16U * 16U + row_in_tile * 16U + dim_in_tile" in fast_vf
    assert "packed_offset" not in fast_vf
    assert "asc_vf_call<head64_fused_value_pack_fast_vf>" in fast_slot
    assert "asc_transpose(" not in fast_slot
    assert "asc_copy_ub2l1(" in fast_slot
    assert "selected_subtile_half * kHead64FusedCanonicalValueTile" in fast_slot
    assert "sizeof(bfloat16_t)" in fast_slot
    assert fast_slot.count("asc_sync_notify(PIPE_V, PIPE_MTE3, EVENT_ID0);") == 1
    assert fast_slot.count("asc_sync_wait(PIPE_V, PIPE_MTE3, EVENT_ID0);") == 1

    assert "if (use_canonical_decode_pack)" in aic
    assert "asc_copy_l12l0b_trans(" in aic
    assert "kHead64FusedCanonicalValueFractals" in aic
    assert "Copy(copy_l1_to_l0b, l0_values, l1_values_subtile);" in aic


def test_v2_fused_canonical_value256_reuses_phase_local_l1():
    source = _v2_fused_source()
    aiv = _function_definition(source, "sparse_attention_head64_fused_aiv(")
    aic = _function_definition(source, "sparse_attention_head64_fused_aic(")
    fast_pack = _function_definition(source, "head64_fused_value_pack_fast_vf(")
    fast_slot = _function_definition(source, "head64_fused_gather_value_fast_slot(")

    assert "kHead64FusedGenericValueTile = kHead64ValueTile" in source
    assert "kHead64FusedCanonicalValueTile = 256" in source
    assert "kHead64FusedMaxValueTile = kHead64FusedCanonicalValueTile" in source

    for function in (aiv, aic):
        assert "head64_fused_is_canonical_v32_decode(plan, causal)" in function
        assert (
            "use_canonical_decode_pack\n"
            "      ? kHead64FusedCanonicalValueTile\n"
            "      : kHead64FusedGenericValueTile"
        ) in function
        assert "value_start += value_tile" in function
        assert "plan.value_head_dim - value_start < value_tile" in function

    assert "dim < kHead64FusedCanonicalValueTile" in fast_pack
    assert "row_block * 16U + dim_block" in fast_pack
    assert "dim_block = dim / 16U" in fast_pack
    assert "kHead64FusedCanonicalValueTile" in fast_slot

    assert "kHead64FusedL1PersistentQueryOffset = 0" in source
    assert "kHead64FusedL1PersistentQueryBytes" in source
    assert "kHead64FusedL1QkQueryOffset = kHead64FusedL1PersistentQueryOffset" in source
    assert "kHead64FusedL1QkKeysOffset" in source
    assert "kHead64FusedL1QkScoresOffset" in source
    assert "kHead64FusedL1QkBytes" in source
    assert (
        "kHead64FusedL1PvProbabilitiesOffset =\n"
        "    kHead64FusedL1PersistentQueryBytes"
    ) in source
    assert "kHead64FusedL1PvValuesOffset" in source
    assert "kHead64FusedL1PvPvOffset" in source
    assert "kHead64FusedL1PvBytes" in source
    assert "kHead64FusedLegacyL1Bytes = kHead64FusedL1PvBytes" in source
    assert "kHead64FusedL1KeysOffset" not in source
    assert aiv.count("kHead64FusedL1QkKeysOffset") == 1
    assert aic.count("kHead64FusedL1QkKeysOffset") == 1


def test_v2_fused_canonical_qk256_preserves_query_and_value256():
    source = _v2_fused_source()
    aiv = _function_definition(source, "sparse_attention_head64_fused_aiv(")
    aic = _function_definition(source, "sparse_attention_head64_fused_aic(")
    prepare_pack = _function_definition(
        source, "head64_fused_key_pack_prepare_vf("
    )
    fast_pack = _function_definition(source, "head64_fused_key_pack_fast_vf(")

    assert "kHead64FusedGenericQkTile = kHead64QkTile" in source
    assert "kHead64FusedCanonicalQkTile = 256" in source
    assert "kHead64FusedMaxQkTile = kHead64FusedCanonicalQkTile" in source
    assert "kHead64FusedCanonicalValueTile = 256" in source
    assert "kHead64FusedL1PersistentQueryOffset = 0" in source
    assert (
        "kHead64FusedL1PvProbabilitiesOffset =\n"
        "    kHead64FusedL1PersistentQueryBytes"
    ) in source
    assert "kHead64FusedLegacyL1Bytes = kHead64FusedL1PvBytes" in source
    assert (
        "kHead64FusedMaxQkTile * kHead64FusedMaxSelectedSubtileHalf * "
        "sizeof(bfloat16_t)"
        in " ".join(source.split())
    )
    assert "kHead64Tile * kHead64FusedMaxQkTile" in aic
    assert "kHead64FusedMaxQkTile * kHead64FusedMaxSelectedSubtile" in aic

    qk_tile_selection = (
        "use_canonical_decode_pack\n"
        "      ? kHead64FusedCanonicalQkTile\n"
        "      : kHead64FusedGenericQkTile"
    )
    for function in (aiv, aic):
        assert qk_tile_selection in function
        assert "k_start += qk_tile" in function
        assert "plan.qk_head_dim - k_start < qk_tile" in function
        assert ": qk_tile" in function

    assert "int32_t current_k" in prepare_pack
    assert "dim < static_cast<uint32_t>(current_k)" in prepare_pack
    assert "int32_t current_k" in fast_pack
    assert "pair_index < static_cast<uint32_t>(current_k) / 2U" in fast_pack
    prepare_call = aiv[aiv.index("asc_vf_call<head64_fused_key_pack_prepare_vf>") :]
    fast_call = aiv[aiv.index("asc_vf_call<head64_fused_key_pack_fast_vf>") :]
    normalized_prepare_call = " ".join(prepare_call.split())
    normalized_fast_call = " ".join(fast_call.split())
    assert (
        "ub_keys_buf, selected_subtile_half, current_k);"
        in normalized_prepare_call
    )
    assert (
        "k_start, selected_subtile_half, current_k);"
        in normalized_fast_call
    )


def test_v2_fused_canonical_selected256_streams_128_and_keeps_generic_selected64():
    source = _v2_fused_source()
    normalized_source = " ".join(source.split())
    predicate = _function_definition(
        source, "head64_fused_is_canonical_v32_decode("
    )
    aiv = _function_definition(source, "sparse_attention_head64_fused_aiv(")
    aic = _function_definition(source, "sparse_attention_head64_fused_aic(")

    assert "kHead64FusedGenericSelectedTile = kHead64SelectedTile" in source
    assert "kHead64FusedCanonicalSelectedTile = 256" in source
    assert "kHead64FusedCanonicalSelectedSubtile = 128" in source
    assert (
        "kHead64FusedGenericSelectedSubtile = kHead64FusedGenericSelectedTile"
        in normalized_source
    )
    assert (
        "kHead64FusedMaxSelectedTile = kHead64FusedCanonicalSelectedTile"
        in normalized_source
    )
    assert (
        "kHead64FusedMaxSelectedSubtile = "
        "kHead64FusedCanonicalSelectedSubtile"
        in normalized_source
    )
    assert "plan.selected_tile == 64" in predicate
    assert "plan.selected_partition_tile_capacity == 8" in predicate

    selected_tile_selection = (
        "use_canonical_decode_pack\n"
        "      ? kHead64FusedCanonicalSelectedTile\n"
        "      : kHead64FusedGenericSelectedTile"
    )
    for function in (aiv, aic):
        assert selected_tile_selection in function
        assert "const int32_t selected_subtile" in function
        assert "? kHead64FusedCanonicalSelectedSubtile" in function
        assert ": kHead64FusedGenericSelectedSubtile" in function
        assert (
            "const int32_t selected_subtile_half = selected_subtile / 2"
            in function
        )
        assert "local_selected_start += selected_tile" in function
        assert "partition_length - local_selected_start < selected_tile" in function

    assert (
        "use_canonical_decode_pack\n"
        "      ? kHead64FusedCanonicalPvGatherSlots\n"
        "      : kHead64FusedGenericPvGatherSlots"
    ) in aiv
    assert "const int32_t pv_gather_slots" in aic


def test_v2_fused_selected256_streaming_uses_phase_local_memory_ownership():
    source = _v2_fused_source()
    normalized_source = " ".join(source.split())
    aic = _function_definition(source, "sparse_attention_head64_fused_aic(")

    assert "kHead64FusedGenericQkGatherSlots = 2" in source
    assert "kHead64FusedCanonicalQkGatherSlots = 1" in source
    assert "kHead64FusedGenericPvGatherSlots = 2" in source
    assert "kHead64FusedCanonicalPvGatherSlots = 1" in source
    assert "kHead64FusedMaxSelectedTile" in source
    assert "kHead64FusedMaxSelectedSubtile" in source
    assert (
        "kHead64FusedL1QkKeyStorageBytes" in source
    )
    assert "kHead64FusedL1PvValueStorageBytes" in source
    assert "kHead64FusedLegacyL1Bytes = kHead64FusedL1PvBytes" in source
    assert "static_assert(kHead64FusedL1PvBytes >= kHead64FusedL1QkBytes)" in source

    assert "kHead64FusedUbPersistentBytes" in source
    assert (
        "kHead64FusedUbQkQueryOffset = kHead64FusedUbPersistentBytes"
        in normalized_source
    )
    assert (
        "kHead64FusedUbPvProbabilitiesOffset = kHead64FusedUbPersistentBytes"
        in normalized_source
    )
    assert (
        "kHead64FusedUbPvValuesOffset = kHead64FusedUbPvProbabilitiesOffset"
        in normalized_source
    )
    assert "kHead64FusedUbBytes = kHead64FusedUbQkBytes" in source
    assert "static_assert(kHead64FusedUbQkBytes >= kHead64FusedUbPvBytes)" in source

    assert "__cb__ bfloat16_t l0_rhs_buf" in aic
    assert "__cc__ float l0_result_buf" in aic
    assert (
        "kHead64FusedMaxQkTile * kHead64FusedMaxSelectedSubtile"
        in aic
    )
    assert "l0_keys_buf" not in aic
    assert "l0_values_buf" not in aic
    assert "l0_scores_buf" not in aic
    assert "l0_pv_buf" not in aic


def test_v2_fused_selected256_packs_64_rows_per_streaming_subtile():
    source = _v2_fused_source()
    prepare = _function_definition(source, "head64_fused_key_pack_prepare_vf(")
    fast_key = _function_definition(source, "head64_fused_key_pack_fast_vf(")
    fast_value = _function_definition(source, "head64_fused_value_pack_fast_vf(")
    gather_generic = _function_definition(
        source, "head64_fused_gather_value_slot("
    )
    gather_value = _function_definition(
        source, "head64_fused_gather_value_fast_slot("
    )
    aiv = _function_definition(source, "sparse_attention_head64_fused_aiv(")
    aic = _function_definition(source, "sparse_attention_head64_fused_aic(")
    normalized_aiv = " ".join(aiv.split())

    for pack in (prepare, fast_key, fast_value):
        assert "int32_t selected_subtile_half" in pack
        assert "local_selected = warp" in pack
        assert (
            "local_selected < static_cast<uint32_t>(selected_subtile_half)"
            in pack
        )
        assert "local_selected += kHead64FusedActiveWarps" in pack

    assert "int32_t selected_subtile_half" in gather_value
    assert (
        "selected_subtile_half * kHead64FusedCanonicalValueTile"
        in gather_value
    )
    assert "sizeof(bfloat16_t)" in gather_value

    assert (
        "selected_begin = "
        "static_cast<int32_t>(subblock_index) * selected_subtile_half"
        in normalized_aiv
    )
    assert "MakeShape(current_k, selected_subtile_half)" in aiv
    assert "MakeShape(selected_subtile_half, current_value)" in gather_generic
    assert "selected_subtile_start += selected_subtile" in aiv
    assert "selected_subtile_start += selected_subtile" in aic
    assert "const int32_t selected_owner_offset = selected_subtile_start / 2" in aiv
    assert "kv_row_offsets + selected_owner_offset" in aiv
    assert "const uint16_t selected_blocks" in aic
    assert "static_cast<uint16_t>(selected_tile / 16)" in aic


def test_v2_fused_selected256_assembles_scores_then_accumulates_pv_subtiles():
    source = _v2_fused_source()
    aiv = _function_definition(source, "sparse_attention_head64_fused_aiv(")
    aic = _function_definition(source, "sparse_attention_head64_fused_aic(")
    normalized_aic = " ".join(aic.split())

    assert aiv.count("asc_vf_call<head64_fused_softmax_vf>") == 1
    assert "for (int32_t selected_subtile_start = 0;" in aic
    assert "current_selected_subtile" in aic
    assert (
        "l1_scores_buf + selected_subtile_start * kHead64Tile" in aic
    )
    assert "MakeShape(kHead64Tile, current_selected_subtile)" in normalized_aic
    assert "MakeShape(current_k, current_selected_subtile)" in normalized_aic
    assert "MakeShape(current_selected_subtile, current_value)" in normalized_aic
    assert "pv_params.k = current_selected_subtile" in aic
    assert "pv_params.cmatrixInitVal = selected_subtile_start == 0" in aic


def test_v2_fused_pv_nz_blocks_use_64b_ub_gaps():
    source = _v2_fused_source()
    output_update = _function_definition(source, "head64_fused_output_update_vf(")
    aic = _function_definition(source, "sparse_attention_head64_fused_aic(")
    normalized_aic = " ".join(aic.split())

    for contract in (
        "kHead64FusedPvBlockGapBytes = 64",
        "kHead64FusedPvBlockGapElements = 16",
        "kHead64FusedMaxValueBlocks - 1",
        "head64_fused_pv_nz_offset(",
    ):
        assert contract in source

    assert source.index("kHead64FusedMaxValueTile =") < source.index(
        "kHead64FusedMaxValueBlocks ="
    )

    assert output_update.count("head64_fused_pv_nz_offset(") == 1
    assert "ub_pv_buf, l1_pv_buf, false, value_blocks, 64, 64, 2" in (
        normalized_aic
    )
    assert (
        "ub_pv_buf, l1_pv_buf + 32 * 16, true, "
        "value_blocks, 64, 64, 2"
    ) in normalized_aic
    assert "ub_scores_buf, l1_scores_buf, false, selected_blocks, 64, 64, 0" in (
        normalized_aic
    )

    row = 0
    row_count = 32
    columns = range(32)

    def nz_offset(column: int, block_gap_elements: int) -> int:
        return (
            (column // 16) * (16 * row_count + block_gap_elements)
            + (row // 16) * 16 * 16
            + (row % 16) * 16
            + column % 16
        )

    retained_resources = [
        (nz_offset(column, 0) * 4 // 8) % 32 for column in columns
    ]
    padded_resources = [
        (nz_offset(column, 16) * 4 // 8) % 32 for column in columns
    ]
    assert len(set(retained_resources)) == 8
    assert max(
        retained_resources.count(resource) for resource in set(retained_resources)
    ) == 4
    assert len(set(padded_resources)) == 16
    assert max(
        padded_resources.count(resource) for resource in set(padded_resources)
    ) == 2


def test_v2_fused_key_fast_uses_aligned_bf16x2_loads_and_stores():
    source = _v2_fused_source()
    prepare = " ".join(
        _function_definition(source, "head64_fused_key_pack_prepare_vf(").split()
    )
    fast = " ".join(
        _function_definition(source, "head64_fused_key_pack_fast_vf(").split()
    )

    for contract in (
        "reinterpret_cast<__gm__ const bfloat16x2_t*>(kv_batch)",
        "reinterpret_cast<__ubuf__ bfloat16x2_t*>(packed_keys)",
        "pair_index = lane",
        "pair_index < static_cast<uint32_t>(current_k) / 2U",
        "pair_index += 32",
        "dim = 2U * pair_index",
        "packed_key_pairs[packed_offset / 2U]",
        "kv_batch_pairs[(row_offset + dimension_start + dim) / 2U]",
        "zero_pair",
    ):
        assert contract in fast

    assert "bfloat16x2_t" not in prepare
    assert "packed_keys[packed_offset]" in prepare


def test_v2_fused_selected256_reuses_single_slot_only_after_l0b_copy():
    aic = _function_definition(
        _v2_fused_source(), "sparse_attention_head64_fused_aic("
    )

    qk_copy = aic.index("Copy(copy_l1_to_l0b, l0_keys")
    qk_copy_done = aic.index(
        "asc_sync_wait(PIPE_MTE1, PIPE_M, EVENT_ID1);", qk_copy
    )
    qk_single_slot_reuse = aic.index(
        "if (has_next_k && qk_gather_slots == 1)", qk_copy
    )
    pv_copy = aic.index("Copy(copy_l1_to_l0b, l0_values")
    pv_copy_done = aic.index(
        "asc_sync_wait(PIPE_MTE1, PIPE_M, EVENT_ID1);", pv_copy
    )
    pv_single_slot_reuse = aic.index(
        "if (has_next_pv_round && pv_gather_slots == 1)", pv_copy
    )

    assert qk_copy < qk_copy_done < qk_single_slot_reuse
    assert pv_copy < pv_copy_done < pv_single_slot_reuse


def test_v2_canonical_p4_combine_reuses_partition_weights_across_dimensions():
    source = _v2_head64_source()
    predicate = _function_definition(
        source, "head64_combine_is_canonical_v32_decode("
    )
    fast_vf = _function_definition(source, "head64_combine_p4_vf(")
    kernel = _function_definition(source, "sparse_attention_head64_combine_kernel(")
    normalized_predicate = " ".join(predicate.split())

    for contract in (
        "plan.batch_size == 2",
        "plan.query_heads == 128",
        "plan.query_tokens == 2",
        "plan.selected_tokens == 2048",
        "plan.value_head_dim == 512",
        "plan.selected_partitions == 4",
        "plan.output_mode == "
        "aten_dsa_sparse_attention_v2::kHead64OutputPartialFloat",
    ):
        assert contract in normalized_predicate

    dimension_loop = fast_vf.index(
        "for (int32_t dim", fast_vf.index("partition_weight3")
    )
    for weight in range(4):
        assert fast_vf.index(f"partition_weight{weight}") < dimension_loop
    assert "__expf" not in fast_vf[dimension_loop:]
    assert "asc_vf_call<head64_combine_p4_vf>" in kernel
    assert "asc_vf_call<head64_combine_vf>" in kernel


def test_v2_vllm_decode_uses_p1_direct_output_and_eight_head64_tasks():
    host = _v2_host_source()
    fused = _v2_fused_source()
    predicate = _function_definition(
        fused, "head64_fused_is_vllm_rolling_decode("
    )
    normalized_predicate = " ".join(predicate.split())

    assert "auto_head64_decode ? 1" in " ".join(host.split())
    assert "selected_partitions == 1" in normalized_predicate
    assert "plan.used_core_num == 8" in normalized_predicate
    assert "plan.task_count == 8" in normalized_predicate
    assert "plan.output_mode == kHead64OutputDirectBfloat16" in normalized_predicate


def test_v2_vllm_decode_uses_selected128_three_slot_rolling_schedule():
    source = _v2_fused_source()
    aiv = _function_definition(
        source, "sparse_attention_head64_fused_vllm_aiv("
    )
    aic = _function_definition(
        source, "sparse_attention_head64_fused_vllm_aic("
    )

    assert "kHead64FusedVllmSelectedTile = 128" in source
    assert "kHead64FusedVllmRollingSlots = 3" in source
    assert "tile_count + kHead64FusedVllmRollingSlots - 1" in aiv
    assert "tile_count + kHead64FusedVllmRollingSlots - 1" in aic
    for stage in ("gather_tile", "softmax_tile", "update_tile"):
        assert stage in aiv
    for stage in ("qk_tile", "pv_tile"):
        assert stage in aic
    assert "% kHead64FusedVllmRollingSlots" in aiv
    assert "% kHead64FusedVllmRollingSlots" in aic


def test_v2_vllm_aic_copies_query_directly_from_gm_to_l1():
    source = _v2_fused_source()
    copy_query = _function_definition(
        source, "head64_vllm_copy_query_gm_to_l1("
    )
    aiv = _function_definition(
        source, "sparse_attention_head64_fused_vllm_aiv("
    )
    aic = _function_definition(
        source, "sparse_attention_head64_fused_vllm_aic("
    )
    kernel = _function_definition(
        source, "sparse_attention_head64_fused_mix12_restored_kernel("
    )

    assert "__gm__ const bfloat16_t* query" in aic
    assert "AscendC::Nd2NzParams params" in copy_query
    assert "params.nValue = 64" in copy_query
    assert "params.dValue = 576" in copy_query
    assert "params.srcDValue = query_tokens * 576" in copy_query
    assert "head64_vllm_copy_query_gm_to_l1(" in aic
    assert "asc_vf_call<head64_fused_query_pack_vf>" not in aiv
    assert "head64_vllm_copy_query_half_ub_to_l1(" not in aiv
    assert "kVllmAivToAicQueryReady" not in aiv
    assert "kVllmAivToAicQueryReady" not in aic
    assert "sparse_attention_head64_fused_vllm_aic(\n          query," in kernel


def test_v2_vllm_gather_and_softmax_read_indices_from_gm():
    source = _v2_fused_source()
    gather = _function_definition(
        source, "head64_fused_vllm_gather_kv_tile("
    )
    softmax = _function_definition(
        source, "head64_fused_vllm_softmax_vf("
    )
    aiv = _function_definition(
        source, "sparse_attention_head64_fused_vllm_aiv("
    )

    copy_row = _function_definition(
        source, "head64_vllm_copy_kv_row_gm_to_ub("
    )
    copy_chunk = _function_definition(
        source, "head64_vllm_copy_kv_chunk_ub_to_gm("
    )

    assert "AscendC::DataCopyPad(" in copy_row
    assert "AscendC::DataCopy(" in copy_chunk
    assert "asc_vf_call<head64_fused_key_pack_prepare_vf>" not in gather
    assert "asc_vf_call<head64_fused_key_pack_fast_vf>" not in gather
    assert "head64_vllm_copy_kv_row_gm_to_ub(" in gather
    assert "head64_vllm_copy_kv_chunk_ub_to_gm(" in gather
    assert "__gm__ const int32_t* indices" in softmax
    assert "indices[indices_row + selected_start + selected]" in softmax
    assert "asc_vf_call<head64_fused_vllm_softmax_vf>" in aiv
    assert "indices," in aiv
    assert "selected_start," in aiv


def test_v2_vllm_softmax_reuses_lane_local_scores_without_shared_metadata():
    softmax = _function_definition(
        _v2_fused_source(), "head64_fused_vllm_softmax_vf("
    )

    assert "float lane_scores[4]" in softmax
    assert "uint32_t lane_valid_mask" in softmax
    assert softmax.count(
        "indices[indices_row + selected_start + selected]"
    ) == 1
    assert softmax.count(
        "scores[local_head * kHead64FusedVllmSelectedTile + selected]"
    ) == 1
    assert "lane_scores[lane_entry]" in softmax
    assert "lane_valid_mask & (1U << lane_entry)" in softmax
    assert "__syncthreads()" not in softmax


def test_v2_vllm_decode_four_aivs_share_each_selected128_gather():
    source = _v2_fused_source()
    kernel = _function_definition(
        source, "sparse_attention_head64_fused_mix12_restored_kernel("
    )
    aiv = _function_definition(
        source, "sparse_attention_head64_fused_vllm_aiv("
    )

    assert kernel.count("sparse_attention_head64_fused_vllm_aiv(") == 1
    assert "AscendC::GetSubBlockIdx()" in aiv
    normalized_aiv = " ".join(aiv.split())
    assert "head_begin = static_cast<int32_t>(subblock_index) * 32" in aiv
    assert "kHead64FusedVllmSelectedQuarter = 32" in source
    assert "const uint32_t aic_in_pair = block_id & 1U" in aiv
    assert "aic_in_pair * 2U + subblock_index" in normalized_aiv
    assert "quarter_index * kHead64FusedVllmSelectedQuarter" in normalized_aiv
    assert "block_id >> 1U" in aiv
    publish = _function_definition(source, "head64_vllm_aiv_publish_gm_slot(")
    assert publish.index("CrossCoreSetFlag<0, PIPE_MTE3>") < publish.index(
        "CrossCoreWaitFlag<0, PIPE_MTE3>"
    )
    assert publish.index("CrossCoreWaitFlag<0, PIPE_MTE3>") < publish.index(
        "CrossCoreSetFlag<4, PIPE_MTE3>"
    )
    assert "CrossCoreWaitFlag<4, PIPE_MTE2>" in source


def test_v2_vllm_decode_reuses_three_shared_gm_kv_slots_and_probability_tails():
    source = _v2_fused_source()
    aiv = _function_definition(
        source, "sparse_attention_head64_fused_vllm_aiv("
    )
    aic = _function_definition(
        source, "sparse_attention_head64_fused_vllm_aic("
    )

    assert "kHead64FusedVllmL1QueryBytes = 73728" in source
    assert "kHead64FusedVllmL1KvSlotBytes = 147456" in source
    assert "kHead64FusedVllmL1Bytes = 516096" in source
    assert "kHead64FusedVllmL1ProbabilityOffset = 131072" in source
    assert "static_assert(kHead64FusedVllmL1Bytes <= 512 * 1024)" in source
    assert "head64_fused_vllm_gather_kv_tile" in aiv
    assert "kHead64FusedVllmGmKvSlotBytes = 147456" in source
    assert "kHead64FusedVllmGmPairCount = 4" in source
    assert "kHead64FusedVllmGmBytes = 1769472" in source
    assert "__gm__ uint8_t* gm_workspace" in aiv
    assert "__gm__ uint8_t* gm_workspace" in aic
    assert "head64_vllm_aic_wait_gm_slot(slot)" in aic
    assert "head64_vllm_copy_kv_gm_to_l1(" in aic
    assert "kHead64FusedVllmL1ProbabilityOffset" in aiv
    assert "MakeFrameLayout<NZLayoutPtn, bfloat16_t>(128, 576)" in aic
    assert "AscendC::DataCopy(l1_tensor, gm_tensor, params)" in source
    assert "MakeFrameLayout<ZNLayoutPtn, bfloat16_t>(current_k, 128)" in aic
    assert "k_start * kHead64FusedVllmSelectedTile" in aic
    assert "MakeShape(128, kHead64FusedVllmValueTile)" in aic


def test_v2_vllm_decode_delivers_qk_and_pv_directly_to_both_aiv_ubs():
    source = _v2_fused_source()
    aic = _function_definition(
        source, "sparse_attention_head64_fused_vllm_aic("
    )

    assert aic.count("asc_copy_l0c2ub(") >= 2
    assert "kHead64FusedVllmUbScoresOffset" in aic
    assert "kHead64FusedVllmUbPvOffset" in aic
    assert aic.count("kHead64FusedVllmDualDstCtl") >= 2
    assert "head64_copy_l0c_to_l1_nz(" not in aic


def test_v2_vllm_decode_handoffs_one_logical_pv512_tile():
    source = _v2_fused_source()
    aiv = _function_definition(
        source, "sparse_attention_head64_fused_vllm_aiv("
    )
    aic = _function_definition(
        source, "sparse_attention_head64_fused_vllm_aic("
    )
    update = _function_definition(
        source, "head64_fused_vllm_output_update_vf("
    )

    assert "kHead64FusedVllmUbPvBytes =\n    32 * 512 * sizeof(float)" in source
    assert "static_assert(kHead64FusedVllmUbBytes == 211584)" in source
    assert "for (int32_t value_start = 0;" not in aiv
    assert "old_scale + slot * 32,\n            512);" in aiv

    pv_loop = aic.index("for (int32_t value_start = 0;")
    pv_free = aic.index(
        "AscendC::CrossCoreWaitFlag<2, PIPE_FIX>(kVllmAivToAicPvFree)"
    )
    pv_copy = aic.index("asc_copy_l0c2ub(", pv_loop)
    pv_ready = aic.index(
        "AscendC::CrossCoreSetFlag<2, PIPE_FIX>(kVllmAicToAivPvReady)"
    )
    assert pv_free < pv_loop < pv_copy < pv_ready
    assert "ub_pv + value_start * 32" in aic
    assert "dim < current_value" in update
    assert "pv_chunk * 32 * kHead64FusedVllmValueTile" in update


def test_v2_l0c_to_l1_uses_linkable_nz_fixpipe():
    source = _v2_fused_source()
    helper = _function_definition(source, "head64_copy_l0c_to_l1_nz(")

    assert "FixpipeParamsC310<AscendC::CO2Layout::NZ>" in helper
    assert "AscendC::Fixpipe<float, float" in helper
    assert "asc_copy_l0c2l1_sync(" not in helper


def test_v2_vllm_decode_uses_upstream_gather_copy_events():
    source = _v2_fused_source()
    gather = _function_definition(
        source, "head64_fused_vllm_gather_kv_tile("
    )
    aiv = _function_definition(
        source, "sparse_attention_head64_fused_vllm_aiv("
    )

    probability_copy = _function_definition(
        source, "head64_vllm_copy_probability_half_ub_to_l1("
    )
    assert "asc_copy_ub2l1(" in probability_copy
    assert "head64_vllm_copy_kv_half_ub_to_l1(" not in gather
    assert "head64_vllm_copy_kv_row_gm_to_ub(" in gather
    assert "head64_vllm_copy_kv_chunk_ub_to_gm(" in gather
    assert "kHead64FusedVllmGatherBatch = 16" in source
    assert "batch_start += kHead64FusedVllmGatherBatch" in gather
    assert "SetFlag<AscendC::HardEvent::MTE2_MTE3>" in gather
    assert "WaitFlag<AscendC::HardEvent::MTE2_MTE3>" in gather
    assert "SetFlag<AscendC::HardEvent::MTE3_MTE2>" in gather
    assert "WaitFlag<AscendC::HardEvent::MTE3_MTE2>" in gather
    assert "head64_vllm_copy_query_half_ub_to_l1(" not in aiv
    assert "head64_vllm_copy_probability_half_ub_to_l1(" in aiv


def test_v2_vllm_workspaces_are_top_level_with_dynamic_ub_base_zero():
    source = _v2_fused_source()
    kernel = _function_definition(
        source, "sparse_attention_head64_fused_mix12_restored_kernel("
    )
    aiv = _function_definition(
        source, "sparse_attention_head64_fused_vllm_aiv("
    )
    aic = _function_definition(
        source, "sparse_attention_head64_fused_vllm_aic("
    )
    launcher = _function_definition(
        source,
        "launch_sparse_attention_head64_fused_hd576_bf16_v2_rolling_restored(",
    )

    assert "__cbuf__ uint8_t l1_workspace[kHead64FusedL1Bytes]" in kernel
    assert "reinterpret_cast<__ubuf__ uint8_t*>(0)" in kernel
    assert "__cbuf__ uint8_t l1_workspace[" not in aiv
    assert "__ubuf__ uint8_t ub_workspace[" not in aiv
    assert "__cbuf__ uint8_t l1_workspace[" not in aic
    assert "__ubuf__ uint8_t ub_workspace[" not in aic
    assert "(void)workspace" not in kernel
    assert "workspace" in aiv
    assert "workspace" in aic
    assert "kHead64FusedVllmUbKvGatherOffset" in source
    assert "kHead64FusedVllmUbKvGatherBytes" in source
    assert "static_assert(kHead64FusedVllmUbBytes == 211584)" in source
    assert "kHead64FusedDynamicUbBytes" in launcher
    assert "static_assert(kHead64FusedDynamicUbBytes <= 216 * 1024)" in source


def test_v2_vllm_mixed_launch_expands_pairs_and_decodes_aiv_block_id():
    source = _v2_fused_source()
    host_source = _v2_host_source()
    aiv = _function_definition(
        source, "sparse_attention_head64_fused_vllm_aiv("
    )
    launcher = _function_definition(
        source,
        "launch_sparse_attention_head64_fused_hd576_bf16_v2_rolling_restored(",
    )
    normalized_aiv = " ".join(aiv.split())
    normalized_launcher = " ".join(launcher.split())
    inactive = _function_definition(
        source, "head64_vllm_aiv_join_gather_barriers("
    )

    assert "kHead64FusedTaskRatio = 2" in source
    assert "GetBlockIdx() / AscendC::GetTaskRatio()" in normalized_aiv
    assert (
        "static_cast<uint32_t>(plan->used_core_num) * "
        "kHead64FusedTaskRatio"
    ) in normalized_launcher
    assert "head64_vllm_aiv_join_gather_barriers(tile_count)" in aiv
    assert "barrier_index < tile_count" in inactive
    assert "CrossCoreSetFlag<0, PIPE_MTE3>" in inactive
    assert "CrossCoreWaitFlag<0, PIPE_MTE3>" in inactive
    assert (
        "launch_sparse_attention_head64_fused_hd576_bf16_v2_rolling_restored("
        in host_source
    )


def test_v2_vllm_source_documents_exact_l1_and_ub_byte_layouts():
    source = _v2_fused_source()
    aiv = _function_definition(
        source, "sparse_attention_head64_fused_vllm_aiv("
    )
    aic = _function_definition(
        source, "sparse_attention_head64_fused_vllm_aic("
    )

    for marker in (
        "VLLM-ROLLING L1 layout",
        "[0, 73728)",
        "[73728, 221184)",
        "[221184, 368640)",
        "[368640, 516096)",
    ):
        assert marker in aic
    for marker in (
        "VLLM-ROLLING AIV UB layout",
        "running max",
        "running output",
        "KV gather staging",
        "QK scores",
        "PV result",
    ):
        assert marker in aiv

    kernel = _function_definition(
        source, "sparse_attention_head64_fused_mix12_restored_kernel("
    )
    for marker in (
        "VLLM-ROLLING shared GM layout",
        "four adjacent-AIC pairs",
        "three 147456-byte slots",
        "128 x 576 BF16 ND",
    ):
        assert marker in kernel


def test_v2_vllm_gm_to_l1_orders_mte2_before_tensor_api_mte1_reads():
    source = _v2_fused_source()
    copy = _function_definition(source, "head64_vllm_copy_kv_gm_to_l1(")
    aic = _function_definition(
        source, "sparse_attention_head64_fused_vllm_aic("
    )

    data_copy = copy.index("AscendC::DataCopy(l1_tensor, gm_tensor, params)")
    set_ready = copy.index("SetFlag<AscendC::HardEvent::MTE2_MTE1>")
    wait_ready = copy.index("WaitFlag<AscendC::HardEvent::MTE2_MTE1>")
    assert data_copy < set_ready < wait_ready
    wait_gm = aic.index("head64_vllm_aic_wait_gm_slot(slot)")
    copy_l1 = aic.index("head64_vllm_copy_kv_gm_to_l1(")
    copy_l0b = aic.index("Copy(copy_l1_to_l0b")
    assert wait_gm < copy_l1 < copy_l0b
