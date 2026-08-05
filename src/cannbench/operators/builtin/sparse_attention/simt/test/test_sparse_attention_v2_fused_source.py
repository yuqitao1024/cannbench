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
    assert "kHead64FusedL1Bytes = kHead64FusedL1PvBytes" in source
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
    assert "kHead64FusedL1Bytes = kHead64FusedL1PvBytes" in source
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
    assert "kHead64FusedL1Bytes = kHead64FusedL1PvBytes" in source
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
