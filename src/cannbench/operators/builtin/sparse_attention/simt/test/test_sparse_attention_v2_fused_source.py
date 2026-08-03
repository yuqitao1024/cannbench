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
    assert "kHead64FusedUbKvRowOffsetsBytes = 32 * sizeof(int32_t)" in source
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


def test_v2_fused_canonical_value_pack_stages_row_major_tiles_then_transposes():
    source = _v2_fused_source()
    fast_vf = _function_definition(source, "head64_fused_value_pack_fast_vf(")
    fast_slot = _function_definition(
        source, "head64_fused_gather_value_fast_slot("
    )

    assert "kHead64FusedUbValueStagingOffset" in source
    assert (
        "kHead64FusedUbValueStagingBytes =\n"
        "    32 * kHead64FusedMaxValueTile * sizeof(bfloat16_t)"
    ) in source
    assert "__ubuf__ bfloat16_t* staged_values" in fast_vf
    assert "tile_index * 16U * 16U + row_in_tile * 16U + dim_in_tile" in fast_vf
    assert "packed_offset" not in fast_vf
    assert "asc_vf_call<head64_fused_value_pack_fast_vf>" in fast_slot
    assert fast_slot.count("asc_transpose(") == 1
    assert "for (uint32_t row_block = 0; row_block < 2; ++row_block)" in fast_slot
    assert "for (uint32_t dim_block = 0; dim_block < 16; ++dim_block)" in fast_slot
    assert "reinterpret_cast<__ubuf__ uint16_t*>" in fast_slot
    assert fast_slot.count("asc_sync_notify(PIPE_V, PIPE_MTE3, EVENT_ID0);") == 2
    assert fast_slot.count("asc_sync_wait(PIPE_V, PIPE_MTE3, EVENT_ID0);") == 2


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
    assert "dim_block < 16" in fast_slot
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
    assert "kHead64FusedMaxQkTile * 32 * sizeof(bfloat16_t)" in source
    assert "kHead64Tile * kHead64FusedMaxQkTile" in aic
    assert "kHead64FusedMaxQkTile * kHead64SelectedTile" in aic

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
    assert "dim < static_cast<uint32_t>(current_k)" in fast_pack
    prepare_call = aiv[aiv.index("asc_vf_call<head64_fused_key_pack_prepare_vf>") :]
    fast_call = aiv[aiv.index("asc_vf_call<head64_fused_key_pack_fast_vf>") :]
    assert "ub_keys_buf,\n              current_k);" in prepare_call
    assert "k_start,\n              current_k);" in fast_call


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
