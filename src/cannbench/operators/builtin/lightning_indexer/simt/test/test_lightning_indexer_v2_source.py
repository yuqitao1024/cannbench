from __future__ import annotations

from pathlib import Path


V2_ROOT = (
    Path(__file__).parents[1]
    / "v2"
    / "aten_dsa_lightning_indexer_v2"
)
DECODE_RADIX_SOURCE = (
    V2_ROOT / "csrc" / "simt" / "lightning_indexer_decode_radix_topk_bfloat16.asc"
)
DISTRIBUTED_TOPK_SOURCE = (
    V2_ROOT
    / "csrc"
    / "simt"
    / "lightning_indexer_decode_distributed_topk_bfloat16.asc"
)
CONTEXT_SHARDED_SOURCE = (
    V2_ROOT / "csrc" / "simt" / "lightning_indexer_context_sharded_family_64x128.asc"
)
PREFILL_Q2_SOURCE = (
    V2_ROOT / "csrc" / "simt" / "lightning_indexer_prefill_q2_family_64x128.asc"
)
PREFILL_FULL_SCORE_SOURCE = (
    V2_ROOT
    / "csrc"
    / "simt"
    / "lightning_indexer_prefill_full_score_family_64x128.asc"
)
V32_DECODE_ACCURACY = Path(__file__).with_name("v32_decode_v2_accuracy.py")
V32_PREFILL_ACCURACY = Path(__file__).with_name("v32_prefill_v2_accuracy.py")


def _function_body(source: str, signature: str) -> str:
    start = source.index(signature)
    body_start = source.index("{", start)
    depth = 0
    for index in range(body_start, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return source[body_start : index + 1]
    raise AssertionError(f"unterminated function: {signature}")


def test_decode_radix_selector_uses_two_bf16_histogram_passes():
    source = DECODE_RADIX_SOURCE.read_text(encoding="utf-8")

    assert "constexpr int32_t kRadixBits = 8;" in source
    assert "constexpr int32_t kRadixPassCount = 2;" in source
    assert "ordered_bf16_key" in source
    assert "score_key > threshold_key" in source
    assert "score_key == threshold_key" in source
    assert "greater_count + equal_rank" in source


def test_decode_radix_selector_compacts_with_one_packed_thread_scan():
    source = DECODE_RADIX_SOURCE.read_text(encoding="utf-8")
    selector = _function_body(source, "select_row_topk_unordered(")

    for contract in (
        "constexpr uint32_t kCountFieldBits = 16U;",
        "constexpr uint32_t kCountFieldMask =",
        "packed_thread_counts",
        "thread_greater_count",
        "thread_equal_count",
        "thread_greater_offset",
        "thread_equal_offset",
        "total_greater_count",
    ):
        assert contract in source

    assert "for (int32_t chunk_start" not in selector
    assert "asc_atomic_add(state + kStateSelectedCount" not in selector
    assert selector.count(
        "for (int32_t offset = 1; offset < kThreadsPerBlock; offset <<= 1)"
    ) == 1
    assert selector.count(
        "output[row_offset + output_slot] = context_index;"
    ) == 2


def test_decode_radix_selector_launches_one_logical_block_per_row():
    source = DECODE_RADIX_SOURCE.read_text(encoding="utf-8")

    assert "row_index = static_cast<int32_t>(blockIdx.x)" in source
    assert "<<<row_count, kDynamicUbufBytes, stream>>>" in source
    assert "row_index +=" not in source


def test_decode_radix_selector_writes_unordered_indices_directly():
    source = DECODE_RADIX_SOURCE.read_text(encoding="utf-8")

    assert "output[row_offset + output_slot] = context_index;" in source
    for forbidden in (
        "bitonic",
        "candidate_scores",
        "basic_api/",
        "kernel_operator.h",
        "AscendC::LocalTensor",
        "SetFlag",
        "WaitFlag",
        "PipeBarrier",
        "CrossCore",
    ):
        assert forbidden not in source


def test_decode_radix_selector_builds_as_an_independent_device_library():
    setup_source = (V2_ROOT.parent / "setup.py").read_text(encoding="utf-8")

    assert "lightning_indexer_decode_radix_topk_bfloat16.asc" in setup_source
    assert '"liblightning_indexer_decode_radix_topk_bfloat16_v2_kernel.so"' in setup_source


def test_decode_distributed_topk_owns_histograms_offsets_and_output_ranges():
    source = DISTRIBUTED_TOPK_SOURCE.read_text(encoding="utf-8")

    for contract in (
        "constexpr int32_t kContextCount = 32768;",
        "constexpr int32_t kTopK = 2048;",
        "constexpr int32_t kCanonicalContextShardCount = 16;",
        "lightning_indexer_decode_distributed_topk_bfloat16_v2_kernel",
        "lightning_indexer_decode_distributed_high_histogram_bfloat16_v2_vf",
        "lightning_indexer_decode_distributed_select_high_bfloat16_v2_vf",
        "lightning_indexer_decode_distributed_low_histogram_bfloat16_v2_vf",
        "lightning_indexer_decode_distributed_select_low_offsets_bfloat16_v2_vf",
        "lightning_indexer_decode_distributed_compact_bfloat16_v2_vf",
        "asc_atomic_add(local_histogram + bucket, 1U);",
        "shard_greater_offset",
        "shard_equal_offset",
        "total_greater_count",
        "equal_count_needed",
        "output[row_offset + output_slot] = context_index;",
    ):
        assert contract in source

    assert source.count("constexpr int32_t kRadixBins = 256;") == 1
    assert source.count("__gm__ uint32_t* histogram") >= 2
    assert "block_index / context_shard_count" in source
    assert "block_index % context_shard_count" in source
    assert "for (int32_t offset = 1; offset < kThreadsPerBlock; offset <<= 1)" in source
    assert "asc_atomic_add(output" not in source
    assert '#include "basic_api/kernel_operator_block_sync_intf.h"' in source
    for forbidden in (
        "kernel_operator.h",
        "AscendC::LocalTensor",
        "SetFlag",
        "WaitFlag",
        "PipeBarrier",
        "CrossCore",
        "asc_sync_inter",
    ):
        assert forbidden not in source
    for removed_kernel in (
        "lightning_indexer_decode_distributed_high_histogram_bfloat16_v2_kernel(",
        "lightning_indexer_decode_distributed_select_high_bfloat16_v2_kernel(",
        "lightning_indexer_decode_distributed_low_histogram_bfloat16_v2_kernel(",
        "lightning_indexer_decode_distributed_select_low_offsets_bfloat16_v2_kernel(",
        "lightning_indexer_decode_distributed_compact_bfloat16_v2_kernel(",
    ):
        assert removed_kernel not in source


def test_decode_distributed_topk_fuses_vfs_with_full_core_barriers():
    source = DISTRIBUTED_TOPK_SOURCE.read_text(encoding="utf-8")
    kernel = _function_body(
        source,
        "lightning_indexer_decode_distributed_topk_bfloat16_v2_kernel(",
    )
    launcher = _function_body(
        source,
        "launch_lightning_indexer_decode_distributed_topk_bfloat16_v2(",
    )

    stages = (
        "lightning_indexer_decode_distributed_high_histogram_bfloat16_v2_vf",
        "lightning_indexer_decode_distributed_select_high_bfloat16_v2_vf",
        "lightning_indexer_decode_distributed_low_histogram_bfloat16_v2_vf",
        "lightning_indexer_decode_distributed_select_low_offsets_bfloat16_v2_vf",
        "lightning_indexer_decode_distributed_compact_bfloat16_v2_vf",
    )
    positions = [kernel.index(stage) for stage in stages]
    assert positions == sorted(positions)
    assert kernel.count("AscendC::SyncAll();") == 4
    assert "extern __ubuf__ uint32_t dynamicStartUB[];" in kernel
    assert "<<<row_count * context_shard_count, kCompactUbufBytes, stream>>>" in launcher

    for producer in stages[:-1]:
        body = _function_body(source, f"{producer}(")
        assert "publish_gm_for_next_stage();" in body
    assert source.count("__builtin_cce_dcci(nullptr, 1, 0);") == 1


def test_decode_distributed_low_reducer_groups_threshold_scan():
    source = DISTRIBUTED_TOPK_SOURCE.read_text(encoding="utf-8")
    reducer = _function_body(
        source,
        "lightning_indexer_decode_distributed_select_low_offsets_bfloat16_v2_vf(",
    )
    normalized_source = " ".join(source.split())
    normalized_reducer = " ".join(reducer.split())

    for contract in (
        "constexpr int32_t kThresholdGroupCount = 16;",
        "constexpr int32_t kThresholdBucketsPerGroup = 16;",
        "kThresholdGroupCount * kThresholdBucketsPerGroup == kRadixBins",
    ):
        assert contract in normalized_source

    for contract in (
        "__ubuf__ uint32_t* threshold_group_counts = shard_greater_counts",
        "if (thread_index < kThresholdGroupCount)",
        "group_bucket_begin = thread_index * kThresholdBucketsPerGroup",
        "threshold_group_counts[thread_index] = group_count",
        "higher_group = thread_index + 1",
        "greater_count < remaining_rank",
        "selected_bucket >= group_bucket_begin",
    ):
        assert contract in normalized_reducer

    assert "if (thread_index == 0)" not in reducer
    assert "selected_bucket = kRadixBins - 1" not in normalized_reducer


def test_decode_distributed_low_reducer_parallelizes_shard_bucket_tails():
    source = DISTRIBUTED_TOPK_SOURCE.read_text(encoding="utf-8")
    reducer = _function_body(
        source,
        "lightning_indexer_decode_distributed_select_low_offsets_bfloat16_v2_vf(",
    )
    normalized_reducer = " ".join(reducer.split())

    assert "context_shard_count != kCanonicalContextShardCount" in reducer

    for contract in (
        "constexpr int32_t kCanonicalBucketGroupsPerShard = 16;",
        "constexpr int32_t kCanonicalBucketsPerGroup = 16;",
        "kCanonicalContextShardCount * kCanonicalBucketGroupsPerShard == "
        "kReducerThreads",
        "kCanonicalBucketGroupsPerShard * kCanonicalBucketsPerGroup == "
        "kRadixBins",
    ):
        assert contract in " ".join(source.split())

    for contract in (
        "partial_shard_index = thread_index / kCanonicalBucketGroupsPerShard",
        "bucket_group = thread_index % kCanonicalBucketGroupsPerShard",
        "group_bucket_begin = bucket_group * kCanonicalBucketsPerGroup",
        "bucket < group_bucket_begin + kCanonicalBucketsPerGroup",
        "combined_histogram[thread_index] = shard_greater_partial",
        "if (bucket_group == 0)",
        "partial_group < kCanonicalBucketGroupsPerShard",
        "prior_shard < shard_index",
        "if (shard_index + 1 == context_shard_count)",
    ):
        assert contract in normalized_reducer

    assert "for (uint32_t bucket = selected_high + 1U;" not in reducer
    assert "for (uint32_t bucket = selected_low + 1U;" not in reducer


def test_decode_distributed_topk_builds_as_an_independent_device_library():
    setup_source = (V2_ROOT.parent / "setup.py").read_text(encoding="utf-8")

    assert "lightning_indexer_decode_distributed_topk_bfloat16.asc" in setup_source
    assert (
        '"liblightning_indexer_decode_distributed_topk_bfloat16_v2_kernel.so"'
        in setup_source
    )


def test_decode_bridge_uses_distributed_topk_only_for_canonical_shape():
    bridge = (V2_ROOT / "csrc" / "lightning_indexer.asc").read_text(
        encoding="utf-8"
    )
    body = _function_body(
        bridge,
        "lightning_indexer_forward_decode_family_64x128_context_sharded_bfloat16(",
    )

    assert (
        "const bool use_distributed_topk =\n"
        "      batch_size == 2 && query_count == 2 && context_shard_count == 16;"
    ) in body
    for workspace in (
        "high_histograms",
        "low_histograms",
        "radix_state",
        "shard_offsets",
    ):
        assert workspace in body

    score_launch = (
        "launch_lightning_indexer_context_sharded_family_64x128_bfloat16_v2("
    )
    fused_topk_launch = (
        "launch_lightning_indexer_decode_distributed_topk_bfloat16_v2("
    )
    assert body.index(score_launch) < body.index(fused_topk_launch)
    for old_launch in (
        "launch_lightning_indexer_decode_distributed_high_histogram_bfloat16_v2(",
        "launch_lightning_indexer_decode_distributed_select_high_bfloat16_v2(",
        "launch_lightning_indexer_decode_distributed_low_histogram_bfloat16_v2(",
        "launch_lightning_indexer_decode_distributed_select_low_offsets_bfloat16_v2(",
        "launch_lightning_indexer_decode_distributed_compact_bfloat16_v2(",
    ):
        assert old_launch not in body
    assert "} else {\n    launch_lightning_indexer_decode_radix_topk_bfloat16_v2(" in body


def test_decode_bridge_launches_score_before_unordered_radix():
    bridge = (V2_ROOT / "csrc" / "lightning_indexer.asc").read_text(
        encoding="utf-8"
    )
    body = _function_body(
        bridge,
        "lightning_indexer_forward_decode_family_64x128_context_sharded_bfloat16(",
    )

    score_launch = (
        "launch_lightning_indexer_context_sharded_family_64x128_bfloat16_v2("
    )
    radix_launch = "launch_lightning_indexer_decode_radix_topk_bfloat16_v2("
    assert body.index(score_launch) < body.index(radix_launch)
    assert "candidate_indices" not in body
    assert "batch_size * query_count" in body


def test_v2_context_sharded_kernel_stops_after_score_production():
    source = CONTEXT_SHARDED_SOURCE.read_text(encoding="utf-8")

    assert "lightning_indexer_context_sharded_postprocess_vf" in source
    for forbidden in (
        "lightning_indexer_context_sharded_topk_vf",
        "needs_local_topk",
        "candidate_indices",
        "shard_candidate_indices",
        "bitonic",
        "asc_sync_inter",
    ):
        assert forbidden not in source


def test_v2_context_sharded_postprocess_uses_all_threads_for_head_reduction():
    source = CONTEXT_SHARDED_SOURCE.read_text(encoding="utf-8")

    assert "constexpr int32_t kWarpSize = 32;" in source
    assert "constexpr int32_t kWarpsPerBlock = kThreadsPerBlock / kWarpSize;" in source
    assert "static_assert(kWarpsPerBlock == kContextTileSize);" in source
    assert "threadIdx.x / kWarpSize" in source
    assert "threadIdx.x % kWarpSize" in source
    assert "lane_index + kWarpSize" in source
    assert "asc_shfl_down" in source


def test_v2_context_sharded_score_uses_capacity_neutral_column_major_layout():
    source = CONTEXT_SHARDED_SOURCE.read_text(encoding="utf-8")
    postprocess = _function_body(
        source, "lightning_indexer_context_sharded_postprocess_vf("
    )
    aic = _function_body(source, "lightning_indexer_context_sharded_aic(")

    for contract in (
        "kColumnMajorUbFixpipeConfig",
        "AscendC::CO2Layout::COLUMN_MAJOR",
        "FixpipeParamsC310<AscendC::CO2Layout::COLUMN_MAJOR>",
        "fixpipe_params.dstStride = kHeadCount;",
        "fixpipe_params.params = {1, 0, 0, 1};",
        "tile_context_index * kHeadCount + first_head_index",
        "tile_context_index * kHeadCount + second_head_index",
    ):
        assert contract in source

    assert aic.count(
        "Fixpipe<bfloat16_t, float, kColumnMajorUbFixpipeConfig>"
    ) == 2
    assert "kRowMajorUbFixpipeConfig" not in source
    assert "head_index * kContextTileSize" not in postprocess
    assert "static_assert(2 * kSharedScoreSlotBytes == 8 * 1024);" in source

    context_index = 0
    row_major_resources = [
        ((head * 32 + context_index) * 2 // 8) % 32 for head in range(32)
    ]
    column_major_resources = [
        ((context_index * 64 + head) * 2 // 8) % 32 for head in range(32)
    ]
    assert len(set(row_major_resources)) == 4
    assert len(set(column_major_resources)) == 8
    assert max(row_major_resources.count(value) for value in set(row_major_resources)) == 8
    assert max(
        column_major_resources.count(value) for value in set(column_major_resources)
    ) == 4


def test_v2_context_sharded_score_pipeline_uses_two_owned_bf16_slots():
    source = CONTEXT_SHARDED_SOURCE.read_text(encoding="utf-8")
    aic = _function_body(
        source, "lightning_indexer_context_sharded_aic("
    )
    aiv = _function_body(
        source, "lightning_indexer_context_sharded_aiv("
    )

    for contract in (
        "constexpr uint16_t kScoreReadyFlag0 = 0;",
        "constexpr uint16_t kScoreReadyFlag1 = 1;",
        "constexpr uint16_t kScoreFreeFlag0 = 2;",
        "constexpr uint16_t kScoreFreeFlag1 = 3;",
        "constexpr uint32_t kSecondQueryL0OffsetBytes =",
        "static_assert(2 * kSharedScoreSlotBytes == 8 * 1024);",
        "AscendC::TPosition::VECCALC",
        "Fixpipe<bfloat16_t, float",
        "quantPre = QuantMode_t::F322BF16",
    ):
        assert contract in source

    assert "if (tile_index >= 2)" in aic
    assert "score_slot_wait_free(slot);" in aic
    assert "score_slot_publish_ready(slot);" in aic
    assert "score_slot_wait_ready(slot);" in aiv
    assert "score_slot_publish_free(slot);" in aiv
    assert aiv.index("score_slot_wait_ready(slot);") < aiv.index(
        "score_slot_publish_free(slot);"
    )
    assert "score_slot_publish_free(0);" not in aiv
    assert "score_slot_publish_free(1);" not in aiv
    assert "fixpipe_params.mSize = kHeadCount;" in aic
    assert "fixpipe_params.dualDstCtl = 0;" in aic
    assert "if (atom_query_rows == kQueryAtomRows)" in aic
    assert "second_query_l0_scores_tensor" in aic
    assert "fixpipe_params.subBlockId = true;" in aic
    assert aic.count("AscendC::Fixpipe<bfloat16_t, float") == 2


def test_v2_decode_accuracy_covers_reduction_order_boundaries():
    source = V32_DECODE_ACCURACY.read_text(encoding="utf-8")

    assert '"near_threshold"' in source
    assert '"negative_scores"' in source
    assert "expected_tied_indices" in source
    assert "tied_threshold selected index set differs from T3" in source


def test_v2_prefill_accuracy_covers_tail_and_reduction_order_boundaries():
    source = V32_PREFILL_ACCURACY.read_text(encoding="utf-8")

    assert "from aten_dsa_lightning_indexer_v2 import ops" in source
    assert '"masked_tail"' in source
    assert '"tied_threshold"' in source
    assert '"near_threshold"' in source
    assert '"negative_scores"' in source


def test_v2_prefill_postprocess_paths_use_all_threads_for_head_reduction():
    for path in (PREFILL_Q2_SOURCE, PREFILL_FULL_SCORE_SOURCE):
        source = path.read_text(encoding="utf-8")

        assert "constexpr int32_t kWarpSize = 32;" in source
        assert (
            "constexpr int32_t kWarpsPerBlock = kThreadsPerBlock / kWarpSize;"
            in source
        )
        assert "static_assert(kWarpsPerBlock == kContextTileSize);" in source
        assert "threadIdx.x / kWarpSize" in source
        assert "threadIdx.x % kWarpSize" in source
        assert "lane_index + kWarpSize" in source
        assert "asc_shfl_down" in source


def test_v2_prefill_q2_keeps_tail_warps_at_the_topk_barrier():
    source = PREFILL_Q2_SOURCE.read_text(encoding="utf-8")
    body = _function_body(
        source,
        "lightning_indexer_prefill_q2_postprocess_vf(",
    )

    assert "if (warp_index < current_context)" in body
    assert body.index("if (warp_index < current_context)") < body.index(
        "asc_syncthreads();"
    )
    assert "return;" not in body
