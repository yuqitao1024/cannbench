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
