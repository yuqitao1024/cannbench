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
