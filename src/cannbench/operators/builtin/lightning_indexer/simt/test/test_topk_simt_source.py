from pathlib import Path

import pytest


@pytest.mark.parametrize("family", ["4x64", "64x128"])
def test_fused_indexer_uses_parallel_ub_topk_merge(family):
    source = Path(
        "src/cannbench/operators/builtin/lightning_indexer/simt/v1/"
        f"aten_dsa_lightning_indexer/csrc/simt/lightning_indexer_fused_family_{family}.asc"
    ).read_text(encoding="utf-8")
    topk_source = Path(
        "src/cannbench/operators/builtin/lightning_indexer/simt/v1/"
        "aten_dsa_lightning_indexer/csrc/simt/lightning_indexer_topk_ub.h"
    ).read_text(encoding="utf-8")

    assert "constexpr int32_t kThreadsPerBlock = 1024;" in source
    assert "constexpr int32_t kThreadsPerBlock = 256;" not in source
    assert "lightning_indexer_merge_topk_ub" in source
    assert "extern __ubuf__ uint32_t dynamicStartUB[];" in source
    assert "asc_syncthreads" in topk_source
    assert "insert_at" not in topk_source
    assert "basic_api/" not in topk_source
    assert "CrossCore" not in topk_source


def test_prefill_radix_topk_selects_bf16_threshold_then_sorts_only_topk():
    source = Path(
        "src/cannbench/operators/builtin/lightning_indexer/simt/v1/"
        "aten_dsa_lightning_indexer/csrc/simt/"
        "lightning_indexer_radix_topk_bfloat16.asc"
    ).read_text(encoding="utf-8")

    for expected in (
        "kRadixBits = 8",
        "kRadixBins = 256",
        "kRadixPassCount = 2",
        "asc_atomic_add",
        "threshold_key",
        "score_key > threshold_key",
        "score_key == threshold_key",
        "candidate_index < other_index",
        "bitonic_size <= kTopK",
        "row_index += kPersistentBlockCount",
    ):
        assert expected in source
    for forbidden in (
        "kSortCapacity = 4096",
        "basic_api/",
        "kernel_operator.h",
        "AscendC::LocalTensor",
    ):
        assert forbidden not in source
