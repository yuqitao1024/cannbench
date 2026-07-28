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
