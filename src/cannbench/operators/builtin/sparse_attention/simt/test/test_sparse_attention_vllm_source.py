from pathlib import Path


VENDOR_DIR = (
    Path(__file__).parents[1]
    / "vllm"
    / "vendor"
    / "vllm_ascend_a5b0ce"
    / "csrc"
    / "attention"
    / "sparse_flash_attention"
)


def test_copied_vllm_tiling_accepts_ascend950():
    source = (VENDOR_DIR / "op_host" / "sparse_flash_attention_def.cpp").read_text(
        encoding="utf-8"
    )

    assert 'AICore().AddConfig("ascend950", aicore_config)' in source


def test_arch35_qk_and_v_gathers_use_simt_vf_with_cache_publication():
    source = (
        VENDOR_DIR
        / "op_kernel"
        / "arch35"
        / "sparse_flash_attention_service_vector_mla.h"
    ).read_text(encoding="utf-8")

    assert "sfa_qk_gather_vf" in source
    assert "sfa_v_gather_vf" in source
    assert "cce::async_invoke<sfa_qk_gather_vf" in source
    assert "cce::async_invoke<sfa_v_gather_vf" in source
    assert "TEventID qkGatherVfDoneId" in source
    assert "TEventID qkGatherLoadDoneId" in source
    assert (
        "qkGatherVfDoneId = GetTPipePtr()->AllocEventID<HardEvent::V_S>()"
        in source
    )

    process_start = source.index("SFAVectorService<TEMPLATE_ARGS>::ProcessSparseKv")
    process_end = source.index(
        "SFAVectorService<TEMPLATE_ARGS>::ProcessVec1", process_start
    )
    gather_sync = source[process_start:process_end]
    assert "SetFlag<HardEvent::V_S>(qkGatherVfDoneId)" in gather_sync
    assert "WaitFlag<HardEvent::V_S>(qkGatherVfDoneId)" in gather_sync
    assert "SetFlag<HardEvent::V_MTE2>(qkGatherLoadDoneId)" in gather_sync
    assert "WaitFlag<HardEvent::V_MTE2>(qkGatherLoadDoneId)" in gather_sync
    assert "vToMte3AttnOutId" not in gather_sync
    assert "DataCopyPad(stage0OutUb" not in gather_sync
    assert "SFA_GATHER_QK_PLANE_ELEMENTS" not in source

    vf_source = source[source.index("sfa_qk_gather_vf"):process_start]
    assert vf_source.count("asc_dcci_entire(packedKv);") == 2
    assert vf_source.count("asc_syncthreads();") == 2
    assert vf_source.count("asc_threadfence();") == 2

    tiling_source = (VENDOR_DIR / "op_host" / "sparse_flash_attention_tiling.cpp").read_text(
        encoding="utf-8"
    )
    assert "constexpr uint32_t D_SIZE = 576;" in tiling_source
    assert "D_SIZE = 576 + 512" not in tiling_source
