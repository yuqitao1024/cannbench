from pathlib import Path


CASE_ROOT = Path(__file__).parent
SOURCE = CASE_ROOT / "vcv_per_aiv_ub_layout.asc"
CMAKE = CASE_ROOT / "CMakeLists.txt"
RUNNER = CASE_ROOT / "scripts" / "run.sh"


def source_text() -> str:
    return SOURCE.read_text(encoding="utf-8")


def test_example_has_one_fixed_vector_cube_vector_pipeline() -> None:
    source = source_text()

    assert "KERNEL_TYPE_MIX_AIC_1_2" in source
    assert "vcv_per_aiv_ub_layout_kernel<<<1, 0, stream>>>" in source
    assert "get_subblockid" in source
    assert "asc_copy_ub2l1_sync(" in source
    assert "asc_mmad_sync(" in source
    assert "asc_copy_l0c2ub_sync(" in source


def test_cube_maps_one_physical_m_half_to_v1() -> None:
    source = source_text()

    assert "#define MATRIX_M 8U" in source
    assert "#define CUBE_M 16U" in source
    assert source.count("asc_copy_l0c2ub_sync(") == 1
    assert "l0_logits + MATRIX_ELEMENTS" not in source
    assert "CUBE_M" in source


def test_v0_and_v1_declare_distinct_static_ub_layouts() -> None:
    source = source_text()

    assert "__ubuf__ bfloat16_t v0_gather_zn" in source
    assert "__ubuf__ float v1_logits_row_major" in source
    assert "__ubuf__ float v1_softmax_scratch" in source
    assert "sub_block_idx == 0U" in source
    assert "sub_block_idx == 1U" in source


def test_pipeline_uses_cross_core_mode_four_only() -> None:
    source = source_text()

    assert "kCrossCoreSyncMode = 4" in source
    assert "CrossCoreSetFlag<kCrossCoreSyncMode" in source
    assert "CrossCoreWaitFlag<kCrossCoreSyncMode" in source
    assert "kV1FlagOffset = 16" in source
    assert "asc_sync_block_wait" not in source
    assert "asc_sync_block_arrive" not in source


def test_experimental_basic_api_exception_is_narrow() -> None:
    source = source_text()

    assert '#include "basic_api/kernel_operator_block_sync_intf.h"' in source
    assert "basic_api/kernel_basic_intf.h" not in source
    assert '#include "kernel_operator.h"' not in source
    assert "AscendC::LocalTensor" not in source
    assert "AscendC::SetFlag" not in source
    assert "AscendC::WaitFlag" not in source
    assert "AscendC::PipeBarrier" not in source


def test_example_has_host_softmax_oracle_and_device_verification() -> None:
    source = source_text()

    assert "build_host_softmax_oracle" in source
    assert "verify_softmax_output" in source
    assert "Verification PASSED" in source
    assert "aclrtSynchronizeStream" in source
    assert "gathered_debug" not in source
    assert "logits_debug" not in source


def test_build_and_runner_target_dav_3510_and_preserve_build_artifacts() -> None:
    cmake = CMAKE.read_text(encoding="utf-8")
    runner = RUNNER.read_text(encoding="utf-8")

    assert 'CMAKE_ASC_ARCHITECTURES "dav-3510"' in cmake
    assert "--npu-arch=${CMAKE_ASC_ARCHITECTURES}" in cmake
    assert "cmake --build" in runner
    assert "Verification PASSED" in runner
    assert "build" in runner
