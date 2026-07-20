from pathlib import Path


V5_ROOT = Path(__file__).resolve().parents[1] / "v5"
V5_PACKAGE = V5_ROOT / "aten_adaptive_max_pool3d_grad_v5"


def test_v5_uses_empty_output_and_exactly_one_device_clear():
    cpp_source = (V5_PACKAGE / "csrc" / "adaptive_max_pool3d_grad.cpp").read_text()
    asc_source = (
        V5_PACKAGE / "csrc" / "simt" / "adaptive_max_pool3d_grad.asc"
    ).read_text()

    assert "at::empty_like" in cpp_source
    assert "at::zeros_like" not in cpp_source
    assert "at::zeros(" not in cpp_source
    assert "adaptive_max_pool3d_grad_v5_zero_kernel" in asc_source
    assert asc_source.count("<<<grid_for_blocks(zero_block_num)") == 1


def test_v5_launches_clear_before_scatter_on_the_same_stream():
    asc_source = (
        V5_PACKAGE / "csrc" / "simt" / "adaptive_max_pool3d_grad.asc"
    ).read_text()

    clear_launch = asc_source.index("<<<grid_for_blocks(zero_block_num)")
    scatter_launch = asc_source.index("<<<grid_for_blocks(grad_block_num)")

    assert clear_launch < scatter_launch
    assert "aclrtStream stream" in asc_source


def test_v5_clear_and_scatter_names_share_the_profile_pattern():
    asc_source = (
        V5_PACKAGE / "csrc" / "simt" / "adaptive_max_pool3d_grad.asc"
    ).read_text()

    assert "adaptive_max_pool3d_grad_v5_zero_kernel" in asc_source
    assert "adaptive_max_pool3d_grad_v5_scatter_kernel" in asc_source


def test_v5_uses_native_bfloat16_scatter_and_clear():
    cpp_source = (V5_PACKAGE / "csrc" / "adaptive_max_pool3d_grad.cpp").read_text()
    asc_source = (
        V5_PACKAGE / "csrc" / "simt" / "adaptive_max_pool3d_grad.asc"
    ).read_text()

    assert "launch_adaptive_max_pool3d_grad_v5_bfloat16" in cpp_source
    assert "self.scalar_type() == at::ScalarType::BFloat16" in cpp_source
    assert "const_data_ptr<at::BFloat16>()" in cpp_source
    assert '#include "simt_api/asc_bf16.h"' in asc_source
    assert "launch_adaptive_max_pool3d_grad_v5<bfloat16_t>" in asc_source
