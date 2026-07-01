from pathlib import Path


SIMT_OP_ROOT = Path(
    "src/cannbench/datasets/data/softmax/simt/v1/"
    "aten_softmax/csrc/simt"
)
SIMT_OP_V2_ROOT = Path(
    "src/cannbench/datasets/data/softmax/simt/v2/"
)


def test_ascend_softmax_keeps_dim_parallel_launch_policy():
    header = (SIMT_OP_ROOT / "occupancy_common.h").read_text()

    assert "dim_threads *= 2" in header
    assert "return {dim_threads, inner_threads};" in header


def test_ascend_softmax_reduction_uses_dynamic_ubuf_not_global_scratch():
    source = (SIMT_OP_ROOT / "spatial_softmax.asc").read_text()

    assert "extern __ubuf__ uint32_t dynamicStartUB[];" in source
    assert "(__ubuf__ accscalar_t*)dynamicStartUB" in source
    assert "spatial_block_reduce_x" in source
    assert "launch.dynamic_ubuf_bytes" in source
    assert "accscalar_t* reduce_workspace" not in source
    assert "at::empty(\n      {\n          launch.grid_x * launch.grid_y" not in source


def test_ascend_softmax_v2_uses_versioned_python_package_and_torch_namespace():
    setup_py = (SIMT_OP_V2_ROOT / "setup.py").read_text()
    ops_py = (SIMT_OP_V2_ROOT / "aten_softmax_v2" / "ops.py").read_text()
    source = (
        SIMT_OP_V2_ROOT
        / "aten_softmax_v2"
        / "csrc"
        / "simt"
        / "spatial_softmax.asc"
    ).read_text()
    header = (
        SIMT_OP_V2_ROOT
        / "aten_softmax_v2"
        / "csrc"
        / "simt"
        / "occupancy_common.h"
    ).read_text()

    assert 'library_name = "aten_softmax_v2"' in setup_py
    assert "torch.ops.aten_softmax_v2" in ops_py
    assert "namespace aten_softmax_v2" in source
    assert "namespace aten_softmax_v2::simt" in header
    assert "aten_softmax_v2_v2" not in source
    assert "aten_softmax_v2_v2" not in header
    assert "TORCH_LIBRARY_FRAGMENT(aten_softmax_v2, m)" in source
    assert "TORCH_LIBRARY_IMPL(aten_softmax_v2, PrivateUse1, m)" in source
