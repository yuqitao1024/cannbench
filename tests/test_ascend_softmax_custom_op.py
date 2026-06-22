from pathlib import Path


CUSTOM_OP_ROOT = Path(
    "src/cannbench/datasets/data/softmax/custom_ops/ascend/default/"
    "aten_softmax/csrc/simt"
)


def test_ascend_softmax_keeps_dim_parallel_launch_policy():
    header = (CUSTOM_OP_ROOT / "occupancy_common.h").read_text()

    assert "dim_threads *= 2" in header
    assert "return {dim_threads, inner_threads};" in header


def test_ascend_softmax_reduction_uses_dynamic_ubuf_not_global_scratch():
    source = (CUSTOM_OP_ROOT / "spatial_softmax.asc").read_text()

    assert "extern __ubuf__ uint32_t dynamicStartUB[];" in source
    assert "(__ubuf__ accscalar_t*)dynamicStartUB" in source
    assert "spatial_block_reduce_x" in source
    assert "launch.dynamic_ubuf_bytes" in source
    assert "accscalar_t* reduce_workspace" not in source
    assert "at::empty(\n      {\n          launch.grid_x * launch.grid_y" not in source
