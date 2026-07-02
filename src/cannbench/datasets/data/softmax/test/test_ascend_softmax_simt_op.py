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


def test_ascend_softmax_v2_splits_rowwise_dispatch_like_cuda():
    source = (
        SIMT_OP_V2_ROOT
        / "aten_softmax_v2"
        / "csrc"
        / "simt"
        / "spatial_softmax.asc"
    ).read_text()

    assert "enum class RowSoftmaxPath" in source
    assert "PersistentLike" in source
    assert "FastLike" in source
    assert "GenericLike" in source
    assert "dim_size <= 2048 && dim_size * scalar_size <= 8192" in source
    assert "aten_softmax_v2::row_softmax_persistent_forward" in source
    assert "aten_softmax_v2::row_softmax_fast_forward" in source
    assert "aten_softmax_v2::row_softmax_generic_forward" in source


def test_ascend_softmax_v2_rowwise_launch_policy_is_not_fixed_to_32_threads():
    source = (
        SIMT_OP_V2_ROOT
        / "aten_softmax_v2"
        / "csrc"
        / "simt"
        / "spatial_softmax.asc"
    ).read_text()

    assert "row_softmax_persistent_block_x" in source
    assert "row_softmax_fast_block_x" in source
    assert "row_softmax_generic_block_x" in source
    assert "constexpr int64_t kCudaFastPathThreads = 512" in source
    assert "constexpr int64_t kCudaGenericMaxThreads = 1024" in source


def test_ascend_softmax_v2_persistent_path_uses_multi_row_block_shape():
    source = (
        SIMT_OP_V2_ROOT
        / "aten_softmax_v2"
        / "csrc"
        / "simt"
        / "spatial_softmax.asc"
    ).read_text()

    assert "row_softmax_persistent_forward_kernel" in source
    assert "row_softmax_persistent_block_y" in source
    assert "threadIdx.y" in source
    assert "dim3(block_x, block_y)" in source
    assert "blockIdx.x * blockDim.y + threadIdx.y" in source


def test_ascend_softmax_v2_persistent_path_uses_cuda_style_register_warp_kernel():
    source = (
        SIMT_OP_V2_ROOT
        / "aten_softmax_v2"
        / "csrc"
        / "simt"
        / "spatial_softmax.asc"
    ).read_text()
    persistent_start = source.index("row_softmax_persistent_forward_kernel")
    fast_start = source.index("row_softmax_fast_forward_kernel")
    persistent_source = source[persistent_start:fast_start]

    assert "persistent_warp_reduce" in source
    assert "asc_shfl_xor" in source
    assert "constexpr int64_t kWarpIterations" in persistent_source
    assert "constexpr int64_t kWarpBatch" in persistent_source
    assert "accscalar_t elements[kWarpBatch][kWarpIterations]" in persistent_source
    assert "row_base = (blockDim.y * blockIdx.x + threadIdx.y) * kWarpBatch" in persistent_source
    assert "persistent_warp_reduce<accscalar_t, kWarpBatch, kWarpSize, Max>" in persistent_source
    assert "spatial_block_reduce_x" not in persistent_source


def test_ascend_softmax_v2_fast_path_has_dedicated_multi_row_kernel():
    source = (
        SIMT_OP_V2_ROOT
        / "aten_softmax_v2"
        / "csrc"
        / "simt"
        / "spatial_softmax.asc"
    ).read_text()

    assert "row_softmax_fast_forward_kernel" in source
    assert "path == RowSoftmaxPath::FastLike" in source
    assert "path == RowSoftmaxPath::FastLike\n      ? 1" in source
    assert "aten_softmax_v2::row_softmax_fast_forward" in source


def test_ascend_softmax_v2_fast_path_uses_cuda_style_block_reduce_kernel():
    source = (
        SIMT_OP_V2_ROOT
        / "aten_softmax_v2"
        / "csrc"
        / "simt"
        / "spatial_softmax.asc"
    ).read_text()
    fast_start = source.index("row_softmax_fast_forward_kernel")
    generic_start = source.index("row_softmax_generic_forward_kernel")
    fast_source = source[fast_start:generic_start]

    assert "fast_ilp_reduce" in source
    assert "fast_block_reduce_warp" in source
    assert "fast_block_reduce_warp_inverse" in source
    assert "constexpr int64_t kFastILP" in fast_source
    assert "row_input = input + row * dim_size" in fast_source
    assert "row_output = output + row * dim_size" in fast_source
    assert "for (int64_t offset = threadIdx.x; offset < dim_size; offset += blockDim.x)" in fast_source
    assert "row = blockIdx.x * blockDim.y + threadIdx.y" not in fast_source
    assert "row_softmax_fast_block_x() {\n  constexpr int64_t kCudaFastPathThreads = 512;" in source
    assert "row_softmax_fast_ubuf_bytes" in source
    assert "row_softmax_fast_block_x() / kCudaWarpLaneLimit" in source


def test_ascend_softmax_v2_generic_path_has_dedicated_multi_row_kernel():
    source = (
        SIMT_OP_V2_ROOT
        / "aten_softmax_v2"
        / "csrc"
        / "simt"
        / "spatial_softmax.asc"
    ).read_text()

    assert "row_softmax_generic_forward_kernel" in source
    assert "row_softmax_generic_block_y" in source
    assert "path == RowSoftmaxPath::GenericLike" in source
    assert "aten_softmax_v2::row_softmax_generic_forward" in source


def test_ascend_softmax_v2_spatial_path_keeps_cuda_style_launch_policy():
    header = (
        SIMT_OP_V2_ROOT
        / "aten_softmax_v2"
        / "csrc"
        / "simt"
        / "occupancy_common.h"
    ).read_text()
    source = (
        SIMT_OP_V2_ROOT
        / "aten_softmax_v2"
        / "csrc"
        / "simt"
        / "spatial_softmax.asc"
    ).read_text()

    assert "inner_threads = std::min(inner_size, kMaxThreads)" in header
    assert "inner_threads <= 64 && dim_size >= 64" in header
    assert "inner_blocks = ceil_div(inner_size, block_y)" in header
    assert "outer_blocks = ceil_div(max_active_blocks, inner_blocks)" in header
    assert "block_x == 1 ? 0 : block_threads * accscalar_size" in header
    assert "cunn_spatial_softmax_forward_kernel" in source
    assert "spatial_block_reduce_x" in source
    assert "blockIdx.y * blockDim.y + threadIdx.y" in source


def test_ascend_softmax_accuracy_script_targets_v2_by_default():
    source = (
        Path("src/cannbench/datasets/data/softmax/test")
        / "ascend_softmax_accuracy.py"
    ).read_text()

    assert 'parser.add_argument("--simt-package", default="aten_softmax_v2")' in source
    assert 'parser.add_argument("--simt-label", default="simt_v2")' in source
    assert "from aten_softmax import ops" not in source
