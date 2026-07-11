from pathlib import Path


SIMT_OP_ROOT = Path(
    "src/cannbench/operators/builtin/softmax/simt/v1/"
    "aten_softmax/csrc/simt"
)
SIMT_OP_V2_ROOT = Path(
    "src/cannbench/operators/builtin/softmax/simt/v2/"
)
SIMT_OP_V3_ROOT = Path(
    "src/cannbench/operators/builtin/softmax/simt/v3/"
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
    persistent_start = source.index("row_softmax_persistent_forward_kernel")
    fast_start = source.index("row_softmax_fast_forward_kernel")
    persistent_source = source[persistent_start:fast_start]

    assert "row_softmax_persistent_forward_kernel" in source
    assert "row_softmax_persistent_block_y" in source
    assert "threadIdx.y" in persistent_source
    assert "dim3(block_x, block_y)" in source
    assert "blockDim.y * blockIdx.x + threadIdx.y" in persistent_source


def test_ascend_softmax_v3_persistent_path_uses_shape_aware_threads_per_block():
    source = (
        SIMT_OP_V3_ROOT
        / "aten_softmax_v3"
        / "csrc"
        / "simt"
        / "spatial_softmax.asc"
    ).read_text()

    assert "inline int64_t row_softmax_persistent_threads_per_block(int64_t dim_size)" in source
    assert "if (dim_size == 512) {\n    return 512;" in source
    assert "if (dim_size == 1024) {\n    return 256;" in source
    assert "return 1024;" in source
    assert "row_softmax_persistent_block_y(block_x, dim_size)" in source


def test_ascend_softmax_v3_persistent_kernel_uses_shape_aware_launch_bounds():
    source = (
        SIMT_OP_V3_ROOT
        / "aten_softmax_v3"
        / "csrc"
        / "simt"
        / "spatial_softmax.asc"
    ).read_text()
    vf_template = source[
        source.rindex("int64_t kThreadsPerBlock", 0, source.index("row_softmax_persistent_forward_vf"))
        : source.index("row_softmax_fast_forward_vf")
    ]
    kernel_template = source[
        source.rindex("int64_t kThreadsPerBlock", 0, source.index("row_softmax_persistent_forward_kernel"))
        : source.index("row_softmax_fast_forward_kernel")
    ]

    assert "__launch_bounds__(kThreadsPerBlock)" in vf_template
    assert "int64_t kThreadsPerBlock" in vf_template
    assert "int64_t kThreadsPerBlock" in kernel_template
    assert "row_softmax_persistent_forward_kernel" in kernel_template


def test_ascend_softmax_v3_isolates_512_and_1024_persistent_kernels():
    source = (
        SIMT_OP_V3_ROOT
        / "aten_softmax_v3"
        / "csrc"
        / "simt"
        / "spatial_softmax.asc"
    ).read_text()
    persistent_512 = (
        SIMT_OP_V3_ROOT
        / "aten_softmax_v3"
        / "csrc"
        / "simt"
        / "persistent_512.asc"
    ).read_text()
    persistent_1024 = (
        SIMT_OP_V3_ROOT
        / "aten_softmax_v3"
        / "csrc"
        / "simt"
        / "persistent_1024.asc"
    ).read_text()

    assert "void dispatch_row_persistent_forward_kernel_512_fp16(" in source
    assert "void dispatch_row_persistent_forward_kernel_512_fp32(" in source
    assert "dispatch_row_persistent_forward_kernel_512_fp16(" in source
    assert "dispatch_row_persistent_forward_kernel_512_fp32(" in source
    assert 'TORCH_CHECK(false, "unsupported 512-thread persistent dtype combination");' in source
    assert "__launch_bounds__(512)" in persistent_512
    assert "dispatch_row_persistent_forward_kernel_with_512_threads" in persistent_512
    assert "__launch_bounds__(1024)" in persistent_1024
    assert "dispatch_row_persistent_forward_kernel_with_1024_threads" in persistent_1024


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
    assert "for (int64_t offset = threadIdx.x; offset < dim_size;" in fast_source
    assert "offset += blockDim.x" in fast_source
    assert "row = blockIdx.x * blockDim.y + threadIdx.y" not in fast_source
    assert "row_softmax_fast_block_x() {\n  constexpr int64_t kCudaFastPathThreads = 512;" in source
    assert "row_softmax_fast_ubuf_bytes" in source
    assert "row_softmax_fast_block_x() / kCudaWarpLaneLimit" in source


def test_ascend_softmax_v3_fast_path_uses_1024_threads_per_block():
    source = (
        SIMT_OP_V3_ROOT
        / "aten_softmax_v3"
        / "csrc"
        / "simt"
        / "spatial_softmax.asc"
    ).read_text()

    assert "row_softmax_fast_block_x() {\n  constexpr int64_t kCudaFastPathThreads = 1024;" in source


def test_ascend_softmax_v3_fast_path_uses_real_ilp_and_shifted_half2():
    source = (
        SIMT_OP_V3_ROOT
        / "aten_softmax_v3"
        / "csrc"
        / "simt"
        / "spatial_softmax.asc"
    ).read_text()
    fast_start = source.index("row_softmax_fast_forward_vf")
    reg_start = source.index("row_softmax_fast_reg_forward_vf")
    fast_source = source[fast_start:reg_start]

    assert "constexpr int64_t kFastILP = 4;" in fast_source
    assert "offset = threadIdx.x * kILP" in source
    assert "offset += blockDim.x * kILP" in source
    assert "fast_ilp_softmax_write" in source
    assert "fast_fp16x2_reduce" in source
    assert "fast_fp16x2_write" in source
    assert "const int64_t fp16x2_shift = (row * dim_size) % 2;" in fast_source
    assert "aligned_data = data + shift" in source
    assert "aligned_input = input + shift" in source


def test_ascend_softmax_v3_fast_path_has_register_cache_variant():
    source = (
        SIMT_OP_V3_ROOT
        / "aten_softmax_v3"
        / "csrc"
        / "simt"
        / "spatial_softmax.asc"
    ).read_text()

    assert "row_softmax_fast_reg_forward_vf" in source
    assert "row_softmax_fast_reg_forward_kernel" in source
    assert "row_softmax_fast_register_count" in source
    assert "use_register_like_row_softmax_fast" in source
    assert "register_count > 1 && register_count <= 8" in source
    assert "scalar_t elements[kRegCount]" in source
    assert "dispatch_row_fast_forward_kernel" in source
    assert "launch_row_fast_reg_forward_kernel" in source


def test_ascend_softmax_v3_uses_mixed_simd_simt_vf_launch_model():
    source = (
        SIMT_OP_V3_ROOT
        / "aten_softmax_v3"
        / "csrc"
        / "simt"
        / "spatial_softmax.asc"
    ).read_text()

    assert "__simt_vf__ __launch_bounds__(kThreadsPerBlock) inline void row_softmax_persistent_forward_vf" in source
    assert "__simt_vf__ __launch_bounds__(1024) inline void row_softmax_fast_forward_vf" in source
    assert "__simt_vf__ __launch_bounds__(1024) inline void row_softmax_fast_reg_forward_vf" in source
    assert "__simt_vf__ inline void cunn_spatial_softmax_forward_vf" in source
    assert "__global__ __vector__ void row_softmax_persistent_forward_kernel" in source
    assert "__global__ __vector__ void row_softmax_fast_forward_kernel" in source
    assert "__global__ __vector__ void row_softmax_fast_reg_forward_kernel" in source
    assert "__global__ __vector__ void cunn_spatial_softmax_forward_kernel" in source
    assert "asc_vf_call<row_softmax_persistent_forward_vf" in source
    assert "asc_vf_call<row_softmax_fast_forward_vf" in source
    assert "asc_vf_call<row_softmax_fast_reg_forward_vf" in source
    assert "asc_vf_call<cunn_spatial_softmax_forward_vf" in source
    assert "<<<grid_x,\n         0,\n         acl_stream>>>" in source
    assert "<<<grid_x,\n         dynamic_ubuf_bytes,\n         acl_stream>>>" in source
    assert "<<<launch.grid_x * launch.grid_y,\n           launch.dynamic_ubuf_bytes,\n           acl_stream>>>" in source


def test_ascend_softmax_v3_rowwise_grid_cap_is_shape_aware():
    source = (
        SIMT_OP_V3_ROOT
        / "aten_softmax_v3"
        / "csrc"
        / "simt"
        / "spatial_softmax.asc"
    ).read_text()
    rowwise_start = source.index("inline int64_t row_softmax_grid_x")
    rowwise_end = source.index("inline const char* row_softmax_api_name")
    rowwise_source = source[rowwise_start:rowwise_end]

    launch_start = source.index("void launch_row_forward_impl")
    launch_end = source.index("template <template <typename, typename, typename> class Epilogue>")
    launch_source = source[launch_start:launch_end]

    assert "constexpr int64_t kRowSoftmaxPhysicalGridXLimit = 64" in rowwise_source
    assert "constexpr int64_t kRowSoftmaxGridXLimit = 32768" in rowwise_source
    assert "row_softmax_persistent_grid_x(outer_size, block_y, warp_batch, dim_size)" in launch_source
    assert "row_softmax_grid_x(outer_size)" in launch_source
    assert "dim_size <= 256 || dim_size == 512 || dim_size == 1024" in rowwise_source


def test_ascend_softmax_v2_fast_path_has_fp16_half2_x4_vector_path():
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

    assert "fast_fp16x4_reduce" in source
    assert "fast_fp16x4_write" in source
    assert "half2" in source
    assert "__low2float" in source
    assert "__high2float" in source
    assert "__floats2half2_rn" in source
    assert "dim_size % 4 == 0" in fast_source
    assert "std::is_same_v<scalar_t, __fp16>" in fast_source


def test_ascend_softmax_v2_generic_path_is_explicitly_unimplemented():
    source = (
        SIMT_OP_V2_ROOT
        / "aten_softmax_v2"
        / "csrc"
        / "simt"
        / "spatial_softmax.asc"
    ).read_text()
    generic_start = source.index("row_softmax_generic_forward_kernel")
    spatial_start = source.index("cunn_spatial_softmax_forward_kernel")
    generic_source = source[generic_start:spatial_start]

    assert "row_softmax_generic_forward_kernel" in source
    assert "path == RowSoftmaxPath::GenericLike" in source
    assert "aten_softmax_v2::row_softmax_generic_forward" in source
    assert "generic row-wise softmax is not implemented or accuracy-verified" in source
    assert "spatial_block_reduce_x" not in generic_source
    assert "output[row_offset + d]" not in generic_source


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


def test_ascend_softmax_v2_spatial_kernel_matches_cuda_layout_shape():
    source = (
        SIMT_OP_V2_ROOT
        / "aten_softmax_v2"
        / "csrc"
        / "simt"
        / "spatial_softmax.asc"
    ).read_text()
    spatial_start = source.index("cunn_spatial_softmax_forward_kernel")
    launch_start = source.index("launch_spatial_forward_impl")
    spatial_source = source[spatial_start:launch_start]

    assert "const int64_t outer_stride = inner_size * dim_size" in spatial_source
    assert "const int64_t dim_stride = inner_size" in spatial_source
    assert "outer_offset = outer_index * outer_stride" in spatial_source
    assert "inner_index = blockIdx.y * blockDim.y + threadIdx.y" in spatial_source
    assert "data_offset = outer_offset + inner_index" in spatial_source
    assert "if (blockDim.x > 1)" in spatial_source
    assert "} else {" in spatial_source
    assert "shared_offset" not in spatial_source
    assert "input[data_offset + d * dim_stride]" in spatial_source
    assert "output[data_offset + d * dim_stride]" in spatial_source
    assert "spatial_block_reduce_x<accscalar_t, Max>" in spatial_source
    assert "spatial_block_reduce_x<accscalar_t, Add>" in spatial_source


def test_ascend_softmax_v2_spatial_serial_branch_keeps_cuda_loop_shape():
    source = (
        SIMT_OP_V2_ROOT
        / "aten_softmax_v2"
        / "csrc"
        / "simt"
        / "spatial_softmax.asc"
    ).read_text()
    spatial_start = source.index("cunn_spatial_softmax_forward_kernel")
    launch_start = source.index("launch_spatial_forward_impl")
    spatial_source = source[spatial_start:launch_start]
    serial_source = spatial_source[spatial_source.index("} else {") :]

    assert "for (int64_t d = threadIdx.x; d < dim_size; d += blockDim.x)" in serial_source
    assert "for (int64_t d = 0; d < dim_size; ++d)" not in serial_source


def test_ascend_softmax_v2_spatial_reduce_helper_applies_y_offset_like_cuda():
    source = (
        SIMT_OP_V2_ROOT
        / "aten_softmax_v2"
        / "csrc"
        / "simt"
        / "spatial_softmax.asc"
    ).read_text()
    helper_start = source.index("spatial_block_reduce_x")
    row_reduce_start = source.index("row_block_reduce")
    helper_source = source[helper_start:row_reduce_start]
    spatial_start = source.index("cunn_spatial_softmax_forward_kernel")
    launch_start = source.index("launch_spatial_forward_impl")
    spatial_source = source[spatial_start:launch_start]

    assert "shared += threadIdx.y * blockDim.x" in helper_source
    assert "shared_offset" not in helper_source
    assert "shared_offset" not in spatial_source


def test_ascend_softmax_accuracy_script_targets_v2_by_default():
    source = (
        Path("src/cannbench/operators/builtin/softmax/simt/test")
        / "ascend_softmax_accuracy.py"
    ).read_text()

    assert 'parser.add_argument("--simt-package", default="aten_softmax_v2")' in source
    assert 'parser.add_argument("--simt-label", default="simt_v2")' in source
    assert "from aten_softmax import ops" not in source
