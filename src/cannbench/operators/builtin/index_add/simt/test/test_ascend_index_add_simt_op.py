from pathlib import Path


SIMT_INDEX_ADD_ROOT = (
    Path(__file__).resolve().parents[1]
    / "v1"
    / "aten_index_add"
    / "csrc"
)
SIMT_INDEX_ADD_V2_ROOT = (
    Path(__file__).resolve().parents[1]
    / "v2"
    / "aten_index_add_v2"
    / "csrc"
)


def test_index_add_simt_v1_has_native_float16_path():
    cpp_source = (SIMT_INDEX_ADD_ROOT / "index_add.cpp").read_text()
    asc_source = (SIMT_INDEX_ADD_ROOT / "simt" / "index_add.asc").read_text()

    assert "launch_index_add_half" in cpp_source
    assert "launch_index_add_half" in asc_source
    assert "self.scalar_type() == at::ScalarType::Half" in cpp_source
    assert "const_data_ptr<at::Half>()" in cpp_source
    assert "mutable_data_ptr<at::Half>()" in cpp_source
    assert "void run_index_add_float(\n    const at::Tensor& index,\n    const at::Tensor& source,\n    at::Tensor& output,\n    float alpha," in cpp_source
    assert "void run_index_add_half(\n    const at::Tensor& index,\n    const at::Tensor& source,\n    at::Tensor& output,\n    at::Half alpha," in cpp_source
    assert '#include "simt_api/asc_fp16.h"' in asc_source
    assert "index_add_kernel<half, uint16_t>" in asc_source
    assert "__hmul(source, __ushort_as_half(alpha))" in asc_source
    assert "index_add_kernel<__fp16, float>" not in asc_source

    half_branch = cpp_source[
        cpp_source.index("self.scalar_type() == at::ScalarType::Half") :
        cpp_source.index("auto compute_self = self.contiguous().to(at::kFloat)")
    ]
    assert ".to(at::kFloat)" not in half_branch
    assert "alpha.to<at::Half>()" in half_branch

    float_branch = cpp_source[
        cpp_source.index("self.scalar_type() == at::ScalarType::Float") :
        cpp_source.index("self.scalar_type() == at::ScalarType::Half")
    ]
    assert "alpha.to<float>()" in float_branch


def test_index_add_plugin_routes_v2_to_isolated_module():
    from cannbench.operators import get_operator_plugin

    plugin = get_operator_plugin("index_add")

    assert plugin.simt_module_name("v1") == "aten_index_add"
    assert plugin.simt_module_name(None) == "aten_index_add"
    assert plugin.simt_module_name("v2") == "aten_index_add_v2"


def test_index_add_simt_v2_defines_shape_specialized_fast_paths():
    cpp_source = (SIMT_INDEX_ADD_V2_ROOT / "index_add.cpp").read_text()
    asc_source = (SIMT_INDEX_ADD_V2_ROOT / "simt" / "index_add.asc").read_text()
    setup_py = (SIMT_INDEX_ADD_V2_ROOT.parents[1] / "setup.py").read_text()
    ops_py = (SIMT_INDEX_ADD_V2_ROOT.parents[0] / "ops.py").read_text()

    assert 'library_name = "aten_index_add_v2"' in setup_py
    assert "torch.ops.aten_index_add_v2" in ops_py
    assert "TORCH_LIBRARY_FRAGMENT(aten_index_add_v2, m)" in cpp_source
    assert "TORCH_LIBRARY_IMPL(aten, PrivateUse1" not in cpp_source
    assert "[=, &output]" not in cpp_source
    assert "output_tensor = output" in cpp_source

    assert "launch_index_add_1d_dim0_half" in cpp_source
    assert "launch_index_add_2d_dim0_half" in cpp_source
    assert "launch_index_add_3d_dim0_half" in cpp_source
    assert "launch_index_add_4d_dim3_half" in cpp_source
    assert "launch_index_add_generic_half" in cpp_source

    assert "index_add_1d_dim0_kernel" in asc_source
    assert "index_add_2d_dim0_kernel" in asc_source
    assert "index_add_3d_dim0_kernel" in asc_source
    assert "index_add_4d_dim3_kernel" in asc_source
    assert "index_add_generic_kernel" in asc_source

    assert "if (rank == 1 && wrapped_dim == 0)" in cpp_source
    assert "if (rank == 2 && wrapped_dim == 0 && shape.inner_stride <= 256)" in cpp_source
    assert "if (rank == 3 && wrapped_dim == 0)" in cpp_source
    assert "if (rank == 4 && wrapped_dim == 3)" in cpp_source


def test_index_add_simt_v2_keeps_hidden_accumulation_on_generic_path():
    cpp_source = (SIMT_INDEX_ADD_V2_ROOT / "index_add.cpp").read_text()
    asc_source = (SIMT_INDEX_ADD_V2_ROOT / "simt" / "index_add.asc").read_text()

    assert "launch_index_add_row_inner_half" not in cpp_source
    assert "launch_index_add_row_inner_float" not in cpp_source
    assert "index_add_row_inner_kernel" not in asc_source

    assert "launch_index_add_3d_dim1_half" not in cpp_source
    assert "launch_index_add_3d_dim1_float" not in cpp_source
    assert "index_add_3d_dim1_kernel" not in asc_source
    assert "if (rank == 3 && wrapped_dim == 1)" not in cpp_source
