from pathlib import Path

from cannbench.operators import get_operator_plugin
from cannbench.operators.builtin.adaptivemaxpool3dgrad import (
    _indices_for_backend,
    get_adaptivemaxpool3dgrad_case,
    get_adaptivemaxpool3dgrad_dataset,
)
from cannbench.operators.builtin.adaptivemaxpool3dgrad.materialize import (
    compute_adaptivemaxpool3d_indices,
    materialize_adaptivemaxpool3dgrad_inputs,
)


def test_adaptivemaxpool3dgrad_realistic_dataset_contains_production_shapes():
    dataset = get_adaptivemaxpool3dgrad_dataset("realistic")
    cases = {case.case_id: case for case in dataset.cases}

    assert {
        "ct_resnet_global_pool_backward",
        "video_resnet_clip_global_pool_backward",
        "spatiotemporal_mid_feature_pool_backward",
        "volumetric_segmentation_head_pool_backward",
    }.issubset(cases)
    assert cases["ct_resnet_global_pool_backward"].input_shape == (1, 32, 32, 64, 64)
    assert cases["ct_resnet_global_pool_backward"].output_size == (1, 1, 1)
    assert cases["spatiotemporal_mid_feature_pool_backward"].output_size == (2, 2, 2)


def test_adaptivemaxpool3dgrad_materialization_matches_input_and_output_sizes():
    case = get_adaptivemaxpool3dgrad_case(
        "smoke",
        "tiny_global_pool_backward",
    )

    payload = materialize_adaptivemaxpool3dgrad_inputs(case, dtype="float16", seed=7)

    assert payload["input_shape"] == (1, 2, 2, 2, 2)
    assert payload["output_size"] == (1, 1, 1)
    assert len(payload["input_values"]) == 1 * 2 * 2 * 2 * 2
    assert len(payload["grad_output_values"]) == 1 * 2 * 1 * 1 * 1
    assert len(payload["indices"]) == 1 * 2 * 1 * 1 * 1


def test_adaptivemaxpool3dgrad_indices_match_adaptive_pooling_regions():
    indices = compute_adaptivemaxpool3d_indices(
        (
            9.0,
            2.0,
            7.0,
            1.0,
            3.0,
            8.0,
            4.0,
            6.0,
        ),
        input_shape=(1, 1, 2, 2, 2),
        output_size=(1, 2, 2),
    )

    assert indices == (0, 5, 2, 7)


def test_adaptivemaxpool3dgrad_registers_pytorch_cuda_baseline_plugin():
    plugin = get_operator_plugin("adaptivemaxpool3dgrad")

    assert plugin.spec.name == "adaptivemaxpool3dgrad"
    assert plugin.spec.runner_name == "adaptivemaxpool3dgrad"
    assert plugin.spec.supported_dtypes == ("float32", "float16", "bfloat16")
    assert callable(plugin.build_torch_callable)


def test_adaptivemaxpool3dgrad_profile_selection_covers_cuda_and_cann_names():
    plugin = get_operator_plugin("adaptivemaxpool3dgrad")

    nvidia = plugin.profile_kernel_selection(
        backend="nvidia",
        implementation=None,
        implementation_version=None,
    )
    ascend = plugin.profile_kernel_selection(
        backend="ascend",
        implementation="cann_ops_library",
        implementation_version=None,
    )

    assert "adaptive_max_pool3d_backward" in nvidia.kernel_name_patterns
    assert "adaptivemaxgradinput" in nvidia.kernel_name_patterns
    assert "atomicadaptivemaxgradinput" in nvidia.kernel_name_patterns
    assert nvidia.launch_count == 4
    assert "AdaptiveMaxPool3DGrad" in ascend.kernel_name_patterns
    assert "AdaptiveMaxPool3DGradD" in ascend.kernel_name_patterns


def test_adaptivemaxpool3dgrad_profile_selection_covers_simt_v1_names():
    plugin = get_operator_plugin("adaptivemaxpool3dgrad")

    selection = plugin.profile_kernel_selection(
        backend="ascend",
        implementation="simt",
        implementation_version="v1",
    )

    assert "aten_adaptive_max_pool3d_grad" in selection.kernel_name_patterns
    assert "adaptive_max_pool3d_grad" in selection.kernel_name_patterns


def test_adaptivemaxpool3dgrad_v5_profiles_clear_and_scatter_as_device_total():
    plugin = get_operator_plugin("adaptivemaxpool3dgrad")

    selection = plugin.profile_kernel_selection(
        backend="ascend",
        implementation="simt",
        implementation_version="v5",
    )

    assert selection.kernel_name_patterns == (
        "adaptive_max_pool3d_grad_v5_zero_kernel",
        "adaptive_max_pool3d_grad_v5_scatter_kernel",
    )
    assert selection.aggregate_across_files is True


def test_adaptivemaxpool3dgrad_uses_int32_indices_for_ascend_cann_path():
    class FakeTorch:
        int32 = "int32"

    class FakeIndices:
        def __init__(self):
            self.requested_dtype = None

        def to(self, dtype):
            self.requested_dtype = dtype
            return self

    class FakeDevice:
        type = "npu"

    indices = FakeIndices()
    ctx = type("Ctx", (), {"device": FakeDevice(), "torch": FakeTorch()})()

    assert _indices_for_backend(ctx, indices) is indices
    assert indices.requested_dtype == "int32"


def test_adaptivemaxpool3dgrad_registers_simt_v1_module_name():
    plugin = get_operator_plugin("adaptivemaxpool3dgrad")

    assert plugin.simt_module_name is not None
    assert plugin.simt_module_name(None) == "aten_adaptive_max_pool3d_grad"
    assert plugin.simt_module_name("v1") == "aten_adaptive_max_pool3d_grad"
    assert plugin.simt_module_name("v2") == "aten_adaptive_max_pool3d_grad_v2"
    assert plugin.simt_module_name("v3") == "aten_adaptive_max_pool3d_grad_v3"
    assert plugin.simt_module_name("v4") == "aten_adaptive_max_pool3d_grad_v4"
    assert plugin.simt_module_name("v5") == "aten_adaptive_max_pool3d_grad_v5"
    assert plugin.simt_module_name("unknown") is None


def test_adaptivemaxpool3dgrad_simt_v2_keeps_cann_arch35_simt_branches():
    root = Path(__file__).resolve().parents[1] / "simt" / "v2"
    cpp_source = (
        root
        / "aten_adaptive_max_pool3d_grad_v2"
        / "csrc"
        / "adaptive_max_pool3d_grad.cpp"
    ).read_text()
    asc_source = (
        root
        / "aten_adaptive_max_pool3d_grad_v2"
        / "csrc"
        / "simt"
        / "adaptive_max_pool3d_grad.asc"
    ).read_text()
    setup_py = (root / "setup.py").read_text()
    ops_py = (root / "aten_adaptive_max_pool3d_grad_v2" / "ops.py").read_text()

    assert 'library_name = "aten_adaptive_max_pool3d_grad_v2"' in setup_py
    assert "torch.ops.aten_adaptive_max_pool3d_grad_v2" in ops_py
    assert "TORCH_LIBRARY_FRAGMENT(aten_adaptive_max_pool3d_grad_v2, m)" in cpp_source
    assert "is_overlap" in cpp_source
    assert "deterministic" in cpp_source

    assert "adaptive_max_pool3d_grad_kernel<scalar_t, true>" in asc_source
    assert "adaptive_max_pool3d_grad_kernel<scalar_t, false>" in asc_source
    assert "if (is_overlap)" in asc_source
    assert "asc_atomic_add(&grad_input[input_offset], grad_value)" in asc_source
    assert "grad_input[input_offset] += grad_value" in asc_source


def test_adaptivemaxpool3dgrad_simt_v3_v4_register_block_strategy_variants():
    root = Path(__file__).resolve().parents[1] / "simt"

    v3_cpp = (
        root
        / "v3"
        / "aten_adaptive_max_pool3d_grad_v3"
        / "csrc"
        / "adaptive_max_pool3d_grad.cpp"
    ).read_text()
    v3_asc = (
        root
        / "v3"
        / "aten_adaptive_max_pool3d_grad_v3"
        / "csrc"
        / "simt"
        / "adaptive_max_pool3d_grad.asc"
    ).read_text()
    v4_cpp = (
        root
        / "v4"
        / "aten_adaptive_max_pool3d_grad_v4"
        / "csrc"
        / "adaptive_max_pool3d_grad.cpp"
    ).read_text()
    v4_asc = (
        root
        / "v4"
        / "aten_adaptive_max_pool3d_grad_v4"
        / "csrc"
        / "simt"
        / "adaptive_max_pool3d_grad.asc"
    ).read_text()

    assert "TORCH_LIBRARY_FRAGMENT(aten_adaptive_max_pool3d_grad_v3, m)" in v3_cpp
    assert "TORCH_LIBRARY_FRAGMENT(aten_adaptive_max_pool3d_grad_v4, m)" in v4_cpp
    assert "constexpr int64_t packed_work_items_per_block = 512;" in v3_cpp
    assert "constexpr int64_t packed_work_items_per_block = 128;" in v4_cpp

    for source in (v3_asc, v4_asc):
        assert "grid_for_blocks(grad_block_num)" in source


def test_adaptivemaxpool3dgrad_simt_v2_v3_v4_use_zeros_like_as_the_only_clear():
    root = Path(__file__).resolve().parents[1] / "simt"

    for version in ("v2", "v3", "v4"):
        package = root / version / f"aten_adaptive_max_pool3d_grad_{version}"
        cpp_source = (package / "csrc" / "adaptive_max_pool3d_grad.cpp").read_text()
        asc_source = (
            package / "csrc" / "simt" / "adaptive_max_pool3d_grad.asc"
        ).read_text()

        assert "at::zeros_like" in cpp_source
        assert "init_output_zero_kernel" not in asc_source

        if version in {"v3", "v4"}:
            assert "zero_block_num" not in cpp_source
            assert "zero_block_num" not in asc_source
