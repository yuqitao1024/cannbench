from cannbench.core.profile import read_device_profile
from cannbench.operators import get_operator_plugin


def test_softmax_v3_profile_sums_multistage_kernels_across_files(tmp_path):
    profile_dir = tmp_path / "profile"
    profile_dir.mkdir()
    for name, duration_us in (
        ("row_softmax_fast_large_gmem_max_kernel", 950),
        ("row_softmax_fast_large_gmem_sum_kernel", 1850),
        ("row_softmax_fast_large_gmem_write_kernel", 2900),
    ):
        (profile_dir / f"{name}.csv").write_text(
            "Op Name,Task Duration(us)\n"
            f"aten_softmax_v3::{name},{duration_us}\n"
        )

    selection = get_operator_plugin("softmax").profile_kernel_selection(
        backend="ascend",
        implementation="simt",
        implementation_version="v3",
    )
    summary = read_device_profile(
        profile_dir,
        backend="ascend",
        kernel_selection=selection,
    )

    assert selection.aggregate_across_files
    assert summary.latency_ms == 5.7


def test_softmax_non_v3_profile_keeps_default_per_file_aggregation():
    selection = get_operator_plugin("softmax").profile_kernel_selection(
        backend="ascend",
        implementation="simt",
        implementation_version="v2",
    )

    assert not selection.aggregate_across_files
