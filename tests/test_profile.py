import json

import pytest

from cannbench.core.profile import (
    ProfileKernelSelection,
    ncu_profile_options,
    read_device_profile,
    read_workflow_profile,
    write_device_profile_summary,
)
from cannbench.operators import get_operator_plugin


def test_ncu_profile_options_select_nvtx_range_without_launch_limit():
    options = ncu_profile_options(
        ProfileKernelSelection(nvtx_range="cannbench_operator_dynamic")
    )

    assert options == (
        "--nvtx",
        "--nvtx-include",
        "cannbench_operator_dynamic/",
    )


def test_read_ascend_msprof_csv_duration_summary(tmp_path):
    profile_dir = tmp_path / "profile"
    profile_dir.mkdir()
    (profile_dir / "op_summary.csv").write_text(
        "Op Name,Task Duration(us)\n"
        "softmax,1000\n"
        "softmax,2000\n"
    )

    summary = read_device_profile(profile_dir, backend="ascend")

    assert summary.backend == "ascend"
    assert summary.latency_ms == 1.5
    assert summary.source_files == ("op_summary.csv",)


def test_read_nvidia_ncu_csv_duration_metric_summary(tmp_path):
    profile_dir = tmp_path / "profile"
    profile_dir.mkdir()
    (profile_dir / "ncu.csv").write_text(
        "Kernel Name,Metric Name,Metric Unit,Metric Value\n"
        "softmax,gpu__time_duration.avg,usecond,1000\n"
        "softmax,gpu__time_duration.avg,usecond,2000\n"
    )

    summary = read_device_profile(profile_dir, backend="nvidia")

    assert summary.backend == "nvidia"
    assert summary.latency_ms == 1.5
    assert summary.source_files == ("ncu.csv",)


def test_read_nvidia_ncu_wide_csv_uses_avg_duration_and_unit_row(tmp_path):
    profile_dir = tmp_path / "profile"
    profile_dir.mkdir()
    (profile_dir / "ncu.csv").write_text(
        '"ID","Kernel Name","gpu__dram_cycles_active.avg","gpu__time_duration.avg"\n'
        '"","","cycle","usecond"\n'
        '"1","softmax_kernel","55295584","60.125"\n'
    )

    summary = read_device_profile(profile_dir, backend="nvidia")

    assert summary.backend == "nvidia"
    assert summary.latency_ms == 0.060125
    assert summary.source_files == ("ncu.csv",)


def test_read_device_profile_filters_to_expected_kernel_name(tmp_path):
    profile_dir = tmp_path / "profile"
    profile_dir.mkdir()
    (profile_dir / "OpBasicInfo.csv").write_text(
        "Op Name,Task Duration(us)\n"
        "StatelessRandomNormalV3,400\n"
        "SoftmaxV2_float16_high_precision_1000,58\n"
    )

    summary = read_device_profile(
        profile_dir,
        backend="ascend",
        expected_kernel_name_patterns=("softmax",),
    )

    assert summary.latency_ms == 0.058
    assert summary.source_files == ("OpBasicInfo.csv",)


def test_read_device_profile_can_sum_multiple_matching_kernels(tmp_path):
    profile_dir = tmp_path / "profile"
    profile_dir.mkdir()
    (profile_dir / "OpBasicInfo.csv").write_text(
        "Op Name,Task Duration(us)\n"
        "lightning_indexer_kernel,400\n"
        "sparse_attention_kernel,600\n"
        "unrelated_kernel,999\n"
    )

    summary = read_device_profile(
        profile_dir,
        backend="ascend",
        kernel_selection=ProfileKernelSelection(
            kernel_name_patterns=("lightning_indexer", "sparse_attention"),
        ),
    )

    assert summary.latency_ms == 1.0


def test_read_workflow_profile_partitions_overlapping_names_by_order(tmp_path):
    profile_dir = tmp_path / "profile"
    profile_dir.mkdir()
    (profile_dir / "OpBasicInfo.csv").write_text(
        "Op Name,Task Duration(us)\n"
        "InitKernel,100\n"
        "Cast_indexer,2\n"
        "LightningIndexerMain,30\n"
        "Cast_attention,3\n"
        "SparseFlashAttention,50\n"
        "UnrelatedKernel,999\n"
    )

    summary = read_workflow_profile(
        profile_dir,
        backend="ascend",
        step_selections=(
            ProfileKernelSelection(
                kernel_name_patterns=("cast", "lightning"),
                terminal_kernel_name_patterns=("lightningindexermain",),
            ),
            ProfileKernelSelection(
                kernel_name_patterns=("cast", "sparseflashattention"),
            ),
        ),
    )

    assert [
        component.latency_ms for component in summary.component_summaries
    ] == pytest.approx([0.032, 0.053])
    assert summary.latency_ms == pytest.approx(0.085)
    assert summary.source_files == ("OpBasicInfo.csv",)


def test_read_workflow_profile_requires_non_final_terminal_pattern(tmp_path):
    profile_dir = tmp_path / "profile"
    profile_dir.mkdir()
    (profile_dir / "OpBasicInfo.csv").write_text(
        "Op Name,Task Duration(us)\nIndexerMain,10\nAttentionMain,20\n"
    )

    with pytest.raises(ValueError, match="non-final workflow step 0.*terminal"):
        read_workflow_profile(
            profile_dir,
            backend="ascend",
            step_selections=(
                ProfileKernelSelection(kernel_name_patterns=("indexer",)),
                ProfileKernelSelection(kernel_name_patterns=("attention",)),
            ),
        )


def test_read_workflow_profile_rejects_missing_terminal_kernel(tmp_path):
    profile_dir = tmp_path / "profile"
    profile_dir.mkdir()
    (profile_dir / "OpBasicInfo.csv").write_text(
        "Op Name,Task Duration(us)\nIndexerHelper,10\nAttentionMain,20\n"
    )

    with pytest.raises(ValueError, match="step 0 has no terminal kernel"):
        read_workflow_profile(
            profile_dir,
            backend="ascend",
            step_selections=(
                ProfileKernelSelection(
                    kernel_name_patterns=("indexer",),
                    terminal_kernel_name_patterns=("indexerterminal",),
                ),
                ProfileKernelSelection(kernel_name_patterns=("attention",)),
            ),
        )


def test_read_workflow_profile_rejects_out_of_order_terminals(tmp_path):
    profile_dir = tmp_path / "profile"
    profile_dir.mkdir()
    (profile_dir / "OpBasicInfo.csv").write_text(
        "Op Name,Task Duration(us)\n"
        "SecondTerminal,10\n"
        "FirstTerminal,20\n"
        "FinalMain,30\n"
    )

    with pytest.raises(ValueError, match="terminal boundaries are out of workflow order"):
        read_workflow_profile(
            profile_dir,
            backend="ascend",
            step_selections=(
                ProfileKernelSelection(
                    kernel_name_patterns=("first",),
                    terminal_kernel_name_patterns=("firstterminal",),
                ),
                ProfileKernelSelection(
                    kernel_name_patterns=("second",),
                    terminal_kernel_name_patterns=("secondterminal",),
                ),
                ProfileKernelSelection(kernel_name_patterns=("final",)),
            ),
        )


def test_read_workflow_profile_reads_ncu_wide_rows_in_physical_order(tmp_path):
    profile_dir = tmp_path / "profile"
    profile_dir.mkdir()
    (profile_dir / "ncu.csv").write_text(
        '"ID","Kernel Name","gpu__time_duration.avg"\n'
        '"","","usecond"\n'
        '"1","topk_stage","4"\n'
        '"2","topk_terminal","6"\n'
        '"3","cast_attention","2"\n'
        '"4","flash_attention","8"\n'
    )

    summary = read_workflow_profile(
        profile_dir,
        backend="nvidia",
        step_selections=(
            ProfileKernelSelection(
                kernel_name_patterns=("topk", "cast"),
                terminal_kernel_name_patterns=("topk_terminal",),
            ),
            ProfileKernelSelection(
                kernel_name_patterns=("cast", "flash_attention"),
            ),
        ),
    )

    assert [component.latency_ms for component in summary.component_summaries] == [
        0.01,
        0.01,
    ]
    assert summary.latency_ms == pytest.approx(0.02)


def test_read_device_profile_rejects_unexpected_kernel_name(tmp_path):
    profile_dir = tmp_path / "profile"
    profile_dir.mkdir()
    (profile_dir / "OpBasicInfo.csv").write_text(
        "Op Name,Task Duration(us)\n"
        "StatelessRandomNormalV3,400\n"
    )

    with pytest.raises(ValueError, match="expected profiler kernel"):
        read_device_profile(
            profile_dir,
            backend="ascend",
            expected_kernel_name_patterns=("softmax",),
        )


def test_index_add_plugin_profile_patterns_reject_cast_only_profile(tmp_path):
    profile_dir = tmp_path / "profile"
    profile_dir.mkdir()
    (profile_dir / "OpBasicInfo.csv").write_text(
        "Op Name,Task Duration(us)\n"
        "Cast_39a266ce9a8d7d9ead33f6686e936b72_high_performance_0,2.323\n"
    )

    selection = get_operator_plugin("index_add").profile_kernel_selection(
        backend="ascend",
        implementation="simt",
        implementation_version="v1",
    )

    assert selection.kernel_name_patterns == ("index_add", "aten_index_add")
    with pytest.raises(ValueError, match="expected profiler kernel"):
        read_device_profile(
            profile_dir,
            backend="ascend",
            kernel_selection=selection,
        )


def test_index_add_plugin_profile_patterns_filter_cann_tensor_move(tmp_path):
    profile_dir = tmp_path / "profile"
    profile_dir.mkdir()
    (profile_dir / "OpBasicInfo.csv").write_text(
        "Op Name,Task Duration(us)\n"
        "TensorMove_d2db1a80c523e7e59a032c95969880af_high_performance_2,4.588\n"
        "InplaceIndexAdd_20d0b91f852eb04b8f161ab3cc623d32_high_performance_101001_mix_aiv,9.750\n"
    )

    selection = get_operator_plugin("index_add").profile_kernel_selection(
        backend="ascend",
        implementation="cann_ops_library",
        implementation_version=None,
    )

    assert selection.kernel_name_patterns == ("indexadd", "inplaceindexadd")
    summary = read_device_profile(
        profile_dir,
        backend="ascend",
        kernel_selection=selection,
    )
    assert summary.latency_ms == 0.00975


def test_lightning_indexer_plugin_uses_simt_profile_patterns():
    selection = get_operator_plugin("lightning_indexer").profile_kernel_selection(
        backend="ascend",
        implementation="simt",
        implementation_version="v1",
    )

    assert selection.kernel_name_patterns == (
        "lightning_indexer",
        "aten_dsa_lightning_indexer",
    )


def test_sparse_attention_plugin_uses_simt_profile_patterns():
    selection = get_operator_plugin("sparse_attention").profile_kernel_selection(
        backend="ascend",
        implementation="simt",
        implementation_version="v1",
    )

    assert selection.kernel_name_patterns == (
        "sparse_attention",
        "aten_dsa_sparse_attention",
    )
def test_write_device_profile_summary_json(tmp_path):
    profile_dir = tmp_path / "profile"
    profile_dir.mkdir()
    (profile_dir / "op_summary.csv").write_text("Name,Duration(ms)\nsoftmax,1.25\n")
    summary = read_device_profile(profile_dir, backend="ascend")

    path = write_device_profile_summary(tmp_path / "profile-summary.json", summary)

    payload = json.loads(path.read_text())
    assert payload["backend"] == "ascend"
    assert payload["latency_ms"] == 1.25
