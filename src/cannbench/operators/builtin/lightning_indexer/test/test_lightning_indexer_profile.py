from cannbench.operators import get_operator_plugin


def test_cuda_library_profile_uses_fixed_dynamic_kernel_selection():
    selection = get_operator_plugin("lightning_indexer").profile_kernel_selection(
        backend="nvidia",
        implementation="cuda_library",
        implementation_version=None,
    )

    assert selection.kernel_name_patterns == (
        "lightning_indexer",
        "mqa_logits",
        "topk",
        "radix",
        "sort",
        "elementwise",
        "copy",
        "cast",
    )
    assert selection.launch_count is None
    assert selection.nvtx_range == "cannbench_lightning_indexer_dynamic"
    assert selection.aggregate_across_files is True


def test_vllm_ascend_profile_includes_dynamic_ragged_lowering():
    selection = get_operator_plugin("lightning_indexer").profile_kernel_selection(
        backend="ascend",
        implementation="vllm_ascend",
        implementation_version=None,
    )

    assert selection.kernel_name_patterns == (
        "lightning",
        "indexer",
        "cat",
        "cast",
    )
    assert selection.launch_count == 64
    assert selection.aggregate_across_files is True
