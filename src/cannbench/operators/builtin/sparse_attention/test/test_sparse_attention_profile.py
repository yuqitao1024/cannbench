from cannbench.operators import get_operator_plugin


def test_cuda_library_profile_includes_complete_dynamic_nvtx_range():
    selection = get_operator_plugin("sparse_attention").profile_kernel_selection(
        backend="nvidia",
        implementation="cuda_library",
        implementation_version=None,
    )

    assert selection.kernel_name_patterns == ()
    assert selection.launch_count is None
    assert selection.nvtx_range == "cannbench_sparse_attention_dynamic"
    assert selection.aggregate_across_files is True


def test_vllm_ascend_profile_includes_dynamic_lowering_and_lse():
    selection = get_operator_plugin("sparse_attention").profile_kernel_selection(
        backend="ascend",
        implementation="vllm_ascend",
        implementation_version=None,
    )

    assert selection.kernel_name_patterns == (
        "sparseflashattention",
        "asstrided",
        "transpose",
        "slice",
        "contiguous",
        "cat",
        "cast",
        "log",
        "add",
    )
    assert selection.launch_count == 64
    assert selection.aggregate_across_files is True
