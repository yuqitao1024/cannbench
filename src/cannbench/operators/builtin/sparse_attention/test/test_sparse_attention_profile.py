from cannbench.operators import get_operator_plugin


def test_simt_profile_sums_all_sparse_attention_kernel_stages():
    selection = get_operator_plugin("sparse_attention").profile_kernel_selection(
        backend="ascend",
        implementation="simt",
        implementation_version="v1",
    )

    assert selection.kernel_name_patterns == (
        "sparse_attention",
        "aten_dsa_sparse_attention",
    )
    assert selection.aggregate_across_files is True


def test_vllm_simt_profile_selects_copied_sparse_flash_attention_kernel():
    selection = get_operator_plugin("sparse_attention").profile_kernel_selection(
        backend="ascend",
        implementation="simt",
        implementation_version="vllm",
    )

    assert selection.kernel_name_patterns == ("SparseFlashAttention",)
    assert selection.launch_count is None
    assert selection.aggregate_across_files is True


def test_cuda_library_profile_uses_fixed_dynamic_kernel_selection():
    selection = get_operator_plugin("sparse_attention").profile_kernel_selection(
        backend="nvidia",
        implementation="cuda_library",
        implementation_version=None,
    )

    assert selection.kernel_name_patterns == (
        "sparse_attention",
        "flash_mla",
        "flashmla",
        "elementwise",
        "copy",
        "cast",
        "index",
        "arange",
        "fill",
    )
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
