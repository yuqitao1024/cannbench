from cannbench.operators import get_operator_plugin


def test_cuda_library_profile_includes_deepgemm_logits_and_torch_topk():
    selection = get_operator_plugin("lightning_indexer").profile_kernel_selection(
        backend="nvidia",
        implementation="cuda_library",
        implementation_version=None,
    )

    assert selection.kernel_name_patterns == ("mqa_logits", "topk")
    assert selection.launch_count == 2
