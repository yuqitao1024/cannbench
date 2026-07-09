from pathlib import Path

from cannbench.operators import (
    get_operator_plugin,
    list_operator_names,
    list_operator_plugins,
)


def test_operator_plugins_are_self_contained_packages():
    root = Path("src/cannbench/operators/builtin")

    for operator_name in list_operator_names():
        operator_root = root / operator_name
        assert operator_root.is_dir(), operator_name
        assert (operator_root / "__init__.py").is_file(), operator_name
        assert (operator_root / "cases.py").is_file(), operator_name
        assert (operator_root / "materialize.py").is_file(), operator_name
        assert (operator_root / "data" / "smoke.json").is_file(), operator_name
        assert (operator_root / "data" / "realistic.json").is_file(), operator_name
        assert (operator_root / "data" / "stress.json").is_file(), operator_name


def test_dsa_decode_and_prefill_are_registered_as_fused_operators():
    decode = get_operator_plugin("dsa_decode")
    prefill = get_operator_plugin("dsa_prefill")

    assert decode.spec.name == "dsa_decode"
    assert decode.spec.runner_name == "dsa_decode"
    assert decode.get_dataset("smoke").cases == ()
    assert all(case.phase == "decode" for case in decode.get_dataset("realistic").cases)

    assert prefill.spec.name == "dsa_prefill"
    assert prefill.spec.runner_name == "dsa_prefill"
    assert all(case.phase == "prefill" for case in prefill.get_dataset("smoke").cases)


def test_operator_datasets_are_not_kept_under_legacy_dataset_data_root():
    legacy_root = Path("src/cannbench/datasets/data")

    for operator_name in list_operator_names():
        assert not (legacy_root / operator_name).exists(), operator_name


def test_operator_plugins_cover_registered_operator_names():
    plugin_names = tuple(plugin.spec.name for plugin in list_operator_plugins())

    assert plugin_names == list_operator_names()


def test_operator_plugins_own_external_implementation_hooks():
    softmax = get_operator_plugin("softmax")
    lightning_indexer = get_operator_plugin("lightning_indexer")
    sparse_attention = get_operator_plugin("sparse_attention")

    assert callable(softmax.build_simt_callable)
    assert softmax.simt_module_name("v1") == "aten_softmax"
    assert softmax.simt_module_name("v2") == "aten_softmax_v2"
    assert softmax.simt_module_name("v3") == "aten_softmax_v3"

    assert callable(lightning_indexer.build_simt_callable)
    assert lightning_indexer.simt_module_name("v1") == "aten_dsa_lightning_indexer"
    assert lightning_indexer.simt_module_name("v2") is None
    assert callable(lightning_indexer.build_cuda_library_callable)
    assert callable(lightning_indexer.build_vllm_ascend_callable)
    assert callable(sparse_attention.build_simt_callable)
    assert sparse_attention.simt_module_name("v1") == "aten_dsa_sparse_attention"
    assert sparse_attention.simt_module_name("v2") is None
    assert callable(sparse_attention.build_cuda_library_callable)
    assert callable(sparse_attention.build_vllm_ascend_callable)


def test_lightning_indexer_operator_plugin_registers_simt_hook():
    lightning_indexer = get_operator_plugin("lightning_indexer")

    assert callable(lightning_indexer.build_simt_callable)
    assert lightning_indexer.simt_module_name("v1") == "aten_dsa_lightning_indexer"
    assert lightning_indexer.simt_module_name("v2") is None


def test_operator_plugin_default_profile_kernel_selection_comes_from_plugin():
    plugin = get_operator_plugin("embedding")

    selection = plugin.profile_kernel_selection(
        backend="nvidia",
        implementation=None,
        implementation_version=None,
    )

    assert selection.kernel_name_patterns == ("embedding",)


def test_non_operator_source_does_not_hardcode_builtin_operator_names():
    checked_files = [
        path
        for path in Path("src/cannbench").rglob("*.py")
        if "src/cannbench/operators/" not in path.as_posix()
    ]

    for path in checked_files:
        source = path.read_text(encoding="utf-8")
        for operator_name in list_operator_names():
            assert f'"{operator_name}"' not in source, path
            assert f"'{operator_name}'" not in source, path


def test_torch_backend_base_does_not_own_operator_specific_helpers():
    from cannbench.backends.torch_backend_base import TorchOperatorBackend

    operator_specific_helpers = {
        "_softmax",
        "_index_add",
        "_index_add_index_dtype",
        "_topk",
        "_lightning_indexer",
        "_sparse_attention",
    }

    assert operator_specific_helpers.isdisjoint(TorchOperatorBackend.__dict__)


def test_operator_plugins_own_payload_summary_ordering():
    embedding = get_operator_plugin("embedding")
    case = embedding.build_result_case(embedding.get_case("smoke", "tiny_token_lookup"))

    assert case.payload_summary == (
        "embedding_dim=64, index_shape=32, num_embeddings=128"
    )


def test_core_result_does_not_own_operator_payload_fields():
    source = Path("src/cannbench/core/result.py").read_text(encoding="utf-8")

    assert "embedding_dim" not in source
    assert "num_embeddings" not in source


def test_public_cli_and_config_do_not_own_workflow_dataset_names():
    checked_files = [
        Path("src/cannbench/cli.py"),
        Path("src/cannbench/core/config.py"),
    ]

    for path in checked_files:
        source = path.read_text(encoding="utf-8")
        assert "realistic_decode" not in source
        assert "realistic_prefill" not in source


def test_dsa_workflow_specifics_live_under_operator_packages():
    assert not Path("src/cannbench/operators/builtin/_dsa_fused.py").exists()


def test_index_add_implementation_tests_live_under_operator_package():
    assert not Path("tests/test_index_add_simt.py").exists()
    assert Path(
        "src/cannbench/operators/builtin/index_add/simt/test/"
        "test_ascend_index_add_simt_op.py"
    ).is_file()


def test_softmax_operator_plugin_owns_dataset_and_materialization():
    plugin = get_operator_plugin("softmax")

    dataset = plugin.get_dataset("smoke")
    case = plugin.get_case("smoke", "tiny_logits")
    payload = plugin.materialize_inputs(case, dtype="float16", seed=7)

    assert plugin.spec.name == "softmax"
    assert dataset.name == "smoke"
    assert case.case_id == "tiny_logits"
    assert payload["shape"] == (32, 128)
    assert payload["dim"] == -1
    assert payload["dtype"] == "float16"
    assert len(payload["values"]) == 32 * 128


def test_embedding_operator_plugin_owns_dataset_and_materialization():
    plugin = get_operator_plugin("embedding")

    dataset = plugin.get_dataset("smoke")
    case = plugin.get_case("smoke", "tiny_token_lookup")
    payload = plugin.materialize_inputs(case, dtype="float16", seed=7)

    assert plugin.spec.name == "embedding"
    assert dataset.name == "smoke"
    assert case.case_id == "tiny_token_lookup"
    assert payload["index_shape"] == (32,)
    assert payload["num_embeddings"] == 128
    assert payload["embedding_dim"] == 64
    assert len(payload["indices"]) == 32
