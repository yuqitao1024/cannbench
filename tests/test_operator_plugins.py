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
    assert [case.phase for case in decode.get_dataset("smoke").cases] == ["decode"]

    assert prefill.spec.name == "dsa_prefill"
    assert prefill.spec.runner_name == "dsa_prefill"
    assert [case.phase for case in prefill.get_dataset("smoke").cases] == ["prefill"]


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

    assert callable(lightning_indexer.build_cuda_library_callable)
    assert callable(lightning_indexer.build_vllm_ascend_callable)
    assert callable(sparse_attention.build_cuda_library_callable)
    assert callable(sparse_attention.build_vllm_ascend_callable)


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
