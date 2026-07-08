from cannbench.datasets import get_operator_case
from cannbench.operators import get_operator_plugin, get_operator_spec


def test_operator_case_payload_is_available_for_softmax():
    case = get_operator_case("softmax", "smoke", "tiny_logits")

    assert case.payload == {
        "dimensions": (32, 128),
        "dim": -1,
    }


def test_operator_case_payload_is_available_for_embedding():
    case = get_operator_case("embedding", "smoke", "tiny_token_lookup")

    assert case.payload == {
        "num_embeddings": 128,
        "embedding_dim": 64,
        "index_shape": (32,),
    }


def test_operator_specs_expose_distinct_runner_names():
    assert get_operator_spec("softmax").runner_name == "softmax"
    assert get_operator_spec("embedding").runner_name == "embedding"


def test_lightning_indexer_plugin_exposes_simt_dispatch_callable():
    plugin = get_operator_plugin("lightning_indexer")

    assert callable(plugin.build_simt_callable)


def test_sparse_attention_plugin_exposes_simt_dispatch_callable():
    plugin = get_operator_plugin("sparse_attention")

    assert callable(plugin.build_simt_callable)
