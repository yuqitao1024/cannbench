import pytest

from cannbench.operators import get_operator_spec, list_operator_names
from cannbench.datasets import get_operator_dataset


def test_softmax_operator_spec_is_registered():
    spec = get_operator_spec("softmax")

    assert spec.name == "softmax"
    assert spec.supported_dtypes == ("float32", "float16", "bfloat16")
    assert spec.dataset_namespace == "softmax"


def test_embedding_operator_spec_is_registered():
    spec = get_operator_spec("embedding")

    assert spec.name == "embedding"
    assert spec.supported_dtypes == ("float32", "float16", "bfloat16")
    assert spec.dataset_namespace == "embedding"
    assert spec.runner_name == "embedding"


def test_gather_operator_spec_is_registered():
    spec = get_operator_spec("gather")

    assert spec.name == "gather"
    assert spec.supported_dtypes == ("float32", "float16", "bfloat16")
    assert spec.dataset_namespace == "gather"
    assert spec.runner_name == "gather"


def test_index_select_operator_spec_is_registered():
    spec = get_operator_spec("index_select")

    assert spec.name == "index_select"
    assert spec.supported_dtypes == ("float32", "float16", "bfloat16")
    assert spec.dataset_namespace == "index_select"
    assert spec.runner_name == "index_select"


def test_take_along_dim_operator_spec_is_registered():
    spec = get_operator_spec("take_along_dim")

    assert spec.name == "take_along_dim"
    assert spec.supported_dtypes == ("float32", "float16", "bfloat16")
    assert spec.dataset_namespace == "take_along_dim"
    assert spec.runner_name == "take_along_dim"


def test_list_operator_names_contains_softmax():
    assert "softmax" in list_operator_names()
    assert "embedding" in list_operator_names()
    assert "gather" in list_operator_names()
    assert "index_select" in list_operator_names()
    assert "take_along_dim" in list_operator_names()


def test_unknown_operator_spec_is_rejected():
    with pytest.raises(ValueError, match="Unsupported operator"):
        get_operator_spec("unknown")


def test_embedding_dataset_is_registered():
    dataset = get_operator_dataset("embedding")

    assert dataset.name == "embedding"
    assert dataset.dataset_namespace == "embedding"
    assert len(dataset.get("smoke").cases) == 3


def test_gather_dataset_is_registered():
    dataset = get_operator_dataset("gather")

    assert dataset.name == "gather"
    assert dataset.dataset_namespace == "gather"


def test_index_select_dataset_is_registered():
    dataset = get_operator_dataset("index_select")

    assert dataset.name == "index_select"
    assert dataset.dataset_namespace == "index_select"


def test_take_along_dim_dataset_is_registered():
    dataset = get_operator_dataset("take_along_dim")

    assert dataset.name == "take_along_dim"
    assert dataset.dataset_namespace == "take_along_dim"


def test_softmax_dataset_is_registered():
    dataset = get_operator_dataset("softmax")

    assert dataset.name == "softmax"
    assert dataset.dataset_namespace == "softmax"
