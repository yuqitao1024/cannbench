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


def test_list_operator_names_contains_softmax():
    assert "softmax" in list_operator_names()
    assert "embedding" in list_operator_names()


def test_unknown_operator_spec_is_rejected():
    with pytest.raises(ValueError, match="Unsupported operator"):
        get_operator_spec("unknown")


def test_embedding_dataset_is_registered():
    dataset = get_operator_dataset("embedding")

    assert dataset.name == "embedding"
    assert dataset.dataset_namespace == "embedding"
    assert len(dataset.get("smoke").cases) == 3
