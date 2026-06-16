import pytest

from cannbench.operators import get_operator_spec, list_operator_names


def test_softmax_operator_spec_is_registered():
    spec = get_operator_spec("softmax")

    assert spec.name == "softmax"
    assert spec.supported_dtypes == ("float32", "float16", "bfloat16")
    assert spec.dataset_namespace == "softmax"


def test_list_operator_names_contains_softmax():
    assert "softmax" in list_operator_names()


def test_unknown_operator_spec_is_rejected():
    with pytest.raises(ValueError, match="Unsupported operator"):
        get_operator_spec("unknown")
