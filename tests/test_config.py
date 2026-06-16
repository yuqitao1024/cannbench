import pytest

from cannbench.core.config import OperatorBenchmarkRequest


def test_operator_request_accepts_builtin_dataset_case():
    request = OperatorBenchmarkRequest(
        backend="nvidia",
        op="softmax",
        dtype="float16",
        dataset="smoke",
        case_id="tiny_logits",
        warmup=5,
        iterations=10,
    )

    assert request.dataset == "smoke"
    assert request.case_id == "tiny_logits"
    assert request.dimensions == (32, 128)
    assert request.dim == -1


def test_operator_request_rejects_unknown_dtype():
    with pytest.raises(ValueError, match="Unsupported dtype"):
        OperatorBenchmarkRequest(
            backend="nvidia",
            op="softmax",
            dtype="fp9",
            dataset="smoke",
            case_id="tiny_logits",
            warmup=5,
            iterations=10,
        )


def test_operator_request_defaults_output_formats():
    request = OperatorBenchmarkRequest(
        backend="nvidia",
        op="softmax",
        dtype="float16",
        dataset="smoke",
        case_id="tiny_logits",
        warmup=5,
        iterations=10,
    )

    assert request.output_formats == ("json", "csv", "md")


def test_operator_request_rejects_unknown_dataset():
    with pytest.raises(ValueError, match="Unknown softmax dataset"):
        OperatorBenchmarkRequest(
            backend="nvidia",
            op="softmax",
            dtype="float16",
            dataset="unknown",
            case_id="tiny_logits",
            warmup=5,
            iterations=10,
        )


@pytest.mark.parametrize("case_id", ["", "   "])
def test_operator_request_rejects_empty_case_id(case_id: str):
    with pytest.raises(ValueError, match="case_id must not be empty"):
        OperatorBenchmarkRequest(
            backend="nvidia",
            op="softmax",
            dtype="float16",
            dataset="smoke",
            case_id=case_id,
            warmup=5,
            iterations=10,
        )


def test_operator_request_rejects_negative_warmup():
    with pytest.raises(ValueError, match="warmup must be >= 0"):
        OperatorBenchmarkRequest(
            backend="nvidia",
            op="softmax",
            dtype="float16",
            dataset="smoke",
            case_id="tiny_logits",
            warmup=-1,
            iterations=10,
        )


def test_operator_request_rejects_non_positive_iterations():
    with pytest.raises(ValueError, match="iterations must be > 0"):
        OperatorBenchmarkRequest(
            backend="nvidia",
            op="softmax",
            dtype="float16",
            dataset="smoke",
            case_id="tiny_logits",
            warmup=0,
            iterations=0,
        )


def test_operator_request_rejects_unsupported_output_formats():
    with pytest.raises(ValueError, match="unsupported output format"):
        OperatorBenchmarkRequest(
            backend="nvidia",
            op="softmax",
            dtype="float16",
            dataset="smoke",
            case_id="tiny_logits",
            warmup=0,
            iterations=1,
            output_formats=("json", "yaml"),
        )
