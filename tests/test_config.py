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
    assert request.seed is not None


def test_embedding_request_accepts_builtin_dataset_case():
    request = OperatorBenchmarkRequest(
        backend="nvidia",
        op="embedding",
        dtype="float16",
        dataset="smoke",
        case_id="tiny_token_lookup",
        warmup=5,
        iterations=10,
    )

    assert request.op == "embedding"
    assert request.case_payload == {
        "num_embeddings": 128,
        "embedding_dim": 64,
        "index_shape": (32,),
    }


def test_gather_request_accepts_builtin_dataset_case():
    request = OperatorBenchmarkRequest(
        backend="nvidia",
        op="gather",
        dtype="float16",
        dataset="smoke",
        case_id="tiny_rank2_gather",
        warmup=5,
        iterations=10,
    )

    assert request.op == "gather"
    assert request.case_payload == {
        "input_shape": (32, 64),
        "index_shape": (32, 32),
        "dim": 1,
    }


def test_index_select_request_accepts_builtin_dataset_case():
    request = OperatorBenchmarkRequest(
        backend="nvidia",
        op="index_select",
        dtype="float16",
        dataset="smoke",
        case_id="tiny_rank2_index_select",
        warmup=5,
        iterations=10,
    )

    assert request.op == "index_select"
    assert request.case_payload == {
        "input_shape": (32, 64),
        "index_shape": (16,),
        "dim": 1,
    }


def test_take_along_dim_request_accepts_builtin_dataset_case():
    request = OperatorBenchmarkRequest(
        backend="nvidia",
        op="take_along_dim",
        dtype="float16",
        dataset="smoke",
        case_id="tiny_rank2_take_along_dim",
        warmup=5,
        iterations=10,
    )

    assert request.op == "take_along_dim"
    assert request.case_payload == {
        "input_shape": (32, 64),
        "index_shape": (32, 16),
        "dim": 1,
    }


def test_masked_select_request_accepts_builtin_dataset_case():
    request = OperatorBenchmarkRequest(
        backend="nvidia",
        op="masked_select",
        dtype="float16",
        dataset="smoke",
        case_id="tiny_rank2_masked_select",
        warmup=5,
        iterations=10,
    )

    assert request.op == "masked_select"
    assert request.case_payload == {
        "input_shape": (32, 64),
        "mask_shape": (32, 64),
        "mask_density": 0.5,
    }


def test_cross_entropy_request_accepts_builtin_dataset_case():
    request = OperatorBenchmarkRequest(
        backend="nvidia",
        op="cross_entropy",
        dtype="float16",
        dataset="smoke",
        case_id="tiny_token_classification_loss",
        warmup=5,
        iterations=10,
    )

    assert request.op == "cross_entropy"
    assert request.case_payload == {
        "logits_shape": (32, 128, 64),
        "target_shape": (32, 128),
        "num_classes": 64,
    }


def test_scatter_add_request_accepts_builtin_dataset_case():
    request = OperatorBenchmarkRequest(
        backend="nvidia",
        op="scatter_add",
        dtype="float16",
        dataset="smoke",
        case_id="tiny_rank2_scatter_add",
        warmup=5,
        iterations=10,
    )

    assert request.op == "scatter_add"
    assert request.case_payload == {
        "input_shape": (32, 64),
        "index_shape": (32, 64),
        "src_shape": (32, 64),
        "dim": 1,
    }


def test_index_add_request_accepts_builtin_dataset_case():
    request = OperatorBenchmarkRequest(
        backend="nvidia",
        op="index_add",
        dtype="float16",
        dataset="smoke",
        case_id="tiny_rank2_index_add",
        warmup=5,
        iterations=10,
    )

    assert request.op == "index_add"
    assert request.case_payload == {
        "input_shape": (32, 64),
        "index_shape": (16,),
        "src_shape": (32, 16),
        "dim": 1,
    }


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


def test_operator_request_defaults_seed():
    request = OperatorBenchmarkRequest(
        backend="nvidia",
        op="softmax",
        dtype="float16",
        dataset="smoke",
        case_id="tiny_logits",
        warmup=5,
        iterations=10,
    )

    assert request.seed == 0


def test_operator_request_rejects_unknown_dataset():
    with pytest.raises(ValueError, match="Unknown operator dataset"):
        OperatorBenchmarkRequest(
            backend="nvidia",
            op="softmax",
            dtype="float16",
            dataset="unknown",
            case_id="tiny_logits",
            warmup=5,
            iterations=10,
        )


def test_embedding_request_rejects_unknown_case_id():
    with pytest.raises(ValueError, match="Unknown embedding case"):
        OperatorBenchmarkRequest(
            backend="nvidia",
            op="embedding",
            dtype="float16",
            dataset="smoke",
            case_id="missing",
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
