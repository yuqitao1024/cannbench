import pytest

from cannbench.core.config import OperatorBenchmarkRequest, WorkflowBenchmarkRequest
from cannbench.core.prepared_input import prepare_workflow_input
from cannbench.operators.builtin.dsa_decode import build_dsa_decode_workflow


def test_operator_request_accepts_builtin_dataset_case():
    request = OperatorBenchmarkRequest(
        backend="nvidia",
        op="softmax",
        dtype="float16",
        dataset="smoke",
        case_id="tiny_logits",
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
    )

    assert request.op == "index_add"
    assert request.case_payload == {
        "input_shape": (32, 64),
        "index_shape": (16,),
        "src_shape": (32, 16),
        "dim": 1,
    }


def test_scatter_request_accepts_builtin_dataset_case():
    request = OperatorBenchmarkRequest(
        backend="nvidia",
        op="scatter",
        dtype="float16",
        dataset="smoke",
        case_id="tiny_rank2_scatter",
    )

    assert request.op == "scatter"
    assert request.case_payload == {
        "input_shape": (32, 64),
        "index_shape": (32, 64),
        "src_shape": (32, 64),
        "dim": 1,
    }


def test_index_put_request_accepts_builtin_dataset_case():
    request = OperatorBenchmarkRequest(
        backend="nvidia",
        op="index_put",
        dtype="float16",
        dataset="smoke",
        case_id="tiny_rank2_index_put",
    )

    assert request.op == "index_put"
    assert request.case_payload == {
        "input_shape": (32, 64),
        "index_shapes": ((16,), (16,)),
        "values_shape": (16,),
        "accumulate": False,
    }


def test_operator_request_rejects_unknown_dtype():
    with pytest.raises(ValueError, match="Unsupported dtype"):
        OperatorBenchmarkRequest(
            backend="nvidia",
            op="softmax",
            dtype="fp9",
            dataset="smoke",
            case_id="tiny_logits",
        )


def test_workflow_request_preserves_prepared_workflow_and_implementation():
    prepared = prepare_workflow_input(
        build_dsa_decode_workflow(
            dataset="realistic",
            case_id="deepseek_v32_flashmla_decode_b2_q2_ctx32768_top2048",
            dtype="bfloat16",
            seed=7,
        )
    )

    request = WorkflowBenchmarkRequest(
        backend="ascend",
        prepared=prepared,
        implementation="simt",
        implementation_version="v2",
        aic_metrics=" InstrTimeline ",
    )

    assert request.prepared is prepared
    assert request.implementation == "simt"
    assert request.implementation_version == "v2"
    assert request.aic_metrics == "InstrTimeline"


def test_operator_request_defaults_seed():
    request = OperatorBenchmarkRequest(
        backend="nvidia",
        op="softmax",
        dtype="float16",
        dataset="smoke",
        case_id="tiny_logits",
    )

    assert request.seed == 0


def test_operator_request_defaults_aic_metrics():
    request = OperatorBenchmarkRequest(
        backend="ascend",
        op="softmax",
        dtype="float16",
        dataset="smoke",
        case_id="tiny_logits",
    )

    assert request.aic_metrics == "BasicInfo"


def test_operator_request_accepts_custom_aic_metrics():
    request = OperatorBenchmarkRequest(
        backend="ascend",
        op="softmax",
        dtype="float16",
        dataset="smoke",
        case_id="tiny_logits",
        aic_metrics=" InstrTimeline ",
    )

    assert request.aic_metrics == "InstrTimeline"


@pytest.mark.parametrize("aic_metrics", ["", "   "])
def test_operator_request_rejects_empty_aic_metrics(aic_metrics: str):
    with pytest.raises(ValueError, match="aic_metrics must not be empty"):
        OperatorBenchmarkRequest(
            backend="ascend",
            op="softmax",
            dtype="float16",
            dataset="smoke",
            case_id="tiny_logits",
            aic_metrics=aic_metrics,
        )


def test_operator_request_rejects_nondefault_aic_metrics_for_nvidia():
    with pytest.raises(ValueError, match="only supported for the ascend backend"):
        OperatorBenchmarkRequest(
            backend="nvidia",
            op="softmax",
            dtype="float16",
            dataset="smoke",
            case_id="tiny_logits",
            aic_metrics="Default",
        )


def test_operator_request_rejects_unknown_dataset():
    with pytest.raises(ValueError, match="Unknown softmax dataset"):
        OperatorBenchmarkRequest(
            backend="nvidia",
            op="softmax",
            dtype="float16",
            dataset="unknown",
            case_id="tiny_logits",
        )


def test_embedding_request_rejects_unknown_case_id():
    with pytest.raises(ValueError, match="Unknown embedding case"):
        OperatorBenchmarkRequest(
            backend="nvidia",
            op="embedding",
            dtype="float16",
            dataset="smoke",
            case_id="missing",
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
        )
