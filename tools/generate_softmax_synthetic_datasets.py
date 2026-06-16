from __future__ import annotations

import json
from pathlib import Path

from cannbench.datasets.synthetic import (
    build_softmax_smoke_case,
    build_softmax_stress_case,
)


ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "src" / "cannbench" / "datasets" / "data" / "softmax"


def _case_to_dict(case) -> dict[str, object]:
    return {
        "case_id": case.case_id,
        "family": case.family,
        "shape": list(case.shape),
        "dim": case.dim,
        "source_kind": case.source_kind,
        "source_project": case.source_project,
        "source_model": case.source_model,
        "source_file": case.source_file,
        "source_op": case.source_op,
    }


def write_dataset(name: str, cases: list[dict[str, object]]) -> None:
    path = DATASET_DIR / f"{name}.json"
    path.write_text(
        json.dumps({"name": name, "cases": cases}, indent=2) + "\n"
    )


def main() -> None:
    smoke_cases = [
        _case_to_dict(
            build_softmax_smoke_case(
                case_id="tiny_logits",
                family="lm_logits",
                shape=(32, 128),
                dim=-1,
                source_model="smoke_fixture",
            )
        ),
        _case_to_dict(
            build_softmax_smoke_case(
                case_id="tiny_attention_scores",
                family="attention",
                shape=(2, 4, 8, 8),
                dim=-1,
                source_model="smoke_attention_fixture",
            )
        ),
        _case_to_dict(
            build_softmax_smoke_case(
                case_id="tiny_channel_softmax",
                family="channel_activation",
                shape=(2, 16, 8, 8),
                dim=1,
                source_model="smoke_channel_fixture",
            )
        ),
    ]

    stress_cases = [
        _case_to_dict(
            build_softmax_stress_case(
                case_id="long_context_attention",
                family="attention",
                shape=(1, 32, 4096, 4096),
                dim=-1,
                source_model="llm_attention_boundary",
            )
        ),
        _case_to_dict(
            build_softmax_stress_case(
                case_id="wide_vocab_lm_logits",
                family="lm_logits",
                shape=(8192, 128256),
                dim=1,
                source_model="llm_logits_boundary",
            )
        ),
        _case_to_dict(
            build_softmax_stress_case(
                case_id="moe_router_scores",
                family="router_scores",
                shape=(4096, 128),
                dim=-1,
                source_model="moe_router_boundary",
            )
        ),
        _case_to_dict(
            build_softmax_stress_case(
                case_id="small_reduction_axis",
                family="reduction_edge",
                shape=(16384, 2),
                dim=-1,
                source_model="softmax_small_axis_boundary",
            )
        ),
        _case_to_dict(
            build_softmax_stress_case(
                case_id="vision_window_batch",
                family="vision_attention",
                shape=(2048, 16, 49, 49),
                dim=-1,
                source_model="vision_window_batch_boundary",
            )
        ),
        _case_to_dict(
            build_softmax_stress_case(
                case_id="channelwise_activation_map",
                family="channel_activation",
                shape=(64, 2048, 7, 7),
                dim=1,
                source_model="channel_activation_boundary",
            )
        ),
        _case_to_dict(
            build_softmax_stress_case(
                case_id="beam_search_token_scores",
                family="decode_logits",
                shape=(512, 4, 64000),
                dim=-1,
                source_model="beam_search_token_boundary",
            )
        ),
    ]

    write_dataset("smoke", smoke_cases)
    write_dataset("stress", stress_cases)


if __name__ == "__main__":
    main()
