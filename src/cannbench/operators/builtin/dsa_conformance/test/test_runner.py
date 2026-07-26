from cannbench.operators.builtin.dsa_conformance.runner import (
    canonical_input_descriptor,
    deterministic_pattern,
    load_v32_cases,
)


def test_deterministic_pattern_has_long_nonrepeating_period():
    pattern = deterministic_pattern(seed=7, nonnegative=False)

    assert len(pattern) == 65521
    assert pattern == deterministic_pattern(seed=7, nonnegative=False)
    assert min(pattern) == -32760.0 / 32768.0
    assert max(pattern) == 32760.0 / 32768.0
    assert len(set(pattern)) == 65521


def test_deterministic_pattern_seed_changes_order_instead_of_rotating_pattern():
    first = deterministic_pattern(seed=7, nonnegative=False)
    second = deterministic_pattern(seed=8, nonnegative=False)
    rotation = first.index(second[0])
    rotated_prefix = tuple(
        first[(rotation + index) % len(first)] for index in range(128)
    )

    assert second[:128] != rotated_prefix


def test_prefill_descriptor_records_full_v32_contract():
    indexer_case, sparse_case = load_v32_cases("prefill")

    descriptor = canonical_input_descriptor(indexer_case, sparse_case)

    assert descriptor["query_tokens"] == 4096
    assert descriptor["context_tokens"] == 32768
    assert descriptor["top_k"] == 2048
    assert descriptor["index_shape"] == [1, 4096, 64, 128]
    assert descriptor["attention_query_shape"] == [1, 128, 4096, 576]
    assert descriptor["shared_kv_shape"] == [1, 1, 32768, 576]
    assert descriptor["value_head_dim"] == 512
    assert descriptor["generator"] == "splitmix64-period-v2"


def test_decode_cases_share_sequence_metadata():
    indexer_case, sparse_case = load_v32_cases("decode")

    assert indexer_case.resolved_query_lens == sparse_case.resolved_query_lens
    assert indexer_case.resolved_context_lens == sparse_case.resolved_context_lens
    assert (
        indexer_case.resolved_query_start_positions
        == sparse_case.resolved_query_start_positions
    )
