from cannbench.operators.builtin.index_add.index_diagnostics import (
    summarize_index_add_dataset,
    summarize_index_distribution,
)
from cannbench.operators.builtin.index_add import get_index_add_case, get_index_add_dataset
from cannbench.operators.builtin.index_add.cases import IndexAddCase
from cannbench.operators.builtin.index_add.materialize import materialize_index_add_inputs


def test_summarize_index_distribution_reports_global_and_block_duplicates():
    summary = summarize_index_distribution(
        indices=(0, 1, 1, 2, 2, 2, 7, 7),
        dim_size=8,
        block_size=4,
    )

    assert summary["index_size"] == 8
    assert summary["dim_size"] == 8
    assert summary["unique_count"] == 4
    assert summary["duplicate_count"] == 4
    assert summary["duplicate_ratio"] == 0.5
    assert summary["load_factor"] == 1.0
    assert summary["max_bucket_count"] == 3
    assert summary["max_bucket_ratio"] == 0.375
    assert summary["block_size"] == 4
    assert summary["block_count"] == 2
    assert summary["mean_block_duplicate_ratio"] == 0.375
    assert summary["max_block_duplicate_ratio"] == 0.5
    assert summary["adjacent_equal_ratio"] == 4 / 7
    assert summary["adjacent_non_decreasing_ratio"] == 1.0


def test_summarize_index_add_dataset_includes_realistic_1d_and_4d_cases():
    summaries = summarize_index_add_dataset("realistic", dtype="float16", seed=0)
    by_case = {item["case_id"]: item for item in summaries}

    assert by_case["allenai_longformer_large_index_add_inplace"]["rank"] == 1
    assert by_case["allenai_longformer_large_index_add_inplace"]["wrapped_dim"] == 0
    assert by_case["xlnet_memory_index_add"]["rank"] == 4
    assert by_case["xlnet_memory_index_add"]["wrapped_dim"] == 3
    assert by_case["last_dim_memory_medium_index_add"]["rank"] == 4
    assert by_case["last_dim_memory_medium_index_add"]["wrapped_dim"] == 3


def test_index_add_atomic_1d_dataset_controls_index_contention_patterns():
    dataset = get_index_add_dataset("atomic_1d")
    cases = {case.case_id: case for case in dataset.cases}

    assert set(cases) == {
        "atomic_1d_unique_contiguous",
        "atomic_1d_unique_random_permutation",
        "atomic_1d_random_with_replacement",
    }

    small_unique_contiguous = IndexAddCase(
        case_id="small_unique_contiguous",
        family="test",
        input_shape=(16,),
        index_shape=(16,),
        src_shape=(16,),
        dim=0,
        index_pattern="unique_contiguous",
        source_kind="test",
        source_project="test",
        source_model="test",
        source_file="test",
        source_op="torch.index_add",
    )
    contiguous = materialize_index_add_inputs(
        small_unique_contiguous,
        dtype="float16",
        seed=7,
    )
    assert contiguous["indices"][:8] == tuple(range(8))
    assert len(set(contiguous["indices"])) == len(contiguous["indices"])

    small_unique_permutation = IndexAddCase(
        case_id="small_unique_random_permutation",
        family="test",
        input_shape=(16,),
        index_shape=(16,),
        src_shape=(16,),
        dim=0,
        index_pattern="unique_random_permutation",
        source_kind="test",
        source_project="test",
        source_model="test",
        source_file="test",
        source_op="torch.index_add",
    )
    permutation = materialize_index_add_inputs(
        small_unique_permutation,
        dtype="float16",
        seed=7,
    )
    assert permutation["indices"][:8] != tuple(range(8))
    assert len(set(permutation["indices"])) == len(permutation["indices"])

    replacement_case = get_index_add_case(
        "atomic_1d",
        "atomic_1d_random_with_replacement",
    )
    assert replacement_case.index_pattern == "random_uniform"
    assert replacement_case.index_shape[0] > replacement_case.input_shape[0]
