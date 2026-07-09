from cannbench.operators.builtin.index_add.index_diagnostics import (
    summarize_index_add_dataset,
    summarize_index_distribution,
)


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
