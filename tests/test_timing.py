import pytest

from cannbench.core.timing import summarize_timings_ms


def test_summarize_timings_ms_computes_percentiles_and_average():
    summary = summarize_timings_ms([1.0, 2.0, 3.0, 4.0])

    assert summary["latency_ms_avg"] == 2.5
    assert summary["latency_ms_p50"] == 2.5
    assert summary["latency_ms_p95"] == 3.85


def test_summarize_timings_ms_rejects_empty_input():
    with pytest.raises(ValueError, match="timing samples must not be empty"):
        summarize_timings_ms([])
