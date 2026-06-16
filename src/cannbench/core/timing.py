from statistics import mean


def _percentile(sorted_values: list[float], percentile: float) -> float:
    if len(sorted_values) == 1:
        return sorted_values[0]

    rank = (len(sorted_values) - 1) * percentile
    lower = int(rank)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = rank - lower
    return round(
        sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight,
        10,
    )


def summarize_timings_ms(values: list[float]) -> dict[str, float]:
    if not values:
        raise ValueError("timing samples must not be empty")

    sorted_values = sorted(values)
    return {
        "latency_ms_avg": mean(sorted_values),
        "latency_ms_p50": _percentile(sorted_values, 0.50),
        "latency_ms_p95": _percentile(sorted_values, 0.95),
        "latency_ms_p99": _percentile(sorted_values, 0.99),
    }
