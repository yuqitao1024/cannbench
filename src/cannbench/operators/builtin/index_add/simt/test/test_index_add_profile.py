import json

import pytest

from cannbench.core.profile import (
    ProfileKernelSelection,
    read_device_profile,
    write_device_profile_summary,
)
from cannbench.operators import get_operator_plugin


def test_index_add_plugin_profile_patterns_accept_nvidia_large_index_kernel(tmp_path):
    profile_dir = tmp_path / "profile"
    profile_dir.mkdir()
    (profile_dir / "ncu.csv").write_text(
        "Kernel Name,Metric Name,Metric Unit,Metric Value\n"
        '"void indexFuncLargeIndex<unsigned int, unsigned int, float, float>",'
        "gpu__time_duration.avg,usecond,125\n"
    )

    selection = get_operator_plugin("index_add").profile_kernel_selection(
        backend="nvidia",
        implementation=None,
        implementation_version=None,
    )

    summary = read_device_profile(
        profile_dir,
        backend="nvidia",
        kernel_selection=selection,
    )
    assert summary.sample_count == 1
    assert summary.latency_ms_avg == 0.125