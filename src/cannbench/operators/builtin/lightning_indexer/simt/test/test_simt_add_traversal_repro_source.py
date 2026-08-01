from pathlib import Path


REPRO_ROOT = Path(__file__).parent / "simt_add_traversal_repro"


def test_simt_add_traversal_repro_accumulates_into_initialized_output():
    source = (REPRO_ROOT / "main.asc").read_text(encoding="utf-8")

    assert source.count(
        "output[index] += input_x[index] + input_y[index];"
    ) == 2
    for suffix in range(4):
        assert source.count(
            f"output[index{suffix}] += input_x{suffix} + input_y{suffix};"
        ) == 2

    assert "output[index] = static_cast<float>(index % 31U) * 0.125F;" in source
    assert "golden[index] = output[index];" in source
    assert "golden[index] += input_x[index] + input_y[index];" in source
    assert "aclrtMemcpy(output_device, kBytes, output.data(), kBytes, ACL_MEMCPY_HOST_TO_DEVICE)" in source
    assert "aclrtMemset(output_device" not in source


def test_simt_add_traversal_repro_profiles_with_default_msopprof_replay():
    source = (REPRO_ROOT / "main.asc").read_text(encoding="utf-8")
    readme = (REPRO_ROOT / "README.md").read_text(encoding="utf-8")
    design = (REPRO_ROOT / "DESIGN.md").read_text(encoding="utf-8")

    assert "kReplayProbeIndex" in source
    assert "kMaxValidationAccumulations" in source
    assert "expected_accumulation_count" in source
    assert "accumulation_count == expected_accumulation_count" in source
    assert "iteration < expected_accumulation_count" in source
    assert 'std::cout << "accumulation_count=" << accumulation_count' in source
    assert (
        'std::cout << "expected_accumulation_count="'
        " << expected_accumulation_count" in source
    )
    for document in (readme, design):
        assert "--replay-mode" not in document
        assert "--warm-up" not in document
