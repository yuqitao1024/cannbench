from pathlib import Path


def _v2_fused_source() -> str:
    path = (
        Path(__file__).parents[1]
        / "v2/aten_dsa_sparse_attention_v2/csrc/simt"
        / "sparse_attention_head64_fused_hd576.asc"
    )
    assert path.is_file(), f"missing V2 fused Head64 device source: {path}"
    return path.read_text(encoding="utf-8")


def _function_definition(source: str, start_marker: str) -> str:
    start = source.index(start_marker)
    while True:
        body_start = source.index("{", start)
        declaration_end = source.find(";", start, body_start)
        if declaration_end == -1:
            break
        start = source.index(start_marker, start + len(start_marker))

    depth = 0
    for index in range(body_start, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return source[start : index + 1]
    raise AssertionError(f"unterminated function definition: {start_marker}")


def test_v2_fused_uses_all_1024_threads_and_32_warps():
    source = _v2_fused_source()

    assert "kHead64FusedLaunchThreads = 1024" in source
    assert "kHead64FusedActiveThreads = 1024" in source
    assert "kHead64FusedActiveWarps = 32" in source
    assert "warp + kHead64FusedActiveWarps" not in source


def test_v2_fused_waits_for_output_update_before_pv_release():
    aiv = _function_definition(
        _v2_fused_source(), "sparse_attention_head64_fused_aiv("
    )

    update = aiv.index("asc_vf_call<head64_fused_output_update_vf>")
    done = aiv.index("asc_sync_wait(PIPE_V, PIPE_MTE3, EVENT_ID1);", update)
    release = aiv.index(
        "AscendC::CrossCoreSetFlag<2, PIPE_MTE3>(kAivToAicReady);", done
    )

    assert update < done < release
