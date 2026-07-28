from pathlib import Path


REPRO_ROOT = Path(
    "src/cannbench/operators/builtin/lightning_indexer/simt/test/"
    "msopprof_two_vf_repro"
)


def test_msopprof_two_vf_repro_is_standalone_and_has_a_single_vf_control():
    assert REPRO_ROOT.is_dir(), "standalone msopprof reproduction is missing"
    for filename in (
        "CMakeLists.txt",
        "two_vf_main.asc",
        "single_vf_main.asc",
        "copy_vf.asc",
        "add_one_vf.asc",
    ):
        assert (REPRO_ROOT / filename).is_file(), f"missing {filename}"

    cmake = (REPRO_ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
    two_vf_main = (REPRO_ROOT / "two_vf_main.asc").read_text(encoding="utf-8")
    single_vf_main = (REPRO_ROOT / "single_vf_main.asc").read_text(
        encoding="utf-8"
    )
    copy_vf = (REPRO_ROOT / "copy_vf.asc").read_text(encoding="utf-8")
    add_one_vf = (REPRO_ROOT / "add_one_vf.asc").read_text(encoding="utf-8")

    assert "add_library(two_vf_kernels SHARED" in cmake
    assert "add_library(single_vf_kernel SHARED" in cmake
    assert "add_executable(two_vf_repro" in cmake
    assert "two_vf_main.asc" in cmake
    assert "copy_vf.asc" in cmake
    assert "add_one_vf.asc" in cmake
    assert "add_executable(single_vf_control" in cmake
    assert "single_vf_main.asc" in cmake
    assert "target_link_libraries(two_vf_repro PRIVATE two_vf_kernels)" in cmake
    assert "target_link_libraries(single_vf_control PRIVATE single_vf_kernel)" in cmake
    assert not (REPRO_ROOT / "two_vf_repro.asc").exists()
    assert not (REPRO_ROOT / "single_vf_control.asc").exists()
    assert two_vf_main.count("__simt_vf__") == 0
    assert single_vf_main.count("__simt_vf__") == 0
    assert copy_vf.count("__simt_vf__") == 1
    assert add_one_vf.count("__simt_vf__") == 1
    assert '"--launch-second"' in two_vf_main
    assert (
        "launch_add_one_vf_repro(\n"
        "        input_device, output_device, kElementCount, stream);"
        in two_vf_main
    )

    for source in (two_vf_main, single_vf_main, copy_vf, add_one_vf):
        assert '#include "acl/acl.h"' in source
        for forbidden in ("torch", "Python.h", "cannbench"):
            assert forbidden not in source

    for source in (copy_vf, add_one_vf):
        assert '#include "simt_api/asc_simt.h"' in source
        assert "constexpr int32_t kThreadsPerBlock = 1024;" in source
        assert "__global__ __vector__" in source
        for forbidden in (
            "__global__ __aicore__",
            "KERNEL_TYPE_MIX_AIC_1_2",
            "ASCEND_IS_AIC",
            "ASCEND_IS_AIV",
            "basic_api/",
            "kernel_operator.h",
        ):
            assert forbidden not in source
