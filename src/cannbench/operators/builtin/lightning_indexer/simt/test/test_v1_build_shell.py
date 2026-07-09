from pathlib import Path


def test_lightning_indexer_simt_v1_setup_uses_bisheng_toolchain():
    setup_py = Path(
        "src/cannbench/operators/builtin/lightning_indexer/simt/v1/setup.py"
    ).read_text(encoding="utf-8")

    assert "bisheng" in setup_py
    assert "--enable-simt" in setup_py
    assert 'library_name = "aten_dsa_lightning_indexer"' in setup_py


def test_lightning_indexer_simt_v1_register_has_python_module_entry():
    source = Path(
        "src/cannbench/operators/builtin/lightning_indexer/simt/v1/"
        "aten_dsa_lightning_indexer/csrc/register.asc"
    ).read_text(encoding="utf-8")

    assert "PyInit__C" in source
    assert "PyModuleDef_HEAD_INIT" in source
