import subprocess
import sys
from pathlib import Path


SCRIPT = Path("tools/check_gpu_dsa_env.py")


def test_gpu_dsa_env_check_can_validate_cases_without_gpu():
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--check-cases-only"],
        cwd=Path.cwd(),
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "[OK] cannbench import" in result.stdout
    assert "[OK] workflow dsa_decode/realistic_decode" in result.stdout
    assert "[OK] workflow dsa_prefill/realistic_prefill" in result.stdout
    assert "GPU checks skipped" in result.stdout


def test_gpu_dsa_env_check_reports_adapter_callable_status(tmp_path):
    adapter = tmp_path / "fake_cuda_dsa_adapter.py"
    adapter.write_text(
        "def lightning_indexer(**kwargs):\n"
        "    return None\n"
        "def sparse_attention(**kwargs):\n"
        "    return None\n",
        encoding="utf-8",
    )
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--check-cases-only",
            "--adapter-module",
            "fake_cuda_dsa_adapter",
        ],
        cwd=Path.cwd(),
        env={"PYTHONPATH": f"{tmp_path}:src"},
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "[OK] CUDA DSA adapter module fake_cuda_dsa_adapter" in result.stdout
    assert "[OK] adapter callable lightning_indexer" in result.stdout
    assert "[OK] adapter callable sparse_attention" in result.stdout
