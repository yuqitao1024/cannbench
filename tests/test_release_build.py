from pathlib import Path
import subprocess
import sys

from cannbench.release import stage_release_tree


def test_stage_release_tree_copies_project_files_and_generates_prepared_inputs(tmp_path):
    repo_root = Path.cwd()
    stage_dir = tmp_path / "cannbench-release"

    result = stage_release_tree(
        repo_root=repo_root,
        stage_dir=stage_dir,
        dtype="float16",
        seed=7,
    )

    assert result.stage_dir == stage_dir
    assert (stage_dir / "src" / "cannbench" / "cli.py").is_file()
    assert (stage_dir / "pyproject.toml").is_file()
    assert (stage_dir / "README.md").is_file()
    assert (stage_dir / "LICENSE").is_file()
    assert (stage_dir / "install.sh").is_file()
    assert (stage_dir / "tools" / "check_gpu_dsa_env.py").is_file()
    assert (stage_dir / "deploy" / "systemd" / "cannbench-serve.service").is_file()
    assert (stage_dir / "published" / "index.json").is_file()
    assert (
        stage_dir
        / "published"
        / "opbench-ascend-950pr-cannops-softmax-realistic-float16"
        / "meta"
        / "benchmark-records.json"
    ).is_file()
    assert (stage_dir / "prepared" / "softmax" / "smoke" / "tiny_logits-float16-seed7.json").is_file()
    assert result.prepared_input_count > 0


def test_release_install_assets_target_opt_cannbench(tmp_path):
    repo_root = Path.cwd()
    stage_dir = tmp_path / "cannbench-release"

    stage_release_tree(
        repo_root=repo_root,
        stage_dir=stage_dir,
        dtype="float16",
        seed=7,
    )

    install_script = (stage_dir / "install.sh").read_text(encoding="utf-8")
    service_unit = (stage_dir / "deploy" / "systemd" / "cannbench-serve.service").read_text(encoding="utf-8")

    assert 'INSTALL_DIR="/opt/cannbench"' in install_script
    assert "WorkingDirectory=/opt/cannbench" in service_unit
    assert "PYTHONPATH=/opt/cannbench/src" in service_unit
    assert "/opt/cannbench/cannbench-release" not in install_script
    assert "/opt/cannbench/cannbench-release" not in service_unit


def test_build_release_script_runs_from_repo_root(tmp_path):
    result = subprocess.run(
        [sys.executable, "tools/build_release.py", "--output-dir", str(tmp_path)],
        cwd=Path.cwd(),
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert (tmp_path / "cannbench-release.tar.gz").is_file()
