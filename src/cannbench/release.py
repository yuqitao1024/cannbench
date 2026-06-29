from __future__ import annotations

import shutil
import tarfile
from dataclasses import dataclass
from pathlib import Path

from cannbench.core.prepared_input import (
    build_prepared_operator_input,
    write_prepared_operator_input,
)
from cannbench.operators import list_operator_names
from cannbench.datasets.loader import get_operator_dataset


@dataclass(frozen=True)
class ReleaseStageResult:
    stage_dir: Path
    prepared_input_count: int


def _copy_tree(source: Path, dest: Path) -> None:
    shutil.copytree(
        source,
        dest,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
    )


def stage_release_tree(
    *,
    repo_root: Path,
    stage_dir: Path,
    dtype: str = "float16",
    seed: int = 7,
) -> ReleaseStageResult:
    if stage_dir.exists():
        shutil.rmtree(stage_dir)
    stage_dir.mkdir(parents=True)

    _copy_tree(repo_root / "src", stage_dir / "src")
    for name in ("pyproject.toml", "README.md", "LICENSE", "install.sh"):
        shutil.copy2(repo_root / name, stage_dir / name)
    deploy_dir = repo_root / "deploy"
    if deploy_dir.is_dir():
        _copy_tree(deploy_dir, stage_dir / "deploy")
    web_dist = repo_root / "web" / "dist"
    if web_dist.is_dir():
        _copy_tree(web_dist, stage_dir / "web" / "dist")

    prepared_count = 0
    for op_name in list_operator_names():
        dataset = get_operator_dataset(op_name)
        for split in ("smoke", "realistic", "stress"):
            for case in dataset.get(split).cases:
                prepared_path = (
                    stage_dir
                    / "prepared"
                    / op_name
                    / split
                    / f"{case.case_id}-{dtype}-seed{seed}.json"
                )
                prepared = build_prepared_operator_input(
                    op=op_name,
                    dtype=dtype,
                    dataset=split,
                    case_id=case.case_id,
                    seed=seed,
                )
                write_prepared_operator_input(prepared_path, prepared)
                prepared_count += 1
    return ReleaseStageResult(stage_dir=stage_dir, prepared_input_count=prepared_count)


def build_release_archive(
    *,
    repo_root: Path,
    output_path: Path,
    stage_dir: Path,
    dtype: str = "float16",
    seed: int = 7,
) -> Path:
    result = stage_release_tree(
        repo_root=repo_root,
        stage_dir=stage_dir,
        dtype=dtype,
        seed=seed,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output_path, "w:gz") as archive:
        archive.add(result.stage_dir, arcname=result.stage_dir.name)
    return output_path
