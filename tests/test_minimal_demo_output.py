from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PRUNER = ROOT / "scripts" / "prune_demo_outputs.py"
KEEP = {"edit.png", "coarse_mask.png", "contour_refined_mask.png"}


def make_sample(run_root: Path, defect: str, index: int, *, complete: bool = True) -> Path:
    sample_dir = run_root / defect / "ref_000" / f"{index:03d}"
    sample_dir.mkdir(parents=True)
    for name in sorted(KEEP):
        if complete or name != "contour_refined_mask.png":
            (sample_dir / name).write_bytes(b"required")
    (sample_dir / "metadata.json").write_text("{}", encoding="utf-8")
    (sample_dir / "soft_mask.npy").write_bytes(b"debug")
    debug_dir = sample_dir / "attention_steps"
    debug_dir.mkdir()
    (debug_dir / "step_01.png").write_bytes(b"debug")
    return sample_dir


def run_pruner(run_root: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(PRUNER), str(run_root)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_pruner_keeps_only_submission_artifacts(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    sample_dirs = [
        make_sample(run_root, "crack", 0),
        make_sample(run_root, "hole", 0),
    ]
    for name in (
        "run_config.json",
        "run_config_history.jsonl",
        "run_log.csv",
        "adaptive_log.csv",
        "run_console.log",
    ):
        (run_root / name).write_text("private runtime paths", encoding="utf-8")

    completed = run_pruner(run_root)

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "Pruned 2 sample directories" in completed.stdout
    for sample_dir in sample_dirs:
        assert {path.name for path in sample_dir.iterdir()} == KEEP
    assert {path.name for path in run_root.iterdir()} == {"crack", "hole"}


def test_pruner_validates_every_sample_before_deleting(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    complete_sample = make_sample(run_root, "crack", 0)
    incomplete_sample = make_sample(run_root, "hole", 0, complete=False)
    root_log = run_root / "run_log.csv"
    root_log.write_text("keep on failure", encoding="utf-8")

    completed = run_pruner(run_root)

    assert completed.returncode != 0
    assert "contour_refined_mask.png" in completed.stderr
    assert (complete_sample / "metadata.json").is_file()
    assert (incomplete_sample / "metadata.json").is_file()
    assert root_log.is_file()


def test_pruner_rejects_symlinked_sample_without_touching_target(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    safe_sample = make_sample(run_root, "hole", 0)
    external_sample = make_sample(tmp_path / "external", "crack", 0)
    linked_sample = run_root / "crack" / "ref_000" / "000"
    linked_sample.parent.mkdir(parents=True)
    linked_sample.symlink_to(external_sample, target_is_directory=True)

    completed = run_pruner(run_root)

    assert completed.returncode != 0
    assert "outside run root or uses a symlink" in completed.stderr
    assert (safe_sample / "metadata.json").is_file()
    assert (external_sample / "metadata.json").is_file()


def test_pruner_rejects_sample_below_symlinked_parent(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir()
    external_sample = make_sample(tmp_path / "external", "crack", 0)
    (run_root / "crack").symlink_to(
        external_sample.parents[1], target_is_directory=True
    )

    completed = run_pruner(run_root)

    assert completed.returncode != 0
    assert "outside run root or uses a symlink" in completed.stderr
    assert (external_sample / "metadata.json").is_file()
