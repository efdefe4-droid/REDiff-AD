from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


def test_core_runtime_uses_in_context_entrypoints() -> None:
    legacy_run = ROOT / ("run_" + "insert_anything.py")
    legacy_batch = ROOT / ("batch_" + "insert_anything.py")
    assert (ROOT / "run_in_context.py").is_file()
    assert (ROOT / "batch_in_context.py").is_file()
    assert not legacy_run.exists()
    assert not legacy_batch.exists()

    assert "run_in_context" in imported_modules(ROOT / "batch_in_context.py")
    for relative in (
        "generation_attention/batch_visualize_flux_attention.py",
        "generation_attention/visualize_flux_attention.py",
    ):
        assert "run_in_context" in imported_modules(ROOT / relative)


def test_evaluation_entrypoints_use_in_context_names() -> None:
    new_paths = (
        ROOT / "eval_diversity/run_in_context_eval.sh",
        ROOT / "eval_downstream/run_in_context_classification.sh",
        ROOT / "eval_downstream/run_in_context_localization.sh",
    )
    old_paths = (
        ROOT / "eval_diversity" / ("run_" + "insertanything_eval.sh"),
        ROOT / "eval_downstream" / ("run_" + "insertanything_classification.sh"),
        ROOT / "eval_downstream" / ("run_" + "insertanything_localization.sh"),
    )
    assert all(path.is_file() for path in new_paths)
    assert not any(path.exists() for path in old_paths)

    completed = subprocess.run(
        [sys.executable, "eval_downstream/prepare_reflex_classification_data.py", "--help"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0
    assert "--in-context-results-root" in completed.stdout
    assert "in_context" in completed.stdout
