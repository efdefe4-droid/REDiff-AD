from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNTIME_DIRS = [ROOT, ROOT / "generation_attention", ROOT / "scripts", ROOT / "eval_diversity", ROOT / "eval_downstream"]
FORBIDDEN_PATH_MARKERS = ("/home/", "/media/")


def python_sources() -> list[Path]:
    return sorted(path for path in ROOT.rglob("*.py") if not any(part.startswith(".") for part in path.relative_to(ROOT).parts))


def imported_roots(tree: ast.AST) -> set[str]:
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".")[0])
    return roots


def test_python_sources_parse_and_keep_eval_boundary() -> None:
    for path in python_sources():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        imports = imported_roots(tree)
        relative = path.relative_to(ROOT)
        if relative.parts[0] in {"eval_diversity", "eval_downstream"}:
            assert "generation_attention" not in imports
            assert "run_in_context" not in imports
        if relative.parts[0] == "generation_attention" or relative.name in {"run_in_context.py", "batch_in_context.py"}:
            assert "eval_diversity" not in imports
            assert "eval_downstream" not in imports


def test_runtime_sources_do_not_contain_personal_absolute_paths() -> None:
    candidates: set[Path] = set()
    for directory in RUNTIME_DIRS:
        if directory == ROOT:
            candidates.update(path for path in directory.glob("*.py"))
        else:
            candidates.update(directory.rglob("*.py"))
            candidates.update(directory.rglob("*.sh"))
    for path in sorted(candidates):
        text = path.read_text(encoding="utf-8")
        assert not any(marker in text for marker in FORBIDDEN_PATH_MARKERS), path
