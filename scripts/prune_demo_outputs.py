#!/usr/bin/env python3
"""Keep only the three submission-facing images in each demo sample."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


KEEP_SAMPLE_FILES = frozenset(
    {
        "edit.png",
        "coarse_mask.png",
        "contour_refined_mask.png",
    }
)
RUNTIME_ROOT_FILES = frozenset(
    {
        "adaptive_log.csv",
        "run_config.json",
        "run_config_history.jsonl",
        "run_console.log",
        "run_log.csv",
        "validation_report.json",
    }
)


def sample_directories(run_root: Path) -> list[Path]:
    return sorted(
        path
        for path in run_root.glob("*/ref_*/*")
        if path.is_dir() and path.name.isdigit()
    )


def validate_samples(sample_dirs: list[Path]) -> list[str]:
    errors: list[str] = []
    if not sample_dirs:
        return ["no sample directories found"]
    for sample_dir in sample_dirs:
        for name in sorted(KEEP_SAMPLE_FILES):
            path = sample_dir / name
            if not path.is_file() or path.stat().st_size == 0:
                errors.append(f"{sample_dir}: missing or empty {name}")
    return errors


def remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def prune_run(run_root: Path) -> int:
    run_root = run_root.expanduser().resolve()
    if not run_root.is_dir():
        print(f"ERROR: run root is not a directory: {run_root}", file=sys.stderr)
        return 1

    sample_dirs = sample_directories(run_root)
    errors = validate_samples(sample_dirs)
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        print("No files were deleted.", file=sys.stderr)
        return 1

    for sample_dir in sample_dirs:
        for path in sample_dir.iterdir():
            if path.name not in KEEP_SAMPLE_FILES:
                remove_path(path)
    for name in RUNTIME_ROOT_FILES:
        path = run_root / name
        if path.exists() or path.is_symlink():
            remove_path(path)

    print(f"Pruned {len(sample_dirs)} sample directories under {run_root}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    args = parser.parse_args()
    return prune_run(args.run_root)


if __name__ == "__main__":
    raise SystemExit(main())
