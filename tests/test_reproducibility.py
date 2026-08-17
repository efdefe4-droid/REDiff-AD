from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_recorded_commits_and_hashes() -> None:
    config = json.loads((ROOT / "configs" / "reproducibility.json").read_text(encoding="utf-8"))
    assert config["review_anonymization"] == {
        "double_blind": True,
        "identifying_provenance": "withheld_until_review_completion",
    }
    for source in ("generation_base", "t2r_patch_source", "eval_source"):
        assert config[source]["path"] == "withheld_for_double_blind_review"
        assert config[source]["git_commit"] == "withheld_for_double_blind_review"
    for relative_path, expected in config["hashes"].items():
        assert sha256(ROOT / relative_path) == expected, relative_path
