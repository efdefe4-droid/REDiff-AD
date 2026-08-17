from __future__ import annotations

import csv
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs"
BLOCK_RE = re.compile(r"^(transformer_blocks|single_transformer_blocks)_(\d+)_attn$")
REQUIRED_COLUMNS = {
    "layer_index",
    "block",
    "count",
    "frequency",
    "high_count",
    "low_count",
    "mean_hist_iou",
    "mean_ap",
    "mean_auc",
    "samples",
}


def read_rows(name: str) -> list[dict[str, str]]:
    with (CONFIG_DIR / name).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames is not None
        assert REQUIRED_COLUMNS.issubset(reader.fieldnames)
        return list(reader)


def test_block_frequency_invariants() -> None:
    rows = read_rows("block_frequency_t2r.csv")
    assert rows
    layers: set[int] = set()
    blocks: set[str] = set()
    for row in rows:
        layer = int(row["layer_index"])
        block = row["block"]
        match = BLOCK_RE.fullmatch(block)
        assert match, block
        family, block_index_text = match.groups()
        block_index = int(block_index_text)
        expected_layer = block_index if family == "transformer_blocks" else 19 + block_index
        assert layer == expected_layer
        assert layer not in layers
        assert block not in blocks
        layers.add(layer)
        blocks.add(block)

        count = int(row["count"])
        assert count == int(row["high_count"]) + int(row["low_count"])
        assert len(row["samples"].split()) == count
        assert math.isclose(float(row["frequency"]), count / 30.0, rel_tol=0.0, abs_tol=1e-12)
        for metric in ("mean_hist_iou", "mean_ap", "mean_auc"):
            assert 0.0 <= float(row[metric]) <= 1.0


def test_frozen_t2r_top10_matches_csv_order() -> None:
    rows = read_rows("block_frequency_t2r.csv")
    frozen = [
        line.strip()
        for line in (CONFIG_DIR / "top10_t2r_blocks.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(frozen) == 10
    assert frozen == [row["block"] for row in rows[:10]]
