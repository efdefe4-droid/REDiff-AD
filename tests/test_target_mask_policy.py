from __future__ import annotations

import ast
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from generation_attention.target_mask_policy import (
    reference_vertical_mask,
    reference_vertical_mixed_variant,
)


ROOT = Path(__file__).resolve().parents[1]


def mask_centroid_y(mask: np.ndarray) -> float:
    ys, _xs = np.nonzero(mask)
    assert ys.size > 0
    return float(ys.mean())


def test_reference_vertical_mixed_schedule_is_exactly_120_to_60() -> None:
    variants = [reference_vertical_mixed_variant(global_ordinal) for global_ordinal in range(180)]

    assert variants[:6] == [
        "reference_up",
        "reference_down",
        "random_object",
        "reference_up",
        "reference_down",
        "random_object",
    ]
    assert Counter(variants) == {
        "reference_up": 60,
        "reference_down": 60,
        "random_object": 60,
    }
    assert sum(variant.startswith("reference_") for variant in variants) == 120


def test_reference_vertical_mixed_schedule_uses_defect_global_ordinal() -> None:
    # Deliberately use pair sizes that are not divisible by three.  The policy
    # must continue the defect-level sequence instead of restarting at ref
    # boundaries.
    samples_per_reference = [17, 23, 19, 41, 80]
    assert sum(samples_per_reference) == 180

    variants: list[str] = []
    first_variant_by_reference: list[str] = []
    defect_global_ordinal = 0
    for pair_count in samples_per_reference:
        first_variant_by_reference.append(
            reference_vertical_mixed_variant(defect_global_ordinal)
        )
        variants.extend(
            reference_vertical_mixed_variant(defect_global_ordinal + local_index)
            for local_index in range(pair_count)
        )
        defect_global_ordinal += pair_count

    assert first_variant_by_reference == [
        "reference_up",
        "random_object",
        "reference_down",
        "random_object",
        "reference_down",
    ]
    assert Counter(variants) == {
        "reference_up": 60,
        "reference_down": 60,
        "random_object": 60,
    }


def test_batch_passes_a_defect_global_ordinal_to_the_mixed_policy() -> None:
    source_path = ROOT / "generation_attention" / "batch_visualize_flux_attention.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "reference_vertical_mixed_variant"
    ]

    assert calls, "batch generator must invoke reference_vertical_mixed_variant"
    for call in calls:
        assert call.args
        input_names = {
            node.id
            for node in ast.walk(call.args[0])
            if isinstance(node, ast.Name)
        }
        assert any(
            "global" in name or "defect" in name
            for name in input_names
        ), (
            "mixed policy must receive a defect-global ordinal; passing the "
            "per-reference sample_index restarts the 0/1/2 schedule"
        )

    recorded_metadata_keys = {
        key.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Dict)
        for key in node.keys
        if isinstance(key, ast.Constant) and isinstance(key.value, str)
    }
    assert "defect_global_ordinal" in recorded_metadata_keys
    source_text = source_path.read_text(encoding="utf-8")
    assert "sample_index\n                            - args.start_index" in source_text


def test_reference_vertical_mixed_rejects_negative_global_ordinal() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        reference_vertical_mixed_variant(-1)


@pytest.mark.parametrize(
    ("variant", "expected_direction"),
    [("reference_up", -1), ("reference_down", 1)],
)
def test_reference_mask_transform_metadata_direction_and_nonempty(
    variant: str,
    expected_direction: int,
) -> None:
    reference = np.zeros((120, 96), dtype=np.uint8)
    reference[48:64, 36:58] = 1

    transformed, metadata = reference_vertical_mask(
        reference,
        variant,
        dilate_iterations=5,
        vertical_shift_ratio=0.05,
    )

    expected_shift = expected_direction * round(reference.shape[0] * 0.05)
    assert transformed.shape == reference.shape
    assert transformed.dtype == np.uint8
    assert set(np.unique(transformed)).issubset({0, 1})
    assert int(transformed.sum()) > 0
    assert expected_direction * (mask_centroid_y(transformed) - mask_centroid_y(reference)) > 0

    assert metadata["variant"] == variant
    assert metadata["dilate_kernel_size"] == 7
    assert metadata["dilate_iterations"] == 5
    assert metadata["vertical_shift_ratio"] == 0.05
    assert metadata["vertical_shift_pixels"] == expected_shift
    assert metadata["original_area"] == int(reference.sum())
    assert int(metadata["dilated_area"]) > int(metadata["original_area"])
    assert int(metadata["transformed_area"]) == int(transformed.sum())
    assert 0.0 < float(metadata["visible_fraction_after_shift"]) <= 1.0


@pytest.mark.parametrize("variant", ["reference_up", "reference_down"])
def test_reference_mask_transform_is_reproducible(variant: str) -> None:
    reference = np.zeros((83, 79), dtype=np.uint8)
    reference[31:47, 28:45] = 1

    first_mask, first_metadata = reference_vertical_mask(
        reference,
        variant,
        dilate_iterations=5,
        vertical_shift_ratio=0.05,
    )
    second_mask, second_metadata = reference_vertical_mask(
        reference.copy(),
        variant,
        dilate_iterations=5,
        vertical_shift_ratio=0.05,
    )

    np.testing.assert_array_equal(first_mask, second_mask)
    assert first_metadata == second_metadata
    assert int(first_mask.sum()) > 0


def test_reference_mask_transform_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        reference_vertical_mask(
            np.zeros((64, 64), dtype=np.uint8),
            "reference_up",
            dilate_iterations=5,
            vertical_shift_ratio=0.05,
        )
