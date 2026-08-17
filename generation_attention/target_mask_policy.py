"""Deterministic, lightweight target-mask policies used by generation.

This module intentionally depends only on NumPy and Pillow so the routing and
mask transform can be tested in CPU-only CI without importing the FLUX stack.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter


REFERENCE_VERTICAL_MIXED_VARIANTS = (
    "reference_up",
    "reference_down",
    "random_object",
)


def reference_vertical_mixed_variant(sample_index: int) -> str:
    """Return the exact 2:1 reference-transform/original-mask schedule.

    Indices 0/1 use the conditioning reference mask shifted up/down; index 2
    uses the original random-object sampler.  Repeating this three-sample
    cycle makes interrupted/resumed runs deterministic.
    """
    if sample_index < 0:
        raise ValueError("sample_index must be non-negative")
    return REFERENCE_VERTICAL_MIXED_VARIANTS[sample_index % 3]


def reference_vertical_mask(
    ref_mask: np.ndarray,
    variant: str,
    *,
    dilate_iterations: int,
    vertical_shift_ratio: float,
) -> tuple[np.ndarray, dict[str, object]]:
    """Enlarge a binary reference mask and translate it vertically.

    Translation is clipped at the canvas boundary; it never wraps around.
    ``reference_up`` uses a negative y offset and ``reference_down`` a
    positive one.  The returned metadata records both the requested transform
    and any area lost at the boundary.
    """
    if variant not in {"reference_up", "reference_down"}:
        raise ValueError(f"unsupported reference-mask variant: {variant!r}")
    if dilate_iterations < 0:
        raise ValueError("dilate_iterations must be non-negative")
    if not 0.0 <= vertical_shift_ratio < 1.0:
        raise ValueError("vertical_shift_ratio must be in [0, 1)")

    binary = (np.asarray(ref_mask) > 0).astype(np.uint8)
    if binary.ndim != 2:
        raise ValueError(f"ref_mask must be a 2-D array, got shape={binary.shape}")
    height, width = binary.shape
    if height == 0 or width == 0 or int(binary.sum()) == 0:
        raise ValueError("ref_mask must be non-empty")

    dilated = binary
    for _ in range(int(dilate_iterations)):
        dilated = (
            np.asarray(
                Image.fromarray(dilated * 255, mode="L").filter(ImageFilter.MaxFilter(7)),
                dtype=np.uint8,
            )
            > 0
        ).astype(np.uint8)

    shift_magnitude = int(round(height * float(vertical_shift_ratio)))
    shift_pixels = -shift_magnitude if variant == "reference_up" else shift_magnitude
    transformed = np.zeros_like(dilated)
    if shift_pixels < 0:
        offset = min(height, -shift_pixels)
        if offset < height:
            transformed[: height - offset] = dilated[offset:]
    elif shift_pixels > 0:
        offset = min(height, shift_pixels)
        if offset < height:
            transformed[offset:] = dilated[: height - offset]
    else:
        transformed[:] = dilated

    transformed_area = int(transformed.sum())
    if transformed_area == 0:
        raise ValueError(
            "reference-mask transform produced an empty mask; reduce vertical_shift_ratio"
        )
    original_area = int(binary.sum())
    dilated_area = int(dilated.sum())
    metadata: dict[str, object] = {
        "variant": variant,
        "dilate_kernel_size": 7,
        "dilate_iterations": int(dilate_iterations),
        "vertical_shift_ratio": float(vertical_shift_ratio),
        "vertical_shift_pixels": int(shift_pixels),
        "original_area": original_area,
        "dilated_area": dilated_area,
        "transformed_area": transformed_area,
        "visible_fraction_after_shift": float(transformed_area / max(1, dilated_area)),
    }
    return transformed, metadata
