#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from diffusers.models.attention_processor import Attention
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from run_insert_anything import (
    DEFAULT_LORA_WEIGHT,
    crop_back,
    dtype_from_name,
    load_pipelines,
    prepare_diptych,
)
from generation_attention.mask_refinement import (
    refine_active_contour_mask,
    refine_boundary_preserve_mask,
    refine_direct_attention_mask,
    refine_q80_appearance_mask,
)


DEFAULT_SOURCE_IMAGE = "datasets/hazelnut/train/good/000.png"
DEFAULT_SOURCE_MASK = "datasets/hazelnut/ground_truth/crack/000_mask.png"
DEFAULT_REF_IMAGE = "datasets/hazelnut/test/crack/000.png"
DEFAULT_REF_MASK = "datasets/hazelnut/ground_truth/crack/000_mask.png"
DEFAULT_TOP15_BLOCKS_FILE = "configs/top10_t2r_blocks.txt"
DEFAULT_BLOCK_FREQUENCY_CSV = "configs/block_frequency_t2r.csv"


def safe_name(name: str) -> str:
    name = name.replace(".", "_")
    name = re.sub(r"[^A-Za-z0-9_+-]+", "_", name)
    return name.strip("_")


def normalize_to_u8(values: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(values, torch.Tensor):
        arr = values.detach().float().cpu().numpy()
    else:
        arr = values.astype(np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    lo = float(arr.min())
    hi = float(arr.max())
    if hi - lo < 1e-12:
        return np.zeros_like(arr, dtype=np.uint8)
    return np.clip((arr - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)


def normalize01_np(values: np.ndarray) -> np.ndarray:
    arr = np.nan_to_num(values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    lo = float(arr.min())
    hi = float(arr.max())
    if hi - lo < 1e-12:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - lo) / (hi - lo)).astype(np.float32)


def colorize_u8(gray: np.ndarray) -> np.ndarray:
    color_bgr = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)
    return cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)


def save_attention_image(
    values: torch.Tensor,
    path: Path,
    out_size: tuple[int, int],
    target_base_rgb: np.ndarray | None = None,
) -> dict[str, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = normalize_to_u8(values)
    image = cv2.resize(image, out_size, interpolation=cv2.INTER_CUBIC)
    color = colorize_u8(image)
    Image.fromarray(color).save(path)

    saved = {"heatmap": str(path)}
    if target_base_rgb is not None:
        overlay_dir = path.parent.parent / f"{path.parent.name}_overlay"
        overlay_dir.mkdir(parents=True, exist_ok=True)
        base = cv2.resize(target_base_rgb, out_size, interpolation=cv2.INTER_CUBIC)
        overlay = np.clip(0.55 * base.astype(np.float32) + 0.45 * color.astype(np.float32), 0, 255).astype(np.uint8)
        overlay_path = overlay_dir / path.name
        Image.fromarray(overlay).save(overlay_path)
        saved["overlay"] = str(overlay_path)
    return saved


def save_raw_map(values: torch.Tensor, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = values.detach().float().cpu().numpy().astype(np.float32)
    np.save(path, arr)
    return str(path)


def pca_rgb_from_tokens(tokens: torch.Tensor, grid_h: int, grid_w: int) -> np.ndarray:
    features = tokens.detach().float().cpu()
    features = features - features.mean(dim=0, keepdim=True)
    try:
        _, _, vh = torch.linalg.svd(features, full_matrices=False)
        proj = features @ vh[:3].T
    except RuntimeError:
        proj = torch.zeros((features.shape[0], 3), dtype=torch.float32)
    lo = proj.min(dim=0, keepdim=True).values
    hi = proj.max(dim=0, keepdim=True).values
    proj = (proj - lo) / (hi - lo + 1e-8)
    return (proj.reshape(grid_h, grid_w, 3).numpy() * 255).astype(np.uint8)


def save_pca_image(tokens: torch.Tensor, path: Path, grid_h: int, grid_w: int, out_size: tuple[int, int]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    rgb = pca_rgb_from_tokens(tokens, grid_h, grid_w)
    rgb = cv2.resize(rgb, out_size, interpolation=cv2.INTER_NEAREST)
    Image.fromarray(rgb).save(path)
    return str(path)


def read_block_list(path: str | Path, limit: int | None = None) -> list[str]:
    blocks = [
        line.strip()
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    return blocks[:limit] if limit is not None else blocks


def read_dominant_polarities(path: str | Path) -> dict[str, str]:
    csv_path = Path(path)
    if not csv_path.is_file():
        return {}
    import csv

    polarities: dict[str, str] = {}
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            block = str(row.get("block", "")).strip()
            if not block:
                continue
            high = int(float(row.get("high_count") or 0))
            low = int(float(row.get("low_count") or 0))
            polarities[block] = "high" if high > low else "low"
    return polarities


def load_binary_mask(path: str | Path, shape_hw: tuple[int, int]) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Could not read mask: {path}")
    if mask.shape[:2] != shape_hw:
        mask = cv2.resize(mask, (shape_hw[1], shape_hw[0]), interpolation=cv2.INTER_NEAREST)
    return mask > 128


def object_roi_from_image(image_path: str | Path, shape_hw: tuple[int, int]) -> np.ndarray:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    image = cv2.resize(image, (shape_hw[1], shape_hw[0]), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, otsu_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = otsu_mask > 0
    if float(mask.mean()) > 0.85:
        mask = gray > np.percentile(gray, 55)
    elif float(mask.mean()) < 0.01:
        mask = gray > np.percentile(gray, 75)
    mask = largest_component(mask)
    kernel = np.ones((31, 31), dtype=np.uint8)
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel, iterations=2).astype(bool)
    return fill_holes(mask)


def map_attention_to_original(
    attn_map: np.ndarray,
    source_shape_hw: tuple[int, int],
    extra_sizes: np.ndarray,
    crop_box: np.ndarray,
) -> np.ndarray:
    source_h, source_w = source_shape_hw
    h1, w1, h2, w2 = [int(v) for v in extra_sizes]
    y1, y2, x1, x2 = [int(v) for v in crop_box]

    padded = cv2.resize(attn_map.astype(np.float32), (w2, h2), interpolation=cv2.INTER_CUBIC)
    if w1 == h1:
        unpadded = padded
    elif w1 < w2:
        pad1 = int((w2 - w1) / 2)
        unpadded = padded[:, pad1 : pad1 + w1]
    else:
        pad1 = int((h2 - h1) / 2)
        unpadded = padded[pad1 : pad1 + h1, :]

    crop_h = y2 - y1
    crop_w = x2 - x1
    if unpadded.shape[:2] != (crop_h, crop_w):
        unpadded = cv2.resize(unpadded, (crop_w, crop_h), interpolation=cv2.INTER_CUBIC)

    original = np.zeros((int(source_h), int(source_w)), dtype=np.float32)
    original[y1:y2, x1:x2] = unpadded
    return normalize01_np(original)


def fill_holes(mask: np.ndarray) -> np.ndarray:
    mask_u8 = mask.astype(np.uint8)
    padded = np.pad(mask_u8, 1, mode="constant", constant_values=0)
    flood = padded.copy()
    h, w = flood.shape
    cv2.floodFill(flood, np.zeros((h + 2, w + 2), np.uint8), (0, 0), 1)
    holes = (flood == 0).astype(np.uint8)
    return np.maximum(padded, holes)[1:-1, 1:-1].astype(bool)


def largest_component(mask: np.ndarray) -> np.ndarray:
    mask_u8 = mask.astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, 8)
    if n <= 1:
        return mask.astype(bool)
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == largest


def threshold_soft_mask(
    soft: np.ndarray,
    roi: np.ndarray,
    scale: float,
    offset: float,
    component_mode: str,
    fill: bool,
    close_iterations: int,
    dilate_iterations: int,
) -> tuple[np.ndarray, float, float]:
    score = normalize01_np(soft)
    valid = score[roi.astype(bool)]
    if valid.size == 0:
        pred = np.zeros_like(score, dtype=bool)
        return pred, 0.0, 0.0
    if float(valid.max() - valid.min()) < 1e-8:
        otsu_threshold = float(valid.mean())
    else:
        values_u8 = (valid * 255.0).clip(0, 255).astype(np.uint8).reshape(-1, 1)
        threshold_u8, _ = cv2.threshold(values_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        otsu_threshold = float(threshold_u8 / 255.0)
    threshold = float(np.clip(otsu_threshold * scale + offset, 0.0, 1.0))
    pred = np.logical_and(score >= threshold, roi)
    if component_mode == "largest":
        pred = largest_component(pred)
    elif component_mode == "max_energy" and pred.any():
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(pred.astype(np.uint8), connectivity=8)
        if num_labels > 2:
            candidates: list[tuple[float, int, int]] = []
            for label in range(1, num_labels):
                component = labels == label
                energy = float(np.maximum(score[component] - threshold, 0.0).sum())
                area = int(stats[label, cv2.CC_STAT_AREA])
                candidates.append((energy, area, label))
            selected_label = max(candidates)[2]
            pred = labels == selected_label
    elif component_mode != "all":
        raise ValueError(f"Unsupported component_mode: {component_mode}")
    if close_iterations > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        pred = cv2.morphologyEx(pred.astype(np.uint8), cv2.MORPH_CLOSE, kernel, iterations=close_iterations).astype(bool)
    if fill:
        pred = fill_holes(pred)
    if dilate_iterations > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        pred = cv2.dilate(pred.astype(np.uint8), kernel, iterations=dilate_iterations).astype(bool)
    return pred, threshold, otsu_threshold


def save_mask_overlay(image: np.ndarray, mask: np.ndarray, out_path: Path, color=(255, 0, 0), alpha: float = 0.45) -> None:
    overlay = image.copy().astype(np.float32)
    color_arr = np.array(color, dtype=np.float32)
    overlay[mask] = (1.0 - alpha) * overlay[mask] + alpha * color_arr
    contours, _ = cv2.findContours((mask.astype(np.uint8) * 255), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    overlay_u8 = np.clip(overlay, 0, 255).astype(np.uint8)
    cv2.drawContours(overlay_u8, contours, -1, (255, 255, 255), 2)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(overlay_u8).save(out_path)


def save_soft_overlay(image: np.ndarray, soft: np.ndarray, out_path: Path, alpha: float = 0.45) -> None:
    score_u8 = (normalize01_np(soft) * 255.0).clip(0, 255).astype(np.uint8)
    heat = colorize_u8(score_u8)
    if heat.shape[:2] != image.shape[:2]:
        heat = cv2.resize(heat, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_CUBIC)
    overlay = np.clip((1.0 - alpha) * image.astype(np.float32) + alpha * heat.astype(np.float32), 0, 255).astype(np.uint8)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(overlay).save(out_path)


def coarse_mask_for_refine(coarse_mask: np.ndarray) -> tuple[np.ndarray, dict]:
    enabled = os.environ.get("REFINE_COARSE_OPEN", "1").lower() not in {"0", "false", "no"}
    kernel_size = max(1, int(os.environ.get("REFINE_COARSE_OPEN_KERNEL", "3")))
    if kernel_size % 2 == 0:
        kernel_size += 1
    iterations = max(0, int(os.environ.get("REFINE_COARSE_OPEN_ITER", "1")))
    original = coarse_mask.astype(bool)
    if enabled and iterations > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        cleaned = cv2.erode(original.astype(np.uint8), kernel, iterations=iterations)
        cleaned = cv2.dilate(cleaned, kernel, iterations=iterations).astype(bool)
    else:
        cleaned = original

    return cleaned, {
        "enabled": bool(enabled and iterations > 0),
        "operation": "erode_then_dilate",
        "kernel": "ellipse",
        "kernel_size": int(kernel_size),
        "iterations": int(iterations),
        "input_area": int(original.sum()),
        "output_area": int(cleaned.sum()),
        "removed_area": int(original.sum() - cleaned.sum()),
    }


def infer_diptych_grid(image_token_count: int, height: int, width: int) -> tuple[int, int]:
    aspect = width / float(height)
    grid_h = int(round(math.sqrt(image_token_count / aspect)))
    grid_w = int(round(grid_h * aspect))
    if grid_h * grid_w != image_token_count:
        # Fallback for packed latent layouts. Keep aspect when possible.
        factors = []
        for h in range(1, int(math.sqrt(image_token_count)) + 1):
            if image_token_count % h == 0:
                w = image_token_count // h
                factors.append((abs((w / h) - aspect), h, w))
        if not factors:
            raise ValueError(f"Cannot infer grid for {image_token_count} image tokens")
        _, grid_h, grid_w = min(factors)
    if grid_w % 2 != 0:
        raise ValueError(f"Expected even diptych grid width, got {grid_h}x{grid_w}")
    return grid_h, grid_w


def reference_shape_map_from_image(
    masked_ref_image: np.ndarray,
    method: str = "sobel",
    foreground_threshold: int = 250,
) -> np.ndarray:
    image = np.asarray(masked_ref_image)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected RGB reference image, got shape={image.shape}")
    image_u8 = image.astype(np.uint8)
    gray = cv2.cvtColor(image_u8, cv2.COLOR_RGB2GRAY)
    foreground = np.any(image_u8 < int(foreground_threshold), axis=2).astype(np.uint8)
    if foreground.any():
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        foreground = cv2.morphologyEx(foreground, cv2.MORPH_OPEN, kernel, iterations=1)
        foreground = cv2.dilate(foreground, kernel, iterations=1)

    method = method.lower()
    if method == "foreground":
        shape = foreground.astype(np.float32)
    elif method == "canny":
        values = gray[foreground.astype(bool)] if foreground.any() else gray.reshape(-1)
        median = float(np.median(values)) if values.size else 127.0
        low = int(max(0, (1.0 - 0.33) * median))
        high = int(min(255, (1.0 + 0.33) * median))
        if high <= low:
            low, high = 50, 150
        shape = cv2.Canny(gray, low, high).astype(np.float32) / 255.0
    elif method == "highfreq":
        blur = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), 2.0)
        shape = np.abs(gray.astype(np.float32) - blur)
    elif method == "sobel":
        gray_f = gray.astype(np.float32) / 255.0
        grad_x = cv2.Sobel(gray_f, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray_f, cv2.CV_32F, 0, 1, ksize=3)
        shape = cv2.magnitude(grad_x, grad_y)
    else:
        raise ValueError(f"Unsupported reference shape method: {method}")

    if foreground.any():
        shape = shape * foreground.astype(np.float32)
    shape = normalize01_np(shape.astype(np.float32))
    if float(shape.max()) <= 1e-8 and foreground.any():
        shape = foreground.astype(np.float32)
    return normalize01_np(shape)


SHAPE_K_BLOCK_SCOPE_CHOICES = ("top", "all", "middle", "dual_middle", "single_middle")
SHAPE_K_DUAL_MIDDLE_BLOCKS = tuple(f"transformer_blocks_{idx}_attn" for idx in range(6, 19))
SHAPE_K_SINGLE_MIDDLE_BLOCKS = tuple(f"single_transformer_blocks_{idx}_attn" for idx in range(8, 31))


def resolve_shape_k_blocks(
    explicit_blocks: list[str] | tuple[str, ...] | set[str] | None,
    save_blocks: list[str] | tuple[str, ...] | set[str] | None,
    block_scope: str = "top",
) -> set[str] | None:
    if explicit_blocks:
        return set(explicit_blocks)
    scope = (block_scope or "top").lower()
    if scope == "all":
        return None
    if scope == "middle":
        return set(SHAPE_K_DUAL_MIDDLE_BLOCKS) | set(SHAPE_K_SINGLE_MIDDLE_BLOCKS)
    if scope == "dual_middle":
        return set(SHAPE_K_DUAL_MIDDLE_BLOCKS)
    if scope == "single_middle":
        return set(SHAPE_K_SINGLE_MIDDLE_BLOCKS)
    if scope != "top":
        raise ValueError(f"Unsupported Shape-K block scope: {block_scope}")
    return set(save_blocks) if save_blocks else None


class FluxAttentionRecorder:
    def __init__(
        self,
        out_dir: Path,
        height: int,
        width: int,
        save_kinds: set[str],
        pca_kinds: set[str],
        target_base_rgb: np.ndarray | None = None,
        max_blocks: int | None = None,
        max_steps: int | None = None,
        save_steps: set[int] | None = None,
        save_blocks: set[str] | None = None,
        save_visuals: bool = True,
        save_raw_maps: bool = False,
        direct_aggregate_kind: str | None = None,
        direct_aggregate_steps: set[int] | None = None,
        adaptive_check_steps: set[int] | None = None,
        adaptive_aggregate_kind: str | None = None,
        adaptive_blocks: set[str] | None = None,
        adaptive_block_polarities: dict[str, str] | None = None,
        record_all_direct_blocks: bool = False,
        direct_block_polarities: dict[str, str] | None = None,
        adaptive_ref_condition: bool = False,
        adaptive_ref_token_start: int = 512,
        num_inference_steps: int | None = None,
        shape_k_enabled: bool = False,
        shape_k_eta: float = 0.5,
        shape_k_start_step: int = -1,
        shape_k_end_step: int = -1,
        shape_k_start_ratio: float = 0.2,
        shape_k_end_ratio: float = 0.7,
        shape_k_blocks: set[str] | None = None,
        shape_k_mode: str = "orthogonal",
        shape_k_suppress_scale: float = 1.0,
        reference_shape_map: np.ndarray | torch.Tensor | None = None,
    ) -> None:
        self.out_dir = out_dir
        self.height = height
        self.width = width
        self.target_size = (width // 2, height)
        self.save_kinds = save_kinds
        self.pca_kinds = pca_kinds
        self.target_base_rgb = target_base_rgb
        self.max_blocks = max_blocks
        self.max_steps = max_steps
        self.save_steps = save_steps
        self.save_blocks = save_blocks
        self.save_visuals = save_visuals
        self.save_raw_maps = save_raw_maps
        self.direct_aggregate_kind = direct_aggregate_kind
        self.direct_aggregate_steps = None if direct_aggregate_steps is None else set(direct_aggregate_steps)
        self.adaptive_check_steps = None if adaptive_check_steps is None else set(adaptive_check_steps)
        self.adaptive_aggregate_kind = adaptive_aggregate_kind
        self.adaptive_blocks = None if adaptive_blocks is None else set(adaptive_blocks)
        self.adaptive_block_polarities = adaptive_block_polarities or {}
        self.record_all_direct_blocks = bool(record_all_direct_blocks)
        self.direct_block_polarities = direct_block_polarities or {}
        self.adaptive_ref_condition = adaptive_ref_condition
        self.adaptive_ref_token_start = max(0, int(adaptive_ref_token_start))
        self.adaptive_ref_stats: dict[int, list[dict[str, object]]] = {}
        self.num_inference_steps = None if num_inference_steps is None else max(1, int(num_inference_steps))
        self.shape_k_enabled = bool(shape_k_enabled)
        self.shape_k_eta = max(0.0, float(shape_k_eta))
        self.shape_k_start_step = int(shape_k_start_step)
        self.shape_k_end_step = int(shape_k_end_step)
        self.shape_k_start_ratio = min(1.0, max(0.0, float(shape_k_start_ratio)))
        self.shape_k_end_ratio = min(1.0, max(0.0, float(shape_k_end_ratio)))
        self.shape_k_blocks = None if shape_k_blocks is None else set(shape_k_blocks)
        self.shape_k_mode = str(shape_k_mode or "orthogonal").lower()
        if self.shape_k_mode not in {"orthogonal", "suppress", "both"}:
            raise ValueError(f"Unsupported Shape-K mode: {shape_k_mode}")
        self.shape_k_suppress_scale = max(0.0, float(shape_k_suppress_scale))
        self.reference_shape_map = None
        if reference_shape_map is not None:
            shape_tensor = torch.as_tensor(reference_shape_map, dtype=torch.float32)
            if shape_tensor.ndim == 3:
                shape_tensor = shape_tensor.mean(dim=-1)
            if shape_tensor.ndim != 2:
                raise ValueError(f"reference_shape_map must be 2D, got shape={tuple(shape_tensor.shape)}")
            self.reference_shape_map = shape_tensor.detach().cpu()
        self.shape_k_vector_cache: dict[tuple[int, int, str, int | None], torch.Tensor | None] = {}
        self.shape_k_weight_cache: dict[tuple[int, int, str, int | None], torch.Tensor | None] = {}
        self.shape_k_applied_calls = 0
        self.shape_k_skipped_no_shape = 0
        self.shape_k_skipped_low_norm = 0
        self.shape_k_step_counts: dict[int, int] = {}
        self.shape_k_block_counts: dict[str, int] = {}
        self.shape_k_delta_ratio_sum = 0.0
        self.shape_k_delta_ratio_max = 0.0
        self.direct_sums: dict[str, torch.Tensor] = {}
        self.direct_counts: dict[str, int] = {}
        self.direct_step_sums: dict[int, dict[str, torch.Tensor]] = {}
        self.direct_step_counts: dict[int, dict[str, int]] = {}
        self.step = 0
        self.records: list[dict[str, object]] = []
        self.last_encoder_token_count = 0

    def after_step(self, step_index: int) -> None:
        self.step = step_index + 1

    def should_record(self, block_name: str) -> bool:
        if self.max_steps is not None and self.step >= self.max_steps:
            return False
        if self.save_steps is not None and self.step not in self.save_steps:
            return False
        block_label = safe_name(block_name)
        if self.save_blocks is None or block_label in self.save_blocks:
            return True
        if (
            self.adaptive_aggregate_kind is not None
            and (self.adaptive_blocks is None or block_label in self.adaptive_blocks)
            and (self.adaptive_check_steps is None or self.step in self.adaptive_check_steps)
        ):
            return True
        # The batch runner requests both selected-block and true all-block masks.
        # Non-selected blocks only need the final aggregate steps; keeping them
        # off adaptive-only steps avoids unnecessary full-model QK recomputation.
        return bool(
            self.record_all_direct_blocks
            and self.direct_aggregate_kind is not None
            and (self.direct_aggregate_steps is None or self.step in self.direct_aggregate_steps)
        )

    def observe_encoder_token_count(self, encoder_token_count: int) -> None:
        if encoder_token_count > 0:
            self.last_encoder_token_count = int(encoder_token_count)

    def _shape_k_start(self) -> int:
        if self.shape_k_start_step >= 0:
            return self.shape_k_start_step
        if self.num_inference_steps is None:
            return 0
        return int(round((self.num_inference_steps - 1) * self.shape_k_start_ratio))

    def _shape_k_end(self) -> int | None:
        if self.shape_k_end_step >= 0:
            return self.shape_k_end_step
        if self.num_inference_steps is None:
            return None
        return int(round((self.num_inference_steps - 1) * self.shape_k_end_ratio))

    def shape_k_active(self) -> bool:
        if not self.shape_k_enabled or self.reference_shape_map is None:
            return False
        has_orthogonal = self.shape_k_mode in {"orthogonal", "both"} and self.shape_k_eta > 0.0
        has_suppress = self.shape_k_mode in {"suppress", "both"} and self.shape_k_suppress_scale > 0.0
        if not (has_orthogonal or has_suppress):
            return False
        if self.step < self._shape_k_start():
            return False
        end_step = self._shape_k_end()
        if end_step is not None and self.step > end_step:
            return False
        return True

    def should_apply_shape_k(self, block_name: str) -> bool:
        if not self.shape_k_active():
            return False
        block_label = safe_name(block_name)
        if self.shape_k_blocks is not None and block_label not in self.shape_k_blocks:
            return False
        return True

    def _reference_shape_vector(
        self,
        grid_h: int,
        half_w: int,
        device: torch.device,
    ) -> torch.Tensor | None:
        if self.reference_shape_map is None:
            return None
        cache_key = (int(grid_h), int(half_w), device.type, device.index)
        if cache_key in self.shape_k_vector_cache:
            return self.shape_k_vector_cache[cache_key]
        shape = self.reference_shape_map.to(device=device, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        resized = torch.nn.functional.interpolate(
            shape,
            size=(int(grid_h), int(half_w)),
            mode="bilinear",
            align_corners=False,
        ).flatten()
        resized = resized - resized.mean()
        norm = resized.norm()
        if float(norm.item()) <= 1e-6:
            self.shape_k_vector_cache[cache_key] = None
            return None
        vector = resized / norm
        self.shape_k_vector_cache[cache_key] = vector
        return vector

    def _reference_shape_weights(
        self,
        grid_h: int,
        half_w: int,
        device: torch.device,
    ) -> torch.Tensor | None:
        if self.reference_shape_map is None:
            return None
        cache_key = (int(grid_h), int(half_w), device.type, device.index)
        if cache_key in self.shape_k_weight_cache:
            return self.shape_k_weight_cache[cache_key]
        shape = self.reference_shape_map.to(device=device, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        resized = torch.nn.functional.interpolate(
            shape,
            size=(int(grid_h), int(half_w)),
            mode="bilinear",
            align_corners=False,
        ).flatten()
        resized = resized - resized.min()
        max_value = resized.max()
        if float(max_value.item()) <= 1e-6:
            self.shape_k_weight_cache[cache_key] = None
            return None
        weights = resized / max_value
        self.shape_k_weight_cache[cache_key] = weights
        return weights


    def apply_shape_orthogonal_k(
        self,
        *,
        block_name: str,
        key: torch.Tensor,
        image_token_count: int,
        encoder_token_count: int,
    ) -> torch.Tensor:
        if not self.should_apply_shape_k(block_name):
            return key
        if image_token_count <= 0:
            self.shape_k_skipped_no_shape += 1
            return key
        try:
            grid_h, grid_w, _, _, ref_key, _ = self._target_indices(
                image_token_count=image_token_count,
                encoder_token_count=encoder_token_count,
                device=key.device,
            )
        except ValueError:
            self.shape_k_skipped_no_shape += 1
            return key
        half_w = grid_w // 2
        key_ref = key[:, ref_key, :, :]
        key_ref_float = key_ref.float()
        corrected = key_ref_float
        applied = False

        if self.shape_k_mode in {"orthogonal", "both"} and self.shape_k_eta > 0.0:
            shape_vector = self._reference_shape_vector(grid_h, half_w, key.device)
            if shape_vector is None:
                if self.shape_k_mode == "orthogonal":
                    self.shape_k_skipped_low_norm += 1
                    return key
            else:
                coeff = torch.einsum("n,bnhc->bhc", shape_vector, corrected)
                shape_component = torch.einsum("n,bhc->bnhc", shape_vector, coeff)
                corrected = corrected - self.shape_k_eta * shape_component
                applied = True

        if self.shape_k_mode in {"suppress", "both"} and self.shape_k_suppress_scale > 0.0:
            weights = self._reference_shape_weights(grid_h, half_w, key.device)
            if weights is None:
                if not applied:
                    self.shape_k_skipped_low_norm += 1
                    return key
            else:
                shape_w = weights.view(1, -1, 1, 1)
                background_w = (1.0 - weights).clamp_min(0.0)
                denom = background_w.sum().clamp_min(1e-6)
                if float(denom.item()) <= 1e-5:
                    anchor = corrected.mean(dim=1, keepdim=True)
                else:
                    anchor = (corrected * background_w.view(1, -1, 1, 1)).sum(dim=1, keepdim=True) / denom
                corrected = corrected - self.shape_k_suppress_scale * shape_w * (corrected - anchor)
                applied = True

        if not applied:
            return key

        delta_ratio = float((corrected - key_ref_float).norm().item() / (key_ref_float.norm().item() + 1e-6))
        key = key.clone()
        key[:, ref_key, :, :] = corrected.to(dtype=key.dtype)

        block_label = safe_name(block_name)
        self.shape_k_applied_calls += 1
        self.shape_k_delta_ratio_sum += delta_ratio
        self.shape_k_delta_ratio_max = max(self.shape_k_delta_ratio_max, delta_ratio)
        self.shape_k_step_counts[self.step] = self.shape_k_step_counts.get(self.step, 0) + 1
        self.shape_k_block_counts[block_label] = self.shape_k_block_counts.get(block_label, 0) + 1
        return key

    def shape_k_summary(self) -> dict[str, object]:
        shape_stats = None
        if self.reference_shape_map is not None:
            shape = self.reference_shape_map.float()
            shape_stats = {
                "height": int(shape.shape[0]),
                "width": int(shape.shape[1]),
                "mean": float(shape.mean().item()),
                "max": float(shape.max().item()),
            }
        return {
            "enabled": self.shape_k_enabled,
            "mode": self.shape_k_mode,
            "eta": self.shape_k_eta,
            "suppress_scale": self.shape_k_suppress_scale,
            "start_step": self._shape_k_start(),
            "end_step": self._shape_k_end(),
            "start_ratio": self.shape_k_start_ratio,
            "end_ratio": self.shape_k_end_ratio,
            "blocks": None if self.shape_k_blocks is None else sorted(self.shape_k_blocks),
            "shape_map": shape_stats,
            "applied_calls": self.shape_k_applied_calls,
            "skipped_no_shape": self.shape_k_skipped_no_shape,
            "skipped_low_norm": self.shape_k_skipped_low_norm,
            "avg_delta_to_key_norm": float(self.shape_k_delta_ratio_sum / self.shape_k_applied_calls) if self.shape_k_applied_calls else 0.0,
            "max_delta_to_key_norm": float(self.shape_k_delta_ratio_max),
            "by_step": {str(k): int(v) for k, v in sorted(self.shape_k_step_counts.items())},
            "by_block": {k: int(v) for k, v in sorted(self.shape_k_block_counts.items())},
        }

    def _target_indices(
        self,
        image_token_count: int,
        encoder_token_count: int,
        device: torch.device,
    ) -> tuple[int, int, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        grid_h, grid_w = infer_diptych_grid(image_token_count, self.height, self.width)
        half_w = grid_w // 2
        spatial_indices = torch.arange(image_token_count, device=device).reshape(grid_h, grid_w)
        ref_indices = spatial_indices[:, :half_w].reshape(-1)
        target_indices = spatial_indices[:, half_w:].reshape(-1)

        image_key_offset = encoder_token_count
        ref_query = ref_indices + image_key_offset
        target_query = target_indices + image_key_offset
        ref_key = ref_indices + image_key_offset
        target_key = target_indices + image_key_offset
        return grid_h, grid_w, ref_query, target_query, ref_key, target_key

    def reduce_maps_from_qk(
        self,
        *,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        image_token_count: int,
        encoder_token_count: int,
        attention_mask: torch.Tensor | None,
        head_dim: int,
        chunk_size: int = 16,
    ) -> dict[str, torch.Tensor]:
        grid_h, grid_w, ref_query, target_query, ref_key, target_key = self._target_indices(
            image_token_count=image_token_count,
            encoder_token_count=encoder_token_count,
            device=query.device,
        )
        half_w = grid_w // 2
        key_t = key.transpose(-2, -1)
        scale = head_dim**-0.5

        reduce_kinds = set(self.save_kinds)
        if self.adaptive_aggregate_kind is not None:
            reduce_kinds.add(self.adaptive_aggregate_kind)
        if self.adaptive_ref_condition:
            reduce_kinds.add("target_to_ref_condition")
        reduced: dict[str, list[torch.Tensor]] = {kind: [] for kind in reduce_kinds}
        condition_image_start = min(max(0, self.adaptive_ref_token_start), encoder_token_count)
        condition_image_key = None
        if encoder_token_count > condition_image_start:
            condition_image_key = torch.arange(condition_image_start, encoder_token_count, device=query.device)

        for start in range(0, target_query.numel(), chunk_size):
            query_idx = target_query[start : start + chunk_size]
            scores = torch.matmul(query[:, :, query_idx, :], key_t) * scale
            if attention_mask is not None:
                scores = scores + attention_mask
            probs = scores.softmax(dim=-1)

            if encoder_token_count > 0 and "target_to_condition" in reduced:
                values = probs[:, :, :, :encoder_token_count].mean(dim=(0, 1, 3)).detach().cpu()
                reduced["target_to_condition"].append(values)

            if "target_to_ref_image" in reduced:
                values = probs[:, :, :, ref_key].mean(dim=(0, 1, 3)).detach().cpu()
                reduced["target_to_ref_image"].append(values)

            if "target_to_target_image" in reduced:
                values = probs[:, :, :, target_key].mean(dim=(0, 1, 3)).detach().cpu()
                reduced["target_to_target_image"].append(values)

            if condition_image_key is not None and "target_to_ref_condition" in reduced:
                values = probs[:, :, :, condition_image_key].mean(dim=(0, 1, 3)).detach().cpu()
                reduced["target_to_ref_condition"].append(values)

            del scores, probs

        if "ref_to_target_image" in reduced:
            for start in range(0, ref_query.numel(), chunk_size):
                query_idx = ref_query[start : start + chunk_size]
                scores = torch.matmul(query[:, :, query_idx, :], key_t) * scale
                if attention_mask is not None:
                    scores = scores + attention_mask
                probs = scores.softmax(dim=-1)
                values = probs[:, :, :, target_key].mean(dim=(0, 1, 3)).detach().cpu()
                reduced["ref_to_target_image"].append(values)
                del scores, probs

        maps: dict[str, torch.Tensor] = {}
        for kind, chunks in reduced.items():
            if chunks:
                maps[kind] = torch.cat(chunks, dim=0).reshape(grid_h, half_w)
        return maps

    def record(
        self,
        *,
        block_name: str,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        image_token_count: int,
        encoder_token_count: int,
        attention_mask: torch.Tensor | None,
        head_dim: int,
    ) -> None:
        self.observe_encoder_token_count(encoder_token_count)

        if self.max_steps is not None and self.step >= self.max_steps:
            return
        if self.save_steps is not None and self.step not in self.save_steps:
            return

        block_label = safe_name(block_name)
        save_block_outputs = self.save_blocks is None or block_label in self.save_blocks
        process_direct = self.direct_aggregate_kind is not None and (save_block_outputs or self.record_all_direct_blocks)
        process_adaptive = self.adaptive_aggregate_kind is not None and (
            self.adaptive_blocks is None or block_label in self.adaptive_blocks
        )
        if not save_block_outputs and not process_direct and not process_adaptive:
            return

        if save_block_outputs:
            block_index = len([r for r in self.records if r.get("step") == self.step])
            if self.max_blocks is not None and block_index >= self.max_blocks:
                return

        grid_h, grid_w = infer_diptych_grid(image_token_count, self.height, self.width)

        saved: dict[str, object] = {}
        maps = self.reduce_maps_from_qk(
            query=query,
            key=key,
            value=value,
            image_token_count=image_token_count,
            encoder_token_count=encoder_token_count,
            attention_mask=attention_mask,
            head_dim=head_dim,
        )

        if self.direct_aggregate_kind is not None and self.direct_aggregate_kind in maps:
            direct_map = maps[self.direct_aggregate_kind].detach().float().cpu()
            if self.direct_aggregate_steps is None or self.step in self.direct_aggregate_steps:
                current = self.direct_sums.get(block_label)
                self.direct_sums[block_label] = direct_map if current is None else current + direct_map
                self.direct_counts[block_label] = self.direct_counts.get(block_label, 0) + 1

        adaptive_kind = self.adaptive_aggregate_kind or self.direct_aggregate_kind
        if adaptive_kind is not None and adaptive_kind in maps:
            adaptive_map = maps[adaptive_kind].detach().float().cpu()
            if self.adaptive_check_steps is None or self.step in self.adaptive_check_steps:
                step_sums = self.direct_step_sums.setdefault(self.step, {})
                step_counts = self.direct_step_counts.setdefault(self.step, {})
                current_step = step_sums.get(block_label)
                step_sums[block_label] = adaptive_map if current_step is None else current_step + adaptive_map
                step_counts[block_label] = step_counts.get(block_label, 0) + 1

        if (
            self.adaptive_ref_condition
            and save_block_outputs
            and (self.adaptive_check_steps is None or self.step in self.adaptive_check_steps)
            and "target_to_ref_condition" in maps
        ):
            condition_image_start = min(max(0, self.adaptive_ref_token_start), encoder_token_count)
            condition_token_count = max(0, encoder_token_count - condition_image_start)
            if condition_token_count > 0:
                ref_condition_mean = float(maps["target_to_ref_condition"].detach().float().mean().item())
                ref_condition_mass = float(ref_condition_mean * condition_token_count)
                self.adaptive_ref_stats.setdefault(self.step, []).append(
                    {
                        "block": block_label,
                        "ref_condition_mass": ref_condition_mass,
                        "ref_condition_mean": ref_condition_mean,
                        "condition_image_tokens": condition_token_count,
                    }
                )

        if not save_block_outputs:
            return

        if "target_to_condition" in maps:
            path = self.out_dir / "target_to_condition" / f"step{self.step:03d}_{block_label}.png"
            saved_item: dict[str, str] = {}
            if self.save_visuals:
                saved_item.update(
                    save_attention_image(
                        maps["target_to_condition"],
                        path,
                        self.target_size,
                        target_base_rgb=self.target_base_rgb,
                    )
                )
            if self.save_raw_maps:
                raw_path = self.out_dir / "target_to_condition_raw" / f"step{self.step:03d}_{block_label}.npy"
                saved_item["raw"] = save_raw_map(maps["target_to_condition"], raw_path)
            saved["target_to_condition"] = saved_item

        if "target_to_ref_image" in maps:
            path = self.out_dir / "target_to_ref_image" / f"step{self.step:03d}_{block_label}.png"
            saved_item = {}
            if self.save_visuals:
                saved_item.update(
                    save_attention_image(
                        maps["target_to_ref_image"],
                        path,
                        self.target_size,
                        target_base_rgb=self.target_base_rgb,
                    )
                )
            if self.save_raw_maps:
                raw_path = self.out_dir / "target_to_ref_image_raw" / f"step{self.step:03d}_{block_label}.npy"
                saved_item["raw"] = save_raw_map(maps["target_to_ref_image"], raw_path)
            saved["target_to_ref_image"] = saved_item

        if "target_to_target_image" in maps:
            path = self.out_dir / "target_to_target_image" / f"step{self.step:03d}_{block_label}.png"
            saved_item = {}
            if self.save_visuals:
                saved_item.update(
                    save_attention_image(
                        maps["target_to_target_image"],
                        path,
                        self.target_size,
                        target_base_rgb=self.target_base_rgb,
                    )
                )
            if self.save_raw_maps:
                raw_path = self.out_dir / "target_to_target_image_raw" / f"step{self.step:03d}_{block_label}.npy"
                saved_item["raw"] = save_raw_map(maps["target_to_target_image"], raw_path)
            saved["target_to_target_image"] = saved_item

        if "ref_to_target_image" in maps:
            path = self.out_dir / "ref_to_target_image" / f"step{self.step:03d}_{block_label}.png"
            saved_item = {}
            if self.save_visuals:
                saved_item.update(
                    save_attention_image(
                        maps["ref_to_target_image"],
                        path,
                        self.target_size,
                        target_base_rgb=self.target_base_rgb,
                    )
                )
            if self.save_raw_maps:
                raw_path = self.out_dir / "ref_to_target_image_raw" / f"step{self.step:03d}_{block_label}.npy"
                saved_item["raw"] = save_raw_map(maps["ref_to_target_image"], raw_path)
            saved["ref_to_target_image"] = saved_item

        if self.pca_kinds:
            grid_h, grid_w, _, target_query, _, _ = self._target_indices(
                image_token_count=image_token_count,
                encoder_token_count=encoder_token_count,
                device=query.device,
            )
            half_w = grid_w // 2
            pca_saved: dict[str, str] = {}
            if "query" in self.pca_kinds:
                tokens = query[:, :, target_query, :].mean(dim=(0, 1))
                path = self.out_dir / "pca_query" / f"step{self.step:03d}_{block_label}.png"
                pca_saved["query"] = save_pca_image(tokens, path, grid_h, half_w, self.target_size)
            if "key" in self.pca_kinds:
                tokens = key[:, :, target_query, :].mean(dim=(0, 1))
                path = self.out_dir / "pca_key" / f"step{self.step:03d}_{block_label}.png"
                pca_saved["key"] = save_pca_image(tokens, path, grid_h, half_w, self.target_size)
            if "value" in self.pca_kinds:
                tokens = value[:, :, target_query, :].mean(dim=(0, 1))
                path = self.out_dir / "pca_value" / f"step{self.step:03d}_{block_label}.png"
                pca_saved["value"] = save_pca_image(tokens, path, grid_h, half_w, self.target_size)
            saved["pca"] = pca_saved

        self.records.append(
            {
                "step": self.step,
                "block": block_name,
                "grid_h": grid_h,
                "grid_w": grid_w,
                "image_tokens": image_token_count,
                "encoder_tokens": encoder_token_count,
                "saved": saved,
            }
        )

    def _direct_soft_mask_from_maps(
        self,
        *,
        direct_sums: dict[str, torch.Tensor],
        direct_counts: dict[str, int],
        selected_blocks: list[str],
        source_shape_hw: tuple[int, int],
        extra_sizes: np.ndarray,
        crop_box: np.ndarray,
        block_polarities: dict[str, str] | None = None,
    ) -> tuple[np.ndarray, dict[str, int]]:
        block_scores: list[np.ndarray] = []
        used_counts: dict[str, int] = {}
        missing: list[str] = []
        for block in selected_blocks:
            count = direct_counts.get(block, 0)
            if count <= 0 or block not in direct_sums:
                missing.append(block)
                continue
            raw = (direct_sums[block] / float(count)).numpy().astype(np.float32)
            mapped = map_attention_to_original(raw, source_shape_hw, extra_sizes, crop_box)
            polarities = self.direct_block_polarities if block_polarities is None else block_polarities
            polarity = polarities.get(block, polarities.get("__default__", "high"))
            score = normalize01_np(mapped)
            if polarity == "low":
                score = 1.0 - score
            block_scores.append(normalize01_np(score))
            used_counts[block] = count

        if not block_scores:
            raise RuntimeError(f"No direct aggregate maps were captured. Missing blocks: {missing}")
        return normalize01_np(np.mean(np.stack(block_scores, axis=0), axis=0)), used_counts

    def direct_soft_mask(
        self,
        *,
        selected_blocks: list[str],
        source_shape_hw: tuple[int, int],
        extra_sizes: np.ndarray,
        crop_box: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, int]]:
        return self._direct_soft_mask_from_maps(
            direct_sums=self.direct_sums,
            direct_counts=self.direct_counts,
            selected_blocks=selected_blocks,
            source_shape_hw=source_shape_hw,
            extra_sizes=extra_sizes,
            crop_box=crop_box,
        )

    def direct_step_soft_mask(
        self,
        *,
        step_index: int,
        selected_blocks: list[str],
        source_shape_hw: tuple[int, int],
        extra_sizes: np.ndarray,
        crop_box: np.ndarray,
        block_polarities: dict[str, str] | None = None,
    ) -> tuple[np.ndarray, dict[str, int]] | None:
        direct_sums = self.direct_step_sums.get(step_index)
        direct_counts = self.direct_step_counts.get(step_index)
        if not direct_sums or not direct_counts:
            return None
        return self._direct_soft_mask_from_maps(
            direct_sums=direct_sums,
            direct_counts=direct_counts,
            selected_blocks=selected_blocks,
            source_shape_hw=source_shape_hw,
            extra_sizes=extra_sizes,
            crop_box=crop_box,
            block_polarities=block_polarities,
        )

    def adaptive_ref_score(self, step_index: int) -> dict[str, object] | None:
        stats = self.adaptive_ref_stats.get(step_index, [])
        if not stats:
            return None
        values = np.array([float(item["ref_condition_mass"]) for item in stats], dtype=np.float32)
        return {
            "score": float(values.mean()),
            "min": float(values.min()),
            "max": float(values.max()),
            "blocks": len(stats),
        }


class RecordingFluxAttnProcessor:
    def __init__(self, recorder: FluxAttentionRecorder, block_name: str, original_processor) -> None:
        self.recorder = recorder
        self.block_name = block_name
        self.original_processor = original_processor

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor | None = None,
        attention_mask: torch.FloatTensor | None = None,
        image_rotary_emb: torch.Tensor | None = None,
    ) -> torch.FloatTensor:
        batch_size = hidden_states.shape[0] if encoder_hidden_states is None else encoder_hidden_states.shape[0]
        if encoder_hidden_states is not None:
            self.recorder.observe_encoder_token_count(encoder_hidden_states.shape[1])

        should_record = self.recorder.should_record(self.block_name)
        should_apply_shape_k = self.recorder.should_apply_shape_k(self.block_name)
        if not should_record and not should_apply_shape_k:
            return self.original_processor(
                attn,
                hidden_states,
                encoder_hidden_states,
                attention_mask,
                image_rotary_emb,
            )

        if encoder_hidden_states is None and self.recorder.last_encoder_token_count > 0:
            encoder_token_count = self.recorder.last_encoder_token_count
            image_token_count = hidden_states.shape[1] - encoder_token_count
        else:
            encoder_token_count = 0 if encoder_hidden_states is None else encoder_hidden_states.shape[1]
            image_token_count = hidden_states.shape[1]

        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        # diffusers==0.32.2 (the pinned project version) uses [B, H, S, D]
        # for FLUX rotary embeddings and scaled-dot-product attention.
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        if attn.added_kv_proj_dim is not None and encoder_hidden_states is not None:
            encoder_query = attn.add_q_proj(encoder_hidden_states)
            encoder_key = attn.add_k_proj(encoder_hidden_states)
            encoder_value = attn.add_v_proj(encoder_hidden_states)

            encoder_query = encoder_query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            encoder_key = encoder_key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            encoder_value = encoder_value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

            if attn.norm_added_q is not None:
                encoder_query = attn.norm_added_q(encoder_query)
            if attn.norm_added_k is not None:
                encoder_key = attn.norm_added_k(encoder_key)

            query = torch.cat([encoder_query, query], dim=2)
            key = torch.cat([encoder_key, key], dim=2)
            value = torch.cat([encoder_value, value], dim=2)

        if image_rotary_emb is not None:
            from diffusers.models.embeddings import apply_rotary_emb

            query = apply_rotary_emb(query, image_rotary_emb)
            key = apply_rotary_emb(key, image_rotary_emb)

        key = self.recorder.apply_shape_orthogonal_k(
            block_name=self.block_name,
            key=key.transpose(1, 2),
            image_token_count=image_token_count,
            encoder_token_count=encoder_token_count,
        ).transpose(1, 2)

        if should_record:
            self.recorder.record(
                block_name=self.block_name,
                query=query,
                key=key,
                value=value,
                image_token_count=image_token_count,
                encoder_token_count=encoder_token_count,
                attention_mask=attention_mask,
                head_dim=head_dim,
            )

        # Pure recording must not change the generated image. Use the exact
        # original processor unless Shape-K intentionally modified the keys.
        if not should_apply_shape_k:
            return self.original_processor(
                attn,
                hidden_states,
                encoder_hidden_states,
                attention_mask,
                image_rotary_emb,
            )

        hidden_states = torch.nn.functional.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
        )
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            encoder_hidden_states, hidden_states = hidden_states.split_with_sizes(
                [encoder_token_count, hidden_states.shape[1] - encoder_token_count], dim=1
            )
            hidden_states = attn.to_out[0](hidden_states.contiguous())
            hidden_states = attn.to_out[1](hidden_states)
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states.contiguous())
            return hidden_states, encoder_hidden_states

        return hidden_states


def is_recordable_attention(module) -> bool:
    return all(
        hasattr(module, attr)
        for attr in ("set_processor", "to_q", "to_k", "to_v", "heads")
    )


def base_attention_processor(processor):
    while isinstance(processor, RecordingFluxAttnProcessor):
        processor = processor.original_processor
    return processor


def register_attention_recorders(pipe, recorder: FluxAttentionRecorder) -> int:
    count = 0
    selected = recorder.save_blocks
    wrap_all_for_shape_k = recorder.shape_k_enabled
    wrap_all_for_direct = recorder.record_all_direct_blocks and recorder.direct_aggregate_kind is not None
    for name, module in pipe.transformer.named_modules():
        if not is_recordable_attention(module):
            continue
        block_label = safe_name(name)
        if not (wrap_all_for_shape_k or wrap_all_for_direct) and selected is not None and block_label not in selected:
            continue
        original_processor = base_attention_processor(getattr(module, "processor", None))
        module.set_processor(RecordingFluxAttnProcessor(recorder, name, original_processor))
        count += 1
    return count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize FLUX attention maps for one Insert-Anything sample.")
    parser.add_argument("--source-image", default=DEFAULT_SOURCE_IMAGE)
    parser.add_argument("--source-mask", default=DEFAULT_SOURCE_MASK)
    parser.add_argument("--ref-image", default=DEFAULT_REF_IMAGE)
    parser.add_argument("--ref-mask", default=DEFAULT_REF_MASK)
    parser.add_argument("--out-dir", default="result_hazelnut_vis/hole")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--num-inference-steps", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--guidance-scale", type=float, default=None)
    parser.add_argument("--max-sequence-length", type=int, default=512)
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cpu-offload", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--sequential-cpu-offload", action="store_true")
    parser.add_argument("--max-blocks", type=int, default=None)
    parser.add_argument("--max-steps-to-save", type=int, default=None)
    parser.add_argument("--save-steps", nargs="+", type=int, default=None)
    parser.add_argument("--save-blocks", nargs="+", default=None)
    parser.add_argument("--clean", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-visuals", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-raw-maps", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--no-pca", action="store_true")
    parser.add_argument(
        "--direct-aggregate-mask",
        action="store_true",
        help="Do not save per-step maps. Stream selected block/step attention into final soft/coarse masks.",
    )
    parser.add_argument(
        "--direct-aggregate-kind",
        choices=["target_to_condition", "target_to_ref_image", "target_to_target_image", "ref_to_target_image"],
        default="target_to_ref_image",
    )
    parser.add_argument(
        "--direct-selected-blocks-file",
        default=DEFAULT_TOP15_BLOCKS_FILE,
        help="Block list for direct aggregate mode. The first --direct-top-k blocks are used.",
    )
    parser.add_argument("--direct-top-k", type=int, default=10)
    parser.add_argument(
        "--direct-block-frequency-csv",
        default=DEFAULT_BLOCK_FREQUENCY_CSV,
        help="Used to choose dominant high/low polarity for fixed blocks.",
    )
    parser.add_argument("--direct-polarity", choices=["dominant", "high", "low"], default="dominant")
    parser.add_argument("--direct-roi", choices=["initial_mask", "object", "all"], default="initial_mask")
    parser.add_argument("--direct-hist-threshold-scale", type=float, default=0.9)
    parser.add_argument("--direct-hist-threshold-offset", type=float, default=0.0)
    parser.add_argument("--direct-component-mode", choices=["all", "largest", "max_energy"], default="all")
    parser.add_argument("--direct-fill-holes", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--direct-close-iterations", type=int, default=1)
    parser.add_argument("--direct-dilate-iterations", type=int, default=1)
    parser.add_argument(
        "--save-kinds",
        nargs="+",
        default=["target_to_condition", "target_to_ref_image", "target_to_target_image"],
        choices=["target_to_condition", "target_to_ref_image", "target_to_target_image", "ref_to_target_image"],
    )
    parser.add_argument(
        "--pca-kinds",
        nargs="+",
        default=["query", "key", "value"],
        choices=["query", "key", "value"],
        help="Save target-token PCA visualizations from attention Q/K/V features.",
    )
    parser.add_argument("--flux-fill-path", default="black-forest-labs/FLUX.1-Fill-dev")
    parser.add_argument("--flux-redux-path", default="black-forest-labs/FLUX.1-Redux-dev")
    parser.add_argument("--lora-path", default="WensongSong/Insert-Anything")
    parser.add_argument("--lora-weight-name", default=DEFAULT_LORA_WEIGHT)
    parser.add_argument(
        "--full-flux-quantize",
        choices=["none", "int8", "int4"],
        default="none",
        help="Quantize diffusers full FLUX after loading while preserving Python attention modules.",
    )
    parser.add_argument(
        "--full-flux-quantize-load-strategy",
        choices=["auto", "direct", "post_load"],
        default="auto",
    )
    parser.add_argument("--full-flux-quantize-redux", action="store_true")
    parser.add_argument("--full-flux-quantize-fuse-lora", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--full-flux-quantize-linear-subclasses", action="store_true")
    parser.add_argument("--full-flux-quantize-min-weight-elements", type=int, default=1)
    parser.add_argument("--shape-k-removal", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--shape-k-eta", type=float, default=0.5)
    parser.add_argument("--shape-k-start-ratio", type=float, default=0.2)
    parser.add_argument("--shape-k-end-ratio", type=float, default=0.7)
    parser.add_argument("--shape-k-start-step", type=int, default=-1)
    parser.add_argument("--shape-k-end-step", type=int, default=-1)
    parser.add_argument("--shape-k-blocks", nargs="+", default=None)
    parser.add_argument("--shape-k-block-scope", choices=SHAPE_K_BLOCK_SCOPE_CHOICES, default="top")
    parser.add_argument("--shape-k-mode", choices=["orthogonal", "suppress", "both"], default="orthogonal")
    parser.add_argument("--shape-k-suppress-scale", type=float, default=1.0)
    parser.add_argument("--shape-k-edge-method", choices=["sobel", "canny", "highfreq", "foreground"], default="sobel")
    parser.add_argument("--shape-k-foreground-threshold", type=int, default=250)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    direct_selected_blocks: list[str] = []
    direct_block_polarities: dict[str, str] = {}
    if args.direct_aggregate_mask:
        direct_selected_blocks = read_block_list(args.direct_selected_blocks_file, args.direct_top_k)
        if not direct_selected_blocks:
            raise ValueError(f"No selected blocks found in {args.direct_selected_blocks_file}")
        if args.direct_polarity == "dominant":
            dominant = read_dominant_polarities(args.direct_block_frequency_csv)
            direct_block_polarities = {
                block: dominant.get(block, "low")
                for block in direct_selected_blocks
            }
        else:
            direct_block_polarities = {block: args.direct_polarity for block in direct_selected_blocks}

        default_steps = [step for step in [5, 10, 15, 20, 25, 30, 35, 40] if step < args.num_inference_steps]
        if not default_steps:
            default_steps = [max(0, args.num_inference_steps - 1)]
        if args.save_steps is None:
            args.save_steps = default_steps
        if args.save_blocks is None:
            args.save_blocks = direct_selected_blocks
        args.save_kinds = [args.direct_aggregate_kind]
        args.save_visuals = False
        args.save_raw_maps = False
        args.no_pca = True

    out_dir = Path(args.out_dir)
    if args.clean and out_dir.exists():
        for name in (
            "target_to_condition",
            "target_to_ref_image",
            "target_to_target_image",
            "ref_to_target_image",
            "target_to_condition_overlay",
            "target_to_ref_image_overlay",
            "target_to_target_image_overlay",
            "ref_to_target_image_overlay",
            "target_to_condition_raw",
            "target_to_ref_image_raw",
            "target_to_target_image_raw",
            "ref_to_target_image_raw",
            "pca_query",
            "pca_key",
            "pca_value",
            "debug_diptych_input.png",
            "debug_diptych_mask.png",
            "debug_masked_reference.png",
            "edit.png",
            "metadata.json",
            "soft_mask.npy",
            "soft_mask.png",
            "soft_mask_heatmap.png",
            "soft_mask_overlay.png",
            "coarse_mask.png",
            "coarse_mask_overlay.png",
            "edit_coarse_mask_overlay.png",
            "coarse_refine_seed_mask.png",
            "edit_coarse_refine_seed_overlay.png",
            "threshold_roi.png",
            "threshold_roi_overlay.png",
            "direct_aggregate_summary.json",
            "refined_mask.png",
            "edit_refined_mask_overlay.png",
            "active_contour_mask.png",
            "active_contour_raw_mask.png",
            "edit_active_contour_overlay.png",
            "active_contour_roi.png",
            "active_contour_edge_gate.png",
            "active_contour_active_roi.png",
            "active_contour_edge_map.png",
            "active_contour_scharr.png",
            "active_contour_wavelet_highfreq.png",
            "q80_appearance_mask.png",
            "edit_q80_appearance_overlay.png",
            "contour_refined_mask.png",
            "contour_refined_mask_raw.png",
            "edit_contour_refined_overlay.png",
        ):
            path = out_dir / name
            if path.is_dir():
                shutil.rmtree(path)
            elif path.exists():
                path.unlink()
    out_dir.mkdir(parents=True, exist_ok=True)

    size = (args.size, args.size)
    diptych_ref_tar, mask_diptych, masked_ref_image, old_tar_image, extra_sizes, crop_box = prepare_diptych(
        args.source_image,
        args.source_mask,
        args.ref_image,
        args.ref_mask,
        size,
    )
    Image.fromarray(masked_ref_image).save(out_dir / "debug_masked_reference.png")
    reference_shape_map = reference_shape_map_from_image(
        masked_ref_image,
        method=args.shape_k_edge_method,
        foreground_threshold=args.shape_k_foreground_threshold,
    )
    Image.fromarray((normalize01_np(reference_shape_map) * 255.0).clip(0, 255).astype(np.uint8)).save(
        out_dir / "debug_reference_shape_map.png"
    )
    diptych_ref_tar.save(out_dir / "debug_diptych_input.png")
    mask_diptych.save(out_dir / "debug_diptych_mask.png")
    diptych_np = np.array(diptych_ref_tar.convert("RGB"))
    target_base_rgb = diptych_np[:, mask_diptych.size[0] // 2 :, :]

    # load_pipelines expects the same argparse fields as run_insert_anything.py.
    args.nunchaku = False
    args.nunchaku_transformer_path = None
    args.nunchaku_lora_path = None
    args.nunchaku_lora_strength = None
    args.flux_fill_path = args.flux_fill_path
    args.flux_redux_path = args.flux_redux_path
    args.lora_path = args.lora_path
    args.lora_weight_name = args.lora_weight_name

    pipe, redux = load_pipelines(args)
    pipe_prior_output = redux(Image.fromarray(masked_ref_image))

    shape_k_blocks = resolve_shape_k_blocks(args.shape_k_blocks, args.save_blocks, args.shape_k_block_scope)
    recorder = FluxAttentionRecorder(
        out_dir=out_dir,
        height=mask_diptych.size[1],
        width=mask_diptych.size[0],
        save_kinds=set(args.save_kinds),
        pca_kinds=set() if args.no_pca else set(args.pca_kinds),
        target_base_rgb=target_base_rgb,
        max_blocks=args.max_blocks,
        max_steps=args.max_steps_to_save,
        save_steps=set(args.save_steps) if args.save_steps else None,
        save_blocks=set(args.save_blocks) if args.save_blocks else None,
        save_visuals=args.save_visuals,
        save_raw_maps=args.save_raw_maps,
        direct_aggregate_kind=args.direct_aggregate_kind if args.direct_aggregate_mask else None,
        direct_block_polarities=direct_block_polarities,
        num_inference_steps=args.num_inference_steps,
        shape_k_enabled=args.shape_k_removal,
        shape_k_eta=args.shape_k_eta,
        shape_k_start_step=args.shape_k_start_step,
        shape_k_end_step=args.shape_k_end_step,
        shape_k_start_ratio=args.shape_k_start_ratio,
        shape_k_end_ratio=args.shape_k_end_ratio,
        shape_k_blocks=shape_k_blocks,
        shape_k_mode=args.shape_k_mode,
        shape_k_suppress_scale=args.shape_k_suppress_scale,
        reference_shape_map=reference_shape_map,
    )
    n_attn = register_attention_recorders(pipe, recorder)
    print(f"[attention] registered {n_attn} attention processors")

    def callback_on_step_end(_pipe, step_index, _timestep, callback_kwargs):
        print(f"[attention] finished step {step_index}", flush=True)
        recorder.after_step(step_index)
        return callback_kwargs

    generator_device = "cuda" if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu"
    generator = torch.Generator(generator_device).manual_seed(args.seed)
    pipe_kwargs = {
        "image": diptych_ref_tar,
        "mask_image": mask_diptych,
        "height": mask_diptych.size[1],
        "width": mask_diptych.size[0],
        "num_inference_steps": args.num_inference_steps,
        "max_sequence_length": args.max_sequence_length,
        "generator": generator,
        "callback_on_step_end": callback_on_step_end,
        "callback_on_step_end_tensor_inputs": ["latents"],
        **pipe_prior_output,
    }
    if args.guidance_scale is not None:
        pipe_kwargs["guidance_scale"] = args.guidance_scale

    edited_image = pipe(**pipe_kwargs).images[0]
    width, height = edited_image.size
    edited_image = edited_image.crop((width // 2, 0, width, height))
    edited_image = np.array(edited_image)
    edited_image = crop_back(edited_image, old_tar_image, extra_sizes, crop_box)
    edited_pil = Image.fromarray(edited_image)
    edited_pil.save(out_dir / "edit.png")

    direct_aggregate_info = None
    if args.direct_aggregate_mask:
        soft_mask, direct_counts = recorder.direct_soft_mask(
            selected_blocks=direct_selected_blocks,
            source_shape_hw=tuple(old_tar_image.shape[:2]),
            extra_sizes=extra_sizes,
            crop_box=crop_box,
        )
        if args.direct_roi == "initial_mask":
            threshold_roi = load_binary_mask(args.source_mask, tuple(old_tar_image.shape[:2]))
        elif args.direct_roi == "object":
            threshold_roi = object_roi_from_image(args.source_image, tuple(old_tar_image.shape[:2]))
        else:
            threshold_roi = np.ones(tuple(old_tar_image.shape[:2]), dtype=bool)

        coarse_mask, hist_threshold, hist_otsu_threshold = threshold_soft_mask(
            soft_mask,
            threshold_roi,
            scale=args.direct_hist_threshold_scale,
            offset=args.direct_hist_threshold_offset,
            component_mode=args.direct_component_mode,
            fill=args.direct_fill_holes,
            close_iterations=args.direct_close_iterations,
            dilate_iterations=args.direct_dilate_iterations,
        )
        soft_u8 = (normalize01_np(soft_mask) * 255.0).clip(0, 255).astype(np.uint8)
        np.save(out_dir / "soft_mask.npy", soft_mask.astype(np.float32))
        Image.fromarray(soft_u8).save(out_dir / "soft_mask.png")
        Image.fromarray(colorize_u8(soft_u8)).save(out_dir / "soft_mask_heatmap.png")
        Image.fromarray((coarse_mask.astype(np.uint8) * 255)).save(out_dir / "coarse_mask.png")
        Image.fromarray((threshold_roi.astype(np.uint8) * 255)).save(out_dir / "threshold_roi.png")
        save_soft_overlay(old_tar_image, soft_mask, out_dir / "soft_mask_overlay.png")
        save_mask_overlay(old_tar_image, coarse_mask, out_dir / "coarse_mask_overlay.png")
        save_mask_overlay(edited_image, coarse_mask, out_dir / "edit_coarse_mask_overlay.png")
        save_mask_overlay(old_tar_image, threshold_roi, out_dir / "threshold_roi_overlay.png", color=(0, 255, 0), alpha=0.25)
        direct_aggregate_info = {
            "mode": "direct_streaming_no_per_step_map_files",
            "kind": args.direct_aggregate_kind,
            "selected_blocks_file": args.direct_selected_blocks_file,
            "top_k": len(direct_selected_blocks),
            "selected_blocks": direct_selected_blocks,
            "block_polarities": direct_block_polarities,
            "block_step_counts": direct_counts,
            "requested_save_steps": args.save_steps,
            "aggregate": "mean_over_selected_steps_then_mean_over_blocks",
            "roi": args.direct_roi,
            "hist_threshold_scale": args.direct_hist_threshold_scale,
            "hist_threshold_offset": args.direct_hist_threshold_offset,
            "hist_threshold": float(hist_threshold),
            "hist_otsu_threshold": float(hist_otsu_threshold),
            "component_mode": args.direct_component_mode,
            "fill_holes": args.direct_fill_holes,
            "close_iterations": args.direct_close_iterations,
            "dilate_iterations": args.direct_dilate_iterations,
            "soft_mask": str(out_dir / "soft_mask.png"),
            "soft_mask_npy": str(out_dir / "soft_mask.npy"),
            "coarse_mask": str(out_dir / "coarse_mask.png"),
            "coarse_mask_overlay": str(out_dir / "coarse_mask_overlay.png"),
            "edit_coarse_mask_overlay": str(out_dir / "edit_coarse_mask_overlay.png"),
        }
        coarse_refine_mask, coarse_refine_info = coarse_mask_for_refine(coarse_mask)
        direct_aggregate_info["coarse_refine_seed"] = coarse_refine_info
        run_pamr_refine = os.environ.get("RUN_PAMR_REFINE", "0").lower() not in {"0", "false", "no"}
        if run_pamr_refine:
            refine_info = refine_direct_attention_mask(
                image_rgb=edited_image,
                soft_mask=soft_mask,
                coarse_mask=coarse_refine_mask,
                object_mask=threshold_roi,
                out_dir=out_dir,
                object_source="threshold_roi",
            )
            refined_overlay_path = out_dir / "edit_refined_mask_overlay.png"
            refined_mask = np.array(Image.open(refine_info["output_mask"]).convert("L")) > 128
            save_mask_overlay(edited_image, refined_mask, refined_overlay_path)
            refine_info["edit_refined_mask_overlay"] = str(refined_overlay_path)
            direct_aggregate_info["refined_mask"] = refine_info
        else:
            refine_info = {"enabled": False, "reason": "RUN_PAMR_REFINE=0"}
            direct_aggregate_info["refined_mask"] = refine_info
            refined_overlay_path = out_dir / "edit_refined_mask_overlay.png"
        active_info = refine_active_contour_mask(
            image_rgb=edited_image,
            coarse_mask=coarse_refine_mask,
            object_mask=threshold_roi,
            out_dir=out_dir,
            object_source="threshold_roi",
        )
        direct_aggregate_info["active_contour_mask"] = active_info
        active_overlay_path = None
        save_active_overlay = os.environ.get("SAVE_ACTIVE_CONTOUR_OVERLAY", "1").lower() not in {"0", "false", "no"}
        if save_active_overlay:
            active_overlay_path = out_dir / "edit_active_contour_overlay.png"
            active_mask = active_info["_refined_mask"]
            save_mask_overlay(edited_image, active_mask, active_overlay_path)
            active_info["edit_active_contour_overlay"] = str(active_overlay_path)

        run_q80_appearance_refine = os.environ.get("RUN_Q80_APPEARANCE_REFINE", "1").lower() not in {"0", "false", "no"}
        if run_q80_appearance_refine:
            q80_edge_map = active_info["_edge_map_u8"]
            q80_coarse_roi = active_info["_coarse_roi_mask"]
            q80_info = refine_q80_appearance_mask(
                image_rgb=edited_image,
                coarse_mask=coarse_refine_mask,
                object_mask=threshold_roi,
                out_dir=out_dir,
                object_source="threshold_roi",
                edge_map_u8=q80_edge_map,
                coarse_roi_mask=q80_coarse_roi,
            )
            q80_mask = np.array(Image.open(q80_info["output_mask"]).convert("L")) > 128
            q80_overlay_path = None
            save_q80_overlay = os.environ.get("SAVE_Q80_OVERLAY", "0").lower() not in {"0", "false", "no"}
            if save_q80_overlay:
                q80_overlay_path = out_dir / "edit_q80_appearance_overlay.png"
                save_mask_overlay(edited_image, q80_mask, q80_overlay_path)
                q80_info["edit_q80_appearance_overlay"] = str(q80_overlay_path)
            direct_aggregate_info["q80_appearance_mask"] = q80_info
        else:
            q80_mask = None
            q80_coarse_roi = None
            q80_info = {"enabled": False, "reason": "RUN_Q80_APPEARANCE_REFINE=0"}
            direct_aggregate_info["q80_appearance_mask"] = q80_info
            q80_overlay_path = None

        run_contour_refine = os.environ.get("RUN_CONTOUR_REFINE", "1").lower() not in {"0", "false", "no"}
        if run_contour_refine and q80_mask is not None and q80_coarse_roi is not None:
            contour_info = refine_boundary_preserve_mask(
                image_rgb=edited_image,
                coarse_mask=coarse_refine_mask,
                edge_refined_mask=q80_mask,
                object_mask=threshold_roi,
                out_dir=out_dir,
                object_source="threshold_roi",
                coarse_roi_mask=q80_coarse_roi,
            )
            contour_overlay_path = out_dir / "edit_contour_refined_overlay.png"
            contour_mask = np.array(Image.open(contour_info["output_mask"]).convert("L")) > 128
            save_mask_overlay(edited_image, contour_mask, contour_overlay_path)
            contour_info["edit_contour_refined_overlay"] = str(contour_overlay_path)
            direct_aggregate_info["contour_refined_mask"] = contour_info
            direct_aggregate_info["recommended_refined_mask"] = contour_info["output_mask"]
        else:
            reason = "RUN_CONTOUR_REFINE=0" if not run_contour_refine else "q80_appearance_mask disabled"
            contour_info = {"enabled": False, "reason": reason}
            direct_aggregate_info["contour_refined_mask"] = contour_info
            direct_aggregate_info["recommended_refined_mask"] = q80_info.get("output_mask")
            contour_overlay_path = out_dir / "edit_contour_refined_overlay.png"

        if run_pamr_refine:
            print(f"[refine] refined_mask={refine_info['output_mask']}", flush=True)
            print(f"[refine] overlay={refined_overlay_path}", flush=True)
        else:
            print("[refine] skipped PAMR refined_mask because RUN_PAMR_REFINE=0", flush=True)
        if active_info["output_mask"]:
            print(f"[refine] active_contour_mask={active_info['output_mask']}", flush=True)
        else:
            print("[refine] active_contour_mask not saved because SAVE_ACTIVE_CONTOUR_MASK=0", flush=True)
        if active_overlay_path is not None:
            print(f"[refine] active_overlay={active_overlay_path}", flush=True)
        if run_q80_appearance_refine:
            print(f"[refine] q80_appearance_mask={q80_info['output_mask']}", flush=True)
            if q80_overlay_path is not None:
                print(f"[refine] q80_overlay={q80_overlay_path}", flush=True)
        else:
            print("[refine] skipped q80_appearance_mask because RUN_Q80_APPEARANCE_REFINE=0", flush=True)
        if run_contour_refine and contour_info.get("enabled", False):
            print(f"[refine] contour_refined_mask={contour_info['output_mask']}", flush=True)
            print(f"[refine] contour_overlay={contour_overlay_path}", flush=True)
        else:
            print(f"[refine] skipped contour_refined_mask because {contour_info.get('reason', 'disabled')}", flush=True)
        direct_aggregate_info["active_contour_mask"] = {
            key: value
            for key, value in direct_aggregate_info["active_contour_mask"].items()
            if not key.startswith("_")
        }
        (out_dir / "direct_aggregate_summary.json").write_text(
            json.dumps(direct_aggregate_info, indent=2),
            encoding="utf-8",
        )
        print(f"[direct] soft_mask={out_dir / 'soft_mask.png'}", flush=True)
        print(f"[direct] coarse_mask={out_dir / 'coarse_mask.png'}", flush=True)
        print(f"[direct] overlay={out_dir / 'coarse_mask_overlay.png'}", flush=True)

    metadata = {
        "source_image": args.source_image,
        "source_mask": args.source_mask,
        "ref_image": args.ref_image,
        "ref_mask": args.ref_mask,
        "seed": args.seed,
        "size": args.size,
        "num_inference_steps": args.num_inference_steps,
        "source_image_shape": list(old_tar_image.shape[:2]),
        "target_extra_sizes": extra_sizes.tolist(),
        "target_crop_box_yyxx": crop_box.tolist(),
        "target_attention_size": [mask_diptych.size[1], mask_diptych.size[0] // 2],
        "attention_processors": n_attn,
        "save_kinds": args.save_kinds,
        "pca_kinds": [] if args.no_pca else args.pca_kinds,
        "save_steps": args.save_steps,
        "save_blocks": args.save_blocks,
        "save_visuals": args.save_visuals,
        "save_raw_maps": args.save_raw_maps,
        "direct_aggregate_mask": direct_aggregate_info,
        "shape_k_removal": recorder.shape_k_summary(),
        "records": recorder.records,
        "attention_note": (
            "FLUX uses joint attention. target_to_condition is target image queries attending to Redux condition "
            "tokens, target_to_ref_image is target image queries attending to left reference-image tokens, and "
            "target_to_target_image is target image-image self attention. ref_to_target_image is left "
            "reference-image queries attending to target-image tokens."
        ),
        "nunchaku_note": "Full FLUX transformer is used because nunchaku fused transformer does not expose per-block attention probabilities.",
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"[output] {out_dir}")


if __name__ == "__main__":
    main()
