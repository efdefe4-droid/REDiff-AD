#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from run_insert_anything import (
    DEFAULT_NUNCHAKU_LORA,
    DEFAULT_NUNCHAKU_TRANSFORMER,
    box2squre,
    crop_back,
    expand_bbox,
    expand_image_mask,
    get_bbox_from_mask,
    load_mask,
    load_pipelines,
    load_rgb,
    pad_to_square,
)


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


class TeeStream:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()

    def isatty(self) -> bool:
        return any(getattr(stream, "isatty", lambda: False)() for stream in self.streams)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch Insert-Anything generation for hazelnut defects.")
    parser.add_argument("--anomalies", nargs="+", required=True)
    parser.add_argument("--ref-ids", "--ref_ids", nargs="+", required=True)
    parser.add_argument("--samples-per-anomaly", "--samples_per_anomaly", type=int, default=500)
    parser.add_argument(
        "--mask-root",
        "--mask_root",
        default=os.environ.get("MASK_ROOT", "datasets/hazelnut/generated_masks"),
    )
    parser.add_argument(
        "--source-root",
        "--source_root",
        default=os.environ.get("SOURCE_IMAGE_ROOT", "datasets/hazelnut/train/good"),
    )
    parser.add_argument(
        "--ref-image-root",
        "--ref_image_root",
        default=os.environ.get("REF_IMAGE_ROOT", "datasets/hazelnut/test"),
    )
    parser.add_argument(
        "--ref-mask-root",
        "--ref_mask_root",
        default=os.environ.get("REF_MASK_ROOT", "datasets/hazelnut/ground_truth"),
    )
    parser.add_argument("--out-root", "--out_root", default="result/hazelnut_batch_nunchaku")
    parser.add_argument("--seed", type=int, default=666)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--num-inference-steps", "--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance-scale", "--guidance_scale", type=float, default=None)
    parser.add_argument("--max-sequence-length", "--max_sequence_length", type=int, default=512)
    parser.add_argument("--local-files-only", "--local_files_only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cpu-offload", "--cpu_offload", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--sequential-cpu-offload", "--sequential_cpu_offload", action="store_true")
    parser.add_argument("--nunchaku", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--nunchaku-transformer-path", "--nunchaku_transformer_path", default=DEFAULT_NUNCHAKU_TRANSFORMER)
    parser.add_argument("--nunchaku-lora-path", "--nunchaku_lora_path", default=DEFAULT_NUNCHAKU_LORA)
    parser.add_argument("--nunchaku-lora-strength", "--nunchaku_lora_strength", type=float, default=1.0)
    parser.add_argument("--nunchaku-precision", "--nunchaku_precision", choices=["auto", "int4", "fp4"], default="auto")
    parser.add_argument("--flux-fill-path", "--flux_fill_path", default="black-forest-labs/FLUX.1-Fill-dev")
    parser.add_argument("--flux-redux-path", "--flux_redux_path", default="black-forest-labs/FLUX.1-Redux-dev")
    parser.add_argument("--lora-path", "--lora_path", default="WensongSong/Insert-Anything")
    parser.add_argument("--lora-weight-name", "--lora_weight_name", default="20250321_steps5000_pytorch_lora_weights.safetensors")
    parser.add_argument("--start-index", "--start_index", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--save-debug-first", "--save_debug_first", action="store_true")
    parser.add_argument(
        "--target-mask-source",
        "--target_mask_source",
        choices=["dataset", "random_object"],
        default="dataset",
        help="dataset reads masks from --mask-root; random_object generates target masks inside the source object support.",
    )
    parser.add_argument("--object-prompt", "--object_prompt", default=None)
    parser.add_argument(
        "--object-support-root",
        "--object_support_root",
        default=None,
        help="Optional directory of precomputed object support masks named like the source images.",
    )
    parser.add_argument(
        "--object-attention-root",
        "--object_attention_root",
        default=None,
        help="Optional directory of precomputed object attention maps named like the source images.",
    )
    parser.add_argument("--object-support-erosion", "--object_support_erosion", type=int, default=8)
    parser.add_argument("--random-mask-area-min-ratio", "--random_mask_area_min_ratio", type=float, default=0.60)
    parser.add_argument("--random-mask-area-max-ratio", "--random_mask_area_max_ratio", type=float, default=1.40)
    parser.add_argument("--random-mask-rotate", "--random_mask_rotate", type=float, default=45.0)
    parser.add_argument("--random-mask-attempts", "--random_mask_attempts", type=int, default=120)
    parser.add_argument("--random-mask-component-spacing", "--random_mask_component_spacing", type=int, default=4)
    parser.add_argument("--random-mask-double-prob", "--random_mask_double_prob", type=float, default=0.0)
    parser.add_argument(
        "--random-mask-placement-filter",
        "--random_mask_placement_filter",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--random-mask-dark-quantile", "--random_mask_dark_quantile", type=float, default=20.0)
    parser.add_argument("--random-mask-max-dark-fraction", "--random_mask_max_dark_fraction", type=float, default=0.35)
    parser.add_argument("--random-mask-edge-quantile", "--random_mask_edge_quantile", type=float, default=90.0)
    parser.add_argument("--random-mask-max-edge-fraction", "--random_mask_max_edge_fraction", type=float, default=0.30)
    parser.add_argument(
        "--random-mask-boundary-margin-ratio",
        "--random_mask_boundary_margin_ratio",
        type=float,
        default=0.015,
    )
    parser.add_argument(
        "--random-mask-max-boundary-fraction",
        "--random_mask_max_boundary_fraction",
        type=float,
        default=0.03,
    )
    parser.add_argument("--save-input-copies", "--save_input_copies", action="store_true")
    parser.add_argument("--log-file", "--log_file", default=None)
    return parser.parse_args()


def natural_key(path: Path) -> tuple[int, str]:
    try:
        return int(path.stem), path.name
    except ValueError:
        return 10**12, path.name


def list_images(root: Path) -> list[Path]:
    files = [p for p in root.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
    return sorted(files, key=natural_key)


def resolve_id_file(root: Path, ref_id: str, suffixes: tuple[str, ...] = ("",)) -> Path:
    candidates = [ref_id]
    try:
        value = int(ref_id)
        candidates.extend([str(value), f"{value:03d}", f"{value:04d}"])
    except ValueError:
        pass

    for stem in dict.fromkeys(candidates):
        for suffix in suffixes:
            for ext in [".png", ".jpg", ".jpeg", ".bmp", ".webp"]:
                path = root / f"{stem}{suffix}{ext}"
                if path.is_file():
                    return path
    raise FileNotFoundError(f"Reference id {ref_id} not found under {root}")


def resolve_matching_file(root: Path, source_path: Path, suffixes: tuple[str, ...] = ("",)) -> Path:
    stems = [source_path.stem]
    try:
        value = int(source_path.stem)
        stems.extend([str(value), f"{value:03d}", f"{value:04d}"])
    except ValueError:
        pass

    for stem in dict.fromkeys(stems):
        for suffix in suffixes:
            for ext in [source_path.suffix, ".png", ".jpg", ".jpeg", ".bmp", ".webp"]:
                if not ext:
                    continue
                path = root / f"{stem}{suffix}{ext}"
                if path.is_file():
                    return path
    raise FileNotFoundError(f"No file matching {source_path.name} found under {root}")


def build_anomaly_refs(anomalies: list[str], ref_ids: list[str]) -> list[tuple[str, str]]:
    if len(ref_ids) == 1 and len(anomalies) > 1:
        ref_ids = ref_ids * len(anomalies)
    if len(anomalies) != len(ref_ids):
        raise ValueError("--anomalies and --ref-ids must have the same length, or provide one ref id for all anomalies.")
    return list(zip(anomalies, ref_ids))


def prepare_reference(ref_image_path: Path, ref_mask_path: Path, size: tuple[int, int]) -> np.ndarray:
    ref_image = load_rgb(ref_image_path)
    ref_mask = load_mask(ref_mask_path)

    ref_box_yyxx = get_bbox_from_mask(ref_mask)
    ref_mask_3 = np.stack([ref_mask, ref_mask, ref_mask], -1)
    masked_ref_image = ref_image * ref_mask_3 + np.ones_like(ref_image) * 255 * (1 - ref_mask_3)

    y1, y2, x1, x2 = ref_box_yyxx
    masked_ref_image = masked_ref_image[y1:y2, x1:x2, :]
    ref_mask = ref_mask[y1:y2, x1:x2]
    masked_ref_image, _ = expand_image_mask(masked_ref_image, ref_mask, ratio=1.3)
    masked_ref_image = pad_to_square(masked_ref_image, pad_value=255, random=False)
    return cv2.resize(masked_ref_image.astype(np.uint8), size).astype(np.uint8)


def single_pixel_mask(mask_path: Path, size: tuple[int, int]) -> np.ndarray:
    width, height = size
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid fallback mask size {size} for {mask_path}")

    rng = random.Random(f"{mask_path}:{width}x{height}")
    x = rng.randrange(width)
    y = rng.randrange(height)
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[y, x] = 1
    print(
        f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] "
        f"WARN empty target mask fallback path={mask_path} shape={height}x{width} pixel=({x},{y})",
        flush=True,
    )
    return mask


def load_target_mask(mask_path: Path, size: tuple[int, int]) -> np.ndarray:
    try:
        return load_mask(mask_path, size=size)
    except ValueError as exc:
        if "Mask is empty after thresholding" not in str(exc):
            raise
        return single_pixel_mask(mask_path, size)


def fill_mask_holes(mask: np.ndarray) -> np.ndarray:
    mask_u8 = (mask > 0).astype(np.uint8)
    padded = np.pad(mask_u8, 1, mode="constant", constant_values=0)
    flood = padded.copy()
    h, w = flood.shape
    cv2.floodFill(flood, np.zeros((h + 2, w + 2), np.uint8), (0, 0), 1)
    holes = (flood == 0).astype(np.uint8)
    filled = np.maximum(padded, holes)[1:-1, 1:-1]
    return filled.astype(np.uint8)


def largest_component(mask: np.ndarray) -> np.ndarray:
    mask_u8 = (mask > 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    if num_labels <= 1:
        return mask_u8
    areas = stats[1:, cv2.CC_STAT_AREA]
    label = int(np.argmax(areas)) + 1
    return (labels == label).astype(np.uint8)


def object_support_from_image(image_rgb: np.ndarray, erosion: int = 8) -> tuple[np.ndarray, np.ndarray, dict]:
    height, width = image_rgb.shape[:2]
    blurred = cv2.GaussianBlur(image_rgb, (5, 5), 0)
    lab = cv2.cvtColor(blurred, cv2.COLOR_RGB2LAB).astype(np.float32)
    border = np.concatenate([lab[:24].reshape(-1, 3), lab[-24:].reshape(-1, 3), lab[:, :24].reshape(-1, 3), lab[:, -24:].reshape(-1, 3)], axis=0)
    bg = np.median(border, axis=0)
    dist = np.linalg.norm(lab - bg.reshape(1, 1, 3), axis=2)
    dist_u8 = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    otsu_threshold, _ = cv2.threshold(dist_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    threshold = max(18, int(otsu_threshold))
    support = (dist_u8 >= threshold).astype(np.uint8)

    kernel_close = np.ones((17, 17), np.uint8)
    kernel_open = np.ones((5, 5), np.uint8)
    support = cv2.morphologyEx(support, cv2.MORPH_CLOSE, kernel_close, iterations=2)
    support = cv2.morphologyEx(support, cv2.MORPH_OPEN, kernel_open, iterations=1)
    support = largest_component(support)
    support = fill_mask_holes(support)

    if erosion > 0:
        kernel_erode = np.ones((3, 3), np.uint8)
        eroded = cv2.erode(support, kernel_erode, iterations=int(erosion))
        if int(eroded.sum()) > max(32, int(support.sum() * 0.15)):
            support = eroded

    if int(support.sum()) == 0:
        support = np.zeros((height, width), dtype=np.uint8)
        cv2.ellipse(support, (width // 2, height // 2), (width // 4, height // 4), 0, 0, 360, 1, -1)

    attention = dist_u8.copy()
    attention[support == 0] = 0
    info = {
        "source": "foreground_attention_proxy",
        "threshold": threshold,
        "erosion": int(erosion),
        "area": int(support.sum()),
        "object_prompt": None,
    }
    return support.astype(np.uint8), attention, info


def load_object_support_from_file(
    support_path: Path,
    attention_path: Path | None,
    size: tuple[int, int],
    erosion: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    support = load_target_mask(support_path, size=size).astype(np.uint8)
    if erosion > 0:
        kernel = np.ones((3, 3), np.uint8)
        eroded = cv2.erode(support, kernel, iterations=int(erosion))
        if int(eroded.sum()) > max(32, int(support.sum() * 0.15)):
            support = eroded

    width, height = size
    if attention_path is not None and attention_path.is_file():
        attention_image = Image.open(attention_path).convert("L").resize((width, height), Image.Resampling.BILINEAR)
        attention = np.asarray(attention_image).astype(np.uint8)
    else:
        attention = (support * 255).astype(np.uint8)
    attention[support == 0] = 0

    return support, attention, {
        "source": "external_object_support",
        "support_path": str(support_path),
        "attention_path": str(attention_path) if attention_path is not None and attention_path.is_file() else None,
        "erosion": int(erosion),
        "area": int(support.sum()),
        "object_prompt": None,
    }


def mask_bbox_np(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        return None
    return int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1


def paste_shape(shape: np.ndarray, canvas_shape: tuple[int, int], center_y: int, center_x: int) -> np.ndarray:
    canvas_h, canvas_w = canvas_shape
    shape_h, shape_w = shape.shape[:2]
    y0 = int(round(center_y - shape_h / 2))
    x0 = int(round(center_x - shape_w / 2))
    y1 = y0 + shape_h
    x1 = x0 + shape_w
    dst_y0 = max(0, y0)
    dst_x0 = max(0, x0)
    dst_y1 = min(canvas_h, y1)
    dst_x1 = min(canvas_w, x1)
    if dst_y0 >= dst_y1 or dst_x0 >= dst_x1:
        return np.zeros((canvas_h, canvas_w), dtype=np.uint8)
    src_y0 = dst_y0 - y0
    src_x0 = dst_x0 - x0
    src_y1 = src_y0 + (dst_y1 - dst_y0)
    src_x1 = src_x0 + (dst_x1 - dst_x0)
    canvas = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
    canvas[dst_y0:dst_y1, dst_x0:dst_x1] = shape[src_y0:src_y1, src_x0:src_x1]
    return canvas


def crop_nonzero(mask: np.ndarray) -> np.ndarray:
    bbox = mask_bbox_np(mask)
    if bbox is None:
        return (mask > 0).astype(np.uint8)
    y0, y1, x0, x1 = bbox
    return (mask[y0:y1, x0:x1] > 0).astype(np.uint8)


def compact_single_component(mask: np.ndarray, close_iterations: int = 1, fill_holes: bool = True) -> np.ndarray:
    mask_u8 = (mask > 0).astype(np.uint8)
    if int(mask_u8.sum()) <= 0:
        return mask_u8
    if close_iterations > 0:
        mask_u8 = cv2.morphologyEx(
            mask_u8,
            cv2.MORPH_CLOSE,
            np.ones((5, 5), np.uint8),
            iterations=int(close_iterations),
        )
    mask_u8 = largest_component(mask_u8)
    if fill_holes:
        mask_u8 = fill_mask_holes(mask_u8)
    return crop_nonzero(mask_u8)


def rotate_binary_shape(shape: np.ndarray, rng: np.random.Generator, max_rotate: float) -> np.ndarray:
    if max_rotate <= 0 or int(shape.sum()) <= 0:
        return (shape > 0).astype(np.uint8)
    angle = float(rng.uniform(-abs(max_rotate), abs(max_rotate)))
    pad = int(max(shape.shape[:2]) * 0.35) + 8
    padded = np.pad((shape > 0).astype(np.uint8), pad, mode="constant", constant_values=0)
    center = (padded.shape[1] / 2, padded.shape[0] / 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(padded, matrix, (padded.shape[1], padded.shape[0]), flags=cv2.INTER_NEAREST, borderValue=0)
    bbox = mask_bbox_np(rotated)
    if bbox is None:
        return (shape > 0).astype(np.uint8)
    y0, y1, x0, x1 = bbox
    return (rotated[y0:y1, x0:x1] > 0).astype(np.uint8)


def random_blob_shape(target_area: int, rng: np.random.Generator, long_shape: bool = False) -> np.ndarray:
    target_area = max(4, int(target_area))
    if long_shape:
        length = max(16, int(np.sqrt(target_area) * float(rng.uniform(4.0, 8.0))))
        thickness = max(2, int(round(target_area / max(1, length))))
        h = max(thickness * 8, int(length * 0.6))
        w = max(length + thickness * 8, 24)
        shape = np.zeros((h, w), dtype=np.uint8)
        points = []
        y = h // 2 + int(rng.integers(-h // 8, h // 8 + 1))
        for i in range(5):
            x = int(round(i * (w - 1) / 4))
            y += int(rng.integers(-max(2, h // 8), max(3, h // 8 + 1)))
            y = max(thickness * 2, min(h - thickness * 2 - 1, y))
            points.append([x, y])
        cv2.polylines(shape, [np.array(points, dtype=np.int32)], False, 1, thickness=thickness)
        shape = cv2.dilate(shape, np.ones((3, 3), np.uint8), iterations=int(rng.integers(0, 2)))
        return (shape > 0).astype(np.uint8)

    radius = max(4, int(np.sqrt(target_area / np.pi)))
    h = max(16, radius * int(rng.integers(4, 7)))
    w = max(16, radius * int(rng.integers(4, 7)))
    center_y = h / 2
    center_x = w / 2
    n_points = int(rng.integers(9, 16))
    angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    rng.shuffle(angles)
    points = []
    aspect = float(rng.uniform(0.55, 1.8))
    for angle in sorted(angles):
        r = radius * float(rng.uniform(0.65, 1.35))
        x = center_x + np.cos(angle) * r * aspect
        y = center_y + np.sin(angle) * r / max(0.35, aspect)
        points.append([int(round(x)), int(round(y))])
    shape = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(shape, [np.array(points, dtype=np.int32)], 1)
    shape = cv2.morphologyEx(shape, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)
    return (shape > 0).astype(np.uint8)


def random_composite_shape(target_area: int, rng: np.random.Generator, long_shape: bool = False) -> np.ndarray:
    target_area = max(4, int(target_area))
    component_count = int(rng.integers(2, 6))
    weights = rng.dirichlet(np.ones(component_count, dtype=np.float32))
    base = max(24, int(np.sqrt(target_area) * float(rng.uniform(4.0, 7.0))))
    if long_shape:
        canvas_h = max(24, int(base * float(rng.uniform(0.7, 1.2))))
        canvas_w = max(32, int(base * float(rng.uniform(1.4, 2.2))))
    else:
        canvas_h = max(24, int(base * float(rng.uniform(0.9, 1.5))))
        canvas_w = max(24, int(base * float(rng.uniform(0.9, 1.5))))

    canvas = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
    for weight in weights:
        component_area = max(4, int(round(target_area * float(weight))))
        component_is_line = bool(long_shape and rng.random() < 0.65)
        component = random_blob_shape(component_area, rng, long_shape=component_is_line)
        center_y = canvas_h // 2 + int(rng.integers(-max(1, canvas_h // 5), max(2, canvas_h // 5 + 1)))
        center_x = canvas_w // 2 + int(rng.integers(-max(1, canvas_w // 5), max(2, canvas_w // 5 + 1)))
        canvas = np.maximum(canvas, paste_shape(component, (canvas_h, canvas_w), center_y, center_x))

    canvas = cv2.morphologyEx(canvas, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)
    if rng.random() < 0.35:
        canvas = cv2.dilate(canvas, np.ones((3, 3), np.uint8), iterations=1)
    canvas = compact_single_component(canvas, close_iterations=1, fill_holes=True)
    if int(canvas.sum()) <= 0:
        return random_blob_shape(target_area, rng, long_shape=long_shape)
    return canvas


def random_pixel_cluster_shape(target_area: int, rng: np.random.Generator, long_shape: bool = False) -> np.ndarray:
    target_area = max(4, int(target_area))
    base = max(24, int(np.sqrt(target_area) * float(rng.uniform(3.5, 6.5))))
    canvas_h = max(24, int(base * float(rng.uniform(0.8, 1.5))))
    canvas_w = max(24, int(base * float(rng.uniform(0.8, 1.5))))
    if long_shape:
        canvas_h = max(24, int(base * float(rng.uniform(0.55, 1.0))))
        canvas_w = max(32, int(base * float(rng.uniform(1.4, 2.4))))

    score = np.zeros((canvas_h, canvas_w), dtype=np.float32)
    blob_count = int(rng.integers(8, 22))
    for _ in range(blob_count):
        cy = int(rng.integers(0, canvas_h))
        cx = int(rng.integers(0, canvas_w))
        radius = max(2, int(np.sqrt(target_area) * float(rng.uniform(0.08, 0.28))))
        color = float(rng.uniform(0.4, 1.0))
        cv2.circle(score, (cx, cy), radius, color, -1)

    noise = rng.random((canvas_h, canvas_w), dtype=np.float32)
    score += cv2.GaussianBlur(noise, (0, 0), sigmaX=float(rng.uniform(1.0, 3.5))) * float(rng.uniform(0.15, 0.45))
    score = cv2.GaussianBlur(score, (0, 0), sigmaX=float(rng.uniform(1.0, 3.0)))
    keep = min(score.size - 1, max(1, int(round(target_area * float(rng.uniform(0.70, 1.30))))))
    threshold = np.partition(score.reshape(-1), -keep)[-keep]
    shape = (score >= threshold).astype(np.uint8)

    if rng.random() < 0.45:
        shape = cv2.morphologyEx(shape, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    shape = cv2.morphologyEx(shape, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)
    shape = largest_component(shape)

    if rng.random() < 0.35 and int(shape.sum()) > 16:
        hole_count = int(rng.integers(1, 4))
        for _ in range(hole_count):
            ys, xs = np.where(shape > 0)
            if len(ys) == 0:
                break
            idx = int(rng.integers(0, len(ys)))
            radius = max(1, int(np.sqrt(target_area) * float(rng.uniform(0.03, 0.11))))
            cv2.circle(shape, (int(xs[idx]), int(ys[idx])), radius, 0, -1)
        shape = largest_component(shape)

    shape = compact_single_component(shape, close_iterations=1, fill_holes=True)
    if int(shape.sum()) <= 0:
        return random_blob_shape(target_area, rng, long_shape=long_shape)
    return shape


def resize_shape_to_area(shape: np.ndarray, target_area: int) -> np.ndarray:
    current_area = int((shape > 0).sum())
    if current_area <= 0:
        return shape.astype(np.uint8)
    scale = np.sqrt(max(1, target_area) / current_area)
    new_h = max(2, int(round(shape.shape[0] * scale)))
    new_w = max(2, int(round(shape.shape[1] * scale)))
    return (cv2.resize(shape.astype(np.uint8), (new_w, new_h), interpolation=cv2.INTER_NEAREST) > 0).astype(np.uint8)


def reference_shape_profile(ref_mask: np.ndarray) -> tuple[np.ndarray, dict]:
    reference_shape = crop_nonzero(ref_mask)
    shape_area = int(reference_shape.sum())
    shape_h, shape_w = reference_shape.shape[:2]
    short_side = max(1, min(shape_h, shape_w))
    long_side = max(shape_h, shape_w)
    bbox_area = max(1, shape_h * shape_w)
    return reference_shape.astype(np.uint8), {
        "reference_bbox_hw": [int(shape_h), int(shape_w)],
        "reference_aspect": float(long_side / short_side),
        "reference_fill_ratio": float(shape_area / bbox_area),
    }


def random_rough_reference_shape(target_area: int, ref_aspect: float, ref_fill_ratio: float, rng: np.random.Generator) -> tuple[np.ndarray, str]:
    target_area = max(4, int(target_area))
    long_profile = ref_aspect >= 1.8 or ref_fill_ratio <= 0.35
    round_profile = ref_aspect <= 1.35 and ref_fill_ratio >= 0.45

    # All reference profiles can sample both round and rectangular masks.
    # The reference profile only nudges probabilities and aspect ranges.
    if long_profile:
        mode_options = [
            "rectangle", "rectangle", "capsule", "thick_line",
            "circle", "ellipse", "rounded_rectangle", "square", "pixel_cluster", "composite", "blob",
        ]
    elif round_profile:
        mode_options = [
            "circle", "circle", "ellipse", "rounded_rectangle", "square",
            "rectangle", "capsule", "pixel_cluster", "composite", "blob",
        ]
    else:
        mode_options = [
            "circle", "ellipse", "rounded_rectangle", "square", "rectangle",
            "capsule", "pixel_cluster", "composite", "blob",
        ]
    mode = str(rng.choice(mode_options))

    if mode == "rectangle":
        if long_profile:
            aspect = float(rng.uniform(1.6, min(max(ref_aspect * 1.5, 2.2), 5.5)))
        elif round_profile:
            aspect = float(rng.uniform(1.0, 2.2))
        else:
            aspect = float(rng.uniform(1.1, 2.8))
    elif mode in {"capsule", "thick_line"}:
        aspect = float(rng.uniform(1.8, min(max(ref_aspect * 1.6, 2.8), 5.8)))
    elif mode == "square":
        aspect = float(rng.uniform(0.9, 1.1))
    else:
        aspect = float(rng.uniform(0.75, 1.6))

    if mode == "blob":
        return random_blob_shape(target_area, rng, long_shape=False), "primitive_blob"
    if mode == "pixel_cluster":
        return random_pixel_cluster_shape(target_area, rng, long_shape=bool(long_profile and rng.random() < 0.6)), "primitive_pixel_cluster"
    if mode == "composite":
        return random_composite_shape(target_area, rng, long_shape=bool(long_profile and rng.random() < 0.6)), "primitive_composite"

    if mode in {"ellipse", "circle"}:
        if mode == "circle":
            aspect = float(rng.uniform(0.9, 1.1))
        h = max(4, int(round(np.sqrt(target_area / max(np.pi * aspect / 4.0, 1e-6)))))
        w = max(4, int(round(h * aspect)))
        canvas_h = h + 8
        canvas_w = w + 8
        shape = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
        cv2.ellipse(shape, (canvas_w // 2, canvas_h // 2), (max(2, w // 2), max(2, h // 2)), 0, 0, 360, 1, -1)
    elif mode == "capsule":
        h = max(4, int(round(np.sqrt(target_area / max(aspect - 0.22, 0.5)))))
        w = max(h + 2, int(round(h * aspect)))
        canvas_h = h + 8
        canvas_w = w + 8
        shape = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
        radius = max(2, h // 2)
        y = canvas_h // 2
        x0 = 4 + radius
        x1 = canvas_w - 4 - radius
        cv2.rectangle(shape, (x0, y - radius), (x1, y + radius), 1, -1)
        cv2.circle(shape, (x0, y), radius, 1, -1)
        cv2.circle(shape, (x1, y), radius, 1, -1)
    elif mode == "thick_line":
        length = max(12, int(round(np.sqrt(target_area * aspect))))
        thickness = max(3, int(round(target_area / max(length, 1))))
        canvas_h = max(thickness * 5, thickness + 12)
        canvas_w = length + thickness * 4 + 8
        shape = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
        y = canvas_h // 2
        cv2.line(shape, (thickness * 2, y), (canvas_w - thickness * 2, y), 1, thickness=thickness)
    else:
        h = max(4, int(round(np.sqrt(target_area / max(aspect, 1e-6)))))
        w = max(4, int(round(h * aspect)))
        canvas_h = h + 8
        canvas_w = w + 8
        shape = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
        if mode == "rounded_rectangle":
            radius = max(2, min(h, w) // 5)
            x0, y0 = 4, 4
            x1, y1 = canvas_w - 5, canvas_h - 5
            cv2.rectangle(shape, (x0 + radius, y0), (x1 - radius, y1), 1, -1)
            cv2.rectangle(shape, (x0, y0 + radius), (x1, y1 - radius), 1, -1)
            for cx, cy in ((x0 + radius, y0 + radius), (x1 - radius, y0 + radius), (x0 + radius, y1 - radius), (x1 - radius, y1 - radius)):
                cv2.circle(shape, (cx, cy), radius, 1, -1)
        else:
            cv2.rectangle(shape, (4, 4), (canvas_w - 5, canvas_h - 5), 1, -1)

    shape = compact_single_component(shape, close_iterations=1, fill_holes=True)
    if int(shape.sum()) <= 0:
        return random_blob_shape(target_area, rng, long_shape=long_profile), "fallback_blob"
    return shape, f"primitive_{mode}"


def random_target_mask_on_object(
    source_image_path: Path,
    ref_mask_path: Path,
    anomaly: str,
    rng: np.random.Generator,
    min_area_ratio: float,
    max_area_ratio: float,
    max_rotate: float,
    attempts: int,
    support_erosion: int,
    object_prompt: str | None,
    double_component_prob: float = 0.0,
    component_spacing: int = 4,
    object_support_path: Path | None = None,
    object_attention_path: Path | None = None,
    placement_filter: bool = False,
    dark_quantile: float = 20.0,
    max_dark_fraction: float = 0.35,
    edge_quantile: float = 90.0,
    max_edge_fraction: float = 0.30,
    boundary_margin_ratio: float = 0.015,
    max_boundary_fraction: float = 0.03,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    image = load_rgb(source_image_path)
    height, width = image.shape[:2]
    if object_support_path is not None:
        support, attention, support_info = load_object_support_from_file(
            object_support_path,
            object_attention_path,
            size=(width, height),
            erosion=support_erosion,
        )
    else:
        support, attention, support_info = object_support_from_image(image, erosion=support_erosion)
    support_info["object_prompt"] = object_prompt

    ref_mask = load_mask(ref_mask_path, size=(width, height))
    ref_area = max(1, int(ref_mask.sum()))
    _reference_shape, reference_info = reference_shape_profile(ref_mask)
    ref_aspect = float(reference_info["reference_aspect"])
    ref_fill_ratio = float(reference_info["reference_fill_ratio"])
    area_low = max(1, int(round(ref_area * min(min_area_ratio, max_area_ratio))))
    area_high = max(area_low, int(round(ref_area * max(min_area_ratio, max_area_ratio))))
    support_coords = np.column_stack(np.where(support > 0))
    if len(support_coords) == 0:
        support_coords = np.array([[height // 2, width // 2]], dtype=np.int64)

    placement_filter = bool(placement_filter)
    support_bool = support > 0
    if placement_filter:
        lab_l = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)[..., 0].astype(np.float32)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
        grad_x = cv2.Scharr(gray, cv2.CV_32F, 1, 0)
        grad_y = cv2.Scharr(gray, cv2.CV_32F, 0, 1)
        gradient = cv2.GaussianBlur(cv2.magnitude(grad_x, grad_y), (0, 0), sigmaX=3.0, sigmaY=3.0)
        support_l = lab_l[support_bool]
        support_gradient = gradient[support_bool]
        dark_cutoff = float(np.percentile(support_l, np.clip(dark_quantile, 0.0, 100.0)))
        edge_cutoff = float(np.percentile(support_gradient, np.clip(edge_quantile, 0.0, 100.0)))
        support_distance = cv2.distanceTransform(support.astype(np.uint8), cv2.DIST_L2, 5)
        boundary_margin = (
            max(4.0, float(round(min(height, width) * boundary_margin_ratio)))
            if boundary_margin_ratio > 0.0
            else 0.0
        )
    else:
        lab_l = np.zeros((height, width), dtype=np.float32)
        gradient = np.zeros((height, width), dtype=np.float32)
        support_distance = np.full((height, width), np.inf, dtype=np.float32)
        dark_cutoff = 0.0
        edge_cutoff = 0.0
        boundary_margin = 0.0

    def placement_quality(mask: np.ndarray) -> tuple[float, float, float, bool]:
        mask_bool = mask > 0
        if not placement_filter or not mask_bool.any():
            return 0.0, 0.0, 0.0, True
        dark_fraction = float((lab_l[mask_bool] < dark_cutoff).mean())
        edge_fraction = float((gradient[mask_bool] > edge_cutoff).mean())
        boundary_fraction = float((support_distance[mask_bool] < boundary_margin).mean())
        quality_ok = bool(
            dark_fraction <= max_dark_fraction
            and edge_fraction <= max_edge_fraction
            and boundary_fraction <= max_boundary_fraction
        )
        return dark_fraction, edge_fraction, boundary_fraction, quality_ok

    double_component_prob = min(1.0, max(0.0, float(double_component_prob)))
    if float(rng.random()) < double_component_prob:
        requested_components = 2
        component_sampling = "probabilistic_double"
    else:
        requested_components = 1
        component_sampling = "single"
    attempts = max(1, int(attempts))
    component_spacing = max(0, int(component_spacing))
    component_area_low = area_low
    component_area_high = max(component_area_low, area_high)

    final_mask = np.zeros((height, width), dtype=np.uint8)
    component_infos: list[dict] = []
    for component_index in range(requested_components):
        occupied = final_mask > 0
        exclusion_mask = occupied.copy()
        effective_spacing = 0
        if occupied.any():
            available_support = None
            spacing_candidates = []
            if component_spacing > 0:
                spacing_candidates.extend([component_spacing, max(1, component_spacing // 2), 1])
            spacing_candidates.append(0)
            for spacing_radius in dict.fromkeys(spacing_candidates):
                if spacing_radius > 0:
                    kernel = cv2.getStructuringElement(
                        cv2.MORPH_ELLIPSE,
                        (spacing_radius * 2 + 1, spacing_radius * 2 + 1),
                    )
                    candidate_exclusion = cv2.dilate(occupied.astype(np.uint8), kernel, iterations=1) > 0
                else:
                    candidate_exclusion = occupied.copy()
                candidate_support = (support > 0) & ~candidate_exclusion
                if candidate_support.any():
                    available_support = candidate_support
                    exclusion_mask = candidate_exclusion
                    effective_spacing = int(spacing_radius)
                    break
            if available_support is None or not available_support.any():
                available_support = support > 0
                exclusion_mask = occupied.copy()
                effective_spacing = 0
        else:
            available_support = support > 0
        available_coords = np.column_stack(np.where(available_support))
        if len(available_coords) == 0:
            available_coords = support_coords

        best_mask = None
        # Preserve the historical sampler byte-for-byte when the optional
        # placement filter is disabled.  This is the path used by the original
        # T2T experiment and is required for paired T2T/T2R replay.
        best_score = -float("inf") if placement_filter else -1.0
        best_info: dict = {}
        best_quality_mask = None
        best_quality_score = -float("inf")
        best_quality_info: dict = {}
        for attempt in range(1, attempts + 1):
            target_area = int(rng.integers(component_area_low, component_area_high + 1))
            shape, mode = random_rough_reference_shape(
                target_area=target_area,
                ref_aspect=ref_aspect,
                ref_fill_ratio=ref_fill_ratio,
                rng=rng,
            )
            shape = rotate_binary_shape(shape, rng, max_rotate)
            shape = resize_shape_to_area(shape, target_area)
            if int(shape.sum()) <= 0:
                continue

            center_y, center_x = available_coords[int(rng.integers(0, len(available_coords)))]
            candidate = paste_shape(shape, (height, width), int(center_y), int(center_x))
            candidate_area = int(candidate.sum())
            if candidate_area <= 0:
                continue
            overlap_area = int((candidate & support).sum())
            object_overlap = overlap_area / max(1, candidate_area)
            existing_overlap = int((candidate & occupied.astype(np.uint8)).sum()) / max(1, candidate_area)
            spacing_overlap = int((candidate & exclusion_mask.astype(np.uint8)).sum()) / max(1, candidate_area)
            clipped = (candidate & support).astype(np.uint8)
            if occupied.any():
                clipped = (clipped & (~exclusion_mask).astype(np.uint8)).astype(np.uint8)
            clipped = largest_component(clipped)
            clipped_area = int(clipped.sum())
            area_ok = component_area_low <= clipped_area <= component_area_high
            area_penalty = 0 if area_ok else abs(clipped_area - target_area) / max(1, target_area)
            if not placement_filter:
                score = object_overlap - 0.25 * area_penalty - 0.75 * existing_overlap - 0.75 * spacing_overlap
                if score > best_score and clipped_area > 0:
                    best_score = score
                    best_mask = clipped
                    best_info = {
                        "component_index": component_index,
                        "attempt": attempt,
                        "mode": mode,
                        "target_area": target_area,
                        "candidate_area": candidate_area,
                        "clipped_area": clipped_area,
                        "object_overlap": object_overlap,
                        "existing_overlap": existing_overlap,
                        "spacing_overlap": spacing_overlap,
                        "effective_component_spacing": effective_spacing,
                    }
                if object_overlap >= 0.995 and existing_overlap <= 0.01 and spacing_overlap <= 0.01 and area_ok:
                    best_info["status"] = "accepted"
                    break
                continue

            dark_fraction, edge_fraction, boundary_fraction, quality_ok = placement_quality(clipped)
            score = (
                object_overlap
                - 0.25 * area_penalty
                - 0.75 * existing_overlap
                - 0.75 * spacing_overlap
                - 0.40 * dark_fraction
                - 0.15 * edge_fraction
                - 0.15 * boundary_fraction
            )
            candidate_info = {
                "component_index": component_index,
                "attempt": attempt,
                "mode": mode,
                "target_area": target_area,
                "candidate_area": candidate_area,
                "clipped_area": clipped_area,
                "object_overlap": object_overlap,
                "existing_overlap": existing_overlap,
                "spacing_overlap": spacing_overlap,
                "effective_component_spacing": effective_spacing,
                "placement_filter": True,
                "dark_fraction": dark_fraction,
                "edge_fraction": edge_fraction,
                "boundary_fraction": boundary_fraction,
                "quality_ok": quality_ok,
            }
            accepted = bool(
                object_overlap >= 0.995
                and existing_overlap <= 0.01
                and spacing_overlap <= 0.01
                and area_ok
                and quality_ok
            )
            if accepted:
                best_score = score
                best_mask = clipped
                best_info = candidate_info
                best_info["status"] = "accepted"
                break
            if score > best_score and clipped_area > 0:
                best_score = score
                best_mask = clipped
                best_info = candidate_info
            if quality_ok and score > best_quality_score and clipped_area > 0:
                best_quality_score = score
                best_quality_mask = clipped
                best_quality_info = candidate_info

        if best_info.get("status") != "accepted" and placement_filter and best_quality_mask is not None:
            best_mask = best_quality_mask
            best_score = best_quality_score
            best_info = best_quality_info
            best_info["status"] = "best_effort_quality_ok"

        if best_mask is None or int(best_mask.sum()) == 0:
            center_y, center_x = available_coords[int(rng.integers(0, len(available_coords)))]
            best_mask = np.zeros((height, width), dtype=np.uint8)
            radius = max(1, int(np.sqrt(component_area_low / np.pi)))
            cv2.circle(best_mask, (int(center_x), int(center_y)), radius, 1, -1)
            best_mask = (best_mask & support).astype(np.uint8)
            if occupied.any():
                best_mask = (best_mask & (~exclusion_mask).astype(np.uint8)).astype(np.uint8)
            if placement_filter:
                dark_fraction, edge_fraction, boundary_fraction, quality_ok = placement_quality(best_mask)
                best_info = {
                    "component_index": component_index,
                    "status": "fallback_circle" if quality_ok else "fallback_circle_quality_failed",
                    "target_area": component_area_low,
                    "clipped_area": int(best_mask.sum()),
                    "object_overlap": float((best_mask & support).sum() / max(1, int(best_mask.sum()))),
                    "existing_overlap": 0.0,
                    "spacing_overlap": 0.0,
                    "effective_component_spacing": effective_spacing,
                    "placement_filter": True,
                    "dark_fraction": dark_fraction,
                    "edge_fraction": edge_fraction,
                    "boundary_fraction": boundary_fraction,
                    "quality_ok": quality_ok,
                }
            else:
                best_info = {
                    "component_index": component_index,
                    "status": "fallback_circle",
                    "target_area": component_area_low,
                    "clipped_area": int(best_mask.sum()),
                    "object_overlap": float((best_mask & support).sum() / max(1, int(best_mask.sum()))),
                    "existing_overlap": 0.0,
                    "spacing_overlap": 0.0,
                    "effective_component_spacing": effective_spacing,
                }
        else:
            fallback_status = "best_effort"
            if placement_filter and not bool(best_info.get("quality_ok", False)):
                fallback_status = "best_effort_quality_failed"
            best_info.setdefault("status", fallback_status)

        if int(best_mask.sum()) > 0:
            final_mask = np.maximum(final_mask, best_mask.astype(np.uint8))
            component_infos.append(best_info)

    if int(final_mask.sum()) == 0:
        center_y, center_x = support_coords[int(rng.integers(0, len(support_coords)))]
        final_mask = np.zeros((height, width), dtype=np.uint8)
        radius = max(1, int(np.sqrt(area_low / np.pi)))
        cv2.circle(final_mask, (int(center_x), int(center_y)), radius, 1, -1)
        final_mask = (final_mask & support).astype(np.uint8)
        if placement_filter:
            dark_fraction, edge_fraction, boundary_fraction, quality_ok = placement_quality(final_mask)
            component_infos = [{
                "component_index": 0,
                "status": "fallback_circle" if quality_ok else "fallback_circle_quality_failed",
                "target_area": area_low,
                "clipped_area": int(final_mask.sum()),
                "placement_filter": True,
                "dark_fraction": dark_fraction,
                "edge_fraction": edge_fraction,
                "boundary_fraction": boundary_fraction,
                "quality_ok": quality_ok,
            }]
        else:
            component_infos = [{
                "component_index": 0,
                "status": "fallback_circle",
                "target_area": area_low,
                "clipped_area": int(final_mask.sum()),
            }]

    connected_count, _, connected_stats, _ = cv2.connectedComponentsWithStats((final_mask > 0).astype(np.uint8), connectivity=8)
    final_component_count = max(0, int(connected_count) - 1)
    component_areas = [int(connected_stats[label, cv2.CC_STAT_AREA]) for label in range(1, connected_count)]
    first_info = component_infos[0] if component_infos else {}
    info = {
        "target_mask_source": "random_object",
        "shape_source": "reference_profile_random_primitives",
        "reference_shape_used": False,
        "single_connected_component": final_component_count <= 1,
        "component_count_requested": requested_components,
        "component_sampling": component_sampling,
        "double_component_prob": double_component_prob,
        "component_count": final_component_count,
        "placement_count": len(component_infos),
        "component_areas": component_areas,
        "components": component_infos,
        "object_support": support_info,
        "reference_area": ref_area,
        **reference_info,
        "area_min": area_low,
        "area_max": area_high,
        "component_area_min": component_area_low,
        "component_area_max": component_area_high,
        "component_spacing": component_spacing,
        "total_area": int(final_mask.sum()),
        **first_info,
    }
    if placement_filter:
        info["placement_filter"] = True
        info["placement_filter_settings"] = {
            "dark_quantile": float(dark_quantile),
            "max_dark_fraction": float(max_dark_fraction),
            "edge_quantile": float(edge_quantile),
            "max_edge_fraction": float(max_edge_fraction),
            "boundary_margin_ratio": float(boundary_margin_ratio),
            "boundary_margin_pixels": float(boundary_margin),
            "max_boundary_fraction": float(max_boundary_fraction),
            "dark_cutoff_lab_l": float(dark_cutoff),
            "edge_cutoff_scharr": float(edge_cutoff),
        }
    return final_mask.astype(np.uint8), support.astype(np.uint8), attention.astype(np.uint8), info

def prepare_target(
    source_image_path: Path,
    target_mask_path: Path,
    masked_ref_image: np.ndarray,
    size: tuple[int, int],
) -> tuple[Image.Image, Image.Image, np.ndarray, np.ndarray, np.ndarray]:
    tar_image = load_rgb(source_image_path)
    tar_mask = load_target_mask(target_mask_path, size=(tar_image.shape[1], tar_image.shape[0]))

    kernel = np.ones((7, 7), np.uint8)
    tar_mask = cv2.dilate(tar_mask, kernel, iterations=2)

    tar_box_yyxx = get_bbox_from_mask(tar_mask)
    tar_box_yyxx = expand_bbox(tar_mask, tar_box_yyxx, ratio=1.2)
    tar_box_yyxx_crop = expand_bbox(tar_image, tar_box_yyxx, ratio=2)
    tar_box_yyxx_crop = box2squre(tar_image, tar_box_yyxx_crop)
    y1, y2, x1, x2 = tar_box_yyxx_crop

    old_tar_image = tar_image.copy()
    tar_image = tar_image[y1:y2, x1:x2, :]
    tar_mask = tar_mask[y1:y2, x1:x2]
    h1, w1 = tar_image.shape[:2]

    tar_mask = pad_to_square(tar_mask, pad_value=0)
    tar_mask = cv2.resize(tar_mask, size, interpolation=cv2.INTER_NEAREST)

    tar_image = pad_to_square(tar_image, pad_value=255)
    h2, w2 = tar_image.shape[:2]
    tar_image = cv2.resize(tar_image, size)

    diptych_ref_tar = np.concatenate([masked_ref_image, tar_image], axis=1)
    tar_mask = np.stack([tar_mask, tar_mask, tar_mask], -1)
    mask_black = np.zeros_like(tar_image)
    mask_diptych = np.concatenate([mask_black, tar_mask], axis=1)
    mask_diptych[mask_diptych == 1] = 255

    return (
        Image.fromarray(diptych_ref_tar),
        Image.fromarray(mask_diptych.astype(np.uint8)),
        old_tar_image,
        np.array([h1, w1, h2, w2]),
        np.array(tar_box_yyxx_crop),
    )


def resized_binary_mask(mask_path: Path, size: tuple[int, int]) -> Image.Image:
    mask = load_target_mask(mask_path, size=size)
    return Image.fromarray((mask * 255).astype(np.uint8))


def save_overlay(edit_image: Image.Image, mask_image: Image.Image, output_path: Path) -> None:
    base = np.asarray(edit_image.convert("RGB")).astype(np.float32)
    mask = np.asarray(mask_image.convert("L")) > 128
    overlay_color = np.array([255, 0, 0], dtype=np.float32)
    base[mask] = base[mask] * 0.55 + overlay_color * 0.45

    mask_u8 = mask.astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    boundary = cv2.dilate(mask_u8, kernel, iterations=1) ^ cv2.erode(mask_u8, kernel, iterations=1)
    base[boundary.astype(bool)] = np.array([255, 255, 0], dtype=np.float32)
    Image.fromarray(np.clip(base, 0, 255).astype(np.uint8)).save(output_path)


def sample_pairs(source_images: list[Path], masks: list[Path], count: int, seed: int) -> list[tuple[Path, Path]]:
    total_pairs = len(source_images) * len(masks)
    if count > total_pairs:
        raise ValueError(f"Requested {count} pairs, but only {total_pairs} unique source/mask pairs are available.")

    rng = random.Random(seed)
    chosen = rng.sample(range(total_pairs), count)
    pairs = []
    for index in chosen:
        source_index = index // len(masks)
        mask_index = index % len(masks)
        pairs.append((source_images[source_index], masks[mask_index]))
    return pairs


def sample_sources(source_images: list[Path], count: int, seed: int) -> list[Path]:
    rng = random.Random(seed)
    if count <= len(source_images):
        return rng.sample(source_images, count)
    return [source_images[rng.randrange(len(source_images))] for _ in range(count)]


def copy_input_files(sample_dir: Path, source_image: Path, ref_image: Path, ref_mask: Path) -> dict[str, str]:
    paths = {
        "source_copy": sample_dir / "source.png",
        "reference_copy": sample_dir / "reference.png",
        "reference_mask_copy": sample_dir / "reference_mask.png",
    }
    shutil.copy2(source_image, paths["source_copy"])
    shutil.copy2(ref_image, paths["reference_copy"])
    shutil.copy2(ref_mask, paths["reference_mask_copy"])
    return {key: str(path) for key, path in paths.items()}


def run_batch() -> None:
    args = parse_args()
    anomaly_refs = build_anomaly_refs(args.anomalies, args.ref_ids)
    mask_root = Path(args.mask_root)
    source_root = Path(args.source_root)
    ref_image_root = Path(args.ref_image_root)
    ref_mask_root = Path(args.ref_mask_root)
    object_support_root = Path(args.object_support_root) if args.object_support_root else None
    object_attention_root = Path(args.object_attention_root) if args.object_attention_root else None
    out_root = Path(args.out_root)
    size = (args.size, args.size)
    out_root.mkdir(parents=True, exist_ok=True)

    text_log_file = None
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    if args.log_file:
        log_file_path = Path(args.log_file)
        log_file_path.parent.mkdir(parents=True, exist_ok=True)
        text_log_file = log_file_path.open("a", encoding="utf-8", buffering=1)
        sys.stdout = TeeStream(original_stdout, text_log_file)
        sys.stderr = TeeStream(original_stderr, text_log_file)

    started_at = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{started_at}] Starting Insert-Anything batch", flush=True)
    print(f"Output root: {out_root}", flush=True)
    print(f"Anomalies/ref ids: {list(zip(args.anomalies, args.ref_ids))}", flush=True)
    print(f"Samples per anomaly: {args.samples_per_anomaly}", flush=True)
    print(f"Source root: {source_root}", flush=True)
    print(f"Target mask root: {mask_root}", flush=True)
    print(f"Reference image root: {ref_image_root}", flush=True)
    print(f"Reference mask root: {ref_mask_root}", flush=True)
    if args.nunchaku:
        print(
            f"Model: nunchaku precision={args.nunchaku_precision}, "
            f"transformer={args.nunchaku_transformer_path}, "
            f"lora={args.nunchaku_lora_path}, lora_strength={args.nunchaku_lora_strength}",
            flush=True,
        )
    else:
        print(
            "Model: full FLUX, "
            f"flux_fill={args.flux_fill_path}, redux={args.flux_redux_path}, "
            f"lora={args.lora_path}/{args.lora_weight_name}",
            flush=True,
        )
    print(
        f"Offload: sequential_cpu_offload={args.sequential_cpu_offload}, cpu_offload={args.cpu_offload}",
        flush=True,
    )

    source_images = list_images(source_root)
    if not source_images:
        raise FileNotFoundError(f"No source images found under {source_root}")

    pipe, redux = load_pipelines(args)
    lora_runtime_audit = getattr(pipe, "_insert_anything_lora_audit", None)
    generator_device = "cuda" if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu"
    print(
        f"Devices: execution={args.device}, generator={generator_device}, "
        f"cuda_available={torch.cuda.is_available()}, cpu_offload={args.cpu_offload}",
        flush=True,
    )
    log_path = out_root / "run_log.csv"
    csv_log_file = log_path.open("w", newline="", encoding="utf-8")
    log_writer = csv.DictWriter(
        csv_log_file,
        fieldnames=[
            "index",
            "anomaly",
            "ref_id",
            "seed",
            "source_image",
            "target_mask",
            "ref_image",
            "ref_mask",
            "target_mask_source",
            "object_support_image",
            "object_attention_image",
            "reference_image_copy",
            "sample_dir",
            "edit_image",
            "target_mask_image",
            "overlay_image",
            "elapsed_sec",
            "status",
            "error",
        ],
    )
    log_writer.writeheader()

    run_config = vars(args).copy()
    run_config["anomaly_refs"] = [{"anomaly": anomaly, "ref_id": ref_id} for anomaly, ref_id in anomaly_refs]
    run_config["num_source_images"] = len(source_images)
    run_config["lora_runtime_audit"] = lora_runtime_audit
    (out_root / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")

    try:
        for anomaly_offset, (anomaly, ref_id) in enumerate(anomaly_refs):
            anomaly_root = mask_root / anomaly
            mask_dir = anomaly_root / "mask"
            ref_image = resolve_id_file(ref_image_root / anomaly, ref_id)
            ref_mask = resolve_id_file(ref_mask_root / anomaly, ref_id, suffixes=("_mask", ""))
            if args.target_mask_source == "dataset":
                target_masks = list_images(mask_dir)
                if not target_masks:
                    raise FileNotFoundError(f"No masks found under {mask_dir}")
            else:
                target_masks = []

            pair_seed = args.seed + anomaly_offset * 1000003
            if args.target_mask_source == "dataset":
                pairs = sample_pairs(source_images, target_masks, args.samples_per_anomaly, pair_seed)
            else:
                pairs = [(source_image, None) for source_image in sample_sources(source_images, args.samples_per_anomaly, pair_seed)]

            anomaly_dir = out_root / anomaly
            anomaly_dir.mkdir(parents=True, exist_ok=True)
            print(
                f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] "
                f"Preparing anomaly={anomaly} ref_id={ref_id} "
                f"ref_image={ref_image} ref_mask={ref_mask} "
                f"target_mask_source={args.target_mask_source} masks={len(target_masks)}",
                flush=True,
            )

            masked_ref_image = prepare_reference(ref_image, ref_mask, size)
            pipe_prior_output = redux(Image.fromarray(masked_ref_image))

            anomaly_config = vars(args).copy()
            anomaly_config.update(
                {
                    "anomaly": anomaly,
                    "ref_id": ref_id,
                    "ref_image": str(ref_image),
                    "ref_mask": str(ref_mask),
                    "num_source_images": len(source_images),
                    "num_target_masks": len(target_masks),
                }
            )
            (anomaly_dir / "config.json").write_text(json.dumps(anomaly_config, indent=2), encoding="utf-8")

            progress = tqdm(enumerate(pairs, start=args.start_index), total=len(pairs), desc=f"{anomaly}/ref_{ref_id}")
            for sample_index, (source_image, target_mask_path) in progress:
                sample_seed = args.seed + anomaly_offset * 1000003 + sample_index
                sample_dir = anomaly_dir / f"{sample_index:03d}"
                sample_dir.mkdir(parents=True, exist_ok=True)
                output_image = sample_dir / "edit.png"
                output_mask = sample_dir / "mask.png"
                output_overlay = sample_dir / "overlay.png"
                sample_metadata = sample_dir / "metadata.json"
                output_object_support = sample_dir / "object_support.png"
                output_object_attention = sample_dir / "object_attention_map.png"
                generated_mask_info = {"target_mask_source": args.target_mask_source}
                input_copy_paths = {}
                object_support_path = None
                object_attention_path = None

                if output_image.exists() and output_mask.exists() and output_overlay.exists() and not args.overwrite:
                    print(
                        f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] "
                        f"SKIP {anomaly}/{sample_index:03d} existing sample_dir={sample_dir}",
                        flush=True,
                    )
                    continue

                start_time = time.time()
                status = "ok"
                error = ""
                print(
                    f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] "
                    f"START {anomaly}/{sample_index:03d} seed={sample_seed} "
                    f"source_image={source_image.name} "
                    f"target_mask={(target_mask_path.name if target_mask_path else 'random_object')} ref={ref_image.name}",
                    flush=True,
                )
                try:
                    if args.save_input_copies:
                        input_copy_paths = copy_input_files(sample_dir, source_image, ref_image, ref_mask)
                    if args.target_mask_source == "random_object":
                        mask_rng = np.random.default_rng(sample_seed + 300000)
                        if object_support_root is not None:
                            object_support_path = resolve_matching_file(object_support_root, source_image)
                        if object_attention_root is not None:
                            try:
                                object_attention_path = resolve_matching_file(object_attention_root, source_image)
                            except FileNotFoundError:
                                object_attention_path = None
                        generated_mask, object_support, object_attention, generated_mask_info = random_target_mask_on_object(
                            source_image_path=source_image,
                            ref_mask_path=ref_mask,
                            anomaly=anomaly,
                            rng=mask_rng,
                            min_area_ratio=args.random_mask_area_min_ratio,
                            max_area_ratio=args.random_mask_area_max_ratio,
                            max_rotate=args.random_mask_rotate,
                            attempts=args.random_mask_attempts,
                            support_erosion=args.object_support_erosion,
                            object_prompt=args.object_prompt,
                            double_component_prob=args.random_mask_double_prob,
                            component_spacing=args.random_mask_component_spacing,
                            object_support_path=object_support_path,
                            object_attention_path=object_attention_path,
                            placement_filter=args.random_mask_placement_filter,
                            dark_quantile=args.random_mask_dark_quantile,
                            max_dark_fraction=args.random_mask_max_dark_fraction,
                            edge_quantile=args.random_mask_edge_quantile,
                            max_edge_fraction=args.random_mask_max_edge_fraction,
                            boundary_margin_ratio=args.random_mask_boundary_margin_ratio,
                            max_boundary_fraction=args.random_mask_max_boundary_fraction,
                        )
                        target_mask_path = sample_dir / "generated_target_mask.png"
                        Image.fromarray((generated_mask * 255).astype(np.uint8)).save(target_mask_path)
                        Image.fromarray((object_support * 255).astype(np.uint8)).save(output_object_support)
                        Image.fromarray(object_attention.astype(np.uint8)).save(output_object_attention)

                    diptych_ref_tar, mask_diptych, old_tar_image, extra_sizes, crop_box = prepare_target(
                        source_image,
                        target_mask_path,
                        masked_ref_image,
                        size,
                    )
                    if args.save_debug_first and sample_index == args.start_index:
                        Image.fromarray(masked_ref_image).save(sample_dir / "debug_masked_reference.png")
                        diptych_ref_tar.save(sample_dir / "debug_diptych_input.png")
                        mask_diptych.save(sample_dir / "debug_diptych_mask.png")

                    generator = torch.Generator(generator_device).manual_seed(sample_seed)
                    pipe_kwargs = {
                        "image": diptych_ref_tar,
                        "mask_image": mask_diptych,
                        "height": mask_diptych.size[1],
                        "width": mask_diptych.size[0],
                        "max_sequence_length": args.max_sequence_length,
                        "generator": generator,
                        "num_inference_steps": args.num_inference_steps,
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
                    edited_pil.save(output_image)

                    target_mask = resized_binary_mask(target_mask_path, edited_pil.size)
                    target_mask.save(output_mask)
                    save_overlay(edited_pil, target_mask, output_overlay)
                except Exception as exc:
                    status = "error"
                    error = repr(exc)

                elapsed = time.time() - start_time
                print(
                    f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] "
                    f"{status.upper()} {anomaly}/{sample_index:03d} elapsed={elapsed:.2f}s "
                    f"edit={output_image} overlay={output_overlay}",
                    flush=True,
                )
                metadata = {
                    "index": sample_index,
                    "anomaly": anomaly,
                    "ref_id": ref_id,
                    "seed": sample_seed,
                    "source_image": str(source_image),
                    "target_mask": str(target_mask_path),
                    "ref_image": str(ref_image),
                    "ref_mask": str(ref_mask),
                    "target_mask_source": args.target_mask_source,
                    "generated_mask_info": generated_mask_info,
                    "lora_runtime_audit": lora_runtime_audit,
                    "object_support_image": str(output_object_support) if output_object_support.exists() else None,
                    "object_attention_image": str(output_object_attention) if output_object_attention.exists() else None,
                    **input_copy_paths,
                    "edit_image": str(output_image),
                    "target_mask_image": str(output_mask),
                    "overlay_image": str(output_overlay),
                    "elapsed_sec": elapsed,
                    "status": status,
                    "error": error,
                }
                sample_metadata.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
                log_writer.writerow(
                    {
                        "index": sample_index,
                        "anomaly": anomaly,
                        "ref_id": ref_id,
                        "seed": sample_seed,
                        "source_image": str(source_image),
                        "target_mask": str(target_mask_path),
                        "ref_image": str(ref_image),
                        "ref_mask": str(ref_mask),
                        "target_mask_source": args.target_mask_source,
                        "object_support_image": str(output_object_support) if output_object_support.exists() else "",
                        "object_attention_image": str(output_object_attention) if output_object_attention.exists() else "",
                        "reference_image_copy": input_copy_paths.get("reference_copy", ""),
                        "sample_dir": str(sample_dir),
                        "edit_image": str(output_image),
                        "target_mask_image": str(output_mask),
                        "overlay_image": str(output_overlay),
                        "elapsed_sec": f"{elapsed:.4f}",
                        "status": status,
                        "error": error,
                    }
                )
                csv_log_file.flush()
                if status != "ok":
                    raise RuntimeError(error)
    finally:
        csv_log_file.close()
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Finished Insert-Anything batch", flush=True)
        if text_log_file is not None:
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            text_log_file.close()


if __name__ == "__main__":
    run_batch()
