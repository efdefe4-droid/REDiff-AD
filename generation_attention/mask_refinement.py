from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


OUTPUT_MASK_NAME = "refined_mask.png"
ACTIVE_CONTOUR_OUTPUT_MASK_NAME = "active_contour_mask.png"
Q80_APPEARANCE_OUTPUT_MASK_NAME = "q80_appearance_mask.png"
CONTOUR_REFINE_OUTPUT_MASK_NAME = "contour_refined_mask.png"
PAMR_ITER = int(os.environ.get("PAMR_ITER", "5000"))
PAMR_SEED_THRESHOLD = float(os.environ.get("PAMR_SEED_THRESHOLD", "0.30"))
PAMR_THRESHOLD = float(os.environ.get("PAMR_THRESHOLD", "0.80"))
PAMR_MAX_SIDE = 512
MGAC_ITER = int(os.environ.get("MGAC_ITER", "120"))
MGAC_SMOOTHING = int(os.environ.get("MGAC_SMOOTHING", "0"))
MGAC_BALLOON = float(os.environ.get("MGAC_BALLOON", "0.0"))
MGAC_ROI_DILATE = int(os.environ.get("MGAC_ROI_DILATE", "17"))
MGAC_EDGE_ALPHA = float(os.environ.get("MGAC_EDGE_ALPHA", "5.0"))
MGAC_SCHARR_WEIGHT = float(os.environ.get("MGAC_SCHARR_WEIGHT", "0.55"))
MGAC_WAVELET_WEIGHT = float(os.environ.get("MGAC_WAVELET_WEIGHT", "0.45"))
MGAC_GATE_PERCENTILE = float(os.environ.get("MGAC_GATE_PERCENTILE", "70"))
MGAC_GATE_DILATE = int(os.environ.get("MGAC_GATE_DILATE", "2"))
MGAC_INIT_ERODE = int(os.environ.get("MGAC_INIT_ERODE", "2"))
MGAC_KEEP_COARSE = os.environ.get("MGAC_KEEP_COARSE", "0").lower() not in {"0", "false", "no"}
MGAC_USE_EDGE_GATE_AS_ROI = os.environ.get("MGAC_USE_EDGE_GATE_AS_ROI", "0").lower() not in {"0", "false", "no"}
MGAC_FINAL_CLOSE = int(os.environ.get("MGAC_FINAL_CLOSE", "0"))
MGAC_FINAL_MIN_AREA = int(os.environ.get("MGAC_FINAL_MIN_AREA", "80"))
MGAC_FINAL_FILL_HOLES = os.environ.get("MGAC_FINAL_FILL_HOLES", "0").lower() not in {"0", "false", "no"}
MGAC_OUTPUT_MODE = os.environ.get("MGAC_OUTPUT_MODE", "mgac").lower()
Q80_APPEARANCE_PERCENTILE = float(os.environ.get("Q80_APPEARANCE_PERCENTILE", "80.0"))
Q80_APPEARANCE_GATE_DILATE = int(os.environ.get("Q80_APPEARANCE_GATE_DILATE", "2"))
Q80_APPEARANCE_MIN_AREA = int(os.environ.get("Q80_APPEARANCE_MIN_AREA", "150"))
Q80_APPEARANCE_FG_ERODE = int(os.environ.get("Q80_APPEARANCE_FG_ERODE", "1"))
Q80_APPEARANCE_BG_RING_DILATE = int(os.environ.get("Q80_APPEARANCE_BG_RING_DILATE", "8"))
Q80_APPEARANCE_KEEP_MARGIN = float(os.environ.get("Q80_APPEARANCE_KEEP_MARGIN", "0.25"))
Q80_APPEARANCE_GROW_RADIUS = int(os.environ.get("Q80_APPEARANCE_GROW_RADIUS", "1"))
Q80_APPEARANCE_ADD_MARGIN = float(os.environ.get("Q80_APPEARANCE_ADD_MARGIN", "0.15"))
Q80_APPEARANCE_MAX_FG_DIST = float(os.environ.get("Q80_APPEARANCE_MAX_FG_DIST", "3.0"))
Q80_APPEARANCE_ROI_DILATE = int(os.environ.get("Q80_APPEARANCE_ROI_DILATE", str(MGAC_ROI_DILATE)))
CONTOUR_REFINE_INNER_ERODE = int(os.environ.get("CONTOUR_REFINE_INNER_ERODE", "1"))
CONTOUR_REFINE_CLIP_TO_COARSE = os.environ.get("CONTOUR_REFINE_CLIP_TO_COARSE", "1").lower() not in {"0", "false", "no"}
CONTOUR_REFINE_EDGE_DILATE = int(os.environ.get("CONTOUR_REFINE_EDGE_DILATE", "2"))
CONTOUR_REFINE_CLOSE = int(os.environ.get("CONTOUR_REFINE_CLOSE", "1"))
CONTOUR_REFINE_FILL_HOLES = os.environ.get("CONTOUR_REFINE_FILL_HOLES", "1").lower() not in {"0", "false", "no"}
CONTOUR_REFINE_COMPONENT_MODE = os.environ.get("CONTOUR_REFINE_COMPONENT_MODE", "all").lower()
SAVE_ACTIVE_CONTOUR_MASK = os.environ.get("SAVE_ACTIVE_CONTOUR_MASK", "0").lower() not in {"0", "false", "no"}
SAVE_ACTIVE_CONTOUR_DEBUG = os.environ.get("SAVE_ACTIVE_CONTOUR_DEBUG", "0").lower() not in {"0", "false", "no"}
SAVE_ACTIVE_CONTOUR_EDGE_MAP = os.environ.get("SAVE_ACTIVE_CONTOUR_EDGE_MAP", "1").lower() not in {"0", "false", "no"}
EPS = 1e-6


class LocalAffinity(nn.Module):
    def __init__(self, dilations: tuple[int, ...] = (1,)) -> None:
        super().__init__()
        self.dilations = dilations
        self.register_buffer("kernel", self._init_aff())

    def _init_aff(self) -> torch.Tensor:
        weight = torch.zeros(8, 1, 3, 3)
        for i in range(weight.size(0)):
            weight[i, 0, 1, 1] = 1
        weight[0, 0, 0, 0] = -1
        weight[1, 0, 0, 1] = -1
        weight[2, 0, 0, 2] = -1
        weight[3, 0, 1, 0] = -1
        weight[4, 0, 1, 2] = -1
        weight[5, 0, 2, 0] = -1
        weight[6, 0, 2, 1] = -1
        weight[7, 0, 2, 2] = -1
        return weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.size()
        x = x.reshape(batch * channels, 1, height, width).float()
        outputs = []
        for dilation in self.dilations:
            padded = F.pad(x, [dilation] * 4, mode="replicate")
            outputs.append(F.conv2d(padded, self.kernel, dilation=dilation))
        affinity = torch.cat(outputs, dim=1)
        return affinity.reshape(batch, channels, -1, height, width)


class LocalAffinityCopy(LocalAffinity):
    def _init_aff(self) -> torch.Tensor:
        weight = torch.zeros(8, 1, 3, 3)
        weight[0, 0, 0, 0] = 1
        weight[1, 0, 0, 1] = 1
        weight[2, 0, 0, 2] = 1
        weight[3, 0, 1, 0] = 1
        weight[4, 0, 1, 2] = 1
        weight[5, 0, 2, 0] = 1
        weight[6, 0, 2, 1] = 1
        weight[7, 0, 2, 2] = 1
        return weight


class LocalStDev(LocalAffinity):
    def _init_aff(self) -> torch.Tensor:
        weight = torch.zeros(9, 1, 3, 3)
        weight[0, 0, 0, 0] = 1
        weight[1, 0, 0, 1] = 1
        weight[2, 0, 0, 2] = 1
        weight[3, 0, 1, 0] = 1
        weight[4, 0, 1, 1] = 1
        weight[5, 0, 1, 2] = 1
        weight[6, 0, 2, 0] = 1
        weight[7, 0, 2, 1] = 1
        weight[8, 0, 2, 2] = 1
        return weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x).std(2, keepdim=True)


class LocalAffinityAbs(LocalAffinity):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.abs(super().forward(x))


class PAMR(nn.Module):
    def __init__(self, num_iter: int = 1, dilations: tuple[int, ...] = (1,)) -> None:
        super().__init__()
        self.num_iter = int(num_iter)
        self.aff_x = LocalAffinityAbs(dilations)
        self.aff_m = LocalAffinityCopy(dilations)
        self.aff_std = LocalStDev(dilations)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x_std = self.aff_std(x)
        affinity = -self.aff_x(x) / (1e-8 + 0.2 * x_std)
        affinity = F.softmax(affinity.mean(1, keepdim=True), dim=2)
        for _ in range(max(1, self.num_iter)):
            mask = (self.aff_m(mask) * affinity).sum(2)
        return mask


def normalize01(values: np.ndarray) -> np.ndarray:
    arr = np.nan_to_num(values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    lo = float(arr.min())
    hi = float(arr.max())
    if hi - lo < 1e-8:
        return np.clip(arr, 0.0, 1.0).astype(np.float32)
    return ((arr - lo) / (hi - lo)).astype(np.float32)


def resize_bool(mask: np.ndarray, shape_hw: tuple[int, int]) -> np.ndarray:
    mask = mask.astype(bool)
    if mask.shape[:2] == shape_hw:
        return mask
    return cv2.resize(mask.astype(np.uint8), (shape_hw[1], shape_hw[0]), interpolation=cv2.INTER_NEAREST) > 0


def resize_score(score: np.ndarray, shape_hw: tuple[int, int]) -> np.ndarray:
    score = score.astype(np.float32)
    if score.shape[:2] == shape_hw:
        return score
    return cv2.resize(score, (shape_hw[1], shape_hw[0]), interpolation=cv2.INTER_CUBIC)


def prepare_object_mask(object_mask: np.ndarray | None, shape_hw: tuple[int, int]) -> np.ndarray:
    if object_mask is None:
        return np.ones(shape_hw, dtype=bool)
    return resize_bool(object_mask, shape_hw)



def save_score_image(score: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((normalize01(score) * 255.0).clip(0, 255).astype(np.uint8), mode="L").save(out_path)


def dilate_binary(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0 or not mask.any():
        return mask.astype(bool)
    k = max(3, int(radius) * 2 + 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)


def erode_binary(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0 or not mask.any():
        return mask.astype(bool)
    k = max(3, int(radius) * 2 + 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.erode(mask.astype(np.uint8), kernel, iterations=1).astype(bool)



def close_binary(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0 or not mask.any():
        return mask.astype(bool)
    k = max(3, int(radius) * 2 + 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel, iterations=1).astype(bool)


def fill_holes_binary(mask: np.ndarray) -> np.ndarray:
    mask_u8 = mask.astype(np.uint8)
    if mask_u8.size == 0 or mask_u8.max() == 0:
        return mask.astype(bool)
    padded = np.pad(mask_u8, ((1, 1), (1, 1)), mode="constant", constant_values=0)
    flood = padded.copy()
    h, w = padded.shape[:2]
    flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    cv2.floodFill(flood, flood_mask, (0, 0), 1)
    holes = flood[1:-1, 1:-1] == 0
    return (mask_u8.astype(bool) | holes).astype(bool)


def connected_component_count(mask: np.ndarray) -> int:
    if not mask.any():
        return 0
    num_labels, _ = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    return max(0, int(num_labels) - 1)


def keep_relevant_components(mask: np.ndarray, coarse_seed: np.ndarray, min_area: int) -> np.ndarray:
    if not mask.any():
        return mask.astype(bool)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if num_labels <= 1:
        return mask.astype(bool)

    kept = np.zeros(mask.shape, dtype=bool)
    largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        component = labels == label
        overlaps_seed = bool((component & coarse_seed).any())
        if label == largest_label or (area >= min_area and overlaps_seed):
            kept |= component
    return kept


def largest_component_mask(mask: np.ndarray) -> np.ndarray:
    if not mask.any():
        return mask.astype(bool)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if num_labels <= 1:
        return mask.astype(bool)
    largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == largest_label


def lab_features(image_rgb: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB).astype(np.float32) / 255.0
    return lab.reshape(-1, 3)


def diagonal_model(features: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = features[mask.reshape(-1)]
    if values.size == 0:
        return np.zeros(features.shape[1], dtype=np.float32), np.ones(features.shape[1], dtype=np.float32)
    return values.mean(axis=0).astype(np.float32), (values.var(axis=0).astype(np.float32) + 1e-4)


def diagonal_distance(
    features: np.ndarray,
    mean: np.ndarray,
    var: np.ndarray,
    shape_hw: tuple[int, int],
) -> np.ndarray:
    diff = features - mean[None, :]
    dist = np.sqrt(np.sum((diff * diff) / (var[None, :] + EPS), axis=1))
    return dist.reshape(shape_hw).astype(np.float32)


def filter_components_by_appearance(
    mask: np.ndarray,
    fg_dist: np.ndarray,
    bg_dist: np.ndarray,
    *,
    min_area: int,
    keep_margin: float,
) -> np.ndarray:
    if not mask.any():
        return mask.astype(bool)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if num_labels <= 1:
        return mask.astype(bool)

    largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    kept = np.zeros(mask.shape, dtype=bool)
    for label in range(1, num_labels):
        component = labels == label
        area = int(stats[label, cv2.CC_STAT_AREA])
        fg_mean = float(fg_dist[component].mean())
        bg_mean = float(bg_dist[component].mean())
        if label == largest_label or (area >= min_area and fg_mean <= bg_mean + keep_margin):
            kept |= component
    return kept


def fill_external_contours(mask: np.ndarray) -> np.ndarray:
    if not mask.any():
        return mask.astype(bool)
    contours, _ = cv2.findContours((mask.astype(np.uint8) * 255), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros(mask.shape, dtype=np.uint8)
    if contours:
        cv2.drawContours(filled, contours, -1, 1, thickness=-1)
    return filled.astype(bool)


def erode_components_preserve_thin(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0 or not mask.any():
        return mask.astype(bool)
    num_labels, labels, _, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if num_labels <= 1:
        return mask.astype(bool)
    protected = np.zeros(mask.shape, dtype=bool)
    for label in range(1, num_labels):
        component = labels == label
        eroded = erode_binary(component, radius)
        protected |= eroded if eroded.any() else component
    return protected


def haar_wavelet_high_frequency(gray: np.ndarray) -> tuple[np.ndarray, str]:
    try:
        import pywt  # type: ignore

        coeffs = pywt.wavedec2(gray.astype(np.float32), "haar", level=1)
        low, (detail_h, detail_v, detail_d) = coeffs
        high = np.sqrt(detail_h * detail_h + detail_v * detail_v + detail_d * detail_d)
        high_up = cv2.resize(high, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_CUBIC)
        low_up = cv2.resize(low, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_CUBIC)
        residual = np.abs(gray - low_up)
        return normalize01(high_up + residual), "pywavelets_haar_level1"
    except Exception:
        h, w = gray.shape[:2]
        pad_h = h % 2
        pad_w = w % 2
        padded = np.pad(gray.astype(np.float32), ((0, pad_h), (0, pad_w)), mode="reflect")
        a = padded[0::2, 0::2]
        b = padded[0::2, 1::2]
        c = padded[1::2, 0::2]
        d = padded[1::2, 1::2]
        detail_h = (a + b - c - d) * 0.5
        detail_v = (a - b + c - d) * 0.5
        detail_d = (a - b - c + d) * 0.5
        high = np.sqrt(detail_h * detail_h + detail_v * detail_v + detail_d * detail_d)
        high_up = cv2.resize(high, (padded.shape[1], padded.shape[0]), interpolation=cv2.INTER_CUBIC)
        return normalize01(high_up[:h, :w]), "manual_haar_level1"


def edited_image_edge_maps(image_rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    scharr_x = cv2.Scharr(gray, cv2.CV_32F, 1, 0)
    scharr_y = cv2.Scharr(gray, cv2.CV_32F, 0, 1)
    scharr = normalize01(np.sqrt(scharr_x * scharr_x + scharr_y * scharr_y))

    wavelet_high, wavelet_method = haar_wavelet_high_frequency(gray)

    edge_strength = normalize01(MGAC_SCHARR_WEIGHT * scharr + MGAC_WAVELET_WEIGHT * wavelet_high)
    edge_stopping = 1.0 / (1.0 + MGAC_EDGE_ALPHA * edge_strength)
    return (
        edge_stopping.astype(np.float32),
        edge_strength.astype(np.float32),
        scharr.astype(np.float32),
        wavelet_high.astype(np.float32),
        wavelet_method,
    )


def edge_gate_from_strength(
    edge_strength: np.ndarray,
    coarse_roi: np.ndarray,
    *,
    gate_percentile: float = MGAC_GATE_PERCENTILE,
    gate_dilate: int = MGAC_GATE_DILATE,
) -> tuple[np.ndarray, np.ndarray, float]:
    edge_u8 = (normalize01(edge_strength) * 255.0).clip(0, 255).astype(np.uint8)
    roi_values = edge_u8[coarse_roi]
    if roi_values.size > 0 and 0.0 < gate_percentile < 100.0:
        gate_threshold = float(np.percentile(roi_values, gate_percentile))
        edge_gate = (edge_u8 > gate_threshold) & coarse_roi
        edge_gate = dilate_binary(edge_gate, gate_dilate) & coarse_roi
    else:
        gate_threshold = 0.0
        edge_gate = coarse_roi.copy()
    return edge_gate.astype(bool), edge_u8, gate_threshold


def refine_edge_gate_mask(
    *,
    image_rgb: np.ndarray,
    coarse_mask: np.ndarray,
    object_mask: np.ndarray | None,
    out_dir: Path,
    object_source: str,
    output_name: str = "mask.png",
    roi_dilate: int = MGAC_ROI_DILATE,
    gate_percentile: float = MGAC_GATE_PERCENTILE,
    gate_dilate: int = MGAC_GATE_DILATE,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    shape_hw = tuple(image_rgb.shape[:2])
    object_mask_full = prepare_object_mask(object_mask, shape_hw)
    coarse_seed = resize_bool(coarse_mask, shape_hw) & object_mask_full
    coarse_roi = dilate_binary(coarse_seed, roi_dilate) & object_mask_full

    _, edge_strength, scharr, wavelet_high, wavelet_method = edited_image_edge_maps(image_rgb)
    edge_gate, edge_u8, gate_threshold = edge_gate_from_strength(
        edge_strength,
        coarse_roi,
        gate_percentile=gate_percentile,
        gate_dilate=gate_dilate,
    )

    output_path = out_dir / output_name
    Image.fromarray((edge_gate.astype(np.uint8) * 255), mode="L").save(output_path)
    edge_path = out_dir / "edge_map.png"
    scharr_path = out_dir / "scharr.png"
    wavelet_path = out_dir / "wavelet_highfreq.png"
    roi_path = out_dir / "coarse_roi.png"
    Image.fromarray(edge_u8, mode="L").save(edge_path)
    save_score_image(scharr, scharr_path)
    save_score_image(wavelet_high, wavelet_path)
    Image.fromarray((coarse_roi.astype(np.uint8) * 255), mode="L").save(roi_path)
    return {
        "output_mask": str(output_path),
        "object_source": object_source,
        "seed_source": "coarse_roi_u8_edge_gate",
        "mask_area": int(edge_gate.sum()),
        "coarse_area": int(coarse_seed.sum()),
        "roi_area": int(coarse_roi.sum()),
        "object_area": int(object_mask_full.sum()),
        "component_count": connected_component_count(edge_gate),
        "edge_map": str(edge_path),
        "scharr_map": str(scharr_path),
        "wavelet_highfreq_map": str(wavelet_path),
        "wavelet_method": wavelet_method,
        "roi_mask": str(roi_path),
        "roi_dilate": roi_dilate,
        "gate_percentile": gate_percentile,
        "gate_dilate": gate_dilate,
        "gate_threshold_u8": gate_threshold,
        "threshold_rule": "edge_u8 > percentile(edge_u8[coarse_roi])",
    }


def refine_q80_appearance_mask(
    *,
    image_rgb: np.ndarray,
    coarse_mask: np.ndarray,
    object_mask: np.ndarray | None,
    out_dir: Path,
    object_source: str,
    output_name: str = Q80_APPEARANCE_OUTPUT_MASK_NAME,
    edge_map_u8: np.ndarray | None = None,
    coarse_roi_mask: np.ndarray | None = None,
    roi_dilate: int = Q80_APPEARANCE_ROI_DILATE,
    percentile: float = Q80_APPEARANCE_PERCENTILE,
    gate_dilate: int = Q80_APPEARANCE_GATE_DILATE,
    min_area: int = Q80_APPEARANCE_MIN_AREA,
    fg_erode: int = Q80_APPEARANCE_FG_ERODE,
    bg_ring_dilate: int = Q80_APPEARANCE_BG_RING_DILATE,
    keep_margin: float = Q80_APPEARANCE_KEEP_MARGIN,
    grow_radius: int = Q80_APPEARANCE_GROW_RADIUS,
    add_margin: float = Q80_APPEARANCE_ADD_MARGIN,
    max_fg_dist: float = Q80_APPEARANCE_MAX_FG_DIST,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    shape_hw = tuple(image_rgb.shape[:2])
    object_mask_full = prepare_object_mask(object_mask, shape_hw)
    coarse_seed = resize_bool(coarse_mask, shape_hw) & object_mask_full
    if coarse_roi_mask is None:
        coarse_roi = dilate_binary(coarse_seed, roi_dilate) & object_mask_full
    else:
        coarse_roi = resize_bool(coarse_roi_mask, shape_hw) & object_mask_full

    if edge_map_u8 is None:
        _, edge_strength, _, _, wavelet_method = edited_image_edge_maps(image_rgb)
        q80_seed, edge_u8, q80_threshold = edge_gate_from_strength(
            edge_strength,
            coarse_roi,
            gate_percentile=percentile,
            gate_dilate=gate_dilate,
        )
        edge_source = "computed_scharr_wavelet"
    else:
        edge_u8 = edge_map_u8
        if edge_u8.ndim == 3:
            edge_u8 = cv2.cvtColor(edge_u8.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        edge_u8 = edge_u8.astype(np.uint8)
        if edge_u8.shape[:2] != shape_hw:
            edge_u8 = cv2.resize(edge_u8, (shape_hw[1], shape_hw[0]), interpolation=cv2.INTER_NEAREST)
        roi_values = edge_u8[coarse_roi]
        if roi_values.size > 0 and 0.0 < percentile < 100.0:
            q80_threshold = float(np.percentile(roi_values, percentile))
            q80_seed = (edge_u8 > q80_threshold) & coarse_roi
            q80_seed = dilate_binary(q80_seed, gate_dilate) & coarse_roi
        else:
            q80_threshold = 0.0
            q80_seed = coarse_roi.copy()
        wavelet_method = "precomputed_edge_map"
        edge_source = "precomputed_edge_map_u8"

    fg_seed = largest_component_mask(q80_seed)
    eroded_fg = erode_binary(fg_seed, fg_erode)
    if eroded_fg.any():
        fg_seed = eroded_fg

    bg_seed = coarse_roi & ~dilate_binary(q80_seed, bg_ring_dilate)
    if not bg_seed.any():
        bg_seed = object_mask_full & ~dilate_binary(q80_seed, bg_ring_dilate)
    if not bg_seed.any():
        bg_seed = ~dilate_binary(q80_seed, bg_ring_dilate)

    features = lab_features(image_rgb)
    fg_mean, fg_var = diagonal_model(features, fg_seed)
    bg_mean, bg_var = diagonal_model(features, bg_seed)
    fg_dist = diagonal_distance(features, fg_mean, fg_var, shape_hw)
    bg_dist = diagonal_distance(features, bg_mean, bg_var, shape_hw)

    cleaned = filter_components_by_appearance(
        q80_seed,
        fg_dist,
        bg_dist,
        min_area=min_area,
        keep_margin=keep_margin,
    ) & coarse_roi
    candidate = dilate_binary(cleaned, grow_radius) & coarse_roi
    appearance_add = candidate & ~cleaned & (fg_dist + add_margin < bg_dist) & (fg_dist <= max_fg_dist)
    refined_mask = cleaned | appearance_add
    refined_mask = filter_components_by_appearance(
        refined_mask,
        fg_dist,
        bg_dist,
        min_area=min_area,
        keep_margin=keep_margin,
    ) & coarse_roi

    output_path = out_dir / output_name
    Image.fromarray((refined_mask.astype(np.uint8) * 255), mode="L").save(output_path)
    return {
        "enabled": True,
        "method": "q80_edge_seed_lab_appearance_g1",
        "output_mask": str(output_path),
        "object_source": object_source,
        "seed_source": "edited_image_scharr_wavelet_q80_inside_coarse_roi",
        "edge_source": edge_source,
        "mask_area": int(refined_mask.sum()),
        "coarse_area": int(coarse_seed.sum()),
        "roi_area": int(coarse_roi.sum()),
        "object_area": int(object_mask_full.sum()),
        "q80_area": int(q80_seed.sum()),
        "appearance_added_area": int(appearance_add.sum()),
        "q80_component_count": connected_component_count(q80_seed),
        "component_count": connected_component_count(refined_mask),
        "wavelet_method": wavelet_method,
        "roi_dilate": roi_dilate,
        "percentile": percentile,
        "gate_dilate": gate_dilate,
        "q80_threshold_u8": q80_threshold,
        "min_area": min_area,
        "fg_erode": fg_erode,
        "bg_ring_dilate": bg_ring_dilate,
        "keep_margin": keep_margin,
        "grow_radius": grow_radius,
        "add_margin": add_margin,
        "max_fg_dist": max_fg_dist,
        "threshold_rule": "edge_u8 > percentile(edge_u8[coarse_roi])",
    }


def refine_boundary_preserve_mask(
    *,
    image_rgb: np.ndarray,
    coarse_mask: np.ndarray,
    edge_refined_mask: np.ndarray,
    object_mask: np.ndarray | None,
    out_dir: Path,
    object_source: str,
    coarse_roi_mask: np.ndarray | None = None,
    output_name: str = CONTOUR_REFINE_OUTPUT_MASK_NAME,
    inner_erode: int = CONTOUR_REFINE_INNER_ERODE,
    roi_dilate: int = Q80_APPEARANCE_ROI_DILATE,
    clip_to_coarse: bool = CONTOUR_REFINE_CLIP_TO_COARSE,
    edge_dilate: int = CONTOUR_REFINE_EDGE_DILATE,
    close_radius: int = CONTOUR_REFINE_CLOSE,
    fill_holes: bool = CONTOUR_REFINE_FILL_HOLES,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    shape_hw = tuple(image_rgb.shape[:2])
    object_mask_full = prepare_object_mask(object_mask, shape_hw)
    coarse_seed = resize_bool(coarse_mask, shape_hw) & object_mask_full
    edge_refined = resize_bool(edge_refined_mask, shape_hw) & object_mask_full
    if coarse_roi_mask is None:
        coarse_roi = dilate_binary(coarse_seed, roi_dilate) & object_mask_full
    else:
        coarse_roi = resize_bool(coarse_roi_mask, shape_hw) & object_mask_full

    if clip_to_coarse:
        clip_mask = coarse_seed
    else:
        clip_mask = coarse_roi

    edge_seed = edge_refined & clip_mask
    edge_support = np.zeros(shape_hw, dtype=bool)
    refined_mask = np.zeros(shape_hw, dtype=bool)

    if coarse_seed.any():
        num_labels, labels, _, _ = cv2.connectedComponentsWithStats(coarse_seed.astype(np.uint8), connectivity=8)
        for label in range(1, num_labels):
            component = labels == label
            component_seed = edge_seed & component
            if not component_seed.any():
                refined_mask |= component
                continue

            component_support = dilate_binary(component_seed, edge_dilate) & component & object_mask_full
            component_support = close_binary(component_support, close_radius) & component & object_mask_full
            component_outer = fill_external_contours(component_support) & component & object_mask_full
            if fill_holes:
                component_outer = fill_holes_binary(component_outer) & component & object_mask_full
            edge_support |= component_support
            refined_mask |= component_outer
        refined_mask = refined_mask & clip_mask & object_mask_full
    else:
        refined_mask = coarse_seed.copy()

    protected_core = np.zeros(shape_hw, dtype=bool)
    boundary_band = clip_mask.copy()
    boundary_refined = refined_mask.copy()

    raw_component_count = connected_component_count(refined_mask)
    raw_output_path: Path | None = None
    if CONTOUR_REFINE_COMPONENT_MODE == "largest" and raw_component_count > 1:
        raw_output_path = out_dir / f"{Path(output_name).stem}_raw{Path(output_name).suffix}"
        Image.fromarray((refined_mask.astype(np.uint8) * 255), mode="L").save(raw_output_path)
        refined_mask = largest_component_mask(refined_mask)
    elif CONTOUR_REFINE_COMPONENT_MODE not in {"all", "largest"}:
        raise ValueError(
            f"CONTOUR_REFINE_COMPONENT_MODE must be all or largest, got {CONTOUR_REFINE_COMPONENT_MODE!r}"
        )

    output_path = out_dir / output_name
    Image.fromarray((refined_mask.astype(np.uint8) * 255), mode="L").save(output_path)
    return {
        "enabled": True,
        "method": "q80_external_contour_fill_g1",
        "output_mask": str(output_path),
        "object_source": object_source,
        "seed_source": "coarse_core_plus_q80_appearance_boundary",
        "mask_area": int(refined_mask.sum()),
        "coarse_area": int(coarse_seed.sum()),
        "edge_refined_area": int(edge_refined.sum()),
        "protected_core_area": int(protected_core.sum()),
        "boundary_band_area": int(boundary_band.sum()),
        "edge_support_area": int(edge_support.sum()),
        "boundary_refined_area": int(boundary_refined.sum()),
        "clip_to_coarse": bool(clip_to_coarse),
        "edge_dilate": int(edge_dilate),
        "close_radius": int(close_radius),
        "fill_holes": bool(fill_holes),
        "outer_contour_fill": True,
        "lost_from_coarse_area": int((coarse_seed & ~refined_mask).sum()),
        "added_outside_coarse_area": int((refined_mask & ~coarse_seed).sum()),
        "kept_coarse_ratio": float((coarse_seed & refined_mask).sum() / max(1, int(coarse_seed.sum()))),
        "roi_area": int(coarse_roi.sum()),
        "object_area": int(object_mask_full.sum()),
        "component_count": connected_component_count(refined_mask),
        "component_mode": CONTOUR_REFINE_COMPONENT_MODE,
        "raw_component_count": raw_component_count,
        "raw_output_mask": str(raw_output_path) if raw_output_path is not None else None,
        "inner_erode": inner_erode,
        "roi_dilate": roi_dilate,
    }


def attention_seed_from_soft_mask(
    soft_mask: np.ndarray,
    shape_hw: tuple[int, int],
    object_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    attention_score = normalize01(resize_score(soft_mask, shape_hw))
    attention_score = normalize01(attention_score * object_mask.astype(np.float32))
    attention_seed = (attention_score >= PAMR_SEED_THRESHOLD) & object_mask
    return attention_score, attention_seed


def resize_for_pamr(
    image_rgb: np.ndarray,
    seed_mask: np.ndarray,
    object_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    height, width = seed_mask.shape[:2]
    max_side = max(height, width)
    if max_side <= PAMR_MAX_SIDE:
        return image_rgb, seed_mask.astype(np.float32), object_mask.astype(bool), False

    scale = PAMR_MAX_SIDE / float(max_side)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    image_small = cv2.resize(image_rgb, (new_width, new_height), interpolation=cv2.INTER_AREA)
    seed_small = cv2.resize(seed_mask.astype(np.uint8), (new_width, new_height), interpolation=cv2.INTER_NEAREST).astype(np.float32)
    object_small = cv2.resize(object_mask.astype(np.uint8), (new_width, new_height), interpolation=cv2.INTER_NEAREST) > 0
    return image_small, seed_small, object_small, True


def run_pamr(image_rgb: np.ndarray, seed_mask: np.ndarray, object_mask: np.ndarray) -> np.ndarray:
    shape_hw = tuple(seed_mask.shape[:2])
    image_pamr, seed_pamr, object_pamr, resized = resize_for_pamr(image_rgb, seed_mask, object_mask)
    image_tensor = torch.tensor(image_pamr, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
    mask_tensor = torch.tensor(seed_pamr, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        score = PAMR(num_iter=PAMR_ITER)(image_tensor, mask_tensor)[0, 0].detach().float().cpu().numpy()
    score = normalize01(score * object_pamr.astype(np.float32))
    if resized:
        score = cv2.resize(score, (shape_hw[1], shape_hw[0]), interpolation=cv2.INTER_LINEAR)
        score = normalize01(score * object_mask.astype(np.float32))
    return score


def refine_direct_attention_mask(
    *,
    image_rgb: np.ndarray,
    soft_mask: np.ndarray,
    coarse_mask: np.ndarray,
    object_mask: np.ndarray | None,
    out_dir: Path,
    object_source: str,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    shape_hw = tuple(image_rgb.shape[:2])
    object_mask_full = prepare_object_mask(object_mask, shape_hw)
    coarse_seed = resize_bool(coarse_mask, shape_hw) & object_mask_full
    attention_score, attention_seed = attention_seed_from_soft_mask(soft_mask, shape_hw, object_mask_full)

    seed_area = int(attention_seed.sum())
    if seed_area == 0:
        refined_mask = np.zeros(shape_hw, dtype=bool)
    else:
        pamr_score = run_pamr(image_rgb, attention_seed.astype(np.float32), object_mask_full)
        refined_mask = (pamr_score >= PAMR_THRESHOLD) & object_mask_full

    output_path = out_dir / OUTPUT_MASK_NAME
    Image.fromarray((refined_mask.astype(np.uint8) * 255), mode="L").save(output_path)
    return {
        "output_mask": str(output_path),
        "object_source": object_source,
        "seed_source": "soft_attention_pamr",
        "mask_area": int(refined_mask.sum()),
        "coarse_area": int(coarse_seed.sum()),
        "attention_seed_area": seed_area,
        "object_area": int(object_mask_full.sum()),
        "pamr_iter": PAMR_ITER,
        "pamr_seed_threshold": PAMR_SEED_THRESHOLD,
        "pamr_threshold": PAMR_THRESHOLD,
        "pamr_max_side": PAMR_MAX_SIDE,
    }


def refine_active_contour_mask(
    *,
    image_rgb: np.ndarray,
    coarse_mask: np.ndarray,
    object_mask: np.ndarray | None,
    out_dir: Path,
    object_source: str,
) -> dict:
    from skimage.segmentation import morphological_geodesic_active_contour

    out_dir.mkdir(parents=True, exist_ok=True)
    shape_hw = tuple(image_rgb.shape[:2])
    object_mask_full = prepare_object_mask(object_mask, shape_hw)
    coarse_seed = resize_bool(coarse_mask, shape_hw) & object_mask_full
    coarse_roi = dilate_binary(coarse_seed, MGAC_ROI_DILATE) & object_mask_full

    edge_stopping, edge_strength, scharr, wavelet_high, wavelet_method = edited_image_edge_maps(image_rgb)
    edge_gate, edge_u8, gate_threshold = edge_gate_from_strength(
        edge_strength,
        coarse_roi,
        gate_percentile=MGAC_GATE_PERCENTILE,
        gate_dilate=MGAC_GATE_DILATE,
    )
    if MGAC_USE_EDGE_GATE_AS_ROI:
        active_roi = edge_gate & object_mask_full
    else:
        active_roi = coarse_roi & object_mask_full
    edge_stopping = normalize01(edge_stopping * active_roi.astype(np.float32))

    init_seed = erode_binary(coarse_seed, MGAC_INIT_ERODE) & active_roi
    if not init_seed.any():
        init_seed = coarse_seed & active_roi
    if not init_seed.any():
        init_seed = coarse_seed.copy()

    if MGAC_OUTPUT_MODE == "edge_gate":
        raw_mask = edge_gate & object_mask_full & coarse_roi
        refined_mask = raw_mask.copy()
    elif not coarse_seed.any():
        refined_mask = np.zeros(shape_hw, dtype=bool)
        raw_mask = refined_mask.copy()
    else:
        init_level_set = init_seed.astype(np.int8)
        evolved = morphological_geodesic_active_contour(
            edge_stopping,
            num_iter=MGAC_ITER,
            init_level_set=init_level_set,
            smoothing=MGAC_SMOOTHING,
            threshold="auto",
            balloon=MGAC_BALLOON,
        ).astype(bool)
        raw_mask = evolved & active_roi
        if MGAC_KEEP_COARSE:
            raw_mask = raw_mask | coarse_seed
        raw_mask = raw_mask & object_mask_full & coarse_roi
        refined_mask = close_binary(raw_mask, MGAC_FINAL_CLOSE)
        if MGAC_FINAL_FILL_HOLES:
            refined_mask = fill_holes_binary(refined_mask)
        refined_mask = keep_relevant_components(refined_mask, coarse_seed, MGAC_FINAL_MIN_AREA)
        refined_mask = refined_mask & object_mask_full & coarse_roi

    output_path = out_dir / ACTIVE_CONTOUR_OUTPUT_MASK_NAME
    raw_path = out_dir / "active_contour_raw_mask.png"
    roi_path = out_dir / "active_contour_roi.png"
    gate_path = out_dir / "active_contour_edge_gate.png"
    active_roi_path = out_dir / "active_contour_active_roi.png"
    edge_path = out_dir / "active_contour_edge_map.png"
    scharr_path = out_dir / "active_contour_scharr.png"
    wavelet_path = out_dir / "active_contour_wavelet_highfreq.png"

    if SAVE_ACTIVE_CONTOUR_MASK:
        Image.fromarray((refined_mask.astype(np.uint8) * 255), mode="L").save(output_path)
    if SAVE_ACTIVE_CONTOUR_DEBUG:
        Image.fromarray((raw_mask.astype(np.uint8) * 255), mode="L").save(raw_path)
        Image.fromarray((coarse_roi.astype(np.uint8) * 255), mode="L").save(roi_path)
        Image.fromarray((edge_gate.astype(np.uint8) * 255), mode="L").save(gate_path)
        Image.fromarray((active_roi.astype(np.uint8) * 255), mode="L").save(active_roi_path)
        save_score_image(scharr, scharr_path)
        save_score_image(wavelet_high, wavelet_path)
    if SAVE_ACTIVE_CONTOUR_EDGE_MAP:
        save_score_image(edge_strength, edge_path)

    return {
        "output_mask": str(output_path) if SAVE_ACTIVE_CONTOUR_MASK else None,
        "object_source": object_source,
        "seed_source": "coarse_mask_mgac_scharr_wavelet",
        "mask_area": int(refined_mask.sum()),
        "raw_mask_area": int(raw_mask.sum()),
        "coarse_area": int(coarse_seed.sum()),
        "roi_area": int(coarse_roi.sum()),
        "gate_area": int(active_roi.sum()),
        "edge_gate_area": int(edge_gate.sum()),
        "init_area": int(init_seed.sum()),
        "object_area": int(object_mask_full.sum()),
        "raw_component_count": connected_component_count(raw_mask),
        "component_count": connected_component_count(refined_mask),
        "raw_mask": str(raw_path) if SAVE_ACTIVE_CONTOUR_DEBUG else None,
        "edge_map": str(edge_path) if SAVE_ACTIVE_CONTOUR_EDGE_MAP else None,
        "scharr_map": str(scharr_path) if SAVE_ACTIVE_CONTOUR_DEBUG else None,
        "wavelet_highfreq_map": str(wavelet_path) if SAVE_ACTIVE_CONTOUR_DEBUG else None,
        "wavelet_method": wavelet_method,
        "roi_mask": str(roi_path) if SAVE_ACTIVE_CONTOUR_DEBUG else None,
        "edge_gate_mask": str(gate_path) if SAVE_ACTIVE_CONTOUR_DEBUG else None,
        "active_roi_mask": str(active_roi_path) if SAVE_ACTIVE_CONTOUR_DEBUG else None,
        "save_active_contour_mask": SAVE_ACTIVE_CONTOUR_MASK,
        "save_active_contour_debug": SAVE_ACTIVE_CONTOUR_DEBUG,
        "save_active_contour_edge_map": SAVE_ACTIVE_CONTOUR_EDGE_MAP,
        "_refined_mask": refined_mask,
        "_edge_map_u8": edge_u8,
        "_coarse_roi_mask": coarse_roi,
        "mgac_iter": MGAC_ITER,
        "mgac_smoothing": MGAC_SMOOTHING,
        "mgac_balloon": MGAC_BALLOON,
        "mgac_roi_dilate": MGAC_ROI_DILATE,
        "mgac_edge_alpha": MGAC_EDGE_ALPHA,
        "mgac_scharr_weight": MGAC_SCHARR_WEIGHT,
        "mgac_wavelet_weight": MGAC_WAVELET_WEIGHT,
        "mgac_gate_percentile": MGAC_GATE_PERCENTILE,
        "mgac_gate_dilate": MGAC_GATE_DILATE,
        "mgac_init_erode": MGAC_INIT_ERODE,
        "mgac_keep_coarse": MGAC_KEEP_COARSE,
        "mgac_use_edge_gate_as_roi": MGAC_USE_EDGE_GATE_AS_ROI,
        "mgac_final_close": MGAC_FINAL_CLOSE,
        "mgac_final_min_area": MGAC_FINAL_MIN_AREA,
        "mgac_final_fill_holes": MGAC_FINAL_FILL_HOLES,
        "mgac_output_mode": MGAC_OUTPUT_MODE,
        "gate_threshold_u8": gate_threshold,
        "threshold_rule": "edge_u8 > percentile(edge_u8[coarse_roi])",
    }
