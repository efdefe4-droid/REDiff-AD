#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from batch_insert_anything import (
    list_images,
    load_object_support_from_file,
    object_support_from_image,
    prepare_reference,
    prepare_target,
    random_target_mask_on_object,
    resolve_id_file,
    resolve_matching_file,
    sample_sources,
)
from run_insert_anything import DEFAULT_LORA_WEIGHT, crop_back, load_pipelines, load_rgb
from generation_attention import mask_refinement as mask_refinement_settings
from generation_attention.mask_refinement import (
    connected_component_count,
    refine_active_contour_mask,
    refine_boundary_preserve_mask,
    refine_direct_attention_mask,
    refine_q80_appearance_mask,
)
from generation_attention.resume_safety import (
    RESUME_CONFIG_SCHEMA_VERSION,
    decide_sample_resume_action,
    ensure_resume_run_config_compatible,
)
from generation_attention.target_mask_policy import (
    reference_vertical_mask,
    reference_vertical_mixed_variant,
)
from generation_attention.visualize_flux_attention import (
    DEFAULT_BLOCK_FREQUENCY_CSV,
    DEFAULT_TOP15_BLOCKS_FILE,
    FluxAttentionRecorder,
    colorize_u8,
    reference_shape_map_from_image,
    resolve_shape_k_blocks,
    SHAPE_K_BLOCK_SCOPE_CHOICES,
    load_binary_mask,
    normalize01_np,
    object_roi_from_image,
    read_block_list,
    read_dominant_polarities,
    register_attention_recorders,
    save_mask_overlay,
    save_soft_overlay,
    threshold_soft_mask,
)


REFINEMENT_SETTING_NAMES = (
    "PAMR_ITER",
    "PAMR_SEED_THRESHOLD",
    "PAMR_THRESHOLD",
    "MGAC_ITER",
    "MGAC_SMOOTHING",
    "MGAC_BALLOON",
    "MGAC_ROI_DILATE",
    "MGAC_EDGE_ALPHA",
    "MGAC_SCHARR_WEIGHT",
    "MGAC_WAVELET_WEIGHT",
    "MGAC_GATE_PERCENTILE",
    "MGAC_GATE_DILATE",
    "MGAC_INIT_ERODE",
    "MGAC_KEEP_COARSE",
    "MGAC_USE_EDGE_GATE_AS_ROI",
    "MGAC_FINAL_CLOSE",
    "MGAC_FINAL_MIN_AREA",
    "MGAC_FINAL_FILL_HOLES",
    "MGAC_OUTPUT_MODE",
    "Q80_APPEARANCE_PERCENTILE",
    "Q80_APPEARANCE_GATE_DILATE",
    "Q80_APPEARANCE_MIN_AREA",
    "Q80_APPEARANCE_FG_ERODE",
    "Q80_APPEARANCE_BG_RING_DILATE",
    "Q80_APPEARANCE_KEEP_MARGIN",
    "Q80_APPEARANCE_GROW_RADIUS",
    "Q80_APPEARANCE_ADD_MARGIN",
    "Q80_APPEARANCE_MAX_FG_DIST",
    "Q80_APPEARANCE_ROI_DILATE",
    "CONTOUR_REFINE_INNER_ERODE",
    "CONTOUR_REFINE_CLIP_TO_COARSE",
    "CONTOUR_REFINE_EDGE_DILATE",
    "CONTOUR_REFINE_CLOSE",
    "CONTOUR_REFINE_FILL_HOLES",
    "CONTOUR_REFINE_COMPONENT_MODE",
    "SAVE_ACTIVE_CONTOUR_MASK",
    "SAVE_ACTIVE_CONTOUR_DEBUG",
    "SAVE_ACTIVE_CONTOUR_EDGE_MAP",
)


def _env_enabled(name: str, default: str) -> bool:
    return os.environ.get(name, default).lower() not in {"0", "false", "no"}


def refinement_runtime_config() -> dict[str, object]:
    """Capture the effective refine configuration, including import-time settings."""
    config = {
        name: getattr(mask_refinement_settings, name)
        for name in REFINEMENT_SETTING_NAMES
    }
    config.update(
        {
            "REFINE_COARSE_OPEN": _env_enabled("REFINE_COARSE_OPEN", "1"),
            "REFINE_COARSE_OPEN_KERNEL": max(1, int(os.environ.get("REFINE_COARSE_OPEN_KERNEL", "3"))),
            "REFINE_COARSE_OPEN_ITER": max(0, int(os.environ.get("REFINE_COARSE_OPEN_ITER", "1"))),
            "RUN_PAMR_REFINE": _env_enabled("RUN_PAMR_REFINE", "0"),
            "RUN_Q80_APPEARANCE_REFINE": _env_enabled("RUN_Q80_APPEARANCE_REFINE", "1"),
            "RUN_CONTOUR_REFINE": _env_enabled("RUN_CONTOUR_REFINE", "1"),
        }
    )
    return config


def build_anomaly_refs(anomalies: list[str], ref_ids: list[str]) -> list[tuple[str, str]]:
    if len(ref_ids) == 1 and len(anomalies) > 1:
        ref_ids = ref_ids * len(anomalies)
    if len(anomalies) != len(ref_ids):
        raise ValueError("--anomalies and --ref-ids must have the same length, or pass one ref id for all anomalies.")
    return list(zip(anomalies, ref_ids))


def sample_resume_expectations(
    args: argparse.Namespace,
    *,
    anomaly: str,
    ref_id: str,
    ref_image: Path,
    ref_mask: Path,
    sample_index: int,
    source_image: Path,
    sample_seed: int,
    selected_blocks: list[str],
) -> dict[str, object]:
    return {
        "expected_seed": sample_seed,
        "expected_source_image": source_image,
        "expected_ref_id": ref_id,
        "expected_ref_image": ref_image,
        "expected_ref_mask": ref_mask,
        "expected_kind": args.direct_aggregate_kind,
        "expected_steps": args.direct_aggregate_steps,
        "expected_selected_blocks": selected_blocks,
        "expected_adaptive_kind": args.adaptive_aggregate_kind if args.adaptive_ref_injection else None,
        "expected_adaptive_selected_blocks": args.adaptive_selected_blocks if args.adaptive_ref_injection else None,
        "expected_adaptive_default_polarity": (
            args.adaptive_block_polarities.get("__default__") if args.adaptive_ref_injection else None
        ),
        "expected_num_inference_steps": args.num_inference_steps,
        "expected_anomaly": anomaly,
        "expected_index": sample_index,
        "expected_target_mask_source": args.target_mask_source,
    }


def preflight_existing_samples_for_resume(
    args: argparse.Namespace,
    *,
    anomaly_refs: list[tuple[str, str]],
    samples_per_pair: list[int],
    source_images: list[Path],
    ref_image_root: Path,
    ref_mask_root: Path,
    out_root: Path,
    selected_blocks: list[str],
) -> None:
    """Validate every requested existing sample before config/log/model writes."""
    if args.overwrite:
        return
    for anomaly_offset, (anomaly, ref_id) in enumerate(anomaly_refs):
        pair_sample_count = samples_per_pair[anomaly_offset]
        if pair_sample_count == 0:
            continue
        ref_image = resolve_id_file(ref_image_root / anomaly, ref_id)
        ref_mask = resolve_id_file(ref_mask_root / anomaly, ref_id, suffixes=("_mask", ""))
        pair_seed = args.seed + anomaly_offset * 1000003
        source_count = args.start_index + pair_sample_count
        sources = sample_sources(source_images, source_count, pair_seed)[args.start_index:]
        for sample_index, source_image in enumerate(sources, start=args.start_index):
            sample_dir = out_root / anomaly / f"ref_{ref_id}" / f"{sample_index:03d}"
            if not sample_dir.exists():
                continue
            sample_seed = args.seed + anomaly_offset * 1000003 + sample_index
            decide_sample_resume_action(
                sample_dir,
                overwrite=False,
                **sample_resume_expectations(
                    args,
                    anomaly=anomaly,
                    ref_id=ref_id,
                    ref_image=ref_image,
                    ref_mask=ref_mask,
                    sample_index=sample_index,
                    source_image=source_image,
                    sample_seed=sample_seed,
                    selected_blocks=selected_blocks,
                ),
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch FLUX Fill generation with streaming attention coarse masks.")
    parser.add_argument("--anomalies", nargs="+", required=True)
    parser.add_argument("--ref-ids", nargs="+", required=True)
    parser.add_argument("--samples-per-anomaly", type=int, default=500)
    parser.add_argument(
        "--samples-per-pair",
        nargs="+",
        type=int,
        default=None,
        help="Optional per (anomaly, ref_id) sample counts. Length must match --anomalies/--ref-ids.",
    )
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--ref-image-root", required=True)
    parser.add_argument("--ref-mask-root", required=True)
    parser.add_argument("--out-root", default="500ps-result_hazelnut_attention_direct_top10_steps5_40")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--save-debug-first", action="store_true")
    parser.add_argument("--save-mask-debug", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--log-file", default=None)
    parser.add_argument("--adaptive-log-file", default=None)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--num-inference-steps", type=int, default=41)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--guidance-scale", type=float, default=None)
    parser.add_argument("--max-sequence-length", type=int, default=512)
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--cpu-offload", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--sequential-cpu-offload", action="store_true")
    parser.add_argument("--empty-cache-each-sample", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--target-mask-source",
        choices=["random_object", "reference_vertical_mixed", "fixed"],
        default="random_object",
    )
    parser.add_argument("--fixed-target-mask", default=None)
    parser.add_argument("--reference-mask-dilate-iterations", type=int, default=5)
    parser.add_argument("--reference-mask-vertical-shift-ratio", type=float, default=0.05)
    parser.add_argument("--object-prompt", default=None)
    parser.add_argument("--object-support-root", default=None)
    parser.add_argument("--object-attention-root", default=None)
    parser.add_argument("--object-support-erosion", type=int, default=8)
    parser.add_argument("--random-mask-area-min-ratio", type=float, default=0.40)
    parser.add_argument("--random-mask-area-max-ratio", type=float, default=2.0)
    parser.add_argument("--random-mask-rotate", type=float, default=45.0)
    parser.add_argument("--random-mask-attempts", type=int, default=160)
    parser.add_argument("--random-mask-component-spacing", type=int, default=24)
    parser.add_argument("--random-mask-double-prob", type=float, default=0.0)
    parser.add_argument(
        "--random-mask-placement-filter",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Reject random target-mask placements dominated by dark, edge, or object-boundary pixels.",
    )
    parser.add_argument("--random-mask-dark-quantile", type=float, default=20.0)
    parser.add_argument("--random-mask-max-dark-fraction", type=float, default=0.35)
    parser.add_argument("--random-mask-edge-quantile", type=float, default=90.0)
    parser.add_argument("--random-mask-max-edge-fraction", type=float, default=0.30)
    parser.add_argument("--random-mask-boundary-margin-ratio", type=float, default=0.015)
    parser.add_argument("--random-mask-max-boundary-fraction", type=float, default=0.03)
    parser.add_argument("--save-steps", nargs="+", type=int, default=None)
    parser.add_argument(
        "--direct-aggregate-steps",
        nargs="+",
        type=int,
        default=None,
        help="Inference steps used for the final direct aggregate mask. Defaults to --save-steps/default direct steps.",
    )
    parser.add_argument(
        "--adaptive-check-steps",
        nargs="+",
        type=int,
        default=None,
        help="Inference steps where adaptive aggregate-mask checks are evaluated. Defaults to --save-steps/default direct steps.",
    )
    parser.add_argument("--max-blocks", type=int, default=None)
    parser.add_argument("--max-steps-to-save", type=int, default=None)
    parser.add_argument("--save-blocks", nargs="+", default=None)
    parser.add_argument("--direct-aggregate-kind", choices=["target_to_condition", "target_to_ref_image", "target_to_target_image", "ref_to_target_image"], default="target_to_ref_image")
    parser.add_argument("--direct-selected-blocks-file", default=DEFAULT_TOP15_BLOCKS_FILE)
    parser.add_argument("--direct-top-k", type=int, default=10)
    parser.add_argument("--direct-block-frequency-csv", default=DEFAULT_BLOCK_FREQUENCY_CSV)
    parser.add_argument("--direct-polarity", choices=["dominant", "high", "low"], default="dominant")
    parser.add_argument("--direct-roi", choices=["initial_mask", "object", "all"], default="initial_mask")
    parser.add_argument("--direct-hist-threshold-scale", type=float, default=0.9)
    parser.add_argument("--direct-hist-threshold-offset", type=float, default=0.0)
    parser.add_argument("--direct-component-mode", choices=["all", "largest", "max_energy"], default="all")
    parser.add_argument("--direct-fill-holes", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--direct-close-iterations", type=int, default=1)
    parser.add_argument("--direct-dilate-iterations", type=int, default=1)
    parser.add_argument("--adaptive-ref-injection", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--adaptive-roi", choices=["initial_mask", "object", "all"], default="initial_mask")
    parser.add_argument("--adaptive-aggregate-kind", choices=["target_to_condition", "target_to_ref_image", "target_to_target_image", "ref_to_target_image"], default="target_to_ref_image")
    parser.add_argument("--adaptive-selected-blocks-file", default=None)
    parser.add_argument("--adaptive-block-frequency-csv", default=None)
    parser.add_argument("--adaptive-polarity", choices=["dominant", "high", "low"], default="dominant")
    parser.add_argument("--adaptive-score-mode", choices=["aggregate_mask", "ref_condition_mass"], default="aggregate_mask")
    parser.add_argument("--adaptive-aggregate-score-kind", choices=["inside_coverage", "inside_mean", "coarse_area_ratio", "shape_strict", "inside_outside_contrast"], default="inside_coverage")
    parser.add_argument("--adaptive-aggregate-min-inside-ratio", type=float, default=0.85)
    parser.add_argument("--adaptive-aggregate-min-inside-mean", type=float, default=0.35)
    parser.add_argument("--adaptive-aggregate-min-area-ratio", type=float, default=0.20)
    parser.add_argument("--adaptive-aggregate-min-contrast-ratio", type=float, default=1.25)
    parser.add_argument("--adaptive-aggregate-min-contrast-inside-mean", type=float, default=0.18)
    parser.add_argument("--adaptive-aggregate-min-contrast-margin", type=float, default=0.055)
    parser.add_argument("--adaptive-aggregate-outside-ring-dilate", type=int, default=32)
    parser.add_argument("--adaptive-ref-token-start", type=int, default=512)
    parser.add_argument("--adaptive-ref-attention-threshold", type=float, default=0.12)
    parser.add_argument("--adaptive-ref-base-scale", type=float, default=1.0)
    parser.add_argument("--adaptive-ref-max-scale", type=float, default=2.0)
    parser.add_argument("--adaptive-ref-boost", type=float, default=0.5)
    parser.add_argument("--adaptive-ref-trigger-min-scale", type=float, default=0.0)
    parser.add_argument("--adaptive-ref-decay", type=float, default=1.0)
    parser.add_argument("--adaptive-ref-decay-min-score", type=float, default=1.0)
    parser.add_argument("--ref-token-noise-std", type=float, default=0.0)
    parser.add_argument("--ref-token-dropout", type=float, default=0.0)
    parser.add_argument("--ref-token-scale-jitter", type=float, default=0.0)
    parser.add_argument("--ref-token-span-dropout", type=float, default=0.0)
    parser.add_argument("--ref-token-span-len", type=int, default=4)
    parser.add_argument("--ref-token-perturb-seed-offset", type=int, default=700000)
    parser.add_argument("--ref-augment-bank-size", type=int, default=1)
    parser.add_argument("--ref-augment-rotate", type=float, default=0.0)
    parser.add_argument("--ref-augment-scale-jitter", type=float, default=0.0)
    parser.add_argument("--ref-augment-translate-ratio", type=float, default=0.0)
    parser.add_argument("--ref-augment-brightness", type=float, default=0.0)
    parser.add_argument("--ref-augment-contrast", type=float, default=0.0)
    parser.add_argument("--ref-augment-seed-offset", type=int, default=800000)
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
    parser.add_argument("--log-attention-steps", action="store_true")
    parser.add_argument("--flux-fill-path", default="black-forest-labs/FLUX.1-Fill-dev")
    parser.add_argument("--flux-redux-path", default="black-forest-labs/FLUX.1-Redux-dev")
    parser.add_argument("--lora-path", default="WensongSong/Insert-Anything")
    parser.add_argument("--lora-weight-name", default=DEFAULT_LORA_WEIGHT)
    parser.add_argument("--full-flux-quantize", choices=["none", "int8", "int4"], default="none")
    return parser.parse_args()


def configure_direct(args: argparse.Namespace) -> tuple[list[str], dict[str, str]]:
    selected_blocks = read_block_list(args.direct_selected_blocks_file, args.direct_top_k)
    if not selected_blocks:
        raise ValueError(f"No selected blocks found in {args.direct_selected_blocks_file}")
    if args.direct_polarity == "dominant":
        dominant = read_dominant_polarities(args.direct_block_frequency_csv)
        # The six FLUX blocks absent from the T2R frequency table are all
        # high-polarity in the full crack/000 block audit. Use that measured
        # fallback so the all-block diagnostic does not invert those maps.
        default_polarity = "high" if args.direct_aggregate_kind == "target_to_ref_image" else "low"
        block_polarities = {"__default__": default_polarity, **dominant}
        for block in selected_blocks:
            block_polarities.setdefault(block, default_polarity)
    else:
        block_polarities = {"__default__": args.direct_polarity}
        for block in selected_blocks:
            block_polarities[block] = args.direct_polarity
    default_steps = [step for step in [5, 10, 15, 20, 25, 30, 35, 40] if step < args.num_inference_steps]
    if not default_steps:
        default_steps = [max(0, args.num_inference_steps - 1)]
    split_steps_given = args.direct_aggregate_steps is not None or args.adaptive_check_steps is not None
    base_steps = list(args.save_steps) if args.save_steps is not None else list(default_steps)
    if args.direct_aggregate_steps is None:
        args.direct_aggregate_steps = list(base_steps)
    if args.adaptive_check_steps is None:
        args.adaptive_check_steps = list(base_steps)
    if args.save_steps is None:
        args.save_steps = [] if split_steps_given else list(base_steps)
    args.capture_steps = sorted(set(args.save_steps) | set(args.direct_aggregate_steps) | set(args.adaptive_check_steps))
    if args.save_blocks is None:
        args.save_blocks = selected_blocks
    adaptive_blocks_file = args.adaptive_selected_blocks_file or args.direct_selected_blocks_file
    adaptive_selected_blocks = read_block_list(adaptive_blocks_file, args.direct_top_k)
    if not adaptive_selected_blocks:
        raise ValueError(f"No adaptive blocks found in {adaptive_blocks_file}")
    if args.adaptive_polarity == "dominant":
        adaptive_frequency_csv = args.adaptive_block_frequency_csv or args.direct_block_frequency_csv
        adaptive_dominant = read_dominant_polarities(adaptive_frequency_csv)
        adaptive_default_polarity = "high" if args.adaptive_aggregate_kind == "target_to_ref_image" else "low"
        adaptive_block_polarities = {"__default__": adaptive_default_polarity, **adaptive_dominant}
        for block in adaptive_selected_blocks:
            adaptive_block_polarities.setdefault(block, adaptive_default_polarity)
    else:
        adaptive_block_polarities = {"__default__": args.adaptive_polarity}
        for block in adaptive_selected_blocks:
            adaptive_block_polarities[block] = args.adaptive_polarity
    args.adaptive_selected_blocks = adaptive_selected_blocks
    args.adaptive_block_polarities = adaptive_block_polarities
    return selected_blocks, block_polarities


def scale_reference_condition_tokens(
    prompt_embeds: torch.Tensor,
    base_prompt_embeds: torch.Tensor,
    token_start: int,
    scale: float,
) -> torch.Tensor:
    token_start = max(0, int(token_start))
    if token_start >= prompt_embeds.shape[1]:
        return prompt_embeds
    scaled = prompt_embeds.clone()
    base = base_prompt_embeds.to(device=scaled.device, dtype=scaled.dtype)
    if base.shape[0] == 1 and scaled.shape[0] > 1:
        base = base.expand(scaled.shape[0], -1, -1)
    token_end = min(scaled.shape[1], base.shape[1])
    if token_start < token_end:
        scaled[:, token_start:token_end, :] = base[:, token_start:token_end, :] * float(scale)
    return scaled


def prompt_embed_generator(prompt_embeds: torch.Tensor, seed: int) -> torch.Generator:
    device = prompt_embeds.device
    if device.type == "cuda":
        generator = torch.Generator(device=device)
    else:
        generator = torch.Generator()
    return generator.manual_seed(int(seed))


def augment_masked_reference_image(
    masked_ref_image: np.ndarray,
    rng: np.random.Generator,
    rotate: float,
    scale_jitter: float,
    translate_ratio: float,
    brightness: float,
    contrast: float,
) -> tuple[np.ndarray, dict[str, object]]:
    height, width = masked_ref_image.shape[:2]
    rotate = max(0.0, float(rotate))
    scale_jitter = max(0.0, float(scale_jitter))
    translate_ratio = max(0.0, float(translate_ratio))
    brightness = max(0.0, float(brightness))
    contrast = max(0.0, float(contrast))

    angle = float(rng.uniform(-rotate, rotate)) if rotate > 0.0 else 0.0
    scale = 1.0 + (float(rng.uniform(-scale_jitter, scale_jitter)) if scale_jitter > 0.0 else 0.0)
    scale = max(0.05, scale)
    translate_x = float(rng.uniform(-translate_ratio, translate_ratio) * width) if translate_ratio > 0.0 else 0.0
    translate_y = float(rng.uniform(-translate_ratio, translate_ratio) * height) if translate_ratio > 0.0 else 0.0

    matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), angle, scale)
    matrix[0, 2] += translate_x
    matrix[1, 2] += translate_y
    augmented = cv2.warpAffine(
        masked_ref_image,
        matrix,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )

    contrast_factor = 1.0 + (float(rng.uniform(-contrast, contrast)) if contrast > 0.0 else 0.0)
    brightness_shift = float(rng.uniform(-brightness, brightness) * 255.0) if brightness > 0.0 else 0.0
    foreground = np.any(augmented < 250, axis=2, keepdims=True)
    adjusted = augmented.astype(np.float32)
    adjusted = 255.0 + (adjusted - 255.0) * contrast_factor
    adjusted = np.where(foreground, adjusted + brightness_shift, adjusted)
    adjusted = np.where(foreground, adjusted, 255.0)
    augmented = np.clip(adjusted, 0, 255).astype(np.uint8)

    info: dict[str, object] = {
        "enabled": True,
        "angle": angle,
        "scale": scale,
        "translate_x": translate_x,
        "translate_y": translate_y,
        "brightness_shift": brightness_shift,
        "contrast_factor": contrast_factor,
    }
    return augmented, info


def perturb_reference_condition_tokens(
    prompt_embeds: torch.Tensor,
    token_start: int,
    noise_std: float,
    dropout: float,
    scale_jitter: float,
    span_dropout: float,
    span_len: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, dict[str, object]]:
    token_start = max(0, int(token_start))
    noise_std = max(0.0, float(noise_std))
    dropout = min(1.0, max(0.0, float(dropout)))
    scale_jitter = max(0.0, float(scale_jitter))
    span_dropout = min(1.0, max(0.0, float(span_dropout)))
    span_len = max(1, int(span_len))
    token_end = prompt_embeds.shape[1]
    enabled = bool(noise_std > 0.0 or dropout > 0.0 or scale_jitter > 0.0 or span_dropout > 0.0)
    info: dict[str, object] = {
        "enabled": enabled,
        "token_start": token_start,
        "token_end": token_end,
        "num_tokens": max(0, token_end - token_start),
        "noise_std": noise_std,
        "dropout": dropout,
        "scale_jitter": scale_jitter,
        "span_dropout": span_dropout,
        "span_len": span_len,
        "dropped_tokens": 0,
        "span_dropped_tokens": 0,
        "total_zeroed_tokens": 0,
    }
    if token_start >= token_end or not enabled:
        return prompt_embeds, info

    perturbed = prompt_embeds.clone()
    ref_tokens = perturbed[:, token_start:token_end, :].float()
    zeroed_tokens = torch.zeros(ref_tokens.shape[:2], dtype=torch.bool, device=ref_tokens.device)

    if scale_jitter > 0.0:
        jitter_scale = 1.0 + torch.randn(
            (*ref_tokens.shape[:2], 1),
            generator=generator,
            device=ref_tokens.device,
            dtype=ref_tokens.dtype,
        ) * scale_jitter
        jitter_scale = jitter_scale.clamp_min(0.0)
        info["scale_min"] = float(jitter_scale.min().item())
        info["scale_max"] = float(jitter_scale.max().item())
        ref_tokens = ref_tokens * jitter_scale

    if noise_std > 0.0:
        token_scale = ref_tokens.std(dim=-1, keepdim=True).clamp_min(1e-6)
        noise = torch.randn(
            ref_tokens.shape,
            generator=generator,
            device=ref_tokens.device,
            dtype=ref_tokens.dtype,
        )
        ref_tokens = ref_tokens + noise * token_scale * noise_std

    if span_dropout > 0.0:
        num_ref_tokens = ref_tokens.shape[1]
        target_drop = max(1, int(round(num_ref_tokens * span_dropout)))
        spans_per_batch = max(1, (target_drop + span_len - 1) // span_len)
        max_start = max(1, num_ref_tokens - span_len + 1)
        starts = torch.randint(
            max_start,
            (ref_tokens.shape[0], spans_per_batch),
            generator=generator,
            device=ref_tokens.device,
        )
        keep = torch.ones(ref_tokens.shape[:2], dtype=torch.bool, device=ref_tokens.device)
        for batch_index in range(ref_tokens.shape[0]):
            for start in starts[batch_index].detach().cpu().tolist():
                end = min(num_ref_tokens, int(start) + span_len)
                keep[batch_index, int(start):end] = False
        span_zeroed = ~keep
        zeroed_tokens |= span_zeroed
        info["span_dropped_tokens"] = int(span_zeroed.sum().item())
        ref_tokens = ref_tokens * keep.unsqueeze(-1)

    if dropout > 0.0:
        keep = torch.rand(
            ref_tokens.shape[:2],
            generator=generator,
            device=ref_tokens.device,
            dtype=ref_tokens.dtype,
        ) >= dropout
        dropped = ~keep
        zeroed_tokens |= dropped
        info["dropped_tokens"] = int(dropped.sum().item())
        ref_tokens = ref_tokens * keep.unsqueeze(-1)

    info["total_zeroed_tokens"] = int(zeroed_tokens.sum().item())
    perturbed[:, token_start:token_end, :] = ref_tokens.to(dtype=perturbed.dtype)
    return perturbed, info


def split_binary_mask_components(mask: np.ndarray) -> list[np.ndarray]:
    mask_u8 = (mask > 0).astype(np.uint8)
    label_count, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    components: list[np.ndarray] = []
    for label in range(1, label_count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area <= 0:
            continue
        components.append((labels == label).astype(np.uint8))
    return components


def strip_runtime_arrays(info: dict) -> dict:
    return {key: value for key, value in info.items() if not str(key).startswith("_")}


def clean_sample_dir(sample_dir: Path) -> None:
    for name in (
        "debug_diptych_input.png", "debug_diptych_mask.png", "debug_masked_reference.png",
        "generated_target_mask.png", "object_support.png", "object_attention_map.png",
        "edit.png", "edit_full.png", "metadata.json",
        "soft_mask.npy", "soft_mask.png", "soft_mask_heatmap.png", "soft_mask_overlay.png",
        "coarse_mask.png", "coarse_mask_overlay.png", "edit_coarse_mask_overlay.png",
        "selected_block_soft_mask.npy", "selected_block_soft_mask.png",
        "selected_block_soft_mask_heatmap.png", "selected_block_soft_mask_overlay.png",
        "selected_block_coarse_mask.png", "selected_block_coarse_mask_overlay.png",
        "selected_block_edit_coarse_mask_overlay.png",
        "all_block_soft_mask.npy", "all_block_soft_mask.png",
        "all_block_soft_mask_heatmap.png", "all_block_soft_mask_overlay.png",
        "all_block_coarse_mask.png", "all_block_coarse_mask_overlay.png",
        "all_block_edit_coarse_mask_overlay.png",
        "coarse_refine_seed_mask.png", "edit_coarse_refine_seed_overlay.png",
        "threshold_roi.png", "threshold_roi_overlay.png", "direct_aggregate_summary.json",
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
        path = sample_dir / name
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()
    for pattern in (
        "pass_*",
        "_internal_pass_*",
        "generated_target_mask_component_*.png",
        "intermediate_after_pass_*.png",
        "_tmp_*.png",
    ):
        for extra_path in sample_dir.glob(pattern):
            if extra_path.is_dir():
                shutil.rmtree(extra_path)
            elif extra_path.exists():
                extra_path.unlink()


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


def aggregate_mask_artifact_paths(out_dir: Path, prefix: str) -> dict[str, Path]:
    stem = f"{prefix}_" if prefix else ""
    return {
        "soft_mask": out_dir / f"{stem}soft_mask.png",
        "soft_mask_npy": out_dir / f"{stem}soft_mask.npy",
        "soft_mask_heatmap": out_dir / f"{stem}soft_mask_heatmap.png",
        "soft_mask_overlay": out_dir / f"{stem}soft_mask_overlay.png",
        "coarse_mask": out_dir / f"{stem}coarse_mask.png",
        "coarse_mask_overlay": out_dir / f"{stem}coarse_mask_overlay.png",
        "edit_coarse_mask_overlay": out_dir / f"{stem}edit_coarse_mask_overlay.png",
    }


def write_aggregate_mask_artifacts(
    paths: dict[str, Path],
    *,
    soft_mask: np.ndarray,
    coarse_mask: np.ndarray,
    old_tar_image: np.ndarray,
    edited_image: np.ndarray,
) -> None:
    soft_u8 = (normalize01_np(soft_mask) * 255.0).clip(0, 255).astype(np.uint8)
    np.save(paths["soft_mask_npy"], soft_mask.astype(np.float32))
    Image.fromarray(soft_u8).save(paths["soft_mask"])
    Image.fromarray(colorize_u8(soft_u8)).save(paths["soft_mask_heatmap"])
    Image.fromarray((coarse_mask.astype(np.uint8) * 255)).save(paths["coarse_mask"])
    save_soft_overlay(old_tar_image, soft_mask, paths["soft_mask_overlay"])
    save_mask_overlay(old_tar_image, coarse_mask, paths["coarse_mask_overlay"])
    save_mask_overlay(edited_image, coarse_mask, paths["edit_coarse_mask_overlay"])


def stringify_paths(paths: dict[str, Path], enabled: bool) -> dict[str, str | None]:
    return {key: str(value) if enabled else None for key, value in paths.items()}


def attention_crop_support(shape_hw: tuple[int, int], crop_box: np.ndarray) -> np.ndarray:
    """Return the source-image region for which the cropped FLUX pass has attention."""
    height, width = shape_hw
    y1, y2, x1, x2 = [int(value) for value in crop_box]
    y1, y2 = max(0, y1), min(height, y2)
    x1, x2 = max(0, x1), min(width, x2)
    support = np.zeros((height, width), dtype=bool)
    if y2 > y1 and x2 > x1:
        support[y1:y2, x1:x2] = True
    return support


def save_direct_masks(
    args,
    *,
    out_dir,
    recorder,
    selected_blocks,
    block_polarities,
    old_tar_image,
    edited_image,
    target_mask_path,
    source_image,
    extra_sizes,
    crop_box,
    object_support=None,
    write_files: bool = True,
    run_refine: bool = True,
    return_arrays: bool = False,
):
    soft_mask, direct_counts = recorder.direct_soft_mask(
        selected_blocks=selected_blocks,
        source_shape_hw=tuple(old_tar_image.shape[:2]),
        extra_sizes=extra_sizes,
        crop_box=crop_box,
    )
    all_blocks = sorted(recorder.direct_sums.keys())
    if all_blocks:
        all_soft_mask, all_direct_counts = recorder.direct_soft_mask(
            selected_blocks=all_blocks,
            source_shape_hw=tuple(old_tar_image.shape[:2]),
            extra_sizes=extra_sizes,
            crop_box=crop_box,
        )
    else:
        all_soft_mask = soft_mask.copy()
        all_direct_counts = dict(direct_counts)

    attention_support = attention_crop_support(tuple(old_tar_image.shape[:2]), crop_box)
    if args.direct_roi == "initial_mask":
        threshold_roi = load_binary_mask(target_mask_path, tuple(old_tar_image.shape[:2]))
    elif args.direct_roi == "object":
        threshold_roi = object_roi_from_image(source_image, tuple(old_tar_image.shape[:2]))
    else:
        threshold_roi = np.ones(tuple(old_tar_image.shape[:2]), dtype=bool)
    # map_attention_to_original fills pixels outside crop_box with zeros. Those
    # are missing attention values, not valid low scores, so exclude them from
    # Otsu even when the requested semantic ROI is "all".
    threshold_roi = threshold_roi.astype(bool) & attention_support
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
    all_coarse_mask, all_hist_threshold, all_hist_otsu_threshold = threshold_soft_mask(
        all_soft_mask,
        threshold_roi,
        scale=args.direct_hist_threshold_scale,
        offset=args.direct_hist_threshold_offset,
        component_mode=args.direct_component_mode,
        fill=args.direct_fill_holes,
        close_iterations=args.direct_close_iterations,
        dilate_iterations=args.direct_dilate_iterations,
    )

    default_paths = aggregate_mask_artifact_paths(out_dir, "")
    selected_paths = aggregate_mask_artifact_paths(out_dir, "selected_block")
    all_paths = aggregate_mask_artifact_paths(out_dir, "all_block")
    soft_path = default_paths["soft_mask"]
    soft_npy_path = default_paths["soft_mask_npy"]
    soft_heatmap_path = default_paths["soft_mask_heatmap"]
    soft_overlay_path = default_paths["soft_mask_overlay"]
    coarse_path = default_paths["coarse_mask"]
    coarse_overlay_path = default_paths["coarse_mask_overlay"]
    edit_coarse_overlay_path = default_paths["edit_coarse_mask_overlay"]
    threshold_roi_path = out_dir / "threshold_roi.png"
    threshold_roi_overlay_path = out_dir / "threshold_roi_overlay.png"
    save_mask_debug = bool(getattr(args, "save_mask_debug", False))

    if write_files:
        out_dir.mkdir(parents=True, exist_ok=True)
        write_aggregate_mask_artifacts(
            default_paths,
            soft_mask=soft_mask,
            coarse_mask=coarse_mask,
            old_tar_image=old_tar_image,
            edited_image=edited_image,
        )
        write_aggregate_mask_artifacts(
            selected_paths,
            soft_mask=soft_mask,
            coarse_mask=coarse_mask,
            old_tar_image=old_tar_image,
            edited_image=edited_image,
        )
        write_aggregate_mask_artifacts(
            all_paths,
            soft_mask=all_soft_mask,
            coarse_mask=all_coarse_mask,
            old_tar_image=old_tar_image,
            edited_image=edited_image,
        )
        Image.fromarray((threshold_roi.astype(np.uint8) * 255)).save(threshold_roi_path)
        save_mask_overlay(old_tar_image, threshold_roi, threshold_roi_overlay_path, color=(0, 255, 0), alpha=0.25)

    info = {
        "mode": "direct_streaming_no_per_step_map_files",
        "kind": args.direct_aggregate_kind,
        "selected_blocks_file": args.direct_selected_blocks_file,
        "top_k": len(selected_blocks),
        "selected_blocks": selected_blocks,
        "block_polarities": block_polarities,
        "block_step_counts": direct_counts,
        "requested_save_steps": args.save_steps,
        "capture_steps": args.capture_steps,
        "direct_aggregate_steps": args.direct_aggregate_steps,
        "adaptive_check_steps": args.adaptive_check_steps,
        "aggregate": "mean_over_direct_aggregate_steps_then_mean_over_blocks",
        "roi": args.direct_roi,
        "hist_threshold_scale": args.direct_hist_threshold_scale,
        "hist_threshold_offset": args.direct_hist_threshold_offset,
        "hist_threshold": float(hist_threshold),
        "hist_otsu_threshold": float(hist_otsu_threshold),
        "component_mode": args.direct_component_mode,
        "fill_holes": args.direct_fill_holes,
        "close_iterations": args.direct_close_iterations,
        "dilate_iterations": args.direct_dilate_iterations,
        "save_mask_debug": save_mask_debug,
        "shape_k_removal": recorder.shape_k_summary(),
        "soft_mask": str(soft_path) if write_files else None,
        "soft_mask_npy": str(soft_npy_path) if write_files else None,
        "soft_mask_heatmap": str(soft_heatmap_path) if write_files else None,
        "soft_mask_overlay": str(soft_overlay_path) if write_files else None,
        "coarse_mask": str(coarse_path) if write_files else None,
        "coarse_mask_overlay": str(coarse_overlay_path) if write_files else None,
        "edit_coarse_mask_overlay": str(edit_coarse_overlay_path) if write_files else None,
        "threshold_roi": str(threshold_roi_path) if write_files else None,
        "threshold_roi_overlay": str(threshold_roi_overlay_path) if write_files else None,
        "selected_block_aggregate_mask": {
            "blocks": selected_blocks,
            "block_step_counts": direct_counts,
            "hist_threshold": float(hist_threshold),
            "hist_otsu_threshold": float(hist_otsu_threshold),
            **stringify_paths(selected_paths, write_files),
        },
        "all_block_aggregate_mask": {
            "blocks": all_blocks,
            "num_blocks": len(all_blocks),
            "block_step_counts": all_direct_counts,
            "hist_threshold": float(all_hist_threshold),
            "hist_otsu_threshold": float(all_hist_otsu_threshold),
            **stringify_paths(all_paths, write_files),
        },
    }
    if return_arrays:
        info["_soft_mask"] = soft_mask.astype(np.float32)
        info["_coarse_mask"] = coarse_mask.astype(bool)
        info["_all_block_soft_mask"] = all_soft_mask.astype(np.float32)
        info["_all_block_coarse_mask"] = all_coarse_mask.astype(bool)
        info["_threshold_roi"] = threshold_roi.astype(bool)

    if not (write_files and run_refine):
        info["refined_mask"] = {"enabled": False, "reason": "internal_pass"}
        info["active_contour_mask"] = {"enabled": False, "reason": "internal_pass"}
        info["q80_appearance_mask"] = {"enabled": False, "reason": "internal_pass"}
        info["contour_refined_mask"] = {"enabled": False, "reason": "internal_pass"}
        info["recommended_refined_mask"] = None
        return info

    refine_object = object_support.astype(bool) if object_support is not None else threshold_roi
    refine_object_source = "object_support" if object_support is not None else "threshold_roi"
    coarse_refine_mask, coarse_refine_info = coarse_mask_for_refine(coarse_mask)
    info["coarse_refine_seed"] = coarse_refine_info
    run_pamr_refine = os.environ.get("RUN_PAMR_REFINE", "0").lower() not in {"0", "false", "no"}
    if run_pamr_refine:
        info["refined_mask"] = refine_direct_attention_mask(
            image_rgb=edited_image,
            soft_mask=soft_mask,
            coarse_mask=coarse_refine_mask,
            object_mask=refine_object,
            out_dir=out_dir,
            object_source=refine_object_source,
        )
        refined_output_mask = info["refined_mask"].get("output_mask")
        if save_mask_debug and refined_output_mask:
            refined_overlay_path = out_dir / "edit_refined_mask_overlay.png"
            refined_mask = np.array(Image.open(refined_output_mask).convert("L")) > 128
            save_mask_overlay(edited_image, refined_mask, refined_overlay_path)
            info["refined_mask"]["edit_refined_mask_overlay"] = str(refined_overlay_path)
    else:
        info["refined_mask"] = {"enabled": False, "reason": "RUN_PAMR_REFINE=0"}

    active_info = refine_active_contour_mask(
        image_rgb=edited_image,
        coarse_mask=coarse_refine_mask,
        object_mask=refine_object,
        out_dir=out_dir,
        object_source=refine_object_source,
    )
    info["active_contour_mask"] = active_info
    save_active_overlay = True
    if save_active_overlay:
        active_overlay_path = out_dir / "edit_active_contour_overlay.png"
        active_mask = active_info["_refined_mask"]
        save_mask_overlay(edited_image, active_mask, active_overlay_path)
        info["active_contour_mask"]["edit_active_contour_overlay"] = str(active_overlay_path)

    run_q80_appearance_refine = os.environ.get("RUN_Q80_APPEARANCE_REFINE", "1").lower() not in {"0", "false", "no"}
    if run_q80_appearance_refine:
        q80_edge_map = active_info["_edge_map_u8"]
        q80_coarse_roi = active_info["_coarse_roi_mask"]
        info["q80_appearance_mask"] = refine_q80_appearance_mask(
            image_rgb=edited_image,
            coarse_mask=coarse_refine_mask,
            object_mask=refine_object,
            out_dir=out_dir,
            object_source=refine_object_source,
            edge_map_u8=q80_edge_map,
            coarse_roi_mask=q80_coarse_roi,
        )
        q80_mask = np.array(Image.open(info["q80_appearance_mask"]["output_mask"]).convert("L")) > 128
    else:
        q80_mask = None
        q80_coarse_roi = None
        info["q80_appearance_mask"] = {"enabled": False, "reason": "RUN_Q80_APPEARANCE_REFINE=0"}

    run_contour_refine = os.environ.get("RUN_CONTOUR_REFINE", "1").lower() not in {"0", "false", "no"}
    if run_contour_refine and q80_mask is not None and q80_coarse_roi is not None:
        info["contour_refined_mask"] = refine_boundary_preserve_mask(
            image_rgb=edited_image,
            coarse_mask=coarse_refine_mask,
            edge_refined_mask=q80_mask,
            object_mask=refine_object,
            out_dir=out_dir,
            object_source=refine_object_source,
            coarse_roi_mask=q80_coarse_roi,
        )
        if save_mask_debug:
            contour_overlay_path = out_dir / "edit_contour_refined_overlay.png"
            contour_mask = np.array(Image.open(info["contour_refined_mask"]["output_mask"]).convert("L")) > 128
            save_mask_overlay(edited_image, contour_mask, contour_overlay_path)
            info["contour_refined_mask"]["edit_contour_refined_overlay"] = str(contour_overlay_path)
        info["recommended_refined_mask"] = info["contour_refined_mask"]["output_mask"]
    else:
        reason = "RUN_CONTOUR_REFINE=0" if not run_contour_refine else "q80_appearance_mask disabled"
        info["contour_refined_mask"] = {"enabled": False, "reason": reason}
        info["recommended_refined_mask"] = info["q80_appearance_mask"].get("output_mask")

    info["active_contour_mask"] = {
        key: value
        for key, value in info["active_contour_mask"].items()
        if not key.startswith("_")
    }
    (out_dir / "direct_aggregate_summary.json").write_text(json.dumps(strip_runtime_arrays(info), indent=2), encoding="utf-8")
    return info


def save_sequential_direct_masks(args, *, out_dir, pass_results, selected_blocks, block_polarities, old_tar_image, edited_image, object_support=None):
    soft_masks = []
    coarse_masks = []
    all_soft_masks = []
    all_coarse_masks = []
    all_blocks_seen: set[str] = set()
    threshold_rois = []
    pass_summaries = []
    for result in pass_results:
        direct_info = result.get("direct_info") or {}
        pass_dir = Path(result["pass_dir"])
        soft_mask = direct_info.get("_soft_mask")
        coarse_mask = direct_info.get("_coarse_mask")
        all_soft_mask = direct_info.get("_all_block_soft_mask")
        all_coarse_mask = direct_info.get("_all_block_coarse_mask")
        threshold_roi = direct_info.get("_threshold_roi")
        soft_path_value = direct_info.get("soft_mask_npy")
        coarse_path_value = direct_info.get("coarse_mask")
        all_block_info = direct_info.get("all_block_aggregate_mask") or {}
        for block in all_block_info.get("blocks", []):
            all_blocks_seen.add(str(block))
        all_soft_path_value = all_block_info.get("soft_mask_npy")
        all_coarse_path_value = all_block_info.get("coarse_mask")
        target_mask_path = Path(result["target_mask_path"])
        if soft_mask is not None:
            soft_masks.append(np.asarray(soft_mask, dtype=np.float32))
        elif soft_path_value and Path(soft_path_value).is_file():
            soft_masks.append(np.load(soft_path_value).astype(np.float32))
        elif (pass_dir / "soft_mask.npy").is_file():
            soft_masks.append(np.load(pass_dir / "soft_mask.npy").astype(np.float32))
        if coarse_mask is not None:
            coarse_masks.append(np.asarray(coarse_mask).astype(bool))
        elif coarse_path_value and Path(coarse_path_value).is_file():
            coarse_masks.append(load_binary_mask(coarse_path_value, tuple(old_tar_image.shape[:2])))
        elif (pass_dir / "coarse_mask.png").is_file():
            coarse_masks.append(load_binary_mask(pass_dir / "coarse_mask.png", tuple(old_tar_image.shape[:2])))

        if all_soft_mask is not None:
            all_soft_masks.append(np.asarray(all_soft_mask, dtype=np.float32))
        elif all_soft_path_value and Path(all_soft_path_value).is_file():
            all_soft_masks.append(np.load(all_soft_path_value).astype(np.float32))
        elif (pass_dir / "all_block_soft_mask.npy").is_file():
            all_soft_masks.append(np.load(pass_dir / "all_block_soft_mask.npy").astype(np.float32))
        elif soft_mask is not None:
            all_soft_masks.append(np.asarray(soft_mask, dtype=np.float32))
        elif soft_path_value and Path(soft_path_value).is_file():
            all_soft_masks.append(np.load(soft_path_value).astype(np.float32))
        elif (pass_dir / "soft_mask.npy").is_file():
            all_soft_masks.append(np.load(pass_dir / "soft_mask.npy").astype(np.float32))

        if all_coarse_mask is not None:
            all_coarse_masks.append(np.asarray(all_coarse_mask).astype(bool))
        elif all_coarse_path_value and Path(all_coarse_path_value).is_file():
            all_coarse_masks.append(load_binary_mask(all_coarse_path_value, tuple(old_tar_image.shape[:2])))
        elif (pass_dir / "all_block_coarse_mask.png").is_file():
            all_coarse_masks.append(load_binary_mask(pass_dir / "all_block_coarse_mask.png", tuple(old_tar_image.shape[:2])))
        elif coarse_mask is not None:
            all_coarse_masks.append(np.asarray(coarse_mask).astype(bool))
        elif coarse_path_value and Path(coarse_path_value).is_file():
            all_coarse_masks.append(load_binary_mask(coarse_path_value, tuple(old_tar_image.shape[:2])))
        elif (pass_dir / "coarse_mask.png").is_file():
            all_coarse_masks.append(load_binary_mask(pass_dir / "coarse_mask.png", tuple(old_tar_image.shape[:2])))

        if threshold_roi is not None:
            threshold_rois.append(np.asarray(threshold_roi).astype(bool))
        else:
            threshold_rois.append(load_binary_mask(target_mask_path, tuple(old_tar_image.shape[:2])))
        keep_pass_files = bool(getattr(args, "save_mask_debug", False))
        pass_summaries.append({
            "pass_index": int(result["pass_index"]),
            "seed": int(result["seed"]),
            "source_image": str(result["source_image"]) if keep_pass_files or int(result["pass_index"]) == 0 else "previous_pass_edit",
            "target_mask": str(target_mask_path) if keep_pass_files else None,
            "pass_dir": str(pass_dir) if keep_pass_files else None,
            "edit_image": None if result.get("edit_image") is None else str(result["edit_image"]),
            "coarse_mask": coarse_path_value if coarse_path_value else None,
            "selected_block_aggregate_mask": strip_runtime_arrays(direct_info.get("selected_block_aggregate_mask", {})),
            "all_block_aggregate_mask": strip_runtime_arrays(direct_info.get("all_block_aggregate_mask", {})),
            "direct_aggregate_mask": strip_runtime_arrays(direct_info),
        })

    if threshold_rois:
        threshold_roi = np.logical_or.reduce([roi.astype(bool) for roi in threshold_rois])
    else:
        threshold_roi = np.zeros(tuple(old_tar_image.shape[:2]), dtype=bool)
    if coarse_masks:
        coarse_mask = np.logical_or.reduce([mask.astype(bool) for mask in coarse_masks])
    else:
        coarse_mask = threshold_roi.copy()
    if soft_masks:
        soft_mask = np.maximum.reduce([normalize01_np(mask) for mask in soft_masks]).astype(np.float32)
    else:
        soft_mask = coarse_mask.astype(np.float32)
    if all_coarse_masks:
        all_coarse_mask = np.logical_or.reduce([mask.astype(bool) for mask in all_coarse_masks])
    else:
        all_coarse_mask = coarse_mask.copy()
    if all_soft_masks:
        all_soft_mask = np.maximum.reduce([normalize01_np(mask) for mask in all_soft_masks]).astype(np.float32)
    else:
        all_soft_mask = all_coarse_mask.astype(np.float32)

    out_dir.mkdir(parents=True, exist_ok=True)
    save_mask_debug = bool(getattr(args, "save_mask_debug", False))
    default_paths = aggregate_mask_artifact_paths(out_dir, "")
    selected_paths = aggregate_mask_artifact_paths(out_dir, "selected_block")
    all_paths = aggregate_mask_artifact_paths(out_dir, "all_block")
    soft_path = default_paths["soft_mask"]
    soft_npy_path = default_paths["soft_mask_npy"]
    soft_heatmap_path = default_paths["soft_mask_heatmap"]
    soft_overlay_path = default_paths["soft_mask_overlay"]
    coarse_path = default_paths["coarse_mask"]
    coarse_overlay_path = default_paths["coarse_mask_overlay"]
    edit_coarse_overlay_path = default_paths["edit_coarse_mask_overlay"]
    threshold_roi_path = out_dir / "threshold_roi.png"
    threshold_roi_overlay_path = out_dir / "threshold_roi_overlay.png"

    write_aggregate_mask_artifacts(
        default_paths,
        soft_mask=soft_mask,
        coarse_mask=coarse_mask,
        old_tar_image=old_tar_image,
        edited_image=edited_image,
    )
    write_aggregate_mask_artifacts(
        selected_paths,
        soft_mask=soft_mask,
        coarse_mask=coarse_mask,
        old_tar_image=old_tar_image,
        edited_image=edited_image,
    )
    write_aggregate_mask_artifacts(
        all_paths,
        soft_mask=all_soft_mask,
        coarse_mask=all_coarse_mask,
        old_tar_image=old_tar_image,
        edited_image=edited_image,
    )
    Image.fromarray((threshold_roi.astype(np.uint8) * 255)).save(threshold_roi_path)
    save_mask_overlay(old_tar_image, threshold_roi, threshold_roi_overlay_path, color=(0, 255, 0), alpha=0.25)

    info = {
        "mode": "sequential_multi_mask",
        "kind": args.direct_aggregate_kind,
        "selected_blocks_file": args.direct_selected_blocks_file,
        "top_k": len(selected_blocks),
        "selected_blocks": selected_blocks,
        "block_polarities": block_polarities,
        "requested_save_steps": args.save_steps,
        "capture_steps": args.capture_steps,
        "direct_aggregate_steps": args.direct_aggregate_steps,
        "adaptive_check_steps": args.adaptive_check_steps,
        "aggregate": "union_over_sequential_passes",
        "roi": args.direct_roi,
        "hist_threshold_scale": args.direct_hist_threshold_scale,
        "hist_threshold_offset": args.direct_hist_threshold_offset,
        "component_mode": args.direct_component_mode,
        "fill_holes": args.direct_fill_holes,
        "close_iterations": args.direct_close_iterations,
        "dilate_iterations": args.direct_dilate_iterations,
        "save_mask_debug": save_mask_debug,
        "shape_k_removal": {
            "enabled": bool(getattr(args, "shape_k_removal", False)),
            "passes": [result.get("shape_k_removal") for result in pass_results],
        },
        "num_passes": len(pass_results),
        "passes": pass_summaries,
        "soft_mask": str(soft_path),
        "soft_mask_npy": str(soft_npy_path),
        "soft_mask_heatmap": str(soft_heatmap_path),
        "soft_mask_overlay": str(soft_overlay_path),
        "coarse_mask": str(coarse_path),
        "coarse_mask_overlay": str(coarse_overlay_path),
        "edit_coarse_mask_overlay": str(edit_coarse_overlay_path),
        "threshold_roi": str(threshold_roi_path),
        "threshold_roi_overlay": str(threshold_roi_overlay_path),
        "selected_block_aggregate_mask": {
            "blocks": selected_blocks,
            **stringify_paths(selected_paths, True),
        },
        "all_block_aggregate_mask": {
            "blocks": sorted(all_blocks_seen),
            "num_blocks": len(all_blocks_seen),
            **stringify_paths(all_paths, True),
        },
    }

    refine_object = object_support.astype(bool) if object_support is not None else threshold_roi
    refine_object_source = "object_support" if object_support is not None else "threshold_roi"
    coarse_refine_mask, coarse_refine_info = coarse_mask_for_refine(coarse_mask)
    info["coarse_refine_seed"] = coarse_refine_info
    run_pamr_refine = os.environ.get("RUN_PAMR_REFINE", "0").lower() not in {"0", "false", "no"}
    if run_pamr_refine:
        info["refined_mask"] = refine_direct_attention_mask(
            image_rgb=edited_image,
            soft_mask=soft_mask,
            coarse_mask=coarse_refine_mask,
            object_mask=refine_object,
            out_dir=out_dir,
            object_source=refine_object_source,
        )
        refined_output_mask = info["refined_mask"].get("output_mask")
        if save_mask_debug and refined_output_mask:
            refined_overlay_path = out_dir / "edit_refined_mask_overlay.png"
            refined_mask = np.array(Image.open(refined_output_mask).convert("L")) > 128
            save_mask_overlay(edited_image, refined_mask, refined_overlay_path)
            info["refined_mask"]["edit_refined_mask_overlay"] = str(refined_overlay_path)
    else:
        info["refined_mask"] = {"enabled": False, "reason": "RUN_PAMR_REFINE=0"}

    active_info = refine_active_contour_mask(
        image_rgb=edited_image,
        coarse_mask=coarse_refine_mask,
        object_mask=refine_object,
        out_dir=out_dir,
        object_source=refine_object_source,
    )
    info["active_contour_mask"] = active_info
    save_active_overlay = True
    if save_active_overlay:
        active_overlay_path = out_dir / "edit_active_contour_overlay.png"
        active_mask = active_info["_refined_mask"]
        save_mask_overlay(edited_image, active_mask, active_overlay_path)
        info["active_contour_mask"]["edit_active_contour_overlay"] = str(active_overlay_path)

    run_q80_appearance_refine = os.environ.get("RUN_Q80_APPEARANCE_REFINE", "1").lower() not in {"0", "false", "no"}
    run_contour_refine = os.environ.get("RUN_CONTOUR_REFINE", "1").lower() not in {"0", "false", "no"}
    component_refines: list[dict[str, object]] = []
    q80_component_masks: list[np.ndarray] = []
    contour_component_masks: list[np.ndarray] = []
    temp_refine_dirs: list[Path] = []

    if run_q80_appearance_refine:
        for pass_list_index, pass_coarse_mask in enumerate(coarse_masks):
            result = pass_results[pass_list_index]
            pass_index = int(result["pass_index"])
            pass_dir = Path(result["pass_dir"])
            pass_refine_dir = pass_dir if save_mask_debug else out_dir / f"_tmp_refine_pass_{pass_index:02d}"
            if not save_mask_debug:
                temp_refine_dirs.append(pass_refine_dir)
            pass_refine_mask, pass_refine_info = coarse_mask_for_refine(pass_coarse_mask)
            pass_active_info = refine_active_contour_mask(
                image_rgb=edited_image,
                coarse_mask=pass_refine_mask,
                object_mask=refine_object,
                out_dir=pass_refine_dir,
                object_source=refine_object_source,
            )
            if save_mask_debug:
                pass_active_overlay_path = pass_refine_dir / "edit_active_contour_overlay.png"
                save_mask_overlay(edited_image, pass_active_info["_refined_mask"], pass_active_overlay_path)
                pass_active_info["edit_active_contour_overlay"] = str(pass_active_overlay_path)

            q80_info = refine_q80_appearance_mask(
                image_rgb=edited_image,
                coarse_mask=pass_refine_mask,
                object_mask=refine_object,
                out_dir=pass_refine_dir,
                object_source=refine_object_source,
                edge_map_u8=pass_active_info["_edge_map_u8"],
                coarse_roi_mask=pass_active_info["_coarse_roi_mask"],
            )
            q80_output_mask = q80_info.get("output_mask")
            q80_mask_component = np.array(Image.open(q80_output_mask).convert("L")) > 128
            q80_component_masks.append(q80_mask_component)

            component_record: dict[str, object] = {
                "pass_index": pass_index,
                "coarse_refine_seed": pass_refine_info,
                "active_contour_mask": strip_runtime_arrays(pass_active_info),
                "q80_appearance_mask": strip_runtime_arrays(q80_info),
            }

            if run_contour_refine:
                contour_info = refine_boundary_preserve_mask(
                    image_rgb=edited_image,
                    coarse_mask=pass_refine_mask,
                    edge_refined_mask=q80_mask_component,
                    object_mask=refine_object,
                    out_dir=pass_refine_dir,
                    object_source=refine_object_source,
                    coarse_roi_mask=pass_active_info["_coarse_roi_mask"],
                )
                contour_output_mask = contour_info.get("output_mask")
                contour_mask_component = np.array(Image.open(contour_output_mask).convert("L")) > 128
                contour_component_masks.append(contour_mask_component)
                if save_mask_debug:
                    pass_contour_overlay_path = pass_refine_dir / "edit_contour_refined_overlay.png"
                    save_mask_overlay(edited_image, contour_mask_component, pass_contour_overlay_path)
                    contour_info["edit_contour_refined_overlay"] = str(pass_contour_overlay_path)
                component_record["contour_refined_mask"] = strip_runtime_arrays(contour_info)

            component_refines.append(component_record)

        if q80_component_masks:
            q80_mask = np.logical_or.reduce([mask.astype(bool) for mask in q80_component_masks])
            q80_output_path = out_dir / "q80_appearance_mask.png"
            Image.fromarray((q80_mask.astype(np.uint8) * 255), mode="L").save(q80_output_path)
            info["q80_appearance_mask"] = {
                "enabled": True,
                "method": "per_pass_q80_appearance_union",
                "output_mask": str(q80_output_path),
                "object_source": refine_object_source,
                "mask_area": int(q80_mask.sum()),
                "component_count": connected_component_count(q80_mask),
                "num_passes": len(q80_component_masks),
                "component_refines": component_refines,
            }
            if save_mask_debug:
                q80_overlay_path = out_dir / "edit_q80_appearance_overlay.png"
                save_mask_overlay(edited_image, q80_mask, q80_overlay_path)
                info["q80_appearance_mask"]["edit_q80_appearance_overlay"] = str(q80_overlay_path)
        else:
            q80_mask = None
            info["q80_appearance_mask"] = {"enabled": False, "reason": "no per-pass q80 masks"}
    else:
        q80_mask = None
        info["q80_appearance_mask"] = {"enabled": False, "reason": "RUN_Q80_APPEARANCE_REFINE=0"}

    if run_contour_refine and contour_component_masks:
        contour_mask = np.logical_or.reduce([mask.astype(bool) for mask in contour_component_masks])
        contour_output_path = out_dir / "contour_refined_mask.png"
        Image.fromarray((contour_mask.astype(np.uint8) * 255), mode="L").save(contour_output_path)
        info["contour_refined_mask"] = {
            "enabled": True,
            "method": "per_pass_contour_refine_union",
            "output_mask": str(contour_output_path),
            "object_source": refine_object_source,
            "mask_area": int(contour_mask.sum()),
            "component_count": connected_component_count(contour_mask),
            "num_passes": len(contour_component_masks),
            "component_refines": component_refines,
        }
        if save_mask_debug:
            contour_overlay_path = out_dir / "edit_contour_refined_overlay.png"
            save_mask_overlay(edited_image, contour_mask, contour_overlay_path)
            info["contour_refined_mask"]["edit_contour_refined_overlay"] = str(contour_overlay_path)
        info["recommended_refined_mask"] = str(contour_output_path)
    else:
        reason = "RUN_CONTOUR_REFINE=0" if not run_contour_refine else "q80_appearance_mask disabled"
        info["contour_refined_mask"] = {"enabled": False, "reason": reason}
        info["recommended_refined_mask"] = info["q80_appearance_mask"].get("output_mask")

    for temp_refine_dir in temp_refine_dirs:
        if temp_refine_dir.exists():
            shutil.rmtree(temp_refine_dir)

    info["active_contour_mask"] = {
        key: value
        for key, value in info["active_contour_mask"].items()
        if not key.startswith("_")
    }
    (out_dir / "direct_aggregate_summary.json").write_text(json.dumps(strip_runtime_arrays(info), indent=2), encoding="utf-8")
    return info

def main() -> None:
    args = parse_args()
    if args.reference_mask_dilate_iterations < 0:
        raise ValueError("--reference-mask-dilate-iterations must be non-negative")
    if not 0.0 <= args.reference_mask_vertical_shift_ratio < 1.0:
        raise ValueError("--reference-mask-vertical-shift-ratio must be in [0, 1)")
    args.nunchaku = False
    args.nunchaku_transformer_path = None
    args.nunchaku_lora_path = None
    args.nunchaku_lora_strength = None
    args.nunchaku_precision = "auto"
    anomaly_refs = build_anomaly_refs(args.anomalies, args.ref_ids)
    if args.samples_per_pair is None:
        samples_per_pair = [int(args.samples_per_anomaly)] * len(anomaly_refs)
    else:
        if len(args.samples_per_pair) != len(anomaly_refs):
            raise ValueError("--samples-per-pair must have the same length as --anomalies/--ref-ids.")
        samples_per_pair = [int(value) for value in args.samples_per_pair]
    if any(value < 0 for value in samples_per_pair):
        raise ValueError("--samples-per-pair values must be non-negative.")
    selected_blocks, block_polarities = configure_direct(args)
    shape_k_blocks = resolve_shape_k_blocks(args.shape_k_blocks, args.save_blocks, args.shape_k_block_scope)
    source_root = Path(args.source_root)
    ref_image_root = Path(args.ref_image_root)
    ref_mask_root = Path(args.ref_mask_root)
    object_support_root = Path(args.object_support_root) if args.object_support_root else None
    object_attention_root = Path(args.object_attention_root) if args.object_attention_root else None
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    size = (args.size, args.size)
    source_images = list_images(source_root)
    if not source_images:
        raise FileNotFoundError(f"No source images found under {source_root}")
    run_config_args = vars(args).copy()
    # Keep pre-policy random_object run configs byte-compatible for safe resume.
    # The transform settings are critical only when that policy is selected.
    if args.target_mask_source != "reference_vertical_mixed":
        run_config_args.pop("reference_mask_dilate_iterations", None)
        run_config_args.pop("reference_mask_vertical_shift_ratio", None)
    run_config = {
        **run_config_args,
        "anomaly_refs": anomaly_refs,
        "samples_per_pair_resolved": samples_per_pair,
        "num_source_images": len(source_images),
        "resume_config_schema_version": RESUME_CONFIG_SCHEMA_VERSION,
        "direct_selected_blocks_resolved": selected_blocks,
        "direct_block_polarities_resolved": block_polarities,
        "adaptive_selected_blocks_resolved": args.adaptive_selected_blocks,
        "adaptive_block_polarities_resolved": args.adaptive_block_polarities,
        "shape_k_blocks_resolved": None if shape_k_blocks is None else sorted(shape_k_blocks),
        "refinement_runtime_config": refinement_runtime_config(),
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    run_config_path = out_root / "run_config.json"
    ensure_resume_run_config_compatible(run_config_path, run_config, overwrite=args.overwrite)
    preflight_existing_samples_for_resume(
        args,
        anomaly_refs=anomaly_refs,
        samples_per_pair=samples_per_pair,
        source_images=source_images,
        ref_image_root=ref_image_root,
        ref_mask_root=ref_mask_root,
        out_root=out_root,
        selected_blocks=selected_blocks,
    )
    run_config_path.write_text(json.dumps(run_config, indent=2), encoding="utf-8")
    with (out_root / "run_config_history.jsonl").open("a", encoding="utf-8") as history_file:
        history_file.write(json.dumps(run_config, sort_keys=True) + "\n")
    log_path = Path(args.log_file) if args.log_file else out_root / "run_log.csv"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("a", newline="", encoding="utf-8")
    log_writer = csv.DictWriter(log_file, fieldnames=["index", "anomaly", "ref_id", "seed", "source_image", "target_mask", "generation_mode", "sequential_passes", "ref_image", "ref_mask", "sample_dir", "edit_image", "coarse_mask", "elapsed_sec", "status", "error"])
    if log_path.stat().st_size == 0:
        log_writer.writeheader()

    adaptive_log_path = Path(args.adaptive_log_file) if args.adaptive_log_file else out_root / "adaptive_log.csv"
    adaptive_log_path.parent.mkdir(parents=True, exist_ok=True)
    adaptive_log_file = adaptive_log_path.open("a", newline="", encoding="utf-8")
    adaptive_log_writer = csv.DictWriter(
        adaptive_log_file,
        fieldnames=[
            "index",
            "anomaly",
            "ref_id",
            "seed",
            "source_image",
            "generation_mode",
            "sequential_passes",
            "sample_dir",
            "adaptive_enabled",
            "adaptive_triggered",
            "adaptive_score_mode",
            "adaptive_aggregate_score_kind",
            "num_adaptive_events",
            "num_boosts",
            "boost_steps",
            "adaptive_event_passes",
            "final_scale",
            "max_scale_after",
            "mean_adaptive_score",
            "min_adaptive_score",
            "threshold",
            "status",
            "error",
        ],
    )
    if adaptive_log_path.stat().st_size == 0:
        adaptive_log_writer.writeheader()
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Loading full FLUX batch pipeline", flush=True)
    pipe, redux = load_pipelines(args)
    lora_runtime_audit = getattr(pipe, "_insert_anything_lora_audit", None)
    run_config["lora_runtime_audit"] = lora_runtime_audit
    run_config_path.write_text(json.dumps(run_config, indent=2), encoding="utf-8")
    with (out_root / "run_config_history.jsonl").open("a", encoding="utf-8") as history_file:
        history_file.write(json.dumps(run_config, sort_keys=True) + "\n")
    generator_device = "cuda" if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu"
    print(
        "Batch attention generation: "
        f"anomalies/ref_ids={anomaly_refs}, samples_per_anomaly={args.samples_per_anomaly}, "
        f"samples_per_pair={samples_per_pair}, direct_aggregate_steps={args.direct_aggregate_steps}, "
        f"adaptive_check_steps={args.adaptive_check_steps}, capture_steps={args.capture_steps}",
        flush=True,
    )
    print(f"Model: full FLUX quantize={args.full_flux_quantize}, cpu_offload={args.cpu_offload}, sequential_cpu_offload={args.sequential_cpu_offload}", flush=True)
    print(
        "Adaptive ref injection: "
        f"enabled={args.adaptive_ref_injection}, roi={args.adaptive_roi}, score_mode={args.adaptive_score_mode}, "
        f"attention_kind={args.adaptive_aggregate_kind}, blocks={len(args.adaptive_selected_blocks)}, aggregate_kind={args.adaptive_aggregate_score_kind}, "
        f"inside_ratio={args.adaptive_aggregate_min_inside_ratio}, inside_mean={args.adaptive_aggregate_min_inside_mean}, "
        f"aggregate_min_area_ratio={args.adaptive_aggregate_min_area_ratio}, token_start={args.adaptive_ref_token_start}, "
        f"ref_threshold={args.adaptive_ref_attention_threshold}, base/max={args.adaptive_ref_base_scale}/{args.adaptive_ref_max_scale}, "
        f"boost={args.adaptive_ref_boost}, decay={args.adaptive_ref_decay}",
        flush=True,
    )
    print(
        "Ref token perturbation: "
        f"noise_std={args.ref_token_noise_std}, dropout={args.ref_token_dropout}, "
        f"scale_jitter={args.ref_token_scale_jitter}, span_dropout={args.ref_token_span_dropout}, "
        f"span_len={args.ref_token_span_len}, seed_offset={args.ref_token_perturb_seed_offset}",
        flush=True,
    )
    print(
        "Ref image augmentation: "
        f"bank_size={args.ref_augment_bank_size}, rotate={args.ref_augment_rotate}, "
        f"scale_jitter={args.ref_augment_scale_jitter}, translate_ratio={args.ref_augment_translate_ratio}, "
        f"brightness={args.ref_augment_brightness}, contrast={args.ref_augment_contrast}, "
        f"seed_offset={args.ref_augment_seed_offset}",
        flush=True,
    )
    shape_k_block_label = (
        args.shape_k_blocks
        if args.shape_k_blocks
        else ("all" if shape_k_blocks is None else f"scope:{args.shape_k_block_scope}({len(shape_k_blocks)})")
    )
    print(
        "Shape-Orthogonal K: "
        f"enabled={args.shape_k_removal}, mode={args.shape_k_mode}, eta={args.shape_k_eta}, "
        f"suppress={args.shape_k_suppress_scale}, ratio={args.shape_k_start_ratio}-{args.shape_k_end_ratio}, "
        f"steps={args.shape_k_start_step}-{args.shape_k_end_step}, "
        f"scope={args.shape_k_block_scope}, edge={args.shape_k_edge_method}, blocks={shape_k_block_label}",
        flush=True,
    )
    defect_sample_offsets: list[int] = []
    defect_sample_totals: dict[str, int] = {}
    for pair_index, (pair_anomaly, _pair_ref_id) in enumerate(anomaly_refs):
        defect_sample_offsets.append(defect_sample_totals.get(pair_anomaly, 0))
        defect_sample_totals[pair_anomaly] = (
            defect_sample_totals.get(pair_anomaly, 0) + samples_per_pair[pair_index]
        )
    try:
        for anomaly_offset, (anomaly, ref_id) in enumerate(anomaly_refs):
            pair_sample_count = samples_per_pair[anomaly_offset]
            if pair_sample_count == 0:
                print(
                    f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] "
                    f"Skipping anomaly={anomaly} ref_id={ref_id} because samples_per_pair=0",
                    flush=True,
                )
                continue
            ref_image = resolve_id_file(ref_image_root / anomaly, ref_id)
            ref_mask = resolve_id_file(ref_mask_root / anomaly, ref_id, suffixes=("_mask", ""))
            pair_seed = args.seed + anomaly_offset * 1000003
            source_count = args.start_index + pair_sample_count
            sources = sample_sources(source_images, source_count, pair_seed)[args.start_index:]
            anomaly_dir = out_root / anomaly / f"ref_{ref_id}"
            anomaly_dir.mkdir(parents=True, exist_ok=True)
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Preparing anomaly={anomaly} ref_id={ref_id} ref_image={ref_image} ref_mask={ref_mask}", flush=True)
            masked_ref_image = prepare_reference(ref_image, ref_mask, size)
            ref_bank_size = max(1, int(args.ref_augment_bank_size))
            ref_condition_bank = []
            for bank_index in range(ref_bank_size):
                if bank_index == 0:
                    bank_ref_image = masked_ref_image
                    bank_info: dict[str, object] = {"enabled": False}
                else:
                    augment_seed = pair_seed + int(args.ref_augment_seed_offset) + bank_index
                    bank_ref_image, bank_info = augment_masked_reference_image(
                        masked_ref_image,
                        np.random.default_rng(augment_seed),
                        args.ref_augment_rotate,
                        args.ref_augment_scale_jitter,
                        args.ref_augment_translate_ratio,
                        args.ref_augment_brightness,
                        args.ref_augment_contrast,
                    )
                    bank_info["seed"] = int(augment_seed)
                bank_info["bank_index"] = int(bank_index)
                bank_shape_map = reference_shape_map_from_image(
                    bank_ref_image,
                    method=args.shape_k_edge_method,
                    foreground_threshold=args.shape_k_foreground_threshold,
                )
                ref_condition_bank.append({
                    "masked_ref_image": bank_ref_image,
                    "reference_shape_map": bank_shape_map,
                    "pipe_prior_output": redux(Image.fromarray(bank_ref_image)),
                    "info": bank_info,
                })
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            progress = tqdm(enumerate(sources, start=args.start_index), total=len(sources), desc=f"{anomaly}/ref_{ref_id}")
            for sample_index, source_image in progress:
                sample_seed = args.seed + anomaly_offset * 1000003 + sample_index
                sample_dir = anomaly_dir / f"{sample_index:03d}"
                sample_dir.mkdir(parents=True, exist_ok=True)
                output_image = sample_dir / "edit.png"
                output_coarse = sample_dir / "coarse_mask.png"
                sample_metadata = sample_dir / "metadata.json"
                resume_action = decide_sample_resume_action(
                    sample_dir,
                    overwrite=args.overwrite,
                    **sample_resume_expectations(
                        args,
                        anomaly=anomaly,
                        ref_id=ref_id,
                        ref_image=ref_image,
                        ref_mask=ref_mask,
                        sample_index=sample_index,
                        source_image=source_image,
                        sample_seed=sample_seed,
                        selected_blocks=selected_blocks,
                    ),
                )
                if resume_action == "skip":
                    log_writer.writerow({"index": sample_index, "anomaly": anomaly, "ref_id": ref_id, "seed": sample_seed, "source_image": str(source_image), "target_mask": str(sample_dir / "threshold_roi.png"), "generation_mode": "", "sequential_passes": "", "ref_image": str(ref_image), "ref_mask": str(ref_mask), "sample_dir": str(sample_dir), "edit_image": str(output_image), "coarse_mask": str(output_coarse), "elapsed_sec": 0.0, "status": "skip", "error": ""})
                    log_file.flush()
                    if args.adaptive_score_mode == "aggregate_mask":
                        if args.adaptive_aggregate_score_kind == "inside_coverage":
                            skip_threshold = args.adaptive_aggregate_min_inside_ratio
                        elif args.adaptive_aggregate_score_kind == "inside_mean":
                            skip_threshold = args.adaptive_aggregate_min_inside_mean
                        elif args.adaptive_aggregate_score_kind == "coarse_area_ratio":
                            skip_threshold = args.adaptive_aggregate_min_area_ratio
                        else:
                            skip_threshold = 1.0
                    else:
                        skip_threshold = args.adaptive_ref_attention_threshold
                    adaptive_log_writer.writerow({"index": sample_index, "anomaly": anomaly, "ref_id": ref_id, "seed": sample_seed, "source_image": str(source_image), "generation_mode": "", "sequential_passes": "", "sample_dir": str(sample_dir), "adaptive_enabled": args.adaptive_ref_injection, "adaptive_triggered": "", "adaptive_score_mode": args.adaptive_score_mode, "adaptive_aggregate_score_kind": args.adaptive_aggregate_score_kind if args.adaptive_score_mode == "aggregate_mask" else "", "num_adaptive_events": "", "num_boosts": "", "boost_steps": "", "final_scale": "", "max_scale_after": "", "mean_adaptive_score": "", "min_adaptive_score": "", "threshold": skip_threshold, "status": "skip", "error": ""})
                    adaptive_log_file.flush()
                    continue
                if args.overwrite:
                    clean_sample_dir(sample_dir)
                start_time = time.time()
                status = "ok"
                error = ""
                target_mask_path = sample_dir / ("generated_target_mask.png" if args.save_mask_debug else "_tmp_generated_target_mask.png")
                temporary_paths: list[Path] = []
                if not args.save_mask_debug:
                    temporary_paths.append(target_mask_path)
                object_support_image = sample_dir / "object_support.png"
                object_attention_image = sample_dir / "object_attention_map.png"
                generated_mask_info = {"target_mask_source": args.target_mask_source}
                direct_aggregate_info = None
                adaptive_history: list[dict[str, object]] = []
                adaptive_final_scale = None
                ref_token_perturbation: dict[str, object] = {"enabled": False}
                ref_image_augmentation: dict[str, object] = {"enabled": False, "selected_bank_index": 0, "bank_size": 1}
                n_attn = 0
                try:
                    object_support_path = resolve_matching_file(object_support_root, source_image) if object_support_root is not None else None
                    object_attention_path = None
                    if object_attention_root is not None:
                        try:
                            object_attention_path = resolve_matching_file(object_attention_root, source_image)
                        except FileNotFoundError:
                            object_attention_path = None
                    if args.target_mask_source == "fixed":
                        if args.fixed_target_mask is None:
                            raise ValueError("--fixed-target-mask is required when --target-mask-source fixed")
                        source_rgb = load_rgb(source_image)
                        source_shape = tuple(source_rgb.shape[:2])
                        generated_mask = load_binary_mask(args.fixed_target_mask, source_shape).astype(np.uint8)
                        if object_support_path is not None:
                            object_support, object_attention, support_info = load_object_support_from_file(
                                object_support_path,
                                object_attention_path,
                                size=(source_shape[1], source_shape[0]),
                                erosion=args.object_support_erosion,
                            )
                        else:
                            object_support, object_attention, support_info = object_support_from_image(
                                source_rgb,
                                erosion=args.object_support_erosion,
                            )
                        support_info["object_prompt"] = args.object_prompt
                        generated_mask_info = {
                            "target_mask_source": "fixed",
                            "fixed_target_mask": args.fixed_target_mask,
                            "target_area": int(generated_mask.sum()),
                            "object_overlap": float((generated_mask & object_support.astype(np.uint8)).sum() / max(1, int(generated_mask.sum()))),
                            "object_support": support_info,
                            "status": "loaded_fixed_mask",
                        }
                    elif args.target_mask_source == "reference_vertical_mixed":
                        defect_global_ordinal = (
                            defect_sample_offsets[anomaly_offset]
                            + sample_index
                            - args.start_index
                        )
                        mixed_variant = reference_vertical_mixed_variant(defect_global_ordinal)
                        mixed_policy = {
                            "schedule": ["reference_up", "reference_down", "random_object"],
                            "cycle_length": 3,
                            "cycle_position": int(defect_global_ordinal % 3),
                            "defect_global_ordinal": int(defect_global_ordinal),
                            "variant": mixed_variant,
                            "reference_fraction": "2/3",
                            "original_random_object_fraction": "1/3",
                        }
                        if mixed_variant == "random_object":
                            mask_rng = np.random.default_rng(sample_seed + 300000)
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
                            generated_mask_info = {
                                **generated_mask_info,
                                "target_mask_source": "reference_vertical_mixed",
                                "base_target_mask_source": "random_object",
                                "mixed_policy": mixed_policy,
                            }
                        else:
                            source_rgb = load_rgb(source_image)
                            source_shape = tuple(source_rgb.shape[:2])
                            reference_binary = load_binary_mask(ref_mask, source_shape).astype(np.uint8)
                            generated_mask, transform_info = reference_vertical_mask(
                                reference_binary,
                                mixed_variant,
                                dilate_iterations=args.reference_mask_dilate_iterations,
                                vertical_shift_ratio=args.reference_mask_vertical_shift_ratio,
                            )
                            if object_support_path is not None:
                                object_support, object_attention, support_info = load_object_support_from_file(
                                    object_support_path,
                                    object_attention_path,
                                    size=(source_shape[1], source_shape[0]),
                                    erosion=args.object_support_erosion,
                                )
                            else:
                                object_support, object_attention, support_info = object_support_from_image(
                                    source_rgb,
                                    erosion=args.object_support_erosion,
                                )
                            support_info["object_prompt"] = args.object_prompt
                            generated_mask_info = {
                                "target_mask_source": "reference_vertical_mixed",
                                "base_target_mask_source": "reference_mask",
                                "conditioning_reference_mask": str(ref_mask),
                                "reference_shape_used": True,
                                "mixed_policy": mixed_policy,
                                "reference_transform": transform_info,
                                "target_area": int(generated_mask.sum()),
                                "object_overlap": float(
                                    (generated_mask & object_support.astype(np.uint8)).sum()
                                    / max(1, int(generated_mask.sum()))
                                ),
                                "object_support": support_info,
                                "status": "loaded_enlarged_shifted_reference_mask",
                            }
                    else:
                        mask_rng = np.random.default_rng(sample_seed + 300000)
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
                    Image.fromarray((generated_mask * 255).astype(np.uint8)).save(target_mask_path)
                    Image.fromarray((object_support * 255).astype(np.uint8)).save(object_support_image)
                    Image.fromarray(object_attention.astype(np.uint8)).save(object_attention_image)
                    if len(ref_condition_bank) > 1:
                        ref_bank_rng = np.random.default_rng(sample_seed + int(args.ref_augment_seed_offset))
                        ref_bank_index = int(ref_bank_rng.integers(0, len(ref_condition_bank)))
                    else:
                        ref_bank_index = 0
                    ref_bank_entry = ref_condition_bank[ref_bank_index]
                    sample_masked_ref_image = ref_bank_entry["masked_ref_image"]
                    sample_reference_shape_map = ref_bank_entry["reference_shape_map"]
                    ref_image_augmentation = dict(ref_bank_entry["info"])
                    ref_image_augmentation["selected_bank_index"] = int(ref_bank_index)
                    ref_image_augmentation["bank_size"] = int(len(ref_condition_bank))
                    def run_generation_pass(
                        pass_source_image: Path,
                        pass_target_mask_path: Path,
                        pass_out_dir: Path,
                        pass_seed: int,
                        pass_index: int,
                        save_debug: bool,
                        write_outputs: bool,
                        run_refine: bool,
                    ) -> dict[str, object]:
                        nonlocal n_attn, ref_token_perturbation
                        if save_debug or write_outputs:
                            pass_out_dir.mkdir(parents=True, exist_ok=True)
                        diptych_ref_tar, mask_diptych, old_tar_image, extra_sizes, crop_box = prepare_target(
                            pass_source_image,
                            pass_target_mask_path,
                            sample_masked_ref_image,
                            size,
                        )
                        if save_debug:
                            Image.fromarray(sample_masked_ref_image).save(pass_out_dir / "debug_masked_reference.png")
                            Image.fromarray((normalize01_np(sample_reference_shape_map) * 255.0).clip(0, 255).astype(np.uint8)).save(
                                pass_out_dir / "debug_reference_shape_map.png"
                            )
                            diptych_ref_tar.save(pass_out_dir / "debug_diptych_input.png")
                            mask_diptych.save(pass_out_dir / "debug_diptych_mask.png")
                        target_base_rgb = np.array(diptych_ref_tar.convert("RGB"))[:, mask_diptych.size[0] // 2 :, :]
                        recorder = FluxAttentionRecorder(
                            out_dir=pass_out_dir,
                            height=mask_diptych.size[1],
                            width=mask_diptych.size[0],
                            save_kinds={args.direct_aggregate_kind},
                            pca_kinds=set(),
                            target_base_rgb=target_base_rgb,
                            max_blocks=args.max_blocks,
                            max_steps=args.max_steps_to_save,
                            save_steps=set(args.capture_steps) if args.capture_steps else None,
                            save_blocks=set(args.save_blocks) if args.save_blocks else None,
                            save_visuals=False,
                            save_raw_maps=False,
                            direct_aggregate_kind=args.direct_aggregate_kind,
                            direct_aggregate_steps=set(args.direct_aggregate_steps) if args.direct_aggregate_steps else None,
                            adaptive_check_steps=set(args.adaptive_check_steps) if args.adaptive_check_steps else None,
                            adaptive_aggregate_kind=args.adaptive_aggregate_kind if args.adaptive_ref_injection and args.adaptive_score_mode == "aggregate_mask" else None,
                            adaptive_blocks=set(args.adaptive_selected_blocks) if args.adaptive_ref_injection and args.adaptive_score_mode == "aggregate_mask" else None,
                            adaptive_block_polarities=args.adaptive_block_polarities,
                            record_all_direct_blocks=True,
                            direct_block_polarities=block_polarities,
                            adaptive_ref_condition=args.adaptive_ref_injection and args.adaptive_score_mode == "ref_condition_mass",
                            adaptive_ref_token_start=args.adaptive_ref_token_start,
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
                            reference_shape_map=sample_reference_shape_map,
                        )
                        n_attn = register_attention_recorders(pipe, recorder)
                        captured_steps = set(args.capture_steps) if args.capture_steps else None
                        prior_kwargs = dict(ref_bank_entry["pipe_prior_output"])
                        base_prompt_embeds = prior_kwargs.get("prompt_embeds")
                        ref_token_perturb_enabled = bool(
                            args.ref_token_noise_std > 0.0
                            or args.ref_token_dropout > 0.0
                            or args.ref_token_scale_jitter > 0.0
                            or args.ref_token_span_dropout > 0.0
                        )
                        if base_prompt_embeds is None and ref_token_perturb_enabled:
                            raise RuntimeError("Reference token perturbation requires Redux prompt_embeds.")
                        if base_prompt_embeds is not None:
                            perturb_seed = sample_seed + int(args.ref_token_perturb_seed_offset)
                            perturb_generator = prompt_embed_generator(base_prompt_embeds, perturb_seed)
                            base_prompt_embeds, pass_ref_token_perturbation = perturb_reference_condition_tokens(
                                base_prompt_embeds,
                                args.adaptive_ref_token_start,
                                args.ref_token_noise_std,
                                args.ref_token_dropout,
                                args.ref_token_scale_jitter,
                                args.ref_token_span_dropout,
                                args.ref_token_span_len,
                                perturb_generator,
                            )
                            pass_ref_token_perturbation["seed"] = int(perturb_seed)
                            if pass_index == 0:
                                ref_token_perturbation = pass_ref_token_perturbation
                            prior_kwargs["prompt_embeds"] = base_prompt_embeds
                        base_ref_scale = max(0.0, float(args.adaptive_ref_base_scale))
                        max_ref_scale = max(base_ref_scale, float(args.adaptive_ref_max_scale))
                        ref_scale = base_ref_scale
                        pass_adaptive_final_scale = None
                        adaptive_threshold_roi = None
                        if args.adaptive_ref_injection:
                            if base_prompt_embeds is None:
                                raise RuntimeError("Adaptive reference injection requires Redux prompt_embeds.")
                            prior_kwargs["prompt_embeds"] = scale_reference_condition_tokens(
                                base_prompt_embeds,
                                base_prompt_embeds,
                                args.adaptive_ref_token_start,
                                ref_scale,
                            )
                            if args.adaptive_score_mode == "aggregate_mask":
                                attention_support = attention_crop_support(tuple(old_tar_image.shape[:2]), crop_box)
                                if args.adaptive_roi == "initial_mask":
                                    adaptive_threshold_roi = load_binary_mask(pass_target_mask_path, tuple(old_tar_image.shape[:2]))
                                elif args.adaptive_roi == "object":
                                    adaptive_threshold_roi = object_roi_from_image(pass_source_image, tuple(old_tar_image.shape[:2]))
                                else:
                                    adaptive_threshold_roi = np.ones(tuple(old_tar_image.shape[:2]), dtype=bool)
                                adaptive_threshold_roi = adaptive_threshold_roi.astype(bool) & attention_support

                        def callback_on_step_end(_pipe, step_index, _timestep, callback_kwargs):
                            nonlocal ref_scale, pass_adaptive_final_scale, adaptive_final_scale
                            if args.log_attention_steps and (captured_steps is None or step_index in captured_steps or step_index == args.num_inference_steps - 1):
                                tag = "captured" if captured_steps is None or step_index in captured_steps else "done"
                                print(f"[attention] {anomaly}/{sample_index:03d} pass={pass_index} step {step_index} ({tag})", flush=True)
                            if args.adaptive_ref_injection:
                                stat = None
                                threshold = max(float(args.adaptive_ref_attention_threshold), 1e-8)
                                if args.adaptive_score_mode == "aggregate_mask":
                                    step_soft_result = recorder.direct_step_soft_mask(
                                        step_index=step_index,
                                        selected_blocks=args.adaptive_selected_blocks,
                                        source_shape_hw=tuple(old_tar_image.shape[:2]),
                                        extra_sizes=extra_sizes,
                                        crop_box=crop_box,
                                        block_polarities=args.adaptive_block_polarities,
                                    )
                                    if step_soft_result is not None and adaptive_threshold_roi is not None:
                                        step_soft_mask, step_counts = step_soft_result
                                        step_coarse_mask, step_hist_threshold, step_otsu_threshold = threshold_soft_mask(
                                            step_soft_mask,
                                            adaptive_threshold_roi,
                                            scale=args.direct_hist_threshold_scale,
                                            offset=args.direct_hist_threshold_offset,
                                            component_mode=args.direct_component_mode,
                                            fill=args.direct_fill_holes,
                                            close_iterations=args.direct_close_iterations,
                                            dilate_iterations=args.direct_dilate_iterations,
                                        )
                                        roi_mask = adaptive_threshold_roi.astype(bool)
                                        roi_area = max(1, int(roi_mask.sum()))
                                        coarse_area = int(step_coarse_mask.sum())
                                        score_map = normalize01_np(step_soft_mask)
                                        inside_values = score_map[roi_mask]
                                        inside_mean = float(inside_values.mean()) if inside_values.size else 0.0
                                        inside_coverage = float((inside_values >= float(step_hist_threshold)).mean()) if inside_values.size else 0.0
                                        coarse_area_ratio = float(coarse_area / roi_area)
                                        object_mask = object_support.astype(bool) if object_support is not None else np.ones_like(roi_mask, dtype=bool)
                                        outside_ring_dilate = max(0, int(args.adaptive_aggregate_outside_ring_dilate))
                                        if outside_ring_dilate > 0:
                                            kernel_size = 2 * outside_ring_dilate + 1
                                            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
                                            outside_mask = cv2.dilate(roi_mask.astype(np.uint8), kernel, iterations=1).astype(bool)
                                            outside_mask = outside_mask & object_mask & ~roi_mask
                                        else:
                                            outside_mask = object_mask & ~roi_mask
                                        if not outside_mask.any():
                                            outside_mask = ~roi_mask
                                        outside_values = score_map[outside_mask]
                                        outside_mean = float(outside_values.mean()) if outside_values.size else 0.0
                                        contrast_ratio = float((inside_mean + 1e-6) / (outside_mean + 1e-6))
                                        contrast_margin = float(inside_mean - outside_mean)
                                        if args.adaptive_aggregate_score_kind == "inside_coverage":
                                            score = inside_coverage
                                            threshold = max(float(args.adaptive_aggregate_min_inside_ratio), 1e-8)
                                        elif args.adaptive_aggregate_score_kind == "inside_mean":
                                            score = inside_mean
                                            threshold = max(float(args.adaptive_aggregate_min_inside_mean), 1e-8)
                                        elif args.adaptive_aggregate_score_kind == "coarse_area_ratio":
                                            score = coarse_area_ratio
                                            threshold = max(float(args.adaptive_aggregate_min_area_ratio), 1e-8)
                                        elif args.adaptive_aggregate_score_kind == "inside_outside_contrast":
                                            inside_ratio_target = max(float(args.adaptive_aggregate_min_inside_ratio), 1e-8)
                                            contrast_ratio_target = max(float(args.adaptive_aggregate_min_contrast_ratio), 1e-8)
                                            contrast_inside_target = max(float(args.adaptive_aggregate_min_contrast_inside_mean), 1e-8)
                                            contrast_margin_target = max(float(args.adaptive_aggregate_min_contrast_margin), 1e-8)
                                            score = min(
                                                inside_coverage / inside_ratio_target,
                                                inside_mean / contrast_inside_target,
                                                contrast_ratio / contrast_ratio_target,
                                                contrast_margin / contrast_margin_target,
                                            )
                                            threshold = 1.0
                                        else:
                                            inside_ratio_target = max(float(args.adaptive_aggregate_min_inside_ratio), 1e-8)
                                            inside_mean_target = max(float(args.adaptive_aggregate_min_inside_mean), 1e-8)
                                            area_ratio_target = max(float(args.adaptive_aggregate_min_area_ratio), 1e-8)
                                            score = min(
                                                inside_coverage / inside_ratio_target,
                                                inside_mean / inside_mean_target,
                                                coarse_area_ratio / area_ratio_target,
                                            )
                                            threshold = 1.0
                                        stat = {
                                            "score": score,
                                            "min": score,
                                            "max": score,
                                            "blocks": len(step_counts),
                                            "aggregate_score_kind": args.adaptive_aggregate_score_kind,
                                            "inside_mean": inside_mean,
                                            "inside_coverage": inside_coverage,
                                            "outside_mean": outside_mean,
                                            "outside_area": int(outside_mask.sum()),
                                            "contrast_ratio": contrast_ratio,
                                            "contrast_margin": contrast_margin,
                                            "coarse_area_ratio": coarse_area_ratio,
                                            "coarse_area": coarse_area,
                                            "roi_area": roi_area,
                                            "hist_threshold": float(step_hist_threshold),
                                            "hist_otsu_threshold": float(step_otsu_threshold),
                                        }
                                else:
                                    stat = recorder.adaptive_ref_score(step_index)
                                if stat is not None and "prompt_embeds" in callback_kwargs:
                                    score = float(stat["score"])
                                    old_scale = float(ref_scale)
                                    action = "keep"
                                    if score < threshold:
                                        deficit = max(0.0, (threshold - score) / threshold)
                                        factor = 1.0 + max(0.0, float(args.adaptive_ref_boost)) * deficit
                                        boosted_scale = min(max_ref_scale, max(base_ref_scale, ref_scale * factor))
                                        trigger_min_scale = min(max_ref_scale, max(base_ref_scale, float(args.adaptive_ref_trigger_min_scale)))
                                        if trigger_min_scale > base_ref_scale:
                                            boosted_scale = max(boosted_scale, trigger_min_scale)
                                        ref_scale = boosted_scale
                                        action = "boost"
                                    elif (
                                        float(args.adaptive_ref_decay) < 1.0
                                        and ref_scale > base_ref_scale
                                        and score >= float(args.adaptive_ref_decay_min_score)
                                    ):
                                        ref_scale = max(base_ref_scale, ref_scale * max(0.0, float(args.adaptive_ref_decay)))
                                        action = "decay"
                                    callback_kwargs["prompt_embeds"] = scale_reference_condition_tokens(
                                        callback_kwargs["prompt_embeds"],
                                        base_prompt_embeds,
                                        args.adaptive_ref_token_start,
                                        ref_scale,
                                    )
                                    event = {
                                        "pass_index": int(pass_index),
                                        "step": int(step_index),
                                        "score_mode": args.adaptive_score_mode,
                                        "aggregate_score_kind": stat.get("aggregate_score_kind", ""),
                                        "score": score,
                                        "threshold": float(threshold),
                                        "score_min": float(stat["min"]),
                                        "score_max": float(stat["max"]),
                                        "blocks": int(stat["blocks"]),
                                        "scale_before": old_scale,
                                        "scale_after": float(ref_scale),
                                        "action": action,
                                    }
                                    for key in ("inside_mean", "inside_coverage", "outside_mean", "outside_area", "contrast_ratio", "contrast_margin", "coarse_area_ratio", "coarse_area", "roi_area", "hist_threshold", "hist_otsu_threshold"):
                                        if key in stat:
                                            event[key] = stat[key]
                                    adaptive_history.append(event)
                                    if args.log_attention_steps:
                                        print(
                                            f"[adaptive-ref] {anomaly}/{sample_index:03d} pass={pass_index} step {step_index} "
                                            f"mode={args.adaptive_score_mode} score={score:.4f} threshold={threshold:.4f} "
                                            f"scale={old_scale:.3f}->{ref_scale:.3f} {action}",
                                            flush=True,
                                        )
                            recorder.after_step(step_index)
                            pass_adaptive_final_scale = float(ref_scale)
                            adaptive_final_scale = float(ref_scale)
                            return callback_kwargs

                        generator = torch.Generator(generator_device).manual_seed(pass_seed)
                        callback_tensor_inputs = ["latents", "prompt_embeds"] if args.adaptive_ref_injection else ["latents"]
                        pipe_kwargs = {
                            "image": diptych_ref_tar,
                            "mask_image": mask_diptych,
                            "height": mask_diptych.size[1],
                            "width": mask_diptych.size[0],
                            "num_inference_steps": args.num_inference_steps,
                            "max_sequence_length": args.max_sequence_length,
                            "generator": generator,
                            "callback_on_step_end": callback_on_step_end,
                            "callback_on_step_end_tensor_inputs": callback_tensor_inputs,
                            **prior_kwargs,
                        }
                        if args.guidance_scale is not None:
                            pipe_kwargs["guidance_scale"] = args.guidance_scale
                        edited_pil = pipe(**pipe_kwargs).images[0]
                        width, height = edited_pil.size
                        edited_crop = np.array(edited_pil.crop((width // 2, 0, width, height)))
                        edited_image = crop_back(edited_crop, old_tar_image, extra_sizes, crop_box)
                        pass_edit_path = pass_out_dir / "edit.png" if write_outputs else None
                        if pass_edit_path is not None:
                            pass_out_dir.mkdir(parents=True, exist_ok=True)
                            Image.fromarray(edited_image).save(pass_edit_path)
                        direct_info = save_direct_masks(
                            args,
                            out_dir=pass_out_dir,
                            recorder=recorder,
                            selected_blocks=selected_blocks,
                            block_polarities=block_polarities,
                            old_tar_image=old_tar_image,
                            edited_image=edited_image,
                            target_mask_path=pass_target_mask_path,
                            source_image=pass_source_image,
                            extra_sizes=extra_sizes,
                            crop_box=crop_box,
                            object_support=object_support,
                            write_files=write_outputs,
                            run_refine=run_refine,
                            return_arrays=not write_outputs,
                        )
                        return {
                            "pass_index": int(pass_index),
                            "seed": int(pass_seed),
                            "source_image": pass_source_image,
                            "target_mask_path": pass_target_mask_path,
                            "pass_dir": pass_out_dir,
                            "edit_image": pass_edit_path,
                            "edited_image": edited_image,
                            "edited_crop_image": edited_crop,
                            "old_tar_image": old_tar_image,
                            "direct_info": direct_info,
                            "shape_k_removal": recorder.shape_k_summary(),
                            "adaptive_final_scale": pass_adaptive_final_scale,
                        }

                    component_masks = split_binary_mask_components(generated_mask)
                    use_sequential = len(component_masks) > 1
                    pass_results: list[dict[str, object]] = []
                    if use_sequential:
                        generated_mask_info["generation_mode"] = "sequential_multi_mask"
                        generated_mask_info["sequential_passes"] = len(component_masks)
                        keep_internal_outputs = bool(args.save_mask_debug)
                        current_source_image = source_image
                        for pass_index, component_mask in enumerate(component_masks):
                            if keep_internal_outputs:
                                component_mask_path = sample_dir / f"generated_target_mask_component_{pass_index:02d}.png"
                                pass_dir = sample_dir / f"pass_{pass_index:02d}"
                            else:
                                component_mask_path = sample_dir / f"_tmp_target_component_{pass_index:02d}.png"
                                pass_dir = sample_dir / f"_internal_pass_{pass_index:02d}"
                                temporary_paths.append(component_mask_path)
                            Image.fromarray((component_mask * 255).astype(np.uint8)).save(component_mask_path)
                            pass_seed = sample_seed + pass_index * 10007
                            pass_result = run_generation_pass(
                                current_source_image,
                                component_mask_path,
                                pass_dir,
                                pass_seed,
                                pass_index,
                                save_debug=bool(args.save_debug_first and sample_index == args.start_index and pass_index == 0),
                                write_outputs=keep_internal_outputs,
                                run_refine=False,
                            )
                            pass_results.append(pass_result)
                            if pass_index < len(component_masks) - 1:
                                if keep_internal_outputs:
                                    intermediate_path = sample_dir / f"intermediate_after_pass_{pass_index:02d}.png"
                                else:
                                    intermediate_path = sample_dir / f"_tmp_intermediate_after_pass_{pass_index:02d}.png"
                                    temporary_paths.append(intermediate_path)
                                Image.fromarray(pass_result["edited_image"]).save(intermediate_path)
                                current_source_image = intermediate_path
                        edited_image = pass_results[-1]["edited_image"]
                        Image.fromarray(edited_image).save(output_image)
                        original_source_image = load_rgb(source_image)
                        direct_aggregate_info = save_sequential_direct_masks(
                            args,
                            out_dir=sample_dir,
                            pass_results=pass_results,
                            selected_blocks=selected_blocks,
                            block_polarities=block_polarities,
                            old_tar_image=original_source_image,
                            edited_image=edited_image,
                            object_support=object_support,
                        )
                    else:
                        generated_mask_info["generation_mode"] = "single_mask"
                        pass_result = run_generation_pass(
                            source_image,
                            target_mask_path,
                            sample_dir,
                            sample_seed,
                            0,
                            save_debug=bool(args.save_debug_first and sample_index == args.start_index),
                            write_outputs=True,
                            run_refine=True,
                        )
                        pass_results.append(pass_result)
                        direct_aggregate_info = pass_result["direct_info"]
                except Exception as exc:
                    status = "error"
                    error = repr(exc)
                elapsed = time.time() - start_time
                adaptive_boosts = [event for event in adaptive_history if event.get("action") == "boost" and float(event.get("scale_after", 0.0)) > float(event.get("scale_before", 0.0))]
                adaptive_scores = [float(event["score"]) for event in adaptive_history if "score" in event]
                adaptive_triggered = bool(adaptive_boosts)
                generation_mode = str(generated_mask_info.get("generation_mode", ""))
                sequential_passes = generated_mask_info.get("sequential_passes", 1 if generation_mode == "single_mask" else "")
                adaptive_pass_counts: dict[int, int] = {}
                for event in adaptive_history:
                    pass_id = int(event.get("pass_index", 0))
                    adaptive_pass_counts[pass_id] = adaptive_pass_counts.get(pass_id, 0) + 1
                adaptive_event_passes = " ".join(f"p{pass_id}:{count}" for pass_id, count in sorted(adaptive_pass_counts.items()))
                adaptive_boost_steps = " ".join(f"p{int(event.get('pass_index', 0))}:s{int(event['step'])}" for event in adaptive_boosts)
                if args.adaptive_score_mode == "aggregate_mask":
                    if args.adaptive_aggregate_score_kind == "inside_coverage":
                        adaptive_threshold = args.adaptive_aggregate_min_inside_ratio
                    elif args.adaptive_aggregate_score_kind == "inside_mean":
                        adaptive_threshold = args.adaptive_aggregate_min_inside_mean
                    elif args.adaptive_aggregate_score_kind == "coarse_area_ratio":
                        adaptive_threshold = args.adaptive_aggregate_min_area_ratio
                    else:
                        adaptive_threshold = 1.0
                else:
                    adaptive_threshold = args.adaptive_ref_attention_threshold
                adaptive_max_scale_after = max([float(event.get("scale_after", 0.0)) for event in adaptive_history], default=float(adaptive_final_scale or args.adaptive_ref_base_scale))
                metadata = {"index": sample_index, "anomaly": anomaly, "ref_id": ref_id, "seed": sample_seed, "source_image": str(source_image), "target_mask": str(sample_dir / "threshold_roi.png"), "target_mask_source": args.target_mask_source, "generated_mask_info": generated_mask_info, "lora_runtime_audit": lora_runtime_audit, "object_support_image": str(object_support_image) if object_support_image.exists() else None, "object_attention_image": str(object_attention_image) if object_attention_image.exists() else None, "ref_image": str(ref_image), "ref_mask": str(ref_mask), "size": args.size, "num_inference_steps": args.num_inference_steps, "attention_processors": n_attn, "save_steps": args.save_steps, "capture_steps": args.capture_steps, "direct_aggregate_steps": args.direct_aggregate_steps, "adaptive_check_steps": args.adaptive_check_steps, "save_blocks": args.save_blocks, "direct_aggregate_mask": direct_aggregate_info, "ref_image_augmentation": ref_image_augmentation, "ref_token_perturbation": ref_token_perturbation, "adaptive_ref_injection": {"enabled": args.adaptive_ref_injection, "attention_kind": args.adaptive_aggregate_kind, "selected_blocks": args.adaptive_selected_blocks, "block_polarities": args.adaptive_block_polarities, "triggered": adaptive_triggered, "score_mode": args.adaptive_score_mode, "aggregate_score_kind": args.adaptive_aggregate_score_kind if args.adaptive_score_mode == "aggregate_mask" else "", "num_events": len(adaptive_history), "num_boosts": len(adaptive_boosts), "boost_steps": adaptive_boost_steps, "adaptive_event_passes": adaptive_event_passes, "generation_mode": generation_mode, "sequential_passes": sequential_passes, "token_start": args.adaptive_ref_token_start, "attention_threshold": args.adaptive_ref_attention_threshold, "aggregate_min_inside_ratio": args.adaptive_aggregate_min_inside_ratio, "aggregate_min_inside_mean": args.adaptive_aggregate_min_inside_mean, "aggregate_min_area_ratio": args.adaptive_aggregate_min_area_ratio, "aggregate_min_contrast_ratio": args.adaptive_aggregate_min_contrast_ratio, "aggregate_min_contrast_inside_mean": args.adaptive_aggregate_min_contrast_inside_mean, "aggregate_min_contrast_margin": args.adaptive_aggregate_min_contrast_margin, "aggregate_outside_ring_dilate": args.adaptive_aggregate_outside_ring_dilate, "base_scale": args.adaptive_ref_base_scale, "max_scale": args.adaptive_ref_max_scale, "boost": args.adaptive_ref_boost, "trigger_min_scale": args.adaptive_ref_trigger_min_scale, "decay": args.adaptive_ref_decay, "decay_min_score": args.adaptive_ref_decay_min_score, "final_scale": adaptive_final_scale, "max_scale_after": adaptive_max_scale_after, "events": adaptive_history}, "edit_image": str(output_image), "coarse_mask": str(output_coarse), "elapsed_sec": elapsed, "status": status, "error": error, "nunchaku_note": "Full FLUX transformer is used because nunchaku fused transformer does not expose per-block attention probabilities."}
                sample_metadata.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
                log_writer.writerow({"index": sample_index, "anomaly": anomaly, "ref_id": ref_id, "seed": sample_seed, "source_image": str(source_image), "target_mask": str(sample_dir / "threshold_roi.png"), "generation_mode": generation_mode, "sequential_passes": sequential_passes, "ref_image": str(ref_image), "ref_mask": str(ref_mask), "sample_dir": str(sample_dir), "edit_image": str(output_image), "coarse_mask": str(output_coarse), "elapsed_sec": f"{elapsed:.4f}", "status": status, "error": error})
                log_file.flush()
                adaptive_log_writer.writerow({"index": sample_index, "anomaly": anomaly, "ref_id": ref_id, "seed": sample_seed, "source_image": str(source_image), "generation_mode": generation_mode, "sequential_passes": sequential_passes, "sample_dir": str(sample_dir), "adaptive_enabled": args.adaptive_ref_injection, "adaptive_triggered": adaptive_triggered, "adaptive_score_mode": args.adaptive_score_mode, "adaptive_aggregate_score_kind": args.adaptive_aggregate_score_kind if args.adaptive_score_mode == "aggregate_mask" else "", "num_adaptive_events": len(adaptive_history), "num_boosts": len(adaptive_boosts), "boost_steps": adaptive_boost_steps, "adaptive_event_passes": adaptive_event_passes, "final_scale": "" if adaptive_final_scale is None else f"{float(adaptive_final_scale):.6f}", "max_scale_after": f"{float(adaptive_max_scale_after):.6f}", "mean_adaptive_score": "" if not adaptive_scores else f"{float(np.mean(adaptive_scores)):.6f}", "min_adaptive_score": "" if not adaptive_scores else f"{float(np.min(adaptive_scores)):.6f}", "threshold": adaptive_threshold, "status": status, "error": error})
                adaptive_log_file.flush()
                for temporary_path in temporary_paths:
                    if temporary_path.exists():
                        temporary_path.unlink()
                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {status.upper()} {anomaly}/{sample_index:03d} seed={sample_seed} source={source_image.name} elapsed={elapsed:.2f}s coarse={output_coarse}", flush=True)
                if status != "ok":
                    raise RuntimeError(f"Failed {anomaly}/{sample_index:03d}: {error}")
                if args.empty_cache_each_sample and torch.cuda.is_available():
                    torch.cuda.empty_cache()
            del ref_condition_bank
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    finally:
        log_file.close()
        adaptive_log_file.close()


if __name__ == "__main__":
    main()
