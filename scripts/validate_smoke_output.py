#!/usr/bin/env python3
"""Validate artifacts and runtime settings of a REDiff-AD MVTec run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
T2R_TOP10 = [
    line.strip()
    for line in (REPO_ROOT / "configs" / "top10_t2r_blocks.txt").read_text(encoding="utf-8").splitlines()
    if line.strip()
]
DIRECT_STEPS = [10, 15, 20, 25, 27, 28, 29]
REQUIRED_IMAGES = [
    "edit.png",
    "generated_target_mask.png",
    "object_support.png",
    "coarse_mask.png",
    "selected_block_coarse_mask.png",
    "all_block_coarse_mask.png",
    "q80_appearance_mask.png",
    "contour_refined_mask.png",
]
REQUIRED_FILES = [
    *REQUIRED_IMAGES,
    "metadata.json",
    "direct_aggregate_summary.json",
    "selected_block_soft_mask.npy",
    "all_block_soft_mask.npy",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--defects", nargs="+", default=None)
    parser.add_argument("--samples-per-defect", type=int, default=None)
    parser.add_argument("--expected-quantize", choices=["none", "int8", "int4"], default="int4")
    parser.add_argument("--expected-lora-path", default="WensongSong/Insert-Anything")
    parser.add_argument("--expected-lora-weight", default="20250321_steps5000_pytorch_lora_weights.safetensors")
    parser.add_argument("--expected-cpu-offload", type=int, choices=[0, 1], default=1)
    parser.add_argument("--expected-sequential-cpu-offload", type=int, choices=[0, 1], default=0)
    parser.add_argument("--expected-mask-area-min", type=float, default=0.5)
    parser.add_argument("--expected-mask-area-max", type=float, default=0.9)
    parser.add_argument(
        "--skip-edit-quality-check",
        action="store_true",
        help="Skip source/edit localization sanity check for archived runs whose source dataset is unavailable.",
    )
    parser.add_argument("--report", type=Path, default=None)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def add_check(checks: list[dict[str, Any]], name: str, passed: bool, detail: str) -> None:
    checks.append({"name": name, "passed": bool(passed), "detail": detail})


def validate_mask(path: Path, expected_size: tuple[int, int]) -> tuple[bool, str]:
    array = np.asarray(Image.open(path).convert("L"))
    unique = np.unique(array)
    valid_binary = bool(np.all(np.isin(unique, [0, 255])))
    nonempty = bool(np.any(array > 0))
    size_ok = (array.shape[1], array.shape[0]) == expected_size
    return (
        valid_binary and nonempty and size_ok,
        f"size={array.shape[1]}x{array.shape[0]} values={unique.tolist()} area={int((array > 0).sum())}",
    )


def valid_lora_audit(value: Any) -> bool:
    if not isinstance(value, dict) or value.get("verified") is not True:
        return False
    available = set(value.get("available_adapters", []))
    active = set(value.get("active_adapters", []))
    return bool(available.intersection(active))


def validate_sample(sample_dir: Path, *, skip_edit_quality_check: bool = False) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    missing = [name for name in REQUIRED_FILES if not (sample_dir / name).is_file() or (sample_dir / name).stat().st_size == 0]
    add_check(checks, "required_artifacts", not missing, "missing=" + ",".join(missing) if missing else "all present")
    if missing:
        return {"sample_dir": str(sample_dir), "passed": False, "checks": checks}

    try:
        metadata = read_json(sample_dir / "metadata.json")
        summary = read_json(sample_dir / "direct_aggregate_summary.json")
    except Exception as exc:
        add_check(checks, "json_readable", False, f"{type(exc).__name__}: {exc}")
        return {"sample_dir": str(sample_dir), "passed": False, "checks": checks}

    add_check(
        checks,
        "generation_status",
        metadata.get("status") == "ok" and not metadata.get("error"),
        f"status={metadata.get('status')!r} error={metadata.get('error')!r}",
    )
    ref_ok = metadata.get("ref_id") == "000" and Path(str(metadata.get("ref_image", ""))).name == "000.png"
    add_check(checks, "dataset_reference_000", ref_ok, f"ref_id={metadata.get('ref_id')} ref={metadata.get('ref_image')}")
    sample_lora_audit = metadata.get("lora_runtime_audit")
    add_check(
        checks,
        "lora_runtime_active",
        valid_lora_audit(sample_lora_audit),
        f"audit={sample_lora_audit}",
    )
    add_check(checks, "attention_processors", int(metadata.get("attention_processors", 0)) > 0, str(metadata.get("attention_processors")))

    edit_size = Image.open(sample_dir / "edit.png").size
    source_path = Path(str(metadata.get("source_image", ""))).expanduser()
    source_size = Image.open(source_path).size if source_path.is_file() else None
    size_ok = edit_size == source_size if source_size is not None else edit_size[0] > 0 and edit_size[1] > 0
    add_check(
        checks,
        "edit_size",
        size_ok,
        f"edit={edit_size[0]}x{edit_size[1]} source={source_size if source_size is not None else 'unavailable'}",
    )
    for name in REQUIRED_IMAGES[1:]:
        passed, detail = validate_mask(sample_dir / name, edit_size)
        add_check(checks, f"mask:{name}", passed, detail)

    if not skip_edit_quality_check:
        if source_path.is_file():
            source = np.asarray(Image.open(source_path).convert("RGB").resize(edit_size), dtype=np.float32)
            edit = np.asarray(Image.open(sample_dir / "edit.png").convert("RGB"), dtype=np.float32)
            target = np.asarray(Image.open(sample_dir / "generated_target_mask.png").convert("L")) > 0
            delta = np.abs(edit - source).mean(axis=2)
            inside_mae = float(delta[target].mean()) if np.any(target) else 0.0
            outside_mae = float(delta[~target].mean()) if np.any(~target) else 0.0
            inside_changed = float((delta[target] > 20.0).mean()) if np.any(target) else 0.0
            localized = (
                inside_mae >= 12.0
                and inside_changed >= 0.15
                and inside_mae >= 1.5 * max(outside_mae, 1.0)
            )
            add_check(
                checks,
                "localized_edit_sanity",
                localized,
                f"inside_mae={inside_mae:.3f} outside_mae={outside_mae:.3f} "
                f"inside_pixels_gt20={inside_changed:.3f}",
            )
        else:
            add_check(checks, "localized_edit_sanity", False, f"source image unavailable: {source_path}")

    for name in ("selected_block_soft_mask.npy", "all_block_soft_mask.npy"):
        array = np.load(sample_dir / name, mmap_mode="r")
        valid = array.shape == (edit_size[1], edit_size[0]) and bool(np.isfinite(array).all()) and float(np.max(array)) > float(np.min(array))
        add_check(checks, f"soft_mask:{name}", valid, f"shape={array.shape} min={float(np.min(array)):.6f} max={float(np.max(array)):.6f}")

    add_check(checks, "direct_kind", summary.get("kind") == "target_to_ref_image", f"kind={summary.get('kind')}")
    add_check(checks, "direct_roi", summary.get("roi") == "initial_mask", f"roi={summary.get('roi')}")
    add_check(checks, "direct_component", summary.get("component_mode") == "all", f"component={summary.get('component_mode')}")
    add_check(checks, "threshold_scale", abs(float(summary.get("hist_threshold_scale", -1)) - 0.85) < 1e-8, f"scale={summary.get('hist_threshold_scale')}")
    add_check(checks, "direct_steps", summary.get("direct_aggregate_steps") == DIRECT_STEPS, f"steps={summary.get('direct_aggregate_steps')}")

    selected = summary.get("selected_block_aggregate_mask", {})
    selected_blocks = selected.get("blocks") if isinstance(selected, dict) else None
    add_check(checks, "t2r_top10", selected_blocks == T2R_TOP10, f"blocks={selected_blocks}")
    polarities = summary.get("block_polarities", {})
    default_polarity = polarities.get("__default__") if isinstance(polarities, dict) else None
    add_check(checks, "t2r_default_polarity", default_polarity == "high", f"default={default_polarity}")
    all_block = summary.get("all_block_aggregate_mask", {})
    all_count = int(all_block.get("num_blocks", 0)) if isinstance(all_block, dict) else 0
    all_list = all_block.get("blocks", []) if isinstance(all_block, dict) else []
    add_check(checks, "all_block_capture", all_count == len(all_list) and all_count == 57, f"num_blocks={all_count}")

    adaptive = metadata.get("adaptive_ref_injection", {})
    adaptive_polarities = adaptive.get("block_polarities", {}) if isinstance(adaptive, dict) else {}
    adaptive_ok = (
        isinstance(adaptive, dict)
        and adaptive.get("enabled") is True
        and adaptive.get("attention_kind") == "target_to_ref_image"
        and adaptive.get("selected_blocks") == T2R_TOP10
        and isinstance(adaptive_polarities, dict)
        and adaptive_polarities.get("__default__") == "high"
        and all(adaptive_polarities.get(block) == "high" for block in T2R_TOP10)
        and int(adaptive.get("num_events", 0)) > 0
    )
    add_check(
        checks,
        "adaptive_diversity",
        adaptive_ok,
        f"enabled={adaptive.get('enabled') if isinstance(adaptive, dict) else None} "
        f"kind={adaptive.get('attention_kind') if isinstance(adaptive, dict) else None} "
        f"blocks={adaptive.get('selected_blocks') if isinstance(adaptive, dict) else None} "
        f"events={adaptive.get('num_events') if isinstance(adaptive, dict) else None} "
        f"boosts={adaptive.get('num_boosts') if isinstance(adaptive, dict) else None}",
    )

    shape_k = summary.get("shape_k_removal", {})
    shape_passes = shape_k.get("passes", [shape_k]) if isinstance(shape_k, dict) else []
    shape_ok = bool(shape_passes) and all(
        isinstance(item, dict)
        and item.get("enabled") is True
        and item.get("mode") == "both"
        and len(item.get("blocks", [])) == 36
        and int(item.get("applied_calls", 0)) == 324
        and int(item.get("skipped_no_shape", -1)) == 0
        and int(item.get("skipped_low_norm", -1)) == 0
        for item in shape_passes
    )
    shape_detail = [
        {"blocks": len(item.get("blocks", [])), "calls": item.get("applied_calls")}
        for item in shape_passes
        if isinstance(item, dict)
    ]
    add_check(checks, "shape_k_diversity", shape_ok, f"passes={shape_detail}")

    q80 = summary.get("q80_appearance_mask", {})
    contour = summary.get("contour_refined_mask", {})
    q80_ok = isinstance(q80, dict) and q80.get("enabled") is True and int(q80.get("mask_area", 0)) > 0
    contour_components = contour.get("component_refines", []) if isinstance(contour, dict) else []
    if contour_components:
        component_modes = [
            item.get("contour_refined_mask", {}).get("component_mode")
            for item in contour_components
            if isinstance(item, dict)
        ]
        component_mode_ok = bool(component_modes) and all(mode == "all" for mode in component_modes)
    else:
        component_modes = [contour.get("component_mode")] if isinstance(contour, dict) else []
        component_mode_ok = component_modes == ["all"]
    contour_ok = isinstance(contour, dict) and contour.get("enabled") is True and component_mode_ok and int(contour.get("mask_area", 0)) > 0
    add_check(checks, "q80_refine", q80_ok, f"enabled={q80.get('enabled') if isinstance(q80, dict) else None} area={q80.get('mask_area') if isinstance(q80, dict) else None}")
    add_check(checks, "contour_refine", contour_ok, f"enabled={contour.get('enabled') if isinstance(contour, dict) else None} area={contour.get('mask_area') if isinstance(contour, dict) else None} components={component_modes}")
    recommended = Path(str(summary.get("recommended_refined_mask", ""))).name
    add_check(checks, "recommended_refine", recommended == "contour_refined_mask.png", f"recommended={recommended}")

    return {"sample_dir": str(sample_dir), "passed": all(item["passed"] for item in checks), "checks": checks}


def main() -> int:
    args = parse_args()
    run_root = args.run_root.expanduser().resolve()
    checks: list[dict[str, Any]] = []
    if not run_root.is_dir():
        print(f"FAIL: run root does not exist: {run_root}")
        return 2

    config_path = run_root / "run_config.json"
    if config_path.is_file():
        config = read_json(config_path)
        adaptive_blocks = config.get("adaptive_selected_blocks_resolved")
        adaptive_polarities = config.get("adaptive_block_polarities_resolved")
        adaptive_csv = Path(str(config.get("adaptive_block_frequency_csv", ""))).name
        adaptive_blocks_file = Path(str(config.get("adaptive_selected_blocks_file", ""))).name
        config_ok = (
            config.get("direct_aggregate_kind") == "target_to_ref_image"
            and config.get("adaptive_aggregate_kind") == "target_to_ref_image"
            and adaptive_blocks == T2R_TOP10
            and adaptive_csv == "block_frequency_t2r.csv"
            and adaptive_blocks_file == "top10_t2r_blocks.txt"
            and isinstance(adaptive_polarities, dict)
            and adaptive_polarities.get("__default__") == "high"
            and all(adaptive_polarities.get(block) == "high" for block in T2R_TOP10)
            and config.get("adaptive_ref_injection") is True
            and config.get("shape_k_removal") is True
            and config.get("shape_k_block_scope") == "middle"
            and config.get("num_inference_steps") == 30
            and config.get("full_flux_quantize") == args.expected_quantize
            and config.get("lora_path") == args.expected_lora_path
            and config.get("lora_weight_name") == args.expected_lora_weight
            and config.get("cpu_offload") is bool(args.expected_cpu_offload)
            and config.get("sequential_cpu_offload") is bool(args.expected_sequential_cpu_offload)
            and abs(float(config.get("random_mask_area_min_ratio", -1)) - args.expected_mask_area_min) < 1e-8
            and abs(float(config.get("random_mask_area_max_ratio", -1)) - args.expected_mask_area_max) < 1e-8
            and valid_lora_audit(config.get("lora_runtime_audit"))
        )
        add_check(
            checks,
            "run_config",
            config_ok,
            f"direct={config.get('direct_aggregate_kind')} adaptive={config.get('adaptive_aggregate_kind')} "
            f"adaptive_csv={adaptive_csv} adaptive_blocks={adaptive_blocks} "
            f"shape_scope={config.get('shape_k_block_scope')} steps={config.get('num_inference_steps')} "
            f"quantize={config.get('full_flux_quantize')} "
            f"lora={config.get('lora_path')}/{config.get('lora_weight_name')} "
            f"offload={config.get('cpu_offload')}/{config.get('sequential_cpu_offload')} "
            f"mask_area={config.get('random_mask_area_min_ratio')}-{config.get('random_mask_area_max_ratio')} "
            f"lora_audit={config.get('lora_runtime_audit')}",
        )
    else:
        add_check(checks, "run_config", False, "missing run_config.json")

    sample_dirs = sorted(path.parent for path in run_root.glob("*/ref_000/[0-9]*/metadata.json"))
    if args.defects:
        expected_defects = set(args.defects)
        sample_dirs = [path for path in sample_dirs if path.parents[1].name in expected_defects]
        found_defects = {path.parents[1].name for path in sample_dirs}
        add_check(checks, "expected_defects", found_defects == expected_defects, f"expected={sorted(expected_defects)} found={sorted(found_defects)}")
    if args.samples_per_defect is not None:
        counts: dict[str, int] = {}
        for path in sample_dirs:
            defect = path.parents[1].name
            counts[defect] = counts.get(defect, 0) + 1
        count_ok = bool(counts) and all(value == args.samples_per_defect for value in counts.values())
        add_check(checks, "sample_counts", count_ok, f"expected={args.samples_per_defect} counts={counts}")
    add_check(checks, "samples_found", bool(sample_dirs), f"count={len(sample_dirs)}")

    samples = [
        validate_sample(path, skip_edit_quality_check=args.skip_edit_quality_check)
        for path in sample_dirs
    ]
    passed = all(item["passed"] for item in checks) and bool(samples) and all(item["passed"] for item in samples)
    report = {
        "run_root": str(run_root),
        "passed": passed,
        "num_samples": len(samples),
        "run_checks": checks,
        "samples": samples,
    }
    report_path = args.report or (run_root / "validation_report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    failed_checks = [item for item in checks if not item["passed"]]
    failed_samples = [item for item in samples if not item["passed"]]
    if passed:
        print(f"PASS: {len(samples)} sample(s); report={report_path}")
        return 0
    print(f"FAIL: run_checks={len(failed_checks)} sample_failures={len(failed_samples)}; report={report_path}")
    for item in failed_checks:
        print(f"  run:{item['name']}: {item['detail']}")
    for sample in failed_samples:
        print(f"  sample:{sample['sample_dir']}")
        for item in sample["checks"]:
            if not item["passed"]:
                print(f"    {item['name']}: {item['detail']}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
