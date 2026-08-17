from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
VALIDATOR = ROOT / "scripts" / "validate_smoke_output.py"
TOP10 = [line.strip() for line in (ROOT / "configs" / "top10_t2r_blocks.txt").read_text(encoding="utf-8").splitlines() if line.strip()]
MASK_NAMES = [
    "generated_target_mask.png",
    "object_support.png",
    "coarse_mask.png",
    "selected_block_coarse_mask.png",
    "all_block_coarse_mask.png",
    "q80_appearance_mask.png",
    "contour_refined_mask.png",
]


def make_run(root: Path, image_size: int = 1024) -> Path:
    sample = root / "crack" / "ref_000" / "000"
    sample.mkdir(parents=True)
    source = np.full((image_size, image_size, 3), 180, dtype=np.uint8)
    source_path = root / "source.png"
    Image.fromarray(source).save(source_path)
    mask = np.zeros((image_size, image_size), dtype=np.uint8)
    mask[300:500, 400:600] = 255
    image = source.copy()
    image[300:500, 400:600] = 40
    Image.fromarray(image).save(sample / "edit.png")
    for name in MASK_NAMES:
        Image.fromarray(mask).save(sample / name)
    soft = np.linspace(0.0, 1.0, image_size * image_size, dtype=np.float32).reshape(image_size, image_size)
    np.save(sample / "selected_block_soft_mask.npy", soft)
    np.save(sample / "all_block_soft_mask.npy", soft[::-1])

    metadata = {
        "status": "ok",
        "error": "",
        "ref_id": "000",
        "ref_image": "/dataset/hazelnut/test/crack/000.png",
        "source_image": str(source_path),
        "attention_processors": 57,
        "lora_runtime_audit": {
            "available_adapters": ["default_0"],
            "active_adapters": ["default_0"],
            "verified": True,
        },
        "adaptive_ref_injection": {
            "enabled": True,
            "attention_kind": "target_to_ref_image",
            "selected_blocks": TOP10,
            "block_polarities": {"__default__": "high", **{block: "high" for block in TOP10}},
            "num_events": 12,
            "num_boosts": 0,
        },
    }
    (sample / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    all_blocks = [f"transformer_blocks_{index}_attn" for index in range(19)] + [
        f"single_transformer_blocks_{index}_attn" for index in range(38)
    ]
    shape_blocks = all_blocks[10:46]
    summary = {
        "kind": "target_to_ref_image",
        "roi": "initial_mask",
        "component_mode": "all",
        "hist_threshold_scale": 0.85,
        "direct_aggregate_steps": [10, 15, 20, 25, 27, 28, 29],
        "selected_block_aggregate_mask": {"blocks": TOP10},
        "block_polarities": {"__default__": "high"},
        "all_block_aggregate_mask": {"num_blocks": 57, "blocks": all_blocks},
        "shape_k_removal": {
            "enabled": True,
            "mode": "both",
            "blocks": shape_blocks,
            "applied_calls": 324,
            "skipped_no_shape": 0,
            "skipped_low_norm": 0,
        },
        "q80_appearance_mask": {"enabled": True, "mask_area": 40000},
        "contour_refined_mask": {"enabled": True, "component_mode": "all", "mask_area": 40000},
        "recommended_refined_mask": str(sample / "contour_refined_mask.png"),
    }
    (sample / "direct_aggregate_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    config = {
        "direct_aggregate_kind": "target_to_ref_image",
        "adaptive_aggregate_kind": "target_to_ref_image",
        "adaptive_block_frequency_csv": str(ROOT / "configs" / "block_frequency_t2r.csv"),
        "adaptive_selected_blocks_file": str(ROOT / "configs" / "top10_t2r_blocks.txt"),
        "adaptive_selected_blocks_resolved": TOP10,
        "adaptive_block_polarities_resolved": {
            "__default__": "high",
            **{block: "high" for block in TOP10},
        },
        "adaptive_ref_injection": True,
        "shape_k_removal": True,
        "shape_k_block_scope": "middle",
        "num_inference_steps": 30,
        "full_flux_quantize": "int4",
        "lora_path": "WensongSong/Insert-Anything",
        "lora_weight_name": "20250321_steps5000_pytorch_lora_weights.safetensors",
        "cpu_offload": True,
        "sequential_cpu_offload": False,
        "random_mask_area_min_ratio": 0.5,
        "random_mask_area_max_ratio": 0.9,
        "lora_runtime_audit": {
            "available_adapters": ["default_0"],
            "active_adapters": ["default_0"],
            "verified": True,
        },
    }
    (root / "run_config.json").write_text(json.dumps(config), encoding="utf-8")
    return sample


def run_validator(root: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(VALIDATOR), str(root), "--defects", "crack", "--samples-per-defect", "1"],
        text=True,
        capture_output=True,
        check=False,
    )


def test_validator_accepts_complete_int4_lora_run(tmp_path: Path) -> None:
    root = tmp_path / "run"
    make_run(root)
    completed = run_validator(root)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    report = json.loads((root / "validation_report.json").read_text(encoding="utf-8"))
    assert report["passed"] is True


def test_validator_accepts_pill_sized_source_and_edit(tmp_path: Path) -> None:
    root = tmp_path / "run"
    make_run(root, image_size=800)
    completed = run_validator(root)
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_validator_rejects_wrong_direct_kind(tmp_path: Path) -> None:
    root = tmp_path / "run"
    sample = make_run(root)
    summary_path = sample / "direct_aggregate_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["kind"] = "target_to_target_image"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    assert run_validator(root).returncode == 1


def test_validator_rejects_wrong_adaptive_kind(tmp_path: Path) -> None:
    root = tmp_path / "run"
    make_run(root)
    config_path = root / "run_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["adaptive_aggregate_kind"] = "target_to_target_image"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    assert run_validator(root).returncode == 1


def test_validator_rejects_wrong_adaptive_blocks(tmp_path: Path) -> None:
    root = tmp_path / "run"
    sample = make_run(root)
    metadata_path = sample / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["adaptive_ref_injection"]["selected_blocks"] = TOP10[:-1]
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    assert run_validator(root).returncode == 1


def test_validator_rejects_empty_refine_mask(tmp_path: Path) -> None:
    root = tmp_path / "run"
    sample = make_run(root)
    Image.fromarray(np.zeros((1024, 1024), dtype=np.uint8)).save(sample / "contour_refined_mask.png")
    assert run_validator(root).returncode == 1
