from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def run_object_policy_dry_run(
    tmp_path: Path,
    object_name: str,
    *,
    policy_env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    dataset = tmp_path / object_name
    defect = "defect"
    (dataset / "train" / "good").mkdir(parents=True)
    (dataset / "test" / defect).mkdir(parents=True)
    (dataset / "ground_truth" / defect).mkdir(parents=True)
    (dataset / "train" / "good" / "000.png").touch()
    (dataset / "test" / defect / "001.png").touch()
    (dataset / "ground_truth" / defect / "001_mask.png").touch()

    env = os.environ.copy()
    env.pop("MVTEC_ROOT", None)
    for name in (
        "TARGET_MASK_SOURCE",
        "REFERENCE_MASK_DILATE_ITERATIONS",
        "REFERENCE_MASK_VERTICAL_SHIFT_RATIO",
        "RANDOM_MASK_AREA_MIN_RATIO",
        "RANDOM_MASK_AREA_MAX_RATIO",
    ):
        env.pop(name, None)
    env.update(
        {
            "OBJECT_NAME": object_name,
            "DATASET_ROOT": str(dataset),
            "OUT_ROOT": str(tmp_path / "out"),
            "ANOMALIES_STR": defect,
            "REF_IDS_STR": "001",
            "SAMPLES_PER_ANOMALY": "180",
            "SAMPLES_PER_PAIR_STR": "180",
            "DRY_RUN": "1",
            "LOG_TO_FILE": "0",
        }
    )
    if policy_env:
        env.update(policy_env)
    return subprocess.run(
        ["bash", "scripts/run_hazelnut_t2r.sh"],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_hazelnut_dry_run_contract(tmp_path: Path) -> None:
    dataset = tmp_path / "hazelnut"
    (dataset / "train" / "good").mkdir(parents=True)
    (dataset / "test" / "crack").mkdir(parents=True)
    (dataset / "ground_truth" / "crack").mkdir(parents=True)
    (dataset / "train" / "good" / "000.png").touch()
    (dataset / "test" / "crack" / "000.png").touch()
    (dataset / "ground_truth" / "crack" / "000_mask.png").touch()
    out_root = tmp_path / "out"

    env = os.environ.copy()
    env.pop("MVTEC_ROOT", None)
    env.update(
        {
            "DATASET_ROOT": str(dataset),
            "OUT_ROOT": str(out_root),
            "ANOMALIES_STR": "crack",
            "SAMPLES_PER_ANOMALY": "1",
            "DRY_RUN": "1",
            "LOG_TO_FILE": "0",
        }
    )
    completed = subprocess.run(
        ["bash", "scripts/run_hazelnut_t2r.sh"],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    output = completed.stdout + completed.stderr
    assert "attention:   kind=target_to_ref_image" in output
    assert "adaptive blocks: kind=target_to_ref_image" in output
    assert "quantize:    int4" in output
    assert "offload:     cpu=1 sequential=0" in output
    assert "LoRA:        path=WensongSong/Insert-Anything" in output
    assert "weight=20250321_steps5000_pytorch_lora_weights.safetensors" in output
    assert "scope=middle" in output
    assert "Q80 refine:  run=1" in output
    assert "Contour:     run=1" in output
    assert "cuda_available=not_checked" in output

    frozen_top10 = ROOT / "configs" / "top10_t2r_blocks.txt"
    assert f"file={frozen_top10}" in output
    assert not (out_root / "adaptive_top10_blocks_from_block_frequency.txt").exists()


def test_launcher_forwards_explicit_reference_and_per_pair_count(tmp_path: Path) -> None:
    dataset = tmp_path / "cable"
    (dataset / "train" / "good").mkdir(parents=True)
    (dataset / "test" / "bent_wire").mkdir(parents=True)
    (dataset / "ground_truth" / "bent_wire").mkdir(parents=True)
    (dataset / "train" / "good" / "000.png").touch()
    (dataset / "test" / "bent_wire" / "001.png").touch()
    (dataset / "ground_truth" / "bent_wire" / "001_mask.png").touch()

    env = os.environ.copy()
    env.pop("MVTEC_ROOT", None)
    env.update(
        {
            "OBJECT_NAME": "cable",
            "DATASET_ROOT": str(dataset),
            "OUT_ROOT": str(tmp_path / "out"),
            "ANOMALIES_STR": "bent_wire",
            "REF_IDS_STR": "001",
            "SAMPLES_PER_ANOMALY": "180",
            "SAMPLES_PER_PAIR_STR": "17",
            "DRY_RUN": "1",
            "LOG_TO_FILE": "0",
        }
    )
    completed = subprocess.run(
        ["bash", "scripts/run_hazelnut_t2r.sh"],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    output = completed.stdout + completed.stderr
    assert completed.returncode == 0, output
    assert "refs:     001" in output
    assert "samples:  per-pair counts=17" in output
    assert "samples:     per-pair=17" in output
    assert "ref_000" not in output


def test_screw_launcher_forwards_four_to_seven_reference_area_ratio(tmp_path: Path) -> None:
    completed = run_object_policy_dry_run(
        tmp_path,
        "screw",
        policy_env={
            "RANDOM_MASK_AREA_MIN_RATIO": "4.0",
            "RANDOM_MASK_AREA_MAX_RATIO": "7.0",
            "RANDOM_MASK_DOUBLE_PROB": "0.0",
        },
    )
    output = completed.stdout + completed.stderr

    assert completed.returncode == 0, output
    assert "target mask: source=random_object" in output
    assert "area_ratio=4.0-7.0" in output
    assert "double_prob=0.0" in output


def test_zipper_launcher_forwards_reference_vertical_mixed_policy(tmp_path: Path) -> None:
    completed = run_object_policy_dry_run(
        tmp_path,
        "zipper",
        policy_env={
            "TARGET_MASK_SOURCE": "reference_vertical_mixed",
            "REFERENCE_MASK_DILATE_ITERATIONS": "5",
            "REFERENCE_MASK_VERTICAL_SHIFT_RATIO": "0.05",
        },
    )
    output = completed.stdout + completed.stderr

    assert completed.returncode == 0, output
    assert "target mask: source=reference_vertical_mixed" in output
    assert "ref_dilate=5" in output
    assert "ref_vertical_shift=0.05" in output
    # The 1/3 original branch must retain the pre-existing random-object range.
    assert "area_ratio=0.5-0.9" in output


@pytest.mark.parametrize("object_name", ["cable", "grid", "pill"])
def test_unmodified_objects_keep_original_random_object_policy(
    tmp_path: Path,
    object_name: str,
) -> None:
    completed = run_object_policy_dry_run(tmp_path, object_name)
    output = completed.stdout + completed.stderr

    assert completed.returncode == 0, output
    assert "target mask: source=random_object" in output
    assert "area_ratio=0.5-0.9" in output
    assert "source=reference_vertical_mixed" not in output


def test_bundled_hazelnut_demo_dry_run_uses_all_defects(tmp_path: Path) -> None:
    env = os.environ.copy()
    for name in (
        "MVTEC_ROOT",
        "DATASET_ROOT",
        "ANOMALIES_STR",
        "REF_IDS_STR",
        "SAMPLES_PER_ANOMALY",
        "SEED",
        "RUN_NAME",
        "LOCAL_FILES_ONLY",
    ):
        env.pop(name, None)
    env.update(
        {
            "OUT_ROOT": str(tmp_path / "out"),
            "DRY_RUN": "1",
            "LOG_TO_FILE": "0",
        }
    )
    completed = subprocess.run(
        ["bash", "scripts/run_hazelnut_demo.sh"],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    output = completed.stdout + completed.stderr
    assert completed.returncode == 0, output
    assert f"dataset:  {ROOT / 'demo_assets/mvtec_ad/hazelnut'}" in output
    assert "defects:  crack hole print cut" in output
    assert "refs:     000 000 000 000" in output
    assert "samples:  5 per defect" in output
    assert "models:   local_files_only=0" in output
    assert "DRY_RUN=1: configuration validated; generation was not started." in output


def test_bundled_demo_prunes_successful_generation_to_three_files(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_bash = fake_bin / "bash"
    fake_bash.write_text(
        """#!/bin/sh
set -eu
sample_dir="$OUT_ROOT/crack/ref_000/000"
mkdir -p "$sample_dir/attention_steps"
for name in edit.png coarse_mask.png contour_refined_mask.png metadata.json soft_mask.npy; do
    printf 'generated' > "$sample_dir/$name"
done
printf 'debug' > "$sample_dir/attention_steps/step_01.png"
printf 'runtime' > "$OUT_ROOT/run_log.csv"
printf '%s' "$OVERWRITE" > "$CAPTURE_OVERWRITE"
""",
        encoding="utf-8",
    )
    fake_bash.chmod(0o755)

    out_root = tmp_path / "out"
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{fake_bin}:{env['PATH']}",
            "OUT_ROOT": str(out_root),
            "ANOMALIES_STR": "crack",
            "REF_IDS_STR": "000",
            "SAMPLES_PER_ANOMALY": "1",
            "DEMO_MINIMAL_OUTPUT": "1",
            "LOG_TO_FILE": "0",
            "CAPTURE_OVERWRITE": str(tmp_path / "overwrite.txt"),
        }
    )
    env.pop("OVERWRITE", None)

    completed = subprocess.run(
        ["/bin/bash", "scripts/run_hazelnut_demo.sh"],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    sample_dir = out_root / "crack" / "ref_000" / "000"
    assert {path.name for path in sample_dir.iterdir()} == {
        "edit.png",
        "coarse_mask.png",
        "contour_refined_mask.png",
    }
    assert not (out_root / "run_log.csv").exists()
    assert (tmp_path / "overwrite.txt").read_text(encoding="utf-8") == "1"
