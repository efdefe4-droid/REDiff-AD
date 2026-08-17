from __future__ import annotations

import json
from pathlib import Path

import pytest

from generation_attention.resume_safety import (
    RESUME_REQUIRED_FILES,
    decide_sample_resume_action,
    ensure_resume_run_config_compatible,
    sample_is_complete_for_resume,
)


TOP10 = [f"transformer_blocks_{index}_attn" for index in range(10)]
STEPS = [10, 15, 20, 25, 27, 28, 29]


def expected_identity(tmp_path: Path) -> dict[str, object]:
    return {
        "expected_seed": 309,
        "expected_source_image": tmp_path / "source" / "032.png",
        "expected_ref_id": "000",
        "expected_ref_image": tmp_path / "test" / "crack" / "000.png",
        "expected_ref_mask": tmp_path / "ground_truth" / "crack" / "000_mask.png",
        "expected_kind": "target_to_ref_image",
        "expected_steps": STEPS,
        "expected_selected_blocks": TOP10,
        "expected_adaptive_kind": "target_to_ref_image",
        "expected_adaptive_selected_blocks": TOP10,
        "expected_adaptive_default_polarity": "high",
        "expected_num_inference_steps": 30,
        "expected_anomaly": "crack",
        "expected_index": 0,
    }


def write_complete_sample(
    sample_dir: Path,
    identity: dict[str, object],
    *,
    mutation: str | None = None,
) -> None:
    sample_dir.mkdir(parents=True)
    metadata = {
        "index": identity["expected_index"],
        "anomaly": identity["expected_anomaly"],
        "ref_id": identity["expected_ref_id"],
        "seed": identity["expected_seed"],
        "source_image": str(identity["expected_source_image"]),
        "ref_image": str(identity["expected_ref_image"]),
        "ref_mask": str(identity["expected_ref_mask"]),
        "num_inference_steps": identity["expected_num_inference_steps"],
        "direct_aggregate_steps": list(STEPS),
        "direct_aggregate_mask": {
            "kind": identity["expected_kind"],
            "direct_aggregate_steps": list(STEPS),
            "selected_blocks": list(TOP10),
        },
        "status": "ok",
        "error": "",
        "adaptive_ref_injection": {
            "enabled": True,
            "attention_kind": identity["expected_adaptive_kind"],
            "selected_blocks": list(TOP10),
            "block_polarities": {"__default__": "high", **{block: "high" for block in TOP10}},
        },
    }
    summary = {
        "kind": identity["expected_kind"],
        "direct_aggregate_steps": list(STEPS),
        "selected_blocks": list(TOP10),
        "selected_block_aggregate_mask": {
            "output_mask": "selected_block_coarse_mask.png",
            "blocks": list(TOP10),
        },
        "all_block_aggregate_mask": {"output_mask": "all_block_coarse_mask.png"},
    }

    if mutation == "seed":
        metadata["seed"] = 310
    elif mutation == "source":
        metadata["source_image"] = str(Path(str(identity["expected_source_image"])).with_name("031.png"))
    elif mutation == "ref_id":
        metadata["ref_id"] = "001"
    elif mutation == "ref_paths":
        metadata["ref_image"] = str(Path(str(identity["expected_ref_image"])).with_name("001.png"))
        metadata["ref_mask"] = str(Path(str(identity["expected_ref_mask"])).with_name("001_mask.png"))
    elif mutation == "kind":
        metadata["direct_aggregate_mask"]["kind"] = "target_to_target_image"
        summary["kind"] = "target_to_target_image"
    elif mutation == "steps":
        metadata["direct_aggregate_steps"] = [10, 20]
        metadata["direct_aggregate_mask"]["direct_aggregate_steps"] = [10, 20]
        summary["direct_aggregate_steps"] = [10, 20]
    elif mutation == "top10":
        wrong_blocks = list(TOP10)
        wrong_blocks[-1] = "single_transformer_blocks_0_attn"
        metadata["direct_aggregate_mask"]["selected_blocks"] = wrong_blocks
        summary["selected_blocks"] = wrong_blocks
    elif mutation == "adaptive_kind":
        metadata["adaptive_ref_injection"]["attention_kind"] = "target_to_target_image"
    elif mutation == "adaptive_top10":
        metadata["adaptive_ref_injection"]["selected_blocks"] = list(TOP10[:-1])

    for name in RESUME_REQUIRED_FILES:
        path = sample_dir / name
        if name == "metadata.json":
            path.write_text(json.dumps(metadata), encoding="utf-8")
        elif name == "direct_aggregate_summary.json":
            path.write_text(json.dumps(summary), encoding="utf-8")
        else:
            path.write_bytes(b"artifact")


def test_exact_sample_identity_is_the_only_skip_path(tmp_path: Path) -> None:
    identity = expected_identity(tmp_path)
    sample_dir = tmp_path / "result" / "crack" / "ref_000" / "000"
    write_complete_sample(sample_dir, identity)

    assert sample_is_complete_for_resume(sample_dir, **identity)
    assert decide_sample_resume_action(sample_dir, overwrite=False, **identity) == "skip"


@pytest.mark.parametrize(
    "mutation",
    ["seed", "source", "ref_id", "ref_paths", "kind", "steps", "top10", "adaptive_kind", "adaptive_top10"],
)
def test_mismatched_existing_sample_is_neither_skipped_nor_overwritten(tmp_path: Path, mutation: str) -> None:
    identity = expected_identity(tmp_path)
    sample_dir = tmp_path / "result" / "crack" / "ref_000" / "000"
    write_complete_sample(sample_dir, identity, mutation=mutation)

    assert not sample_is_complete_for_resume(sample_dir, **identity)
    with pytest.raises(RuntimeError, match="Refusing to overwrite or skip unsafe existing sample"):
        decide_sample_resume_action(sample_dir, overwrite=False, **identity)
    assert decide_sample_resume_action(sample_dir, overwrite=True, **identity) == "generate"


def test_partial_existing_sample_requires_explicit_overwrite(tmp_path: Path) -> None:
    sample_dir = tmp_path / "result" / "crack" / "ref_000" / "000"
    sample_dir.mkdir(parents=True)
    (sample_dir / "edit.png").write_bytes(b"partial")

    with pytest.raises(RuntimeError, match="missing_or_empty"):
        decide_sample_resume_action(sample_dir, overwrite=False, **expected_identity(tmp_path))


def test_missing_canonical_refine_artifact_cannot_be_skipped(tmp_path: Path) -> None:
    identity = expected_identity(tmp_path)
    sample_dir = tmp_path / "result" / "crack" / "ref_000" / "000"
    write_complete_sample(sample_dir, identity)
    (sample_dir / "q80_appearance_mask.png").unlink()

    assert not sample_is_complete_for_resume(sample_dir, **identity)
    with pytest.raises(RuntimeError, match="missing_or_empty:q80_appearance_mask.png"):
        decide_sample_resume_action(sample_dir, overwrite=False, **identity)


def test_critical_run_config_mismatch_does_not_replace_existing_config(tmp_path: Path) -> None:
    config_path = tmp_path / "run_config.json"
    existing = {
        "resume_config_schema_version": 1,
        "seed": 309,
        "direct_aggregate_kind": "target_to_ref_image",
        "direct_selected_blocks_resolved": TOP10,
        "adaptive_aggregate_kind": "target_to_ref_image",
        "adaptive_selected_blocks_resolved": TOP10,
        "num_inference_steps": 30,
        "out_root": "old-output",
        "overwrite": True,
        "started_at_utc": "old-time",
    }
    original_text = json.dumps(existing, indent=2)
    config_path.write_text(original_text, encoding="utf-8")
    current = {
        **existing,
        "seed": 310,
        "out_root": "new-spelling-for-same-output",
        "overwrite": False,
        "started_at_utc": "new-time",
    }

    with pytest.raises(RuntimeError, match=r"different critical configuration.*seed"):
        ensure_resume_run_config_compatible(config_path, current, overwrite=False)
    assert config_path.read_text(encoding="utf-8") == original_text


def test_resume_allows_only_noncritical_run_control_changes(tmp_path: Path) -> None:
    config_path = tmp_path / "run_config.json"
    existing = {
        "resume_config_schema_version": 1,
        "seed": 309,
        "direct_aggregate_kind": "target_to_ref_image",
        "direct_selected_blocks_resolved": TOP10,
        "adaptive_aggregate_kind": "target_to_ref_image",
        "adaptive_selected_blocks_resolved": TOP10,
        "num_inference_steps": 30,
        "samples_per_anomaly": 1,
        "start_index": 0,
        "overwrite": True,
        "started_at_utc": "old-time",
    }
    config_path.write_text(json.dumps(existing), encoding="utf-8")
    current = {
        **existing,
        "samples_per_anomaly": 6,
        "start_index": 1,
        "overwrite": False,
        "started_at_utc": "new-time",
    }

    ensure_resume_run_config_compatible(config_path, current, overwrite=False)
