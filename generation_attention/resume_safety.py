"""Pure, dependency-free safety checks for resumable generation runs."""

from __future__ import annotations

import json
from pathlib import Path


RESUME_REQUIRED_FILES = (
    "edit.png",
    "coarse_mask.png",
    "metadata.json",
    "direct_aggregate_summary.json",
    "generated_target_mask.png",
    "object_support.png",
    "selected_block_soft_mask.npy",
    "selected_block_coarse_mask.png",
    "all_block_soft_mask.npy",
    "all_block_coarse_mask.png",
    "q80_appearance_mask.png",
    "contour_refined_mask.png",
)

RESUME_CONFIG_SCHEMA_VERSION = 1
RESUME_NONCRITICAL_CONFIG_KEYS = {
    "adaptive_log_file",
    "cpu_offload",
    "device",
    "empty_cache_each_sample",
    "local_files_only",
    "log_attention_steps",
    "log_file",
    "lora_runtime_audit",
    "out_root",
    "overwrite",
    "samples_per_anomaly",
    "samples_per_pair",
    "samples_per_pair_resolved",
    "sequential_cpu_offload",
    "start_index",
    "started_at_utc",
}
RESUME_PATH_CONFIG_KEYS = {
    "adaptive_block_frequency_csv",
    "adaptive_selected_blocks_file",
    "direct_block_frequency_csv",
    "direct_selected_blocks_file",
    "fixed_target_mask",
    "object_attention_root",
    "object_support_root",
    "ref_image_root",
    "ref_mask_root",
    "source_root",
}


def _normalized_path(value: object) -> object:
    if value in (None, ""):
        return value
    return str(Path(str(value)).expanduser().resolve(strict=False))


def _json_value(value: object) -> object:
    """Normalize tuples and Paths before resume comparisons."""
    return json.loads(json.dumps(value, sort_keys=True, default=str))


def critical_resume_config(config: dict[str, object]) -> dict[str, object]:
    critical: dict[str, object] = {}
    for key, value in config.items():
        if key in RESUME_NONCRITICAL_CONFIG_KEYS:
            continue
        if key in RESUME_PATH_CONFIG_KEYS:
            value = _normalized_path(value)
        critical[key] = _json_value(value)
    return critical


def ensure_resume_run_config_compatible(
    run_config_path: Path,
    current_config: dict[str, object],
    *,
    overwrite: bool,
) -> None:
    """Reject unsafe in-place resumes before replacing the recorded run config."""
    if overwrite or not run_config_path.exists():
        return
    try:
        existing_config = json.loads(run_config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Cannot safely resume: existing run config is unreadable: {run_config_path}. "
            "Use a new --out-root or pass --overwrite to regenerate the requested samples."
        ) from exc
    if not isinstance(existing_config, dict):
        raise RuntimeError(f"Cannot safely resume: {run_config_path} must contain a JSON object.")

    existing_critical = critical_resume_config(existing_config)
    current_critical = critical_resume_config(current_config)
    mismatch_keys = sorted(
        key
        for key in set(existing_critical) | set(current_critical)
        if existing_critical.get(key) != current_critical.get(key)
    )
    if not mismatch_keys:
        return

    details = []
    for key in mismatch_keys[:12]:
        old_text = json.dumps(existing_critical.get(key), sort_keys=True, default=str)
        new_text = json.dumps(current_critical.get(key), sort_keys=True, default=str)
        details.append(f"{key}: existing={old_text[:180]} current={new_text[:180]}")
    if len(mismatch_keys) > len(details):
        details.append(f"... and {len(mismatch_keys) - len(details)} more")
    raise RuntimeError(
        "Cannot safely resume into an output root created with a different critical configuration. "
        + "; ".join(details)
        + ". Use a new --out-root, or pass --overwrite only when regeneration is intentional."
    )


def _path_matches(recorded: object, expected: Path | str | None) -> bool:
    if expected is None:
        return True
    return _normalized_path(recorded) == _normalized_path(expected)


def sample_resume_mismatches(
    sample_dir: Path,
    *,
    expected_seed: int | None = None,
    expected_source_image: Path | str | None = None,
    expected_ref_id: str | None = None,
    expected_ref_image: Path | str | None = None,
    expected_ref_mask: Path | str | None = None,
    expected_kind: str | None = None,
    expected_steps: list[int] | tuple[int, ...] | None = None,
    expected_selected_blocks: list[str] | tuple[str, ...] | None = None,
    expected_adaptive_kind: str | None = None,
    expected_adaptive_selected_blocks: list[str] | tuple[str, ...] | None = None,
    expected_adaptive_default_polarity: str | None = None,
    expected_num_inference_steps: int | None = None,
    expected_anomaly: str | None = None,
    expected_index: int | None = None,
    expected_target_mask_source: str | None = None,
) -> list[str]:
    """Return every reason an existing sample cannot be safely skipped."""
    mismatches = []
    for name in RESUME_REQUIRED_FILES:
        path = sample_dir / name
        if not path.is_file() or path.stat().st_size == 0:
            mismatches.append(f"missing_or_empty:{name}")
    if mismatches:
        return mismatches

    try:
        metadata = json.loads((sample_dir / "metadata.json").read_text(encoding="utf-8"))
        summary = json.loads((sample_dir / "direct_aggregate_summary.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"invalid_json:{type(exc).__name__}"]
    if not isinstance(metadata, dict) or not isinstance(summary, dict):
        return ["metadata_and_summary_must_be_objects"]

    if metadata.get("status") != "ok":
        mismatches.append(f"status:{metadata.get('status')!r}")
    if metadata.get("error"):
        mismatches.append("metadata_error_is_not_empty")
    selected_summary = summary.get("selected_block_aggregate_mask")
    if not selected_summary:
        mismatches.append("missing_selected_block_aggregate_mask")
    if not summary.get("all_block_aggregate_mask"):
        mismatches.append("missing_all_block_aggregate_mask")

    expected_fields = (
        ("seed", metadata.get("seed"), expected_seed),
        ("ref_id", metadata.get("ref_id"), expected_ref_id),
        ("anomaly", metadata.get("anomaly"), expected_anomaly),
        ("index", metadata.get("index"), expected_index),
        ("num_inference_steps", metadata.get("num_inference_steps"), expected_num_inference_steps),
        ("target_mask_source", metadata.get("target_mask_source"), expected_target_mask_source),
    )
    for name, recorded, expected in expected_fields:
        if expected is not None and recorded != expected:
            mismatches.append(f"{name}:recorded={recorded!r}:expected={expected!r}")
    for name, recorded, expected in (
        ("source_image", metadata.get("source_image"), expected_source_image),
        ("ref_image", metadata.get("ref_image"), expected_ref_image),
        ("ref_mask", metadata.get("ref_mask"), expected_ref_mask),
    ):
        if expected is not None and not _path_matches(recorded, expected):
            mismatches.append(f"{name}:recorded={recorded!r}:expected={str(expected)!r}")

    normalized_steps = None if expected_steps is None else [int(step) for step in expected_steps]
    normalized_blocks = None if expected_selected_blocks is None else [str(block) for block in expected_selected_blocks]
    direct_metadata = metadata.get("direct_aggregate_mask")
    if not isinstance(direct_metadata, dict):
        mismatches.append("metadata_direct_aggregate_mask_missing")
        direct_metadata = {}
    if expected_kind is not None:
        if summary.get("kind") != expected_kind:
            mismatches.append(f"summary_kind:{summary.get('kind')!r}:expected={expected_kind!r}")
        if direct_metadata.get("kind") != expected_kind:
            mismatches.append(f"metadata_kind:{direct_metadata.get('kind')!r}:expected={expected_kind!r}")
    if normalized_steps is not None:
        for location, recorded in (
            ("metadata_steps", metadata.get("direct_aggregate_steps")),
            ("summary_steps", summary.get("direct_aggregate_steps")),
            ("metadata_direct_steps", direct_metadata.get("direct_aggregate_steps")),
        ):
            if recorded != normalized_steps:
                mismatches.append(f"{location}:recorded={recorded!r}:expected={normalized_steps!r}")
    if normalized_blocks is not None:
        if summary.get("selected_blocks") != normalized_blocks:
            mismatches.append(
                f"summary_selected_blocks:recorded={summary.get('selected_blocks')!r}:expected={normalized_blocks!r}"
            )
        if direct_metadata.get("selected_blocks") != normalized_blocks:
            mismatches.append(
                "metadata_selected_blocks:"
                f"recorded={direct_metadata.get('selected_blocks')!r}:expected={normalized_blocks!r}"
            )
        if not isinstance(selected_summary, dict) or selected_summary.get("blocks") != normalized_blocks:
            recorded = selected_summary.get("blocks") if isinstance(selected_summary, dict) else None
            mismatches.append(f"selected_mask_blocks:recorded={recorded!r}:expected={normalized_blocks!r}")

    adaptive_metadata = metadata.get("adaptive_ref_injection")
    if any(
        value is not None
        for value in (
            expected_adaptive_kind,
            expected_adaptive_selected_blocks,
            expected_adaptive_default_polarity,
        )
    ):
        if not isinstance(adaptive_metadata, dict) or adaptive_metadata.get("enabled") is not True:
            mismatches.append("metadata_adaptive_ref_injection_missing_or_disabled")
            adaptive_metadata = {}
    if expected_adaptive_kind is not None and adaptive_metadata.get("attention_kind") != expected_adaptive_kind:
        mismatches.append(
            "adaptive_kind:"
            f"recorded={adaptive_metadata.get('attention_kind')!r}:expected={expected_adaptive_kind!r}"
        )
    if expected_adaptive_selected_blocks is not None:
        expected_adaptive_blocks = [str(block) for block in expected_adaptive_selected_blocks]
        if adaptive_metadata.get("selected_blocks") != expected_adaptive_blocks:
            mismatches.append(
                "adaptive_selected_blocks:"
                f"recorded={adaptive_metadata.get('selected_blocks')!r}:expected={expected_adaptive_blocks!r}"
            )
    if expected_adaptive_default_polarity is not None:
        adaptive_polarities = adaptive_metadata.get("block_polarities")
        recorded_default = adaptive_polarities.get("__default__") if isinstance(adaptive_polarities, dict) else None
        if recorded_default != expected_adaptive_default_polarity:
            mismatches.append(
                "adaptive_default_polarity:"
                f"recorded={recorded_default!r}:expected={expected_adaptive_default_polarity!r}"
            )
    return mismatches


def sample_is_complete_for_resume(sample_dir: Path, **expected: object) -> bool:
    """Only skip samples whose artifacts and generation identity match this run."""
    return not sample_resume_mismatches(sample_dir, **expected)


def decide_sample_resume_action(
    sample_dir: Path,
    *,
    overwrite: bool,
    **expected: object,
) -> str:
    """Return ``skip``/``generate`` or reject a non-empty unsafe sample directory."""
    if overwrite:
        return "generate"
    mismatches = sample_resume_mismatches(sample_dir, **expected)
    if not mismatches:
        return "skip"
    if not sample_dir.exists() or not any(sample_dir.iterdir()):
        return "generate"
    raise RuntimeError(
        f"Refusing to overwrite or skip unsafe existing sample {sample_dir}: "
        + "; ".join(mismatches[:12])
        + (f"; ... and {len(mismatches) - 12} more" if len(mismatches) > 12 else "")
        + ". Use a new --out-root or pass --overwrite to regenerate it intentionally."
    )
