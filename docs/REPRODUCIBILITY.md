# Reproducibility contract

The canonical hazelnut run is defined by `scripts/run_hazelnut_t2r.sh`. These values are explicit rather than inherited from Python defaults.

| Setting | Canonical value |
|---|---|
| Defects | crack, hole, print, cut |
| Reference per defect | dataset `000` image and mask |
| Samples per defect | 6 |
| Base seed | 309 |
| Inference | 512 working size, 30 steps |
| Target-mask area | 0.5–0.9 × reference-mask area (INT4 stability profile) |
| Direct attention | T2R, frozen Top-10, initial-mask ROI |
| Direct aggregate steps | 10, 15, 20, 25, 27, 28, 29 |
| All-block attention | T2R, exactly 57 blocks |
| Quantization | bitsandbytes INT4/NF4 |
| LoRA | `WensongSong/Insert-Anything`, `20250321_steps5000_pytorch_lora_weights.safetensors` |
| Adaptive | enabled; T2R frozen Top-10 monitor |
| Shape-K | enabled; middle 36; both; eta 0.30; suppress 0.60 |
| Refinement | Q80 enabled; contour enabled; component mode all |

`configs/reproducibility.json` records the non-identifying experimental contract and source hashes. Identifying source URLs and commit IDs are withheld for double-blind review and will be restored in the archival release. CI recomputes every listed hash so a code/config change cannot silently retain stale provenance.

Generation fails closed if the loaded LoRA has no active adapter. The resolved available/active adapter names are written to `run_config.json` and each sample's `metadata.json`. The smoke validator also compares `edit.png` with its source inside and outside `generated_target_mask.png`, so a structurally complete no-op cannot be reported as a passing run.

The legacy target-mask range `0.40–2.5` from the broader BF16 experiment is still available through environment overrides. It is not the default INT4 profile because one crack/000 large-mask INT4 diagnostic yielded a no-op; the mechanism and cross-defect success rate have not yet been established.

## Seeds

The Python runner offsets the base seed by defect and sample index:

```text
sample_seed = base_seed + defect_offset * 1,000,003 + sample_index
```

This prevents two defect classes from using identical random streams. For the canonical anomaly order, the actual seeds are crack 309–314, hole 1,000,312–1,000,317, print 2,000,315–2,000,320 and cut 3,000,318–3,000,323. Subtracting each class offset gives the normalized local range 309–314.

## Resume safety

Use a new `RUN_NAME` when changing a result-affecting setting. `OVERWRITE=0` resumes complete samples and preserves logs; `OVERWRITE=1` regenerates samples and resets logs. The output validator checks the effective `run_config.json`, sample metadata, runtime attention summaries, and all required masks.

## Model and dataset assets

Models and MVTec AD are intentionally not stored in Git. Record the exact local Hugging Face snapshot/revision in experiment notes when preparing a paper release; repository IDs alone can move over time.
