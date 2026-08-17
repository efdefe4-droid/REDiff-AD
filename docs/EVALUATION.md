# Evaluation

The evaluation launchers support the REDiff-AD output layout directly:

```text
<run-root>/<defect>/ref_<reference-id>/<sample>/edit.png
<run-root>/<defect>/ref_<reference-id>/<sample>/contour_refined_mask.png
```

## Diversity/quality

`eval_diversity/run_insertanything_eval.sh` prepares flat image links and can run KID, IS, and IC-LPIPS. A six-image-per-defect run is useful only for plumbing validation; it is too small for a thesis metric claim.

Create the environment documented in the README and install
`requirements-eval.txt` before running metrics. The launchers default to the
environment name `rediff-ad`; override `CONDA_ENV` when using another name.
They fail with an actionable error when a module is missing and never install
packages implicitly. KID and IC-LPIPS default to seed 2026, and IS reduces the
requested split count to the available image count for small plumbing runs.

## Downstream tasks

- Classification: `eval_downstream/run_insertanything_classification.sh`
- Localization: `eval_downstream/run_insertanything_localization.sh`

Use `PREPARE_ONLY=1` first. The default localization mask is `contour_refined_mask.png`; mask ablations can set `MASK_NAME` to `coarse_mask.png`, `q80_appearance_mask.png`, or `all_block_coarse_mask.png`.

All three launchers derive `RESULT_TAG` from the basename of `RESULT_ROOT`, so separate generation runs do not overwrite each other's metrics, prepared data, logs or checkpoints. Set an explicit unique `RESULT_TAG` when two result roots share a basename. Downstream training defaults to seed 2026.

## Known protocol constraints

- Full metric/training jobs require the evaluation environment and CUDA.
- The first metric run may need pretrained Inception/LPIPS weights.
- Symlink-based prepared data is space efficient but becomes invalid if the generation root moves.
- Before a publication run, pin model snapshots and retain the metric CSV plus command/environment manifest.
