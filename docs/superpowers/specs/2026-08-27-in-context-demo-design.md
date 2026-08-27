# In-Context Naming and Self-Contained Hazelnut Demo Design

## Goal

Rebrand the repository's project-owned Insert-Anything terminology as
In-Context, while preserving technically and legally required upstream model
provenance, and add a directly runnable hazelnut demonstration that does not
require a separate MVTec AD dataset download.

## Scope

This change starts from `origin/main` commit `72da207` on the isolated branch
`feature/in-context-demo`. It does not import any uncommitted changes from the
original `thesis_main` checkout.

The deliverables are:

1. Project-owned runtime files, APIs, CLI options, log messages, evaluation
   launchers, output-layout names, tests, and documentation use In-Context
   terminology.
2. Required upstream LoRA identifiers and third-party attribution retain the
   upstream `WensongSong/Insert-Anything` name where users need the exact model
   source or where provenance is being documented.
3. A repository-local hazelnut demo subset contains 15 normal source images,
   one anomalous reference image for each of four defect classes, and the four
   corresponding masks.
4. A single shell launcher runs the existing REDiff-AD generation pipeline on
   the bundled subset without requiring a separately installed dataset.
5. README and third-party notices explain model prerequisites, demo commands,
   asset provenance, and licensing.

## Non-Goals

- Do not rename the REDiff-AD repository or the T2R, adaptive-injection,
  Shape-K, Q80, or contour-refinement method components.
- Do not change generation mathematics, model weights, attention policies,
  default thresholds, or evaluation algorithms.
- Do not bundle FLUX, Redux, LoRA, metric, or other model weights.
- Do not make a real GPU generation run part of CPU-only CI.
- Do not preserve legacy project-owned filenames, environment variables, or
  CLI aliases containing the old branding. This branch is the clean naming
  transition rather than a compatibility release.

## Naming Contract

Human-facing method text uses `In-Context`. Files and Python identifiers use
`in_context`; shell environment variables use `IN_CONTEXT`; CLI options use
the conventional hyphenated spelling `--in-context-*`.

The primary rename map is:

| Existing name | New name |
| --- | --- |
| `run_insert_anything.py` | `run_in_context.py` |
| `batch_insert_anything.py` | `batch_in_context.py` |
| `run_insertanything_eval.sh` | `run_in_context_eval.sh` |
| `run_insertanything_classification.sh` | `run_in_context_classification.sh` |
| `run_insertanything_localization.sh` | `run_in_context_localization.sh` |
| `insert_anything_sample_dirs` | `in_context_sample_dirs` |
| `--insert-anything-results-root` | `--in-context-results-root` |
| `insert-anything` output-layout value | `in_context` |
| `INSERT_ANYTHING_LORA_PATH` | `IN_CONTEXT_LORA_PATH` |
| `INSERT_ANYTHING_LORA_WEIGHT` | `IN_CONTEXT_LORA_WEIGHT` |
| `_insert_anything_lora_audit` | `_in_context_lora_audit` |

Every import, invocation, hash key, test expectation, and documentation link
must follow renamed files. A source-contract test scans tracked text and
rejects old branding except in an explicit, narrow upstream-provenance
allowlist. Allowed occurrences must refer to the exact third-party model name,
not to the project or its output format.

## Demo Asset Layout

The bundled data lives outside the ignored `datasets/` tree:

```text
demo_assets/mvtec_ad/
  README.md
  LICENSE.md
  MANIFEST.sha256
  hazelnut/
    train/good/000.png ... 014.png
    test/crack/000.png
    test/cut/000.png
    test/hole/000.png
    test/print/000.png
    ground_truth/crack/000_mask.png
    ground_truth/cut/000_mask.png
    ground_truth/hole/000_mask.png
    ground_truth/print/000_mask.png
```

The 23 PNG files are byte-for-byte copies from the authorized local MVTec AD
hazelnut dataset at `<local-mvtec-root>/hazelnut`. Normal images are the first
15 naturally sorted files, `000.png` through `014.png`.
Anomalous references and masks are ID `000` for the four non-good defect
classes. `test/good/000.png` is not included.

The asset README identifies MVTec Software GmbH, links the official MVTec AD
page, cites the dataset paper, states the subset selection, and clearly marks
the images and annotations as CC BY-NC-SA 4.0. `LICENSE.md` records the asset
license and does not claim to license the repository's source code. The
manifest records SHA-256 hashes for all 23 PNG files and is verified by tests.

## Demo Launcher

Create `scripts/run_hazelnut_demo.sh` as a small wrapper around
`scripts/run_hazelnut_t2r.sh`. It resolves all paths relative to the checkout
and exports:

- `DATASET_ROOT=<repo>/demo_assets/mvtec_ad/hazelnut`
- `ANOMALIES_STR='crack hole print cut'`
- `REF_IDS_STR='000 000 000 000'`
- `SAMPLES_PER_ANOMALY=1` by default
- `SEED=309` by default
- `RUN_NAME=hazelnut_in_context_demo_seed<seed>` by default
- `LOCAL_FILES_ONLY=0` by default so a first run may fetch missing model files

The wrapper preserves caller overrides. `DRY_RUN=1 LOG_TO_FILE=0` validates
paths and prints the effective contract without loading a model. A default
real run produces one result for each defect. Setting
`SAMPLES_PER_ANOMALY=15` uses every bundled normal image once per defect in a
deterministically shuffled order.

The script validates the exact demo asset layout before delegating. Missing
assets produce a concise error naming the missing path. Model access remains a
separate prerequisite: users need a CUDA-capable environment, access to FLUX
Fill and Redux, and access to the upstream LoRA repository.

## Runtime and Evaluation Changes

Runtime modules are renamed without changing public generation semantics.
Imports and audit metadata use the new identifiers. Default model repository
and weight values remain unchanged because they locate the required upstream
weights.

Evaluation launchers and classification preparation use the `in_context`
layout label and `--in-context-results-root`. Generated directory structure is
unchanged; only names describing that structure change. Existing REDiff-AD
output directories remain readable when passed through the renamed layout
option because their on-disk sample schema does not change.

`configs/reproducibility.json` updates descriptive text, renamed hash keys,
and all runtime-critical SHA-256 values after implementation. It retains the
exact upstream LoRA path as model provenance.

## Documentation

README's opening architecture description remains REDiff-AD focused. File
lists, installation notes, generation, evaluation, and validation examples
use the new filenames and In-Context terminology. A new self-contained demo
section distinguishes two dependencies:

- Dataset: already included as the licensed hazelnut demo subset.
- Models: downloaded or cached separately and subject to their own access
  terms.

README documents these commands:

```bash
DRY_RUN=1 LOG_TO_FILE=0 bash scripts/run_hazelnut_demo.sh
LOCAL_FILES_ONLY=0 bash scripts/run_hazelnut_demo.sh
SAMPLES_PER_ANOMALY=15 bash scripts/run_hazelnut_demo.sh
```

`THIRD_PARTY_NOTICES.md` is updated from "does not redistribute MVTec" to a
precise statement that the small demo subset is redistributed under its own
CC BY-NC-SA 4.0 notice. It continues to identify the upstream LoRA weights.

## Error Handling and Safety

- The original dirty checkout is never modified; all edits occur in the
  isolated worktree.
- The launcher rejects missing or incomplete demo assets before creating a
  model workload.
- The manifest test detects accidental image replacement, omission, or extra
  tracked PNG files.
- Naming tests prevent project branding from regressing while permitting only
  necessary upstream provenance.
- Binary images are copied without transformation so their hashes and source
  identity remain auditable.
- No model token, local absolute runtime path, checkpoint, output, or log is
  committed.

## Test Strategy

Tests are implemented before their corresponding production changes and are
run through a dedicated development environment rather than modifying the
generation environment.

1. A naming-contract test initially fails on old filenames, imports, CLI
   options, layout values, logs, and documentation; it passes after the rename
   and allows only exact upstream provenance locations.
2. A demo-assets test initially fails because the bundle and manifest do not
   exist; it then verifies the required 23-file set, non-empty images, expected
   1024 x 1024 dimensions, mask modes, and manifest hashes.
3. A launcher-contract test initially fails because the demo script does not
   exist; it then uses the existing dry-run harness to verify the bundled data
   root, all four defects, four `000` references, output path, and model-locality
   override behavior.
4. Existing source-boundary, reproducibility, validator, and launcher tests are
   updated to renamed interfaces and kept green.
5. Every shell file passes `bash -n`; all CPU contract tests pass with pytest.
6. A real GPU generation is documented for users but is not required for CI or
   the branch's CPU verification gate.

## Acceptance Criteria

- The branch is based on the current remote `origin/main`, with no files copied
  from the dirty original checkout except the explicitly selected MVTec assets
  from the external dataset path.
- Project-owned filenames and identifiers contain no old branding.
- Remaining old-name text is limited to required upstream model provenance and
  third-party notices.
- The repository contains exactly 15 bundled normal hazelnut images, four
  defect reference images, and four masks with a valid manifest.
- `scripts/run_hazelnut_demo.sh` passes a dataset-free dry run from a standalone
  checkout and documents first-run model download requirements.
- CPU tests and shell syntax checks pass.
- Changes are committed to `feature/in-context-demo` and pushed to the same
  GitHub remote without modifying `main`.
