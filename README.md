# REDiff-AD

Reference-guided diffusion anomaly editing with target-to-reference (T2R)
attention localization, adaptive reference injection, Shape-K diversity, and
refined anomaly masks.

> **Double-blind review:** author names and identifying source-repository
> metadata are intentionally withheld. They will be restored after review.

## What is included

```text
run_insert_anything.py                  FLUX Fill, Redux and LoRA loading
batch_insert_anything.py                image/mask preparation
generation_attention/                   T2R attention, diversity and refinement
configs/                                frozen T2R blocks and reproducibility data
scripts/run_hazelnut_t2r.sh             main generation launcher
scripts/smoke_hazelnut_t2r.sh           one-image integration smoke test
scripts/validate_smoke_output.py         output/runtime validator
eval_diversity/                          KID, IS and IC-LPIPS
eval_downstream/                         classification and localization
```

The default method uses:

- Direct localization: T2R attention with the frozen Top-10 blocks.
- Adaptive reference injection: the same T2R Top-10 blocks.
- Shape-K diversity: 36 middle blocks, diffusion steps 12–20.
- Mask refinement: Q80 appearance refinement followed by contour refinement.
- FLUX transformer: bitsandbytes INT4/NF4 with Insert-Anything LoRA.

Implementation details are summarized in
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md), and the exact experimental
contract is in [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).

## 1. Installation

Python 3.11 and a CUDA-capable GPU are recommended. The verified generation
stack uses PyTorch 2.6.0 with CUDA 12.4.

```bash
git clone https://github.com/efdefe4-droid/REDiff-AD.git
cd REDiff-AD

conda create -n rediff-ad python=3.11 -y
conda activate rediff-ad

pip install torch==2.6.0 torchvision==0.21.0 \
  --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

For evaluation, install the optional dependencies in the same environment:

```bash
pip install -r requirements-eval.txt
```

The code uses these Hugging Face repositories:

- `black-forest-labs/FLUX.1-Fill-dev`
- `black-forest-labs/FLUX.1-Redux-dev`
- `WensongSong/Insert-Anything`

Request access where required, then authenticate:

```bash
huggingface-cli login
```

Launchers default to local-cache-only mode. On the first run, set
`LOCAL_FILES_ONLY=0` to download the model files; later runs can omit it.

## 2. Dataset

Download MVTec AD from the
[official dataset page](https://www.mvtec.com/research-teaching/datasets/mvtec-ad)
and follow its license terms. Set `MVTEC_ROOT` to the directory containing the
object folders:

```text
<MVTEC_ROOT>/
  hazelnut/
    train/good/*.png
    test/crack/*.png
    test/hole/*.png
    test/print/*.png
    test/cut/*.png
    ground_truth/<defect>/*_mask.png
```

Either export the path or place the data under `datasets/` in this repository:

```bash
export MVTEC_ROOT=/path/to/mvtec_ad
```

No model, dataset, generated image, checkpoint, or log is tracked by Git.

## 3. Quick reproduction

First check all paths and effective parameters without loading a model:

```bash
conda activate rediff-ad
DRY_RUN=1 LOG_TO_FILE=0 bash scripts/smoke_hazelnut_t2r.sh
```

Run one hazelnut `crack` sample with reference `000`, seed 309, and 30 steps:

```bash
LOCAL_FILES_ONLY=0 bash scripts/smoke_hazelnut_t2r.sh
```

The default output is `outputs/smoke_rediff_ad_seed309`. Validate both its
files and recorded runtime configuration:

```bash
python scripts/validate_smoke_output.py \
  outputs/smoke_rediff_ad_seed309 \
  --defects crack --samples-per-defect 1
```

The validator checks active LoRA adapters, Direct/Adaptive T2R Top-10,
all-block attention, Shape-K calls, coarse/Q80/contour masks, and whether the
edit is localized inside the generated target mask.

## 4. Generation

Generate six samples for each hazelnut defect:

```bash
RUN_NAME=hazelnut_rediff_ad \
ANOMALIES_STR='crack hole print cut' \
SAMPLES_PER_ANOMALY=6 \
SEED=309 \
bash scripts/run_hazelnut_t2r.sh
```

Common overrides:

```bash
OBJECT_NAME=hazelnut                  # MVTec object folder
DATASET_ROOT=/path/to/mvtec/hazelnut # alternative to MVTEC_ROOT
REF_IDS_STR='000 000 000 000'         # one reference ID per defect
OUT_ROOT=outputs/my_run
OVERWRITE=0                           # resume matching complete samples
```

The main launcher also accepts another MVTec object through `OBJECT_NAME`,
`ANOMALIES_STR`, and `REF_IDS_STR`; no machine-specific absolute path is built
into the code.

Each sample directory contains:

- `edit.png`: generated image.
- `coarse_mask.png`: selected T2R Top-10 coarse mask.
- `all_block_coarse_mask.png`: all-block diagnostic mask.
- `q80_appearance_mask.png`: first refinement stage.
- `contour_refined_mask.png`: final mask used by downstream evaluation.
- `metadata.json` and `direct_aggregate_summary.json`: reproducibility records.

## 5. Evaluation

All evaluation launchers default to the `rediff-ad` conda environment and the
Insert-Anything output layout. Set the generation and dataset roots once:

```bash
export RESULT_ROOT="$PWD/outputs/hazelnut_rediff_ad"
export MVTEC_ROOT=/path/to/mvtec_ad
```

Check the diversity-evaluation layout without computing metrics:

```bash
OBJ=hazelnut ANOMALIES='crack hole print cut' \
RESULT_ROOT="$RESULT_ROOT" REAL_ROOT="$MVTEC_ROOT" \
MAX_IMAGES=6 DRY_RUN=1 \
bash eval_diversity/run_insertanything_eval.sh
```

Run KID, IS, and IC-LPIPS by removing `DRY_RUN=1`. Individual metrics can be
controlled with `RUN_KID`, `RUN_IS`, and `RUN_LPIPS`.

Prepare and verify localization pairs before training:

```bash
OBJ=hazelnut ANOMALIES='crack hole print cut' \
RESULT_ROOT="$RESULT_ROOT" MVTEC_PATH="$MVTEC_ROOT" \
MASK_NAME=contour_refined_mask.png MAX_IMAGES=6 PREPARE_ONLY=1 \
bash eval_downstream/run_insertanything_localization.sh
```

Use the same command without `PREPARE_ONLY=1` for localization training.
Classification uses the corresponding launcher:

```bash
OBJ=hazelnut ANOMALIES='crack hole print cut' \
RESULT_ROOT="$RESULT_ROOT" MVTEC_PATH="$MVTEC_ROOT" \
MAX_IMAGES=6 PREPARE_ONLY=1 \
bash eval_downstream/run_insertanything_classification.sh
```

See [docs/EVALUATION.md](docs/EVALUATION.md) for metric controls and mask
ablations. Very small smoke runs verify plumbing only and are not sufficient
for reporting paper metrics.

## Verification

CPU-only contract tests do not download models:

```bash
pip install -r requirements-dev.txt
make check
```

`configs/reproducibility.json` records the effective method settings and
SHA-256 hashes of runtime-critical source files. GitHub Actions checks these
contracts on every push.

Third-party model and dataset notices are listed in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). This anonymous review
snapshot does not grant a project-level redistribution license; attribution
and the archival license will be restored after review.
