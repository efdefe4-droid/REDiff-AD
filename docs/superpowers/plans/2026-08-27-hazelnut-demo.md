# Self-Contained Hazelnut Demo Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bundle a licensed 23-image MVTec AD hazelnut subset and provide a one-command In-Context demo that needs no separately downloaded dataset.

**Architecture:** Store the byte-identical demo subset under `demo_assets/mvtec_ad`, protect it with a fixed SHA-256 and image-shape contract, and point a thin shell wrapper at the existing hazelnut generation launcher. Keep model download/authentication separate from dataset availability and document both paths explicitly.

**Tech Stack:** Bash, Python 3.11+, pytest, Pillow, SHA-256 manifests, Markdown.

**Spec:** `docs/superpowers/specs/2026-08-27-in-context-demo-design.md`

## Global Constraints

- Execute after `docs/superpowers/plans/2026-08-27-in-context-naming.md` is complete.
- Work only in the isolated `feature/in-context-demo` worktree.
- Copy exactly 15 normal images (`000.png` through `014.png`), four defect images (`000.png` for crack, hole, print, and cut), and four corresponding `000_mask.png` annotations.
- Do not copy `test/good/000.png`, any additional dataset image, model weight, generated output, log, or local absolute runtime path.
- Demo images and annotations remain under CC BY-NC-SA 4.0 with dedicated attribution; that asset notice does not claim to license repository source code.
- A default demo produces one sample for each defect; `SAMPLES_PER_ANOMALY=15` uses all 15 normal sources once per defect.
- CPU tests and dry runs must not load or download models.

---

### Task 1: Add a fixed, licensed demo-asset contract

**Files:**
- Create: `tests/test_demo_assets.py`
- Create: `demo_assets/mvtec_ad/README.md`
- Create: `demo_assets/mvtec_ad/LICENSE.md`
- Create: `demo_assets/mvtec_ad/MANIFEST.sha256`
- Create: `demo_assets/mvtec_ad/hazelnut/train/good/000.png` through `014.png`
- Create: `demo_assets/mvtec_ad/hazelnut/test/{crack,hole,print,cut}/000.png`
- Create: `demo_assets/mvtec_ad/hazelnut/ground_truth/{crack,hole,print,cut}/000_mask.png`

**Interfaces:**
- Consumes: authorized local MVTec AD files supplied out-of-band through the task-specific `MVTEC_DEMO_SOURCE_ROOT` environment variable.
- Produces: a deterministic 23-PNG dataset-compatible tree plus a verifiable manifest and scoped license notice.

- [ ] **Step 1: Write the failing asset-integrity test**

Create `tests/test_demo_assets.py`. The test must use the literal required
relative-path set and the literal source hashes below, then verify the real
files with Pillow rather than trusting the manifest to describe itself.

```python
from __future__ import annotations

import hashlib
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
ASSET_ROOT = ROOT / "demo_assets" / "mvtec_ad"
HAZELNUT_ROOT = ASSET_ROOT / "hazelnut"

EXPECTED_HASHES = {
    "hazelnut/train/good/000.png": "5e28a714fa36ef5198c683058b607435b792974cdd23f3e0810b887dbdfe7112",
    "hazelnut/train/good/001.png": "612e2884069ce4ca26c96b9e885d54cee1cf3f16256c83a74b985966c4098c4f",
    "hazelnut/train/good/002.png": "96ec0d30fb80620a0c3dc09173a856cda9681a24c383866272a04f05e360c2d2",
    "hazelnut/train/good/003.png": "7c20b778f1226beb943485c0489373a8185fa5debbc8c371398e2d5762189cf4",
    "hazelnut/train/good/004.png": "de53dc392f6cdae2bb25e325de6c159af78552529fc2bdb35a0b8d7d7466e2f3",
    "hazelnut/train/good/005.png": "90a3f7e86066c8558d81b2c2f71ecbb7ad38de78d195deddbb3d78f0348dbb4d",
    "hazelnut/train/good/006.png": "0cfed0af8c0d3b69101e01849aea21ddd791a55e1f032c1a235de353190d47fc",
    "hazelnut/train/good/007.png": "b2679a80a85894a7a7bb465a7fb53f5f03d40d5b5b2fe8e0e516a4624802abb7",
    "hazelnut/train/good/008.png": "7cc3bc930ceca4b58cc45050dd4aed66e42dbc7270f90a275b692adea1031c2b",
    "hazelnut/train/good/009.png": "1502087011ec6a5ec6e089e64506526f1f46fa9e852aba88b2add9b02f99f0ec",
    "hazelnut/train/good/010.png": "4b3cc485775daaf8512305aae869799b6f5659833a4b1176df181c9ff7d9d102",
    "hazelnut/train/good/011.png": "ac46ac9b51f6af9eaf033f03615270ef9fb59608dbfbd8a386871be3a04f0303",
    "hazelnut/train/good/012.png": "ac46f6aa2cc8618d5d04b468390a5c9cd2f331dd3cbbfd80f69b9013f05eef0d",
    "hazelnut/train/good/013.png": "511fb2e58ff750e69aea2aab59b95301e409ca923645e08452e06614b12688a4",
    "hazelnut/train/good/014.png": "7924bcfefa9736647abdf63409ce279d6df11a8e03b0aaf5bc0f9fb4db5ef601",
    "hazelnut/test/crack/000.png": "e26b54282b6b11286760905ca2880af594561b606dfa0643859d6ee6027cf8c2",
    "hazelnut/test/hole/000.png": "b9a1a102db263079fec8bc9836f182253dced9ca178957b7e9bef05cba3312c2",
    "hazelnut/test/print/000.png": "63245ac91c1f83c97e081bbcfde5bb9751b008a994c201d9f18dcd6db3552c87",
    "hazelnut/test/cut/000.png": "901bf3b3ddef6e7d3592b00cfd0f5c512b745ef61e7837ed4e283da2f99c1b8a",
    "hazelnut/ground_truth/crack/000_mask.png": "299566b57b84b9d466f4f7f50a8210e0bd0ff2a4ac5922f456d00cd9169cfbb4",
    "hazelnut/ground_truth/hole/000_mask.png": "0db752dd6620c7c7493fe8b150e2a706d4c851a15a27c919e545552850e85018",
    "hazelnut/ground_truth/print/000_mask.png": "2ad368e89aa0f9e28dd3df9a5581ab9815916447abbb5e13ee0405f666057870",
    "hazelnut/ground_truth/cut/000_mask.png": "5dc3b2dc619d9e0d15fed32cf39b3bc8b2ba425964a00e947745c632ccf73c5b",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_demo_subset_is_exact_and_auditable() -> None:
    actual = {
        path.relative_to(ASSET_ROOT).as_posix()
        for path in HAZELNUT_ROOT.rglob("*.png")
    }
    assert actual == set(EXPECTED_HASHES)
    for relative, expected_hash in EXPECTED_HASHES.items():
        path = ASSET_ROOT / relative
        assert path.stat().st_size > 0
        assert sha256(path) == expected_hash
        with Image.open(path) as image:
            assert image.size == (1024, 1024)
            expected_mode = "L" if "ground_truth" in relative else "RGB"
            assert image.mode == expected_mode

    manifest = (ASSET_ROOT / "MANIFEST.sha256").read_text(encoding="utf-8").splitlines()
    assert manifest == [f"{digest}  {relative}" for relative, digest in EXPECTED_HASHES.items()]
```

- [ ] **Step 2: Run the asset test and verify RED**

```bash
/tmp/rediff-ad-in-context-dev/bin/python -m pytest tests/test_demo_assets.py::test_demo_subset_is_exact_and_auditable -v
```

Expected: FAIL because `demo_assets/mvtec_ad` does not exist.

- [ ] **Step 3: Copy exactly the selected binary assets**

Create the target directories, then copy the files byte-for-byte from the local
dataset. The controller sets `MVTEC_DEMO_SOURCE_ROOT` for the current execution
without recording its machine-specific value in Git. Do not copy directories
or globs broader than the named selection.

```bash
mkdir -p demo_assets/mvtec_ad/hazelnut/train/good
mkdir -p demo_assets/mvtec_ad/hazelnut/test/crack demo_assets/mvtec_ad/hazelnut/test/hole demo_assets/mvtec_ad/hazelnut/test/print demo_assets/mvtec_ad/hazelnut/test/cut
mkdir -p demo_assets/mvtec_ad/hazelnut/ground_truth/crack demo_assets/mvtec_ad/hazelnut/ground_truth/hole demo_assets/mvtec_ad/hazelnut/ground_truth/print demo_assets/mvtec_ad/hazelnut/ground_truth/cut
cp "$MVTEC_DEMO_SOURCE_ROOT/train/good/"0{00..14}.png demo_assets/mvtec_ad/hazelnut/train/good/
cp "$MVTEC_DEMO_SOURCE_ROOT/test/crack/000.png" demo_assets/mvtec_ad/hazelnut/test/crack/000.png
cp "$MVTEC_DEMO_SOURCE_ROOT/test/hole/000.png" demo_assets/mvtec_ad/hazelnut/test/hole/000.png
cp "$MVTEC_DEMO_SOURCE_ROOT/test/print/000.png" demo_assets/mvtec_ad/hazelnut/test/print/000.png
cp "$MVTEC_DEMO_SOURCE_ROOT/test/cut/000.png" demo_assets/mvtec_ad/hazelnut/test/cut/000.png
cp "$MVTEC_DEMO_SOURCE_ROOT/ground_truth/crack/000_mask.png" demo_assets/mvtec_ad/hazelnut/ground_truth/crack/000_mask.png
cp "$MVTEC_DEMO_SOURCE_ROOT/ground_truth/hole/000_mask.png" demo_assets/mvtec_ad/hazelnut/ground_truth/hole/000_mask.png
cp "$MVTEC_DEMO_SOURCE_ROOT/ground_truth/print/000_mask.png" demo_assets/mvtec_ad/hazelnut/ground_truth/print/000_mask.png
cp "$MVTEC_DEMO_SOURCE_ROOT/ground_truth/cut/000_mask.png" demo_assets/mvtec_ad/hazelnut/ground_truth/cut/000_mask.png
```

- [ ] **Step 4: Add the exact manifest and scoped asset documentation**

Write `MANIFEST.sha256` in the same insertion order as `EXPECTED_HASHES`, using
the literal digest and relative path pairs above. Write `README.md` with:

```markdown
# MVTec AD hazelnut demo subset

This directory contains 15 defect-free source images and one reference image
plus pixel-precise mask for each hazelnut defect class (`crack`, `hole`,
`print`, and `cut`). Files are unmodified excerpts from MVTec AD for the
non-commercial, reproducible research demo in this repository.

Source: https://www.mvtec.com/research-teaching/datasets/mvtec-ad

Please cite: Paul Bergmann, Michael Fauser, David Sattlegger, and Carsten
Steger, “MVTec AD — A Comprehensive Real-World Dataset for Unsupervised
Anomaly Detection,” CVPR 2019.

The images and annotations in this directory are distributed under CC
BY-NC-SA 4.0; see `LICENSE.md`. This notice does not grant a license for other
repository contents.
```

Write `LICENSE.md` with the asset scope, MVTec attribution, official source
link, the full license name, and link
`https://creativecommons.org/licenses/by-nc-sa/4.0/`. Do not claim that the
repository source code is licensed by MVTec or under Creative Commons.

- [ ] **Step 5: Run the asset test and verify GREEN**

```bash
/tmp/rediff-ad-in-context-dev/bin/python -m pytest tests/test_demo_assets.py -v
```

Expected: PASS with exactly 23 images, matching hashes, RGB image modes, grayscale mask modes, and a matching manifest.

- [ ] **Step 6: Commit the licensed demo subset**

```bash
git add demo_assets/mvtec_ad tests/test_demo_assets.py
git commit -m "feat: bundle licensed hazelnut demo subset"
```

---

### Task 2: Add the directly runnable demo launcher

**Files:**
- Create: `scripts/run_hazelnut_demo.sh`
- Modify: `tests/test_launcher_contract.py`
- Modify: `Makefile`

**Interfaces:**
- Consumes: `demo_assets/mvtec_ad/hazelnut` and `scripts/run_hazelnut_t2r.sh`.
- Produces: `bash scripts/run_hazelnut_demo.sh`, a four-defect default demo with caller-overridable controls.

- [ ] **Step 1: Write the failing launcher behavior test**

Add this integration test to `tests/test_launcher_contract.py`; it executes the
real shell wrapper with model loading disabled by the existing dry-run path.

```python
def test_bundled_hazelnut_demo_dry_run_uses_all_defects(tmp_path: Path) -> None:
    env = os.environ.copy()
    for name in ("MVTEC_ROOT", "DATASET_ROOT", "ANOMALIES_STR", "REF_IDS_STR"):
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
    assert "samples:  1 per defect" in output
    assert "models:   local_files_only=0" in output
    assert "DRY_RUN=1: configuration validated; generation was not started." in output
```

- [ ] **Step 2: Run the launcher test and verify RED**

```bash
/tmp/rediff-ad-in-context-dev/bin/python -m pytest tests/test_launcher_contract.py::test_bundled_hazelnut_demo_dry_run_uses_all_defects -v
```

Expected: FAIL because `scripts/run_hazelnut_demo.sh` does not exist.

- [ ] **Step 3: Implement the minimal wrapper**

Create executable `scripts/run_hazelnut_demo.sh` with this behavior:

```bash
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd -P)"
DEMO_DATASET_ROOT="$PROJECT_ROOT/demo_assets/mvtec_ad/hazelnut"

required=()
for normal_index in {000..014}; do
    required+=("train/good/${normal_index}.png")
done
required+=(
    "test/crack/000.png" "test/hole/000.png" "test/print/000.png" "test/cut/000.png"
    "ground_truth/crack/000_mask.png" "ground_truth/hole/000_mask.png"
    "ground_truth/print/000_mask.png" "ground_truth/cut/000_mask.png"
)
for relative_path in "${required[@]}"; do
    if [[ ! -s "$DEMO_DATASET_ROOT/$relative_path" ]]; then
        echo "ERROR: missing bundled demo asset: $DEMO_DATASET_ROOT/$relative_path" >&2
        exit 1
    fi
done

export DATASET_ROOT="$DEMO_DATASET_ROOT"
export ANOMALIES_STR="${ANOMALIES_STR:-crack hole print cut}"
export REF_IDS_STR="${REF_IDS_STR:-000 000 000 000}"
export SAMPLES_PER_ANOMALY="${SAMPLES_PER_ANOMALY:-1}"
export SEED="${SEED:-309}"
export RUN_NAME="${RUN_NAME:-hazelnut_in_context_demo_seed${SEED}}"
export OUT_ROOT="${OUT_ROOT:-$PROJECT_ROOT/outputs/$RUN_NAME}"
export LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-0}"

echo "models:   local_files_only=$LOCAL_FILES_ONLY"
exec bash "$PROJECT_ROOT/scripts/run_hazelnut_t2r.sh"
```

Add a `demo` Make target that runs the wrapper and a `demo-dry-run` target that
runs it with `DRY_RUN=1 LOG_TO_FILE=0`. Add both target names to `.PHONY`.

- [ ] **Step 4: Verify GREEN and shell syntax**

```bash
/tmp/rediff-ad-in-context-dev/bin/python -m pytest tests/test_launcher_contract.py -v
bash -n scripts/run_hazelnut_demo.sh
DRY_RUN=1 LOG_TO_FILE=0 bash scripts/run_hazelnut_demo.sh
```

Expected: tests pass, shell syntax exits 0, and the real dry run names the bundled dataset and all four defects without invoking conda or model downloads.

- [ ] **Step 5: Commit the demo launcher**

```bash
git add scripts/run_hazelnut_demo.sh tests/test_launcher_contract.py Makefile
git commit -m "feat: add self-contained hazelnut demo launcher"
```

---

### Task 3: Document the demo and finish repository contracts

**Files:**
- Modify: `README.md`
- Modify: `THIRD_PARTY_NOTICES.md`
- Modify: `docs/ARCHITECTURE.md`
- Modify: `docs/REPRODUCIBILITY.md`
- Modify: `configs/reproducibility.json`
- Modify: `tests/test_reproducibility.py` only if the runtime-critical map changes

**Interfaces:**
- Consumes: the renamed runtime/evaluation entry points and bundled demo launcher.
- Produces: standalone setup instructions, correct third-party notices, and final runtime hashes.

- [ ] **Step 1: Update documentation to the final public interface**

Update the README file tree and all invocation examples to the renamed
`run_in_context.py`, `batch_in_context.py`, and evaluation launchers. Replace
project-owned branding with `In-Context` while retaining the exact upstream
model repository in the model-access list.

Insert a `Self-contained hazelnut demo` section before the full-dataset
instructions with these commands and meanings:

```bash
# Validate paths and effective parameters without loading models.
DRY_RUN=1 LOG_TO_FILE=0 bash scripts/run_hazelnut_demo.sh

# First real run; downloads missing model files after Hugging Face login.
LOCAL_FILES_ONLY=0 bash scripts/run_hazelnut_demo.sh

# Use each of the 15 bundled normal images once per defect.
SAMPLES_PER_ANOMALY=15 bash scripts/run_hazelnut_demo.sh
```

State that no separate dataset download is needed for the demo, while FLUX
Fill, Redux, and the upstream LoRA still require access and local storage.
Retain the full MVTec download instructions for complete experiments.

Update `THIRD_PARTY_NOTICES.md` to say that only the scoped 23-file hazelnut
demo subset is redistributed and point to `demo_assets/mvtec_ad/README.md` and
`LICENSE.md`. Continue stating that the repo does not redistribute model
weights or the remainder of MVTec AD.

Update architecture and reproducibility docs to use renamed source files and
In-Context project terminology.

- [ ] **Step 2: Refresh runtime-critical hashes after final launcher edits**

```bash
sha256sum batch_in_context.py run_in_context.py generation_attention/batch_visualize_flux_attention.py generation_attention/visualize_flux_attention.py generation_attention/mask_refinement.py generation_attention/resume_safety.py generation_attention/target_mask_policy.py scripts/run_attention_direct_top10.sh scripts/run_hazelnut_t2r.sh scripts/validate_smoke_output.py
```

Update only the corresponding values in `configs/reproducibility.json`.

- [ ] **Step 3: Run final content audits**

```bash
rg -n '/home/|/media/' --glob '*.py' --glob '*.sh' --glob '*.md' --glob '*.json' .
rg -n -i 'insert[-_ ]?anything|insertanything' --glob '!docs/superpowers/**' .
git status --short
```

Expected: no personal absolute path appears. Legacy-brand hits are restricted
to the exact upstream Hugging Face source and third-party provenance. Git
status lists only the intended naming, demo, documentation, test, and asset
changes.

- [ ] **Step 4: Run the complete verification gate**

```bash
PATH=/tmp/rediff-ad-in-context-dev/bin:$PATH make check
DRY_RUN=1 LOG_TO_FILE=0 bash scripts/run_hazelnut_demo.sh
cd demo_assets/mvtec_ad
sha256sum -c MANIFEST.sha256
cd ../..
git diff --check
```

Expected: all tests and shell checks pass, the demo dry run exits 0 without a
model load, all 23 hashes verify, and Git reports no whitespace errors.

- [ ] **Step 5: Commit the final documentation and contract updates**

```bash
git add README.md THIRD_PARTY_NOTICES.md VALIDATION.md docs configs tests
git commit -m "docs: document in-context hazelnut demo"
```

- [ ] **Step 6: Review branch contents before push**

```bash
git status --short --branch
git log --oneline --decorate origin/main..HEAD
git diff --stat origin/main...HEAD
git diff --name-status origin/main...HEAD
```

Expected: a clean worktree on `feature/in-context-demo`, a reviewable sequence
of design, naming, asset, launcher, and documentation commits, and no changes
to the original `main` checkout.
