# In-Context Naming Transition Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace project-owned Insert-Anything names with In-Context names without changing generation behavior or hiding required upstream model provenance.

**Architecture:** Rename the two root runtime modules and three evaluation launchers, then update imports, CLI interfaces, environment variables, output-layout labels, audit attributes, tests, and reproducibility hashes as one coordinated interface migration. Preserve the exact upstream Hugging Face model identifier only where runtime lookup or attribution requires it.

**Tech Stack:** Python 3.11+, Bash, pytest, Git, SHA-256 reproducibility manifest.

**Spec:** `docs/superpowers/specs/2026-08-27-in-context-demo-design.md`

## Global Constraints

- Work only in the isolated `feature/in-context-demo` worktree.
- Do not change generation mathematics, model weights, attention policies, thresholds, or evaluation algorithms.
- Human-facing method text uses `In-Context`; files and Python identifiers use `in_context`; shell environment variables use `IN_CONTEXT`; CLI options use `--in-context-*`.
- Do not keep legacy project-owned aliases. Preserve `WensongSong/Insert-Anything` only as the exact upstream model source and in third-party attribution.
- Do not commit tokens, checkpoints, generated outputs, logs, local absolute runtime paths, or virtual environments.
- Use `/tmp/rediff-ad-in-context-dev/bin/python -m pytest` for tests so the generation conda environments remain unchanged.

## Execution Prerequisite

Create an isolated CPU-test environment and install only the checked-in development dependencies:

```bash
python -m venv /tmp/rediff-ad-in-context-dev
/tmp/rediff-ad-in-context-dev/bin/pip install -r requirements-dev.txt
```

Verify the clean baseline before changing runtime files:

```bash
PATH=/tmp/rediff-ad-in-context-dev/bin:$PATH make check
```

The expected result is all existing shell syntax checks and all existing pytest tests passing. If package installation or the baseline fails, stop and report the exact error before continuing.

---

### Task 1: Rename the core runtime entry points

**Files:**
- Create: `tests/test_in_context_interfaces.py`
- Rename: `run_insert_anything.py` to `run_in_context.py`
- Rename: `batch_insert_anything.py` to `batch_in_context.py`
- Modify: `run_in_context.py`
- Modify: `batch_in_context.py`
- Modify: `generation_attention/batch_visualize_flux_attention.py`
- Modify: `generation_attention/visualize_flux_attention.py`
- Modify: `scripts/run_attention_direct_top10.sh`
- Modify: `scripts/run_hazelnut_t2r.sh`
- Modify: `tests/test_source_boundaries.py`

**Interfaces:**
- Consumes: existing `load_pipelines`, image/mask utilities, and generation CLI behavior.
- Produces: importable `run_in_context` and `batch_in_context` modules; `IN_CONTEXT_LORA_PATH`, `IN_CONTEXT_LORA_WEIGHT`, and `_in_context_lora_audit` names; unchanged generation results.

- [ ] **Step 1: Write the failing runtime-interface test**

Create `tests/test_in_context_interfaces.py` with a filesystem and AST contract that names the user-visible break: renamed modules must exist, old root entry points must not exist, all Python sources must parse, and generation modules must import the new root module.

```python
from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


def test_core_runtime_uses_in_context_entrypoints() -> None:
    legacy_run = ROOT / ("run_" + "insert_anything.py")
    legacy_batch = ROOT / ("batch_" + "insert_anything.py")
    assert (ROOT / "run_in_context.py").is_file()
    assert (ROOT / "batch_in_context.py").is_file()
    assert not legacy_run.exists()
    assert not legacy_batch.exists()

    assert "run_in_context" in imported_modules(ROOT / "batch_in_context.py")
    for relative in (
        "generation_attention/batch_visualize_flux_attention.py",
        "generation_attention/visualize_flux_attention.py",
    ):
        assert "run_in_context" in imported_modules(ROOT / relative)
```

- [ ] **Step 2: Run the test and verify RED**

```bash
/tmp/rediff-ad-in-context-dev/bin/python -m pytest tests/test_in_context_interfaces.py::test_core_runtime_uses_in_context_entrypoints -v
```

Expected: FAIL because `run_in_context.py` and `batch_in_context.py` do not exist.

- [ ] **Step 3: Rename files and update runtime-owned identifiers**

Rename the files through Git:

```bash
git mv run_insert_anything.py run_in_context.py
git mv batch_insert_anything.py batch_in_context.py
```

Apply these exact semantic replacements in the listed runtime files and tests:

```text
from run_insert_anything import       -> from run_in_context import
from batch_insert_anything import     -> from batch_in_context import
INSERT_ANYTHING_LORA_PATH             -> IN_CONTEXT_LORA_PATH
INSERT_ANYTHING_LORA_WEIGHT           -> IN_CONTEXT_LORA_WEIGHT
_insert_anything_lora_audit           -> _in_context_lora_audit
Batch Insert-Anything generation      -> Batch In-Context generation
Starting Insert-Anything batch        -> Starting In-Context batch
Finished Insert-Anything batch        -> Finished In-Context batch
one Insert-Anything sample            -> one In-Context sample
Insert-Anything LoRA is not active    -> In-Context LoRA is not active
Insert-Anything LoRA adapters         -> In-Context LoRA adapters
```

Update `tests/test_source_boundaries.py` so the runtime root filenames are
`run_in_context.py` and `batch_in_context.py`, and the forbidden evaluation
import root is `run_in_context`.

Do not alter the literal default model source `WensongSong/Insert-Anything` or
the LoRA weight filename.

- [ ] **Step 4: Run targeted runtime tests and verify GREEN**

```bash
/tmp/rediff-ad-in-context-dev/bin/python -m pytest tests/test_in_context_interfaces.py tests/test_source_boundaries.py tests/test_launcher_contract.py -v
```

Expected: PASS with the launcher's default LoRA source still resolving to the exact upstream repository.

- [ ] **Step 5: Run Python and shell syntax checks**

```bash
/tmp/rediff-ad-in-context-dev/bin/python -m compileall -q run_in_context.py batch_in_context.py generation_attention tests
find scripts eval_diversity eval_downstream -type f -name '*.sh' -print0 | xargs -0 -n1 bash -n
```

Expected: both commands exit 0.

- [ ] **Step 6: Commit the core runtime rename**

```bash
git add run_in_context.py batch_in_context.py generation_attention scripts tests
git commit -m "refactor: rename runtime interfaces to in-context"
```

---

### Task 2: Rename evaluation entry points and layout interfaces

**Files:**
- Modify: `tests/test_in_context_interfaces.py`
- Rename: `eval_diversity/run_insertanything_eval.sh` to `eval_diversity/run_in_context_eval.sh`
- Rename: `eval_downstream/run_insertanything_classification.sh` to `eval_downstream/run_in_context_classification.sh`
- Rename: `eval_downstream/run_insertanything_localization.sh` to `eval_downstream/run_in_context_localization.sh`
- Modify: `eval_downstream/prepare_reflex_classification_data.py`
- Modify: `docs/EVALUATION.md`

**Interfaces:**
- Consumes: the unchanged REDiff-AD generated sample directory schema.
- Produces: `in_context` evaluation layout, `--in-context-results-root`, and three renamed launchers.

- [ ] **Step 1: Extend the interface test for executable evaluation behavior**

Add tests that assert the new shell paths exist, the old paths do not exist,
and the real classification-preparation CLI exposes the renamed option:

```python
import subprocess
import sys


def test_evaluation_entrypoints_use_in_context_names() -> None:
    new_paths = (
        ROOT / "eval_diversity/run_in_context_eval.sh",
        ROOT / "eval_downstream/run_in_context_classification.sh",
        ROOT / "eval_downstream/run_in_context_localization.sh",
    )
    old_paths = (
        ROOT / "eval_diversity" / ("run_" + "insertanything_eval.sh"),
        ROOT / "eval_downstream" / ("run_" + "insertanything_classification.sh"),
        ROOT / "eval_downstream" / ("run_" + "insertanything_localization.sh"),
    )
    assert all(path.is_file() for path in new_paths)
    assert not any(path.exists() for path in old_paths)

    completed = subprocess.run(
        [sys.executable, "eval_downstream/prepare_reflex_classification_data.py", "--help"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0
    assert "--in-context-results-root" in completed.stdout
    assert "in_context" in completed.stdout
```

- [ ] **Step 2: Run the new evaluation test and verify RED**

```bash
/tmp/rediff-ad-in-context-dev/bin/python -m pytest tests/test_in_context_interfaces.py::test_evaluation_entrypoints_use_in_context_names -v
```

Expected: FAIL because the new launchers and CLI option do not exist.

- [ ] **Step 3: Rename launchers and evaluation symbols**

```bash
git mv eval_diversity/run_insertanything_eval.sh eval_diversity/run_in_context_eval.sh
git mv eval_downstream/run_insertanything_classification.sh eval_downstream/run_in_context_classification.sh
git mv eval_downstream/run_insertanything_localization.sh eval_downstream/run_in_context_localization.sh
```

Apply these exact interface replacements across the renamed shell files,
`prepare_reflex_classification_data.py`, and `docs/EVALUATION.md`:

```text
GENERATED_LAYOUT default insert-anything -> in_context
insert_anything_sample_dirs              -> in_context_sample_dirs
args.insert_anything_results_root        -> args.in_context_results_root
--insert-anything-results-root           -> --in-context-results-root
Insert-Anything output/log text          -> In-Context output/log text
old launcher filenames                   -> new launcher filenames
```

Keep other supported layout labels such as `anomaly-diffusion`, `seas`, and
`dualanodiff` unchanged. All non-Reflex layouts continue to consume the common
renamed results-root argument.

- [ ] **Step 4: Run targeted evaluation tests and verify GREEN**

```bash
/tmp/rediff-ad-in-context-dev/bin/python -m pytest tests/test_in_context_interfaces.py tests/test_source_boundaries.py -v
bash -n eval_diversity/run_in_context_eval.sh
bash -n eval_downstream/run_in_context_classification.sh
bash -n eval_downstream/run_in_context_localization.sh
```

Expected: all commands exit 0.

- [ ] **Step 5: Commit the evaluation rename**

```bash
git add eval_diversity eval_downstream docs/EVALUATION.md tests/test_in_context_interfaces.py
git commit -m "refactor: rename evaluation layout to in-context"
```

---

### Task 3: Update validation and reproducibility contracts

**Files:**
- Modify: `VALIDATION.md`
- Modify: `configs/reproducibility.json`
- Modify: `scripts/validate_smoke_output.py`
- Modify: `tests/test_launcher_contract.py`
- Modify: `tests/test_validator.py`
- Modify: `tests/test_reproducibility.py` only if an assertion names a renamed path

**Interfaces:**
- Consumes: renamed runtime and evaluation files from Tasks 1 and 2.
- Produces: consistent validation messages and exact SHA-256 hashes for all runtime-critical files.

- [ ] **Step 1: Update tests to expect In-Context user-facing audit text**

Retain the exact upstream path assertion but rename any project label around it:

```python
assert "LoRA:        path=WensongSong/Insert-Anything" in output
```

Keep validator JSON fields `lora_path`, `lora_weight_name`, and
`lora_runtime_audit` unchanged because they describe model provenance and a
generic audit record rather than project branding.

- [ ] **Step 2: Run validation and reproducibility tests and verify RED**

```bash
/tmp/rediff-ad-in-context-dev/bin/python -m pytest tests/test_launcher_contract.py tests/test_validator.py tests/test_reproducibility.py -v
```

Expected: the reproducibility test fails because renamed files and modified runtime files no longer match the recorded hash map.

- [ ] **Step 3: Update validation prose and reproducibility keys**

Change project-owned validation prose to `In-Context LoRA`. In
`configs/reproducibility.json`, rename the two root hash keys to
`batch_in_context.py` and `run_in_context.py`, change the eval-source role to
`in-context evaluation fixes`, and retain the exact `lora_path` value.

Compute hashes from the worktree:

```bash
sha256sum batch_in_context.py run_in_context.py generation_attention/batch_visualize_flux_attention.py generation_attention/visualize_flux_attention.py generation_attention/mask_refinement.py generation_attention/resume_safety.py generation_attention/target_mask_policy.py scripts/run_attention_direct_top10.sh scripts/run_hazelnut_t2r.sh scripts/validate_smoke_output.py
```

Replace every corresponding value in `configs/reproducibility.json` with the
printed digest. Do not add documentation, tests, or evaluation launchers to
the runtime-critical hash map.

- [ ] **Step 4: Run the full CPU contract suite and verify GREEN**

```bash
PATH=/tmp/rediff-ad-in-context-dev/bin:$PATH make check
```

Expected: all shell syntax checks and all pytest tests pass.

- [ ] **Step 5: Audit remaining legacy-brand text**

```bash
rg -n -i 'insert[-_ ]?anything|insertanything' run_in_context.py batch_in_context.py generation_attention scripts eval_diversity eval_downstream tests configs VALIDATION.md
```

Expected: remaining hits in this runtime-focused scan are limited to exact
`WensongSong/Insert-Anything` model-source values; no filename, import, CLI,
environment variable, layout value, or project log uses the legacy brand.

- [ ] **Step 6: Commit the validated naming transition**

```bash
git add VALIDATION.md configs/reproducibility.json scripts tests
git commit -m "test: align in-context validation contracts"
```
