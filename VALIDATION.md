# Validation status

The public REDiff-AD contract is covered by CPU-only CI and a GPU integration
smoke performed before release.

- Shell syntax, Python syntax, frozen T2R configuration and source hashes: PASS.
- Dry-run wiring: Direct T2R, Adaptive T2R, INT4, LoRA, Shape-K, Q80 and contour
  refinement: PASS.
- GPU smoke: active Insert-Anything LoRA, 57 attention blocks, 36 × 9 Shape-K
  calls, localized edit, coarse mask and both refinement stages: PASS.
- Full benchmark reproduction: run by the reviewer with the commands in
  `README.md`; models, datasets and generated outputs are not distributed.

The validator rejects missing artifacts, inactive LoRA, incompatible attention
settings and structurally complete no-op edits.
