# Architecture

The repository intentionally keeps the original research-module boundaries so results are not changed by a large refactor.

```text
scripts/run_hazelnut_t2r.sh
        |
        v
scripts/run_attention_direct_top10.sh
        |
        v
generation_attention/batch_visualize_flux_attention.py
   |              |                 |
   |              |                 +--> mask_refinement.py
   |              |                      Q80 -> contour mask
   |              +--> visualize_flux_attention.py
   |                   T2R recorder, Adaptive, Shape-K, all blocks
   +--> batch_insert_anything.py -> run_insert_anything.py
                                      FLUX Fill, Redux, INT4, LoRA
```

## Attention streams

The two attention consumers remain separate, but now use the same T2R stream and frozen Top-10:

| Consumer | Kind | Block source | Purpose |
|---|---|---|---|
| Direct selected mask | `target_to_ref_image` | frozen T2R Top-10 | Main coarse localization |
| Direct all-block mask | `target_to_ref_image` | all 57 FLUX blocks | Diagnostic/ablation mask |
| Adaptive monitor | `target_to_ref_image` | frozen T2R Top-10 | Control reference-token scaling from target-to-reference attention |

Adaptive reads attention during diffusion and can change reference-token scale, so switching it from the historical T2T monitor to T2R changes generation itself. Direct and Adaptive share the block list but keep separate aggregation state and consumers.

## Diversity and refinement

- Adaptive reference injection is enabled and records checks even when no boost is triggered.
- Shape-K uses the original `middle` scope: 36 blocks, steps 12–20 for a 30-step run, 324 expected calls per pass.
- The primary coarse mask is refined by the unchanged Q80 appearance stage and contour-fill stage.
- `contour_refined_mask.png` is the recommended downstream mask.

## INT4 mode

The default is the full diffusers FLUX transformer quantized with bitsandbytes NF4. It is not Nunchaku. The batch attention recorder needs per-block processors, so `batch_visualize_flux_attention.py` explicitly disables the fused Nunchaku path. Insert-Anything LoRA is loaded after the quantized transformer is assembled.

After loading, available and active PEFT adapter names are aligned and audited. Generation stops before sampling if no registered LoRA adapter is active. The audit is persisted in the run config and sample metadata.

## Resume boundary

`generation_attention/resume_safety.py` is a dependency-free guard used before run-config/log writes and before model loading. A resume may skip a sample only when all canonical artifacts exist and its seed, source, reference, attention kind, aggregate steps and frozen T2R Top-10 match the current request. A non-empty mismatch is rejected unless regeneration was explicitly requested with `OVERWRITE=1`.

## Launcher boundary

`run_hazelnut_t2r.sh` retains the experimental filename but is parameterized by
`OBJECT_NAME`, dataset root, anomaly names, explicit reference IDs, and sample
counts. It is the only public generation entry point; scheduling a collection
of objects does not require a second generation implementation.
