# Third-party assets and provenance

This double-blind snapshot contains a cleaned research implementation derived from an upstream generation baseline, a verified T2R patch, and selected evaluation code. Identifying repository URLs and commit IDs are withheld during review and will be restored with full attribution in the archival release. Non-identifying code hashes and the experimental contract remain recorded in `configs/reproducibility.json`.

The repository redistributes a 23-file hazelnut demo subset from MVTec AD:

- 15 defect-free source images;
- the `000` image and pixel-precise mask for each of `crack`, `hole`, `print`,
  and `cut`.

These unmodified files are located under `demo_assets/mvtec_ad/hazelnut/` and
are distributed under CC BY-NC-SA 4.0. Their source, attribution, checksums,
and asset-specific license are recorded in `demo_assets/mvtec_ad/README.md`,
`demo_assets/mvtec_ad/LICENSE.md`, and `demo_assets/mvtec_ad/MANIFEST.sha256`.
That license applies to the bundled dataset assets, not to other repository
contents.

The repository does not redistribute:

- the remainder of the MVTec AD dataset;
- FLUX.1-Fill-dev or FLUX.1-Redux-dev weights;
- Insert-Anything LoRA weights;
- downloaded metric backbones or checkpoints.

Those assets retain their own licenses and access terms. Python dependencies also retain their respective licenses. This review snapshot is not a project-level license grant and is provided only for peer-review inspection. Full attribution and applicable redistribution terms must be restored before an archival public release.
