# Model release artifacts

This directory contains lightweight model-release metadata, metrics, reports, and figures for the final and comparison runs.

Included runs:

- `hybrid/` — H01 accident hybrid full-frame + local-crop line.
- `accident_convnext_d02/` — D02 conservative ConvNeXt accident line.
- `severity/` — S13 severity R3D line.
- `accident_a03r_no_harmonization/` — comparison accident baseline.
- `severity_b01_baseline/` — comparison severity baseline.

The actual PyTorch checkpoint files (`model_best.pt`, `model_last.pt`) are intentionally not included in the repository. The `checkpoint_file` fields in `summary.json` and `report.md` document the expected checkpoint filename from the original run, not an included binary artifact. Store large weights externally or via Git LFS/release assets if deployment requires them.
