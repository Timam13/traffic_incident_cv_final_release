# Demo data

This folder contains a small curated demonstration set for testing the UI and the inference pipeline.

It is not a training dataset and must not be treated as an official benchmark split.

## Structure

- `videos/positive/` — accident demo clips.
- `videos/negative/` — no-accident demo clips.
- `assessments/` — precomputed cascade outputs for the demo clips.
- `ui_examples.json` — metadata used by the demo UI to list example cases.

## Notes

The assessment files are precomputed demo artifacts. They are included only to make the UI demonstration reproducible without rerunning the full pipeline.

The video paths inside the assessment files point to `demo_data/videos/...` so that the demo folder remains portable inside the repository.
