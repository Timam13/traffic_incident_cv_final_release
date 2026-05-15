from __future__ import annotations

import argparse
import json

from _bootstrap import ensure_src_path

ensure_src_path()

from traffic_incident_cv.config import load_config, to_experiment_config
from traffic_incident_cv.pipelines.accident import AccidentDetectionPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Train accident detection model.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    exp = to_experiment_config(config)
    payload = {
        "experiment_id": exp.experiment_id,
        "task": exp.task,
        "model_family": exp.model.get("family"),
        "manifest_paths": exp.manifest_paths,
        "next_steps": [
            "Instantiate dataset and dataloaders",
            "Instantiate model family for accident branch",
            "Run trainer.fit()",
            "Save metrics/report/checkpoint bundle",
        ],
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if args.dry_run:
        return
    pipeline = AccidentDetectionPipeline(config=config)
    pipeline.fit()
    metrics = pipeline.validate()
    print(
        json.dumps(
            {
                "experiment_id": exp.experiment_id,
                "output_dir": str(pipeline.latest_output_dir),
                "metrics": metrics.metrics,
                "metadata": metrics.metadata,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
