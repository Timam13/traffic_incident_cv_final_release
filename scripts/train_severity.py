from __future__ import annotations

import argparse
import json

from _bootstrap import ensure_src_path

ensure_src_path()

from traffic_incident_cv.config import load_config, to_experiment_config
from traffic_incident_cv.pipelines.severity import SeverityClassificationPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Train severity classification model.")
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
            "Filter accident-only severity dataset",
            "Instantiate multi-class model",
            "Run trainer.fit()",
            "Generate macro-F1 / confusion matrix report",
        ],
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if args.dry_run:
        return
    pipeline = SeverityClassificationPipeline(config=config)
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
