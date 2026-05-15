from __future__ import annotations

import argparse
import json

from _bootstrap import ensure_src_path

ensure_src_path()

from traffic_incident_cv.config import read_yaml
from traffic_incident_cv.data.manifest import save_manifest
from traffic_incident_cv.data.manifest_builders import build_artifacts_from_config, summarize_artifacts
from traffic_incident_cv.data.split_policy import assert_no_camera_leakage


def main() -> None:
    parser = argparse.ArgumentParser(description="Build JSONL manifests from dataset structure.")
    parser.add_argument("--config", required=True, help="Path to dataset YAML config")
    parser.add_argument("--dry-run", action="store_true", help="Do not write files; only print summary")
    args = parser.parse_args()

    cfg = read_yaml(args.config)
    artifacts = build_artifacts_from_config(cfg)
    all_records = next((artifact.records for artifact in artifacts if artifact.name == "all"), [])
    if all_records:
        assert_no_camera_leakage(all_records)
    payload = {
        "dataset": cfg["dataset"]["name"],
        "builder": cfg["dataset"].get("builder", cfg["dataset"]["name"]),
        "artifacts": summarize_artifacts(artifacts),
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if args.dry_run:
        return
    for artifact in artifacts:
        if artifact.output_path:
            save_manifest(artifact.output_path, artifact.records)


if __name__ == "__main__":
    main()
