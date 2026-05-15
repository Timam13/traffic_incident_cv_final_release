from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import ensure_src_path

ensure_src_path()

from traffic_incident_cv.config import read_yaml
from traffic_incident_cv.data.audit import audit_dataset_from_config
from traffic_incident_cv.data.manifest import save_manifest
from traffic_incident_cv.data.manifest_builders import build_artifacts_from_config, summarize_artifacts
from traffic_incident_cv.utils.io import ensure_dir, write_json


DEFAULT_CONFIGS = [
    "configs/datasets/ai_city.yaml",
    "configs/datasets/balanced_accident.yaml",
    "configs/datasets/accident_binary_clip_bundle_bridge_v1_8fps_160.yaml"
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit datasets and prepare manifests for the accident vertical slice.")
    parser.add_argument("--configs", nargs="+", default=DEFAULT_CONFIGS, help="Dataset config paths to audit/build in order")
    parser.add_argument("--dry-run", action="store_true", help="Only audit and print readiness report")
    parser.add_argument(
        "--report-path",
        default="outputs/data_preflight/readiness_report.json",
        help="Where to save the readiness report when not running dry-run",
    )
    args = parser.parse_args()

    report: dict[str, object] = {
        "ready": True,
        "configs": args.configs,
        "datasets": [],
        "artifacts": {},
        "next_steps": [],
    }
    datasets_payload: list[dict[str, object]] = []
    artifacts_payload: dict[str, dict[str, dict[str, int]]] = {}
    manifest_cache: dict[str, list[object]] = {}

    for config_path in args.configs:
        cfg = read_yaml(config_path)
        audit = audit_dataset_from_config(cfg, manifest_cache=manifest_cache)
        datasets_payload.append(audit.to_dict())
        if not audit.ready:
            report["ready"] = False
            continue
        artifacts = build_artifacts_from_config(cfg, manifest_cache=manifest_cache)  # type: ignore[arg-type]
        artifacts_payload[cfg["dataset"]["name"]] = summarize_artifacts(artifacts)
        for artifact in artifacts:
            if artifact.output_path:
                manifest_cache[artifact.output_path] = artifact.records
        if not args.dry_run:
            for artifact in artifacts:
                if artifact.output_path:
                    save_manifest(artifact.output_path, artifact.records)

    report["datasets"] = datasets_payload
    report["artifacts"] = artifacts_payload
    report["next_steps"] = (
        [
            "Run python scripts/train_accident.py --config configs/experiments/accident_convnext.yaml",
            "Run python scripts/train_severity.py --config configs/experiments/severity_convnext.yaml",
        ]
        if report["ready"]
        else [
            "Fix the dataset errors listed in this report",
            "Rerun python scripts/prepare_data.py",
        ]
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))

    if args.dry_run:
        if not report["ready"]:
            raise SystemExit(1)
        return

    report_path = Path(args.report_path)
    ensure_dir(report_path.parent)
    write_json(report_path, report)
    if not report["ready"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
