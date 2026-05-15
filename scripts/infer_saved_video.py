from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import ensure_src_path

ensure_src_path()

from traffic_incident_cv.inference.saved_run_inference import (
    assess_video_cascade,
    discover_default_run_dir,
    write_assessment_bundle,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run saved accident/severity models on an arbitrary video.")
    parser.add_argument("--video", required=True)
    parser.add_argument("--accident-run-dir")
    parser.add_argument("--severity-run-dir")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--accident-threshold", type=float, default=0.5)
    parser.add_argument("--window-span-frames", type=int, default=None)
    parser.add_argument("--step-frames", type=int, default=None)
    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument("--output-dir")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    accident_run_dir = Path(args.accident_run_dir).resolve() if args.accident_run_dir else discover_default_run_dir(project_root, "accident")
    if accident_run_dir is None:
        raise FileNotFoundError("No usable accident run directory was found. Provide --accident-run-dir explicitly.")
    severity_run_dir = (
        Path(args.severity_run_dir).resolve()
        if args.severity_run_dir
        else discover_default_run_dir(project_root, "severity")
    )
    payload = assess_video_cascade(
        video_path=args.video,
        accident_run_dir=accident_run_dir,
        severity_run_dir=severity_run_dir,
        device_name=args.device,
        accident_threshold=args.accident_threshold,
        window_span_frames=args.window_span_frames,
        step_frames=args.step_frames,
        max_windows=args.max_windows,
    )
    bundle_paths = {}
    if args.output_dir:
        bundle_paths = write_assessment_bundle(
            output_dir=args.output_dir,
            payload=payload,
            uploaded_video_path=args.video,
        )
    print(
        json.dumps(
            {
                "accident_run_dir": str(accident_run_dir),
                "severity_run_dir": str(severity_run_dir) if severity_run_dir else None,
                "bundle_paths": bundle_paths,
                "result": payload,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
