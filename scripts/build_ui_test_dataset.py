from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

from _bootstrap import ensure_src_path

ensure_src_path()

from traffic_incident_cv.inference.saved_run_inference import (  # noqa: E402
    assess_video_cascade,
    discover_default_run_dir,
    write_assessment_bundle,
)
from traffic_incident_cv.utils.io import ensure_dir  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[1]
HARMONIZED_ROOT = PROJECT_ROOT / "outputs" / "harmonized_videos" / "bridge_v1_8fps_160"
OUTPUT_ROOT = PROJECT_ROOT / "demo_data"


CASES = [
    {
        "case_id": "accident_major_defense_reference",
        "title": "Accident: major severity (defense reference)",
        "kind": "positive",
        "source": HARMONIZED_ROOT / "balanced_accident" / "test" / "major" / "2021_02_030_cuted.mp4",
        "target_rel": Path("videos/positive/accident_major_defense_reference.mp4"),
        "expected_accident_label": "accident",
        "expected_severity_label": "major",
        "notes": "Curated major-severity clip with stable accident detection and dominant major votes in the current H01 + S13 cascade.",
        "trim": None,
    },
    {
        "case_id": "accident_moderate_defense_reference_1",
        "title": "Accident: moderate severity (reference 1)",
        "kind": "positive",
        "source": HARMONIZED_ROOT / "balanced_accident" / "test" / "moderate" / "2019_06_12_cuted.mp4",
        "target_rel": Path("videos/positive/accident_moderate_defense_reference_1.mp4"),
        "expected_accident_label": "accident",
        "expected_severity_label": "moderate",
        "notes": "Strong moderate case with confident accident detection and repeated moderate votes.",
        "trim": None,
    },
    {
        "case_id": "accident_moderate_defense_reference_2",
        "title": "Accident: moderate severity (reference 2)",
        "kind": "positive",
        "source": HARMONIZED_ROOT / "balanced_accident" / "test" / "moderate" / "2021_01_10_cuted.mp4",
        "target_rel": Path("videos/positive/accident_moderate_defense_reference_2.mp4"),
        "expected_accident_label": "accident",
        "expected_severity_label": "moderate",
        "notes": "Compact moderate example that keeps the UI timeline short while preserving the expected severity label.",
        "trim": None,
    },
    {
        "case_id": "accident_moderate_defense_reference_3",
        "title": "Accident: moderate severity (reference 3)",
        "kind": "positive",
        "source": HARMONIZED_ROOT / "balanced_accident" / "test" / "moderate" / "2022_03_02_cuted.mp4",
        "target_rel": Path("videos/positive/accident_moderate_defense_reference_3.mp4"),
        "expected_accident_label": "accident",
        "expected_severity_label": "moderate",
        "notes": "Short moderate clip selected specifically because the current cascade keeps both accident and severity predictions stable.",
        "trim": None,
    },
    {
        "case_id": "accident_minor_defense_reference_1",
        "title": "Accident: minor severity (reference 1)",
        "kind": "positive",
        "source": HARMONIZED_ROOT / "balanced_accident" / "test" / "minor" / "2019_07_16_cuted.mp4",
        "target_rel": Path("videos/positive/accident_minor_defense_reference_1.mp4"),
        "expected_accident_label": "accident",
        "expected_severity_label": "minor",
        "notes": "Minor-severity clip with several positive windows, useful for showing the score timeline and consistent minor votes.",
        "trim": None,
    },
    {
        "case_id": "accident_minor_defense_reference_2",
        "title": "Accident: minor severity (reference 2)",
        "kind": "positive",
        "source": HARMONIZED_ROOT / "balanced_accident" / "test" / "minor" / "2021_02_150_cuted.mp4",
        "target_rel": Path("videos/positive/accident_minor_defense_reference_2.mp4"),
        "expected_accident_label": "accident",
        "expected_severity_label": "minor",
        "notes": "Second minor case with a longer context span and multiple minor votes for the severity branch.",
        "trim": None,
    },
    {
        "case_id": "no_accident_reference_cam14",
        "title": "No accident: AI City cam_14",
        "kind": "negative",
        "source": HARMONIZED_ROOT / "ai_city" / "test" / "cam_14.mp4",
        "target_rel": Path("videos/negative/no_accident_reference_cam14.mp4"),
        "expected_accident_label": "no_accident",
        "expected_severity_label": None,
        "notes": "Regular negative example after harmonization, trimmed for a quick UI run.",
        "trim": {"start_sec": 0.0, "duration_sec": 12.0},
    },
    {
        "case_id": "no_accident_hard_cam15",
        "title": "Hard negative: AI City cam_15",
        "kind": "hard_negative",
        "source": HARMONIZED_ROOT / "ai_city" / "test" / "cam_15.mp4",
        "target_rel": Path("videos/negative/no_accident_hard_cam15.mp4"),
        "expected_accident_label": "no_accident",
        "expected_severity_label": None,
        "notes": "Historically difficult negative camera. Included to show that the current demo configuration avoids false alarms here.",
        "trim": {"start_sec": 0.0, "duration_sec": 12.0},
    },
    {
        "case_id": "no_accident_hard_cam8",
        "title": "Hard negative: AI City cam_8",
        "kind": "hard_negative",
        "source": HARMONIZED_ROOT / "ai_city" / "test" / "cam_8.mp4",
        "target_rel": Path("videos/negative/no_accident_hard_cam8.mp4"),
        "expected_accident_label": "no_accident",
        "expected_severity_label": None,
        "notes": "Historically difficult negative camera. Selected because H01 stays below threshold on the trimmed defense clip.",
        "trim": {"start_sec": 0.0, "duration_sec": 12.0},
    },
    {
        "case_id": "no_accident_hard_cam13",
        "title": "Hard negative: AI City cam_13",
        "kind": "hard_negative",
        "source": HARMONIZED_ROOT / "ai_city" / "test" / "cam_13.mp4",
        "target_rel": Path("videos/negative/no_accident_hard_cam13.mp4"),
        "expected_accident_label": "no_accident",
        "expected_severity_label": None,
        "notes": "Another hard negative used to demonstrate low accident scores and the absence of severity output.",
        "trim": {"start_sec": 0.0, "duration_sec": 12.0},
    },
]


def _as_project_relative(path: str | Path | None) -> str | None:
    if path is None:
        return None

    path_obj = Path(path)

    try:
        rel = path_obj.resolve().relative_to(PROJECT_ROOT)
        return rel.as_posix()
    except (OSError, ValueError):
        return str(path).replace("\\", "/")


def _relativize_payload_paths(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _relativize_payload_paths(item) for key, item in value.items()}

    if isinstance(value, list):
        return [_relativize_payload_paths(item) for item in value]

    if isinstance(value, str):
        normalized = value.replace("\\", "/")
        project_root = PROJECT_ROOT.as_posix()

        if normalized.startswith(project_root + "/"):
            return normalized.removeprefix(project_root + "/")

        return normalized

    return value


def _require_ffmpeg() -> str:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise FileNotFoundError("`ffmpeg` was not found in PATH, but it is required to cut the UI test videos.")
    return ffmpeg


def _prepare_video(src: Path, dst: Path, trim: dict[str, float] | None, ffmpeg: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)

    if not src.exists():
        if dst.exists():
            return
        raise FileNotFoundError(
            f"Source video was not found: {src}. "
            f"The target video also does not exist yet: {dst}. "
            "Place harmonized videos under outputs/harmonized_videos/bridge_v1_8fps_160 "
            "or keep the existing demo_data/videos files."
        )

    if trim is None:
        shutil.copy2(src, dst)
        return

    start_sec = float(trim.get("start_sec", 0.0))
    duration_sec = float(trim.get("duration_sec", 12.0))
    cmd = [
        ffmpeg,
        "-y",
        "-ss",
        f"{start_sec:.3f}",
        "-t",
        f"{duration_sec:.3f}",
        "-i",
        str(src),
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "24",
        "-pix_fmt",
        "yuv420p",
        str(dst),
    ]
    completed = subprocess.run(cmd, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(
            f"`ffmpeg` could not prepare {dst.name}: {completed.stderr[-2000:]}"
        )


def _write_readme(output_root: Path, examples_payload: dict[str, object]) -> None:
    rows = []
    for example in examples_payload["examples"]:
        rows.append(
            f"| `{example['case_id']}` | {example['title']} | {example['kind']} | "
            f"{example['expected_accident_label']} | {example.get('expected_severity_label') or '—'} | {example['notes']} |"
        )

    content = "\n".join(
        [
            "# UI test dataset",
            "",
            "This directory contains a compact set of preprocessed videos in the `bridge_v1_8fps_160` scheme,",
            "prepared for reproducible demo UI testing.",
            "",
            "## Structure",
            "",
            "- `videos/`: short videos that are fast to analyze;",
            "- `assessments/`: reference bundle results from the current local models;",
            "- `ui_examples.json`: manifest automatically picked up by the demo UI.",
            "",
            "## Case set",
            "",
            "| case_id | Title | Type | Expected accident | Expected severity | Purpose |",
            "| --- | --- | --- | --- | --- | --- |",
            *rows,
            "",
            "## Usage",
            "",
            "1. Run `python scripts/launch_demo_ui.py`.",
            "2. Buttons from `ui_examples.json` will appear in the quick examples section.",
            "3. For each example you can compare the live result with the reference bundle in `assessments/`.",
            "",
        ]
    )

    (output_root / "README.md").write_text(content, encoding="utf-8")


def main() -> int:
    ffmpeg = _require_ffmpeg()
    output_root = ensure_dir(OUTPUT_ROOT)
    ensure_dir(output_root / "videos")
    assessments_root = ensure_dir(output_root / "assessments")

    accident_run_dir = discover_default_run_dir(PROJECT_ROOT, "accident")
    severity_run_dir = discover_default_run_dir(PROJECT_ROOT, "severity")

    if accident_run_dir is None:
        raise RuntimeError("No local accident checkpoint was found for building the UI test dataset.")

    examples = []

    for case in CASES:
        dst_video = output_root / case["target_rel"]
        _prepare_video(Path(case["source"]), dst_video, case["trim"], ffmpeg)

        payload = assess_video_cascade(
            video_path=dst_video,
            accident_run_dir=accident_run_dir,
            severity_run_dir=severity_run_dir,
            device_name="cuda",
            accident_threshold=0.5,
            max_windows=12,
        )
        payload = _relativize_payload_paths(payload)

        bundle_dir = assessments_root / str(case["case_id"])
        if bundle_dir.exists():
            shutil.rmtree(bundle_dir)

        bundle_paths = write_assessment_bundle(
            output_dir=bundle_dir,
            payload=payload,
            uploaded_video_path=None,
        )

        assessment_json = Path(bundle_paths["assessment_json"])
        report_html = Path(bundle_paths["report_html"])

        examples.append(
            {
                "case_id": case["case_id"],
                "title": case["title"],
                "kind": case["kind"],
                "video_path": _as_project_relative(dst_video),
                "expected_accident_label": case["expected_accident_label"],
                "expected_severity_label": case["expected_severity_label"],
                "notes": case["notes"],
                "assessment_dir": _as_project_relative(bundle_dir),
                "assessment_json": _as_project_relative(assessment_json),
                "report_html": _as_project_relative(report_html),
            }
        )

    examples_payload = {
        "dataset_id": "ui_test_dataset_bridge_v1_8fps_160",
        "prepared_scheme": "bridge_v1_8fps_160",
        "accident_run_dir": _as_project_relative(accident_run_dir),
        "severity_run_dir": _as_project_relative(severity_run_dir) if severity_run_dir else None,
        "examples": examples,
    }

    (output_root / "ui_examples.json").write_text(
        json.dumps(examples_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_readme(output_root, examples_payload)
    print(json.dumps(examples_payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())