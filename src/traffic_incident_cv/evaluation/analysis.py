from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from traffic_incident_cv.data.manifest import load_manifest
from traffic_incident_cv.domain.schemas import ClipRecord
from traffic_incident_cv.utils.io import ensure_dir, write_json

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def discover_resolved_manifests(run_dir: str | Path) -> dict[str, Path]:
    run_path = Path(run_dir)
    manifests: dict[str, Path] = {}
    for split in ("train", "val", "test"):
        candidate = run_path / f"resolved_{split}.jsonl"
        if candidate.exists():
            manifests[split] = candidate
    return manifests


def _duration_bucket(duration_sec: float | None) -> str:
    if duration_sec is None:
        return "unknown"
    if duration_sec < 30:
        return "short"
    if duration_sec <= 300:
        return "standard"
    return "long"


def _fps_bucket(fps: float | None) -> str:
    if fps is None:
        return "unknown"
    if fps < 10:
        return "low_fps"
    if fps <= 15:
        return "standard_fps"
    return "high_fps"


def _label_for_task(record: ClipRecord) -> str:
    task = str(record.task)
    if task == "severity":
        return record.severity_label or "unknown"
    return str(record.binary_label or "unknown")


def summarize_records(records: list[ClipRecord]) -> dict[str, Any]:
    dataset_counter = Counter(record.dataset for record in records)
    label_counter = Counter(_label_for_task(record) for record in records)
    severity_counter = Counter(record.severity_label or "unknown" for record in records if record.severity_label is not None)
    tag_counter = Counter(tag for record in records for tag in record.tags)
    duration_counter = Counter(_duration_bucket(record.duration_sec) for record in records)
    fps_counter = Counter(_fps_bucket(record.fps) for record in records)
    camera_counter = Counter(record.camera_id for record in records if record.camera_id)
    augmented_count = sum(1 for record in records if any("aug" in tag for tag in record.tags))
    real_count = sum(1 for record in records if "real_video" in record.tags)
    return {
        "count": len(records),
        "datasets": dict(sorted(dataset_counter.items())),
        "labels": dict(sorted(label_counter.items())),
        "severity_labels": dict(sorted(severity_counter.items())),
        "tags": dict(tag_counter.most_common()),
        "duration_buckets": dict(sorted(duration_counter.items())),
        "fps_buckets": dict(sorted(fps_counter.items())),
        "camera_groups": len(camera_counter),
        "augmented_count": augmented_count,
        "real_video_count": real_count,
    }


def _dominant_share(counts: dict[str, int]) -> float:
    if not counts:
        return 0.0
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    return max(counts.values()) / total


def _minority_share(counts: dict[str, int]) -> float:
    non_zero = [value for value in counts.values() if value > 0]
    if not non_zero:
        return 0.0
    total = sum(non_zero)
    return min(non_zero) / total if total > 0 else 0.0


def derive_risks(task: str, metrics: dict[str, float], split_summaries: dict[str, dict[str, Any]]) -> list[dict[str, str]]:
    risks: list[dict[str, str]] = []
    val_summary = split_summaries.get("val", {})
    val_labels = val_summary.get("labels", {})
    val_datasets = val_summary.get("datasets", {})

    minority_share = _minority_share(val_labels)
    if minority_share > 0 and minority_share <= 0.1:
        risks.append(
            {
                "severity": "high",
                "code": "val_class_imbalance",
                "message": "Validation split has extreme class imbalance; headline metrics can look stronger than real deployment quality.",
            }
        )
    if task == "accident" and val_labels.get("no_accident", 0) < 10:
        risks.append(
            {
                "severity": "high",
                "code": "tiny_negative_pool",
                "message": "Validation negative pool is too small for reliable false-alarm estimation.",
            }
        )
    if _dominant_share(val_datasets) > 0.9:
        risks.append(
            {
                "severity": "medium",
                "code": "source_dominance",
                "message": "One source dominates the validation split; source bias needs separate checking.",
            }
        )

    balanced_accuracy = metrics.get("val_balanced_accuracy")
    f1_value = metrics.get("val_f1")
    macro_f1 = metrics.get("val_macro_f1")
    if task == "accident" and balanced_accuracy is not None and f1_value is not None and f1_value - balanced_accuracy > 0.3:
        risks.append(
            {
                "severity": "high",
                "code": "metric_gap",
                "message": "F1 is much higher than balanced accuracy; the baseline likely benefits from label skew.",
            }
        )
    if task == "severity" and macro_f1 is not None and macro_f1 < 0.4:
        risks.append(
            {
                "severity": "medium",
                "code": "weak_multiclass_baseline",
                "message": "Current severity baseline is still weak and needs reweighting or stronger temporal modeling.",
            }
        )
    return risks


def recommend_next_steps(task: str, risks: list[dict[str, str]]) -> list[str]:
    risk_codes = {risk["code"] for risk in risks}

    if task == "accident":
        steps = [
            "Use D02 as the conservative zero-false-positive accident line and H01 as the stronger aggregate hybrid line.",
            "Report the D02-vs-H01 trade-off explicitly: D02 controls false positives better, while H01 improves recall and F1.",
            "Run long-stream false-alarm analysis on normal traffic videos before treating the model as deployment-ready.",
            "Keep the detector-driven YOLO branch as future work unless detector-guided crops are fully validated.",
        ]
        if "tiny_negative_pool" in risk_codes:
            steps.insert(0, "Do not rely on headline accuracy until the negative pool is large enough for false-alarm analysis.")
        return steps

    if task == "severity":
        steps = [
            "Use S13 as the primary severity line and B01 only as a historical baseline.",
            "Report macro-F1, balanced accuracy, and the confusion matrix because severity classes are visually close.",
            "Inspect minor-vs-moderate and moderate-vs-major confusions separately.",
            "Avoid selecting a severity run only by validation performance; held-out test behavior should dominate.",
        ]
        if "weak_multiclass_baseline" in risk_codes:
            steps.insert(0, "Treat the ConvNeXt severity baseline as a reference point rather than the final model.")
        return steps

    return ["Treat this branch as a future extension and keep it outside the defended empirical comparison."]


def _plot_counter(counter: dict[str, int], title: str, output_path: Path) -> Path | None:
    if plt is None or not counter:
        return None
    labels = list(counter.keys())
    values = list(counter.values())
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(labels, values, color="#1f4e79")
    ax.set_title(title)
    ax.set_ylabel("Count")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def write_review_markdown(review: dict[str, Any], output_dir: str | Path) -> Path:
    output_path = ensure_dir(output_dir) / "research_review.md"
    lines = [
        f"# {review['experiment_id']} research review",
        "",
        "## Headline Metrics",
        "",
    ]
    for key, value in review["metrics"].items():
        if isinstance(value, float) and math.isnan(value):
            formatted = "nan"
        elif isinstance(value, float):
            formatted = f"{value:.4f}"
        else:
            formatted = str(value)
        lines.append(f"- **{key}**: {formatted}")

    lines.extend(["", "## Risks", ""])
    if review["risks"]:
        for risk in review["risks"]:
            lines.append(f"- **{risk['severity']} / {risk['code']}**: {risk['message']}")
    else:
        lines.append("- No major research risks flagged from the saved artifacts.")

    lines.extend(["", "## Split Analysis", ""])
    for split, summary in review["splits"].items():
        datasets = summary.get("datasets", {})
        labels = summary.get("labels", {})
        severity_labels = summary.get("severity_labels", {})
        duration_buckets = summary.get("duration_buckets", {})
        fps_buckets = summary.get("fps_buckets", {})
        camera_groups = summary.get("camera_groups", 0)
        augmented_count = summary.get("augmented_count", 0)
        lines.extend(
            [
                f"### {split}",
                "",
                f"- count: {summary.get('count', 0)}",
                f"- datasets: {datasets}",
                f"- labels: {labels}",
                f"- severity_labels: {severity_labels}",
                f"- duration_buckets: {duration_buckets}",
                f"- fps_buckets: {fps_buckets}",
                f"- camera_groups: {camera_groups}",
                f"- augmented_count: {augmented_count}",
                "",
            ]
        )

    lines.extend(["## Next Steps", ""])
    for step in review["recommended_next_steps"]:
        lines.append(f"- {step}")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def build_research_review(run_dir: str | Path) -> dict[str, Any]:
    run_path = Path(run_dir)
    metrics_payload = load_json(run_path / "metrics.json")
    config_payload = load_json(run_path / "config.json")
    manifest_summary = load_json(run_path / "manifest_summary.json") if (run_path / "manifest_summary.json").exists() else {}

    split_records: dict[str, list[ClipRecord]] = {}
    for split, manifest_path in discover_resolved_manifests(run_path).items():
        split_records[split] = load_manifest(manifest_path)

    split_summaries = {split: summarize_records(records) for split, records in split_records.items()}
    for split, summary in manifest_summary.items():
        split_summaries.setdefault(split, {})
        split_summaries[split].setdefault("count", summary.get("count", 0))
        split_summaries[split].setdefault("labels", summary.get("labels", {}))

    experiment = config_payload.get("experiment", {})
    experiment_id = experiment.get("id") or config_payload.get("experiment_id") or run_path.name
    task = str(experiment.get("task") or config_payload.get("task") or "unknown")
    metrics = metrics_payload.get("metrics", {})
    risks = derive_risks(task=task, metrics=metrics, split_summaries=split_summaries)
    return {
        "experiment_id": experiment_id,
        "task": task,
        "run_dir": str(run_path),
        "metrics": metrics,
        "best_epoch": metrics_payload.get("metadata", {}).get("best_epoch"),
        "history_length": metrics_payload.get("metadata", {}).get("history_length"),
        "splits": split_summaries,
        "risks": risks,
        "recommended_next_steps": recommend_next_steps(task, risks),
    }


def export_research_review(run_dir: str | Path, output_dir: str | Path | None = None) -> dict[str, Any]:
    run_path = Path(run_dir)
    review = build_research_review(run_path)
    target_dir = ensure_dir(output_dir or (run_path / "research_review"))
    assets_dir = ensure_dir(target_dir / "assets")

    val_summary = review["splits"].get("val", {})
    charts: dict[str, str] = {}
    label_chart = _plot_counter(val_summary.get("labels", {}), "Validation Label Balance", assets_dir / "val_label_balance.png")
    dataset_chart = _plot_counter(val_summary.get("datasets", {}), "Validation Dataset Mix", assets_dir / "val_dataset_mix.png")
    tag_chart = _plot_counter(dict(list(val_summary.get("tags", {}).items())[:8]), "Validation Top Tags", assets_dir / "val_top_tags.png")
    for chart_name, chart_path in {
        "val_label_balance": label_chart,
        "val_dataset_mix": dataset_chart,
        "val_top_tags": tag_chart,
    }.items():
        if chart_path is not None:
            charts[chart_name] = str(chart_path)
    review["charts"] = charts

    summary_path = target_dir / "research_review.json"
    write_json(summary_path, review)
    report_path = write_review_markdown(review, target_dir)
    return {
        "summary_path": str(summary_path),
        "report_path": str(report_path),
        "output_dir": str(target_dir),
        "charts": charts,
        "experiment_id": review["experiment_id"],
        "task": review["task"],
    }
