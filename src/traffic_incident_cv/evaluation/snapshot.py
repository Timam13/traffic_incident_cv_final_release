from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from traffic_incident_cv.utils.io import ensure_dir, write_json

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None


def discover_reviews(root: str | Path) -> list[Path]:
    root_path = Path(root)
    return sorted(root_path.rglob("research_review.json"))


def load_review(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _headline_metric(review: dict[str, Any]) -> tuple[str, float | None]:
    metrics = review.get("metrics", {})
    task = review.get("task")
    candidates = {
        "accident": ["val_f1", "val_pr_auc", "val_balanced_accuracy"],
        "severity": ["val_macro_f1", "val_accuracy", "val_balanced_accuracy"],
    }
    for key in candidates.get(task, []):
        if key in metrics:
            return key, metrics[key]
    for key, value in metrics.items():
        if key.startswith("val_"):
            return key, value
    return "n/a", None


def build_snapshot(review_paths: list[str | Path]) -> dict[str, Any]:
    reviews = [load_review(path) for path in review_paths]
    experiments: list[dict[str, Any]] = []
    for review in reviews:
        metric_name, metric_value = _headline_metric(review)
        experiments.append(
            {
                "experiment_id": review.get("experiment_id"),
                "task": review.get("task"),
                "headline_metric": metric_name,
                "headline_value": metric_value,
                "risk_codes": [risk["code"] for risk in review.get("risks", [])],
                "recommended_next_steps": review.get("recommended_next_steps", []),
                "run_dir": review.get("run_dir"),
            }
        )
    return {
        "review_count": len(reviews),
        "experiments": experiments,
    }


def _plot_snapshot(experiments: list[dict[str, Any]], output_path: Path) -> Path | None:
    if plt is None or not experiments:
        return None
    labels = [item["experiment_id"] for item in experiments]
    values = [float(item["headline_value"] or 0.0) for item in experiments]
    colors = ["#1f4e79" if item["task"] == "accident" else "#8a4f08" for item in experiments]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.bar(labels, values, color=colors)
    ax.set_ylabel("Headline metric")
    ax.set_title("Research Snapshot")
    ax.set_ylim(0.0, max(values + [1.0]))
    ax.grid(axis="y", alpha=0.2)
    ax.tick_params(axis="x", rotation=12)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def write_snapshot_markdown(snapshot: dict[str, Any], output_dir: str | Path) -> Path:
    output_path = ensure_dir(output_dir) / "research_snapshot.md"
    lines = [
        "# Research Snapshot",
        "",
        f"- review_count: {snapshot['review_count']}",
        "",
        "## Experiments",
        "",
    ]
    for item in snapshot["experiments"]:
        lines.append(
            f"- **{item['experiment_id']}** ({item['task']}): {item['headline_metric']}={item['headline_value']}"
        )
        lines.append(f"  risks: {item['risk_codes']}")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def export_snapshot(review_root: str | Path, output_dir: str | Path) -> dict[str, Any]:
    review_paths = discover_reviews(review_root)
    snapshot = build_snapshot(review_paths)
    target_dir = ensure_dir(output_dir)
    chart_path = _plot_snapshot(snapshot["experiments"], target_dir / "assets" / "research_snapshot.png")
    if chart_path is not None:
        snapshot["chart"] = str(chart_path)
    snapshot["review_paths"] = [str(path) for path in review_paths]
    summary_path = target_dir / "research_snapshot.json"
    write_json(summary_path, snapshot)
    report_path = write_snapshot_markdown(snapshot, target_dir)
    return {
        "summary_path": str(summary_path),
        "report_path": str(report_path),
        "chart_path": str(chart_path) if chart_path is not None else None,
        "review_count": snapshot["review_count"],
    }
