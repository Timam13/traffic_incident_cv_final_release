from __future__ import annotations

from pathlib import Path

from traffic_incident_cv.domain.schemas import MetricBundle
from traffic_incident_cv.utils.io import ensure_dir


def _format_metric(value: float) -> str:
    return "nan" if value != value else f"{value:.4f}"


def write_markdown_report(metrics: MetricBundle, output_dir: str, title: str = "Experiment report") -> Path:
    output_dir = ensure_dir(output_dir)
    report_path = output_dir / "report.md"
    lines = [f"# {title}", "", "## Metrics", ""]
    for key, value in metrics.metrics.items():
        lines.append(f"- **{key}**: {_format_metric(value)}")
    confusion_matrix = metrics.metadata.get("confusion_matrix")
    if confusion_matrix:
        lines.extend(["", "## Confusion Matrix", "", "```text", str(confusion_matrix), "```"])
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path
