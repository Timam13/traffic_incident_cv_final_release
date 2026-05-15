from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Iterable, Sequence

from traffic_incident_cv.evaluation.metrics import multiclass_metrics
from traffic_incident_cv.utils.io import ensure_dir, write_json, write_jsonl

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None


DEFAULT_LABELS = ("minor", "moderate", "major")


def _safe_float(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return number


def build_prediction_rows(
    *,
    clip_ids: Iterable[str],
    y_true: Iterable[int],
    y_score: Sequence[Sequence[float]],
    labels: Sequence[str] = DEFAULT_LABELS,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for clip_id, target, score_row in zip(clip_ids, y_true, y_score, strict=True):
        scores = [float(item) for item in score_row]
        predicted_index = max(range(len(scores)), key=lambda index: scores[index])
        rows.append(
            {
                "clip_id": clip_id,
                "task": "severity",
                "predicted_label": str(labels[predicted_index]),
                "score": float(scores[predicted_index]),
                "probabilities": {
                    str(label): float(scores[index])
                    for index, label in enumerate(labels)
                },
                "metadata": {
                    "target_index": int(target),
                    "target_label": str(labels[int(target)]),
                    "predicted_index": int(predicted_index),
                },
            }
        )
    return rows


def write_confusion_matrix_plot(
    confusion_matrix: list[list[int]],
    output_path: str | Path,
    labels: Sequence[str] = DEFAULT_LABELS,
) -> str | None:
    if plt is None:
        return None
    matrix = confusion_matrix or [[0 for _ in labels] for _ in labels]
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    im = ax.imshow(matrix, cmap="YlGnBu")
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(len(labels)), labels=labels)
    ax.set_yticks(range(len(labels)), labels=labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Severity Confusion Matrix")

    for row_index, row in enumerate(matrix):
        for col_index, value in enumerate(row):
            ax.text(col_index, row_index, str(value), ha="center", va="center", color="#122018", fontsize=11)

    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return str(output)


def write_probability_margin_plot(
    y_score: Sequence[Sequence[float]],
    output_path: str | Path,
) -> str | None:
    if plt is None or not y_score:
        return None
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    margins: list[float] = []
    for row in y_score:
        ranked = sorted((float(item) for item in row), reverse=True)
        if len(ranked) >= 2:
            margins.append(ranked[0] - ranked[1])
        elif ranked:
            margins.append(ranked[0])

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.hist(margins, bins=20, color="#1f6f5f", alpha=0.85)
    ax.set_title("Prediction Margin Distribution")
    ax.set_xlabel("Top-1 minus Top-2 probability")
    ax.set_ylabel("Count")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return str(output)


def write_post_eval_report(
    *,
    output_dir: str | Path,
    split: str,
    metrics: dict[str, Any],
    charts: dict[str, str],
    metadata: dict[str, Any],
) -> str:
    target_dir = ensure_dir(output_dir)
    report_path = target_dir / f"{split}_post_eval_report.md"
    lines = [
        f"# {split.upper()} severity post-eval",
        "",
        "## Metrics",
        "",
    ]
    for key, value in metrics.items():
        number = _safe_float(value)
        if math.isfinite(number):
            lines.append(f"- **{key}**: {number:.4f}")
        else:
            lines.append(f"- **{key}**: {value}")
    lines.extend(["", "## Metadata", ""])
    for key in ("record_count", "resolved_manifest_path", "device", "eval_clip_sampler"):
        value = metadata.get(key)
        if value is not None:
            lines.append(f"- **{key}**: {value}")
    if charts:
        lines.extend(["", "## Charts", ""])
        for chart_name, chart_path in charts.items():
            lines.append(f"- **{chart_name}**: {chart_path}")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(report_path)


def export_severity_post_eval_bundle(
    *,
    output_dir: str | Path,
    split: str,
    metrics_payload: dict[str, Any],
    predictions: list[dict[str, Any]],
    y_score: Sequence[Sequence[float]],
    labels: Sequence[str] = DEFAULT_LABELS,
) -> dict[str, Any]:
    target_dir = ensure_dir(output_dir)
    assets_dir = ensure_dir(target_dir / "assets")

    metrics_path = target_dir / f"{split}_metrics.json"
    predictions_path = target_dir / f"{split}_predictions.jsonl"
    write_json(metrics_path, metrics_payload)
    write_jsonl(predictions_path, predictions)

    charts: dict[str, str] = {}
    confusion_matrix = metrics_payload.get("metadata", {}).get("confusion_matrix", [])
    confusion_chart = write_confusion_matrix_plot(
        confusion_matrix,
        assets_dir / f"{split}_confusion_matrix.png",
        labels=labels,
    )
    margin_chart = write_probability_margin_plot(
        y_score,
        assets_dir / f"{split}_margin_histogram.png",
    )
    if confusion_chart:
        charts["confusion_matrix"] = confusion_chart
    if margin_chart:
        charts["prediction_margin"] = margin_chart

    report_path = write_post_eval_report(
        output_dir=target_dir,
        split=split,
        metrics=metrics_payload.get("metrics", {}),
        charts=charts,
        metadata=metrics_payload.get("metadata", {}),
    )
    return {
        "metrics_path": str(metrics_path),
        "predictions_path": str(predictions_path),
        "report_path": report_path,
        "charts": charts,
        "output_dir": str(target_dir),
    }

