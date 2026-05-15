from __future__ import annotations

import math
import json
from pathlib import Path
from typing import Any, Iterable

from traffic_incident_cv.evaluation.metrics import classification_metrics
from traffic_incident_cv.utils.io import ensure_dir, write_json, write_jsonl

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None


DEFAULT_LABELS = ("no_accident", "accident")


def _safe_float(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return number


def _finite_or_none(value: Any) -> float | None:
    number = _safe_float(value)
    return number if math.isfinite(number) else None


def threshold_sweep(
    y_true: Iterable[int],
    y_score: Iterable[float],
    thresholds: Iterable[float] | None = None,
) -> dict[str, Any]:
    yt = [int(item) for item in y_true]
    ys = [float(item) for item in y_score]
    threshold_values = [round(float(item), 4) for item in (thresholds or [index / 20 for index in range(1, 20)])]
    rows: list[dict[str, Any]] = []
    best_row: dict[str, Any] | None = None
    best_key: tuple[float, float, float] | None = None

    for threshold in threshold_values:
        y_pred = [1 if score >= threshold else 0 for score in ys]
        bundle = classification_metrics(yt, y_pred, y_score=ys, average="binary")
        row = {
            "threshold": threshold,
            **bundle.metrics,
            "confusion_matrix": bundle.metadata.get("confusion_matrix", []),
        }
        rows.append(row)
        balanced_accuracy = _finite_or_none(row.get("balanced_accuracy")) or float("-inf")
        f1_value = _finite_or_none(row.get("f1")) or float("-inf")
        recall_value = _finite_or_none(row.get("recall")) or float("-inf")
        ranking = (balanced_accuracy, f1_value, recall_value)
        if best_key is None or ranking > best_key:
            best_key = ranking
            best_row = row

    return {
        "thresholds": rows,
        "best": best_row,
    }


def build_prediction_rows(
    *,
    clip_ids: Iterable[str],
    y_true: Iterable[int],
    y_score: Iterable[float],
    threshold: float,
    labels: tuple[str, str] = DEFAULT_LABELS,
) -> list[dict[str, Any]]:
    negative_label, positive_label = labels
    rows: list[dict[str, Any]] = []
    for clip_id, target, score in zip(clip_ids, y_true, y_score, strict=True):
        predicted_target = 1 if float(score) >= threshold else 0
        rows.append(
            {
                "clip_id": clip_id,
                "task": "accident",
                "predicted_label": positive_label if predicted_target == 1 else negative_label,
                "score": float(score),
                "probabilities": {
                    negative_label: float(1.0 - float(score)),
                    positive_label: float(score),
                },
                "metadata": {
                    "target_index": int(target),
                    "target_label": positive_label if int(target) == 1 else negative_label,
                    "predicted_index": predicted_target,
                    "threshold": float(threshold),
                },
            }
        )
    return rows


def write_confusion_matrix_plot(
    confusion_matrix: list[list[int]],
    output_path: str | Path,
    labels: tuple[str, str] = DEFAULT_LABELS,
) -> str | None:
    if plt is None:
        return None
    matrix = confusion_matrix or [[0, 0], [0, 0]]
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    im = ax.imshow(matrix, cmap="YlGnBu")
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(len(labels)), labels=labels)
    ax.set_yticks(range(len(labels)), labels=labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Test Confusion Matrix")

    for row_index, row in enumerate(matrix):
        for col_index, value in enumerate(row):
            ax.text(col_index, row_index, str(value), ha="center", va="center", color="#122018", fontsize=12)

    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return str(output)


def write_threshold_sweep_plot(
    sweep_payload: dict[str, Any],
    output_path: str | Path,
) -> str | None:
    if plt is None:
        return None
    rows = list(sweep_payload.get("thresholds", []))
    if not rows:
        return None
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    thresholds = [row["threshold"] for row in rows]
    balanced_accuracy = [_safe_float(row.get("balanced_accuracy")) for row in rows]
    f1_values = [_safe_float(row.get("f1")) for row in rows]
    precision_values = [_safe_float(row.get("precision")) for row in rows]
    recall_values = [_safe_float(row.get("recall")) for row in rows]

    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    ax.plot(thresholds, balanced_accuracy, label="balanced_accuracy", color="#1f6f5f", linewidth=2.4)
    ax.plot(thresholds, f1_values, label="f1", color="#bb5a24", linewidth=2.2)
    ax.plot(thresholds, precision_values, label="precision", color="#5d668a", linewidth=1.8, alpha=0.95)
    ax.plot(thresholds, recall_values, label="recall", color="#987f1d", linewidth=1.8, alpha=0.95)
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Metric value")
    ax.set_title("Threshold Sweep")
    ax.set_ylim(0.0, 1.02)
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, ncols=2)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return str(output)


def write_post_eval_report(
    *,
    output_dir: str | Path,
    split: str,
    threshold: float,
    metrics: dict[str, Any],
    best_threshold_row: dict[str, Any] | None,
    charts: dict[str, str],
) -> str:
    target_dir = ensure_dir(output_dir)
    report_path = target_dir / f"{split}_post_eval_report.md"
    lines = [
        f"# {split.upper()} post-eval",
        "",
        f"- threshold: {threshold:.4f}",
        "",
        "## Metrics",
        "",
    ]
    for key, value in metrics.items():
        if isinstance(value, float):
            lines.append(f"- **{key}**: {value:.4f}")
        else:
            lines.append(f"- **{key}**: {value}")
    if best_threshold_row is not None:
        lines.extend(["", "## Best Threshold Sweep Point", ""])
        for key in ("threshold", "balanced_accuracy", "f1", "precision", "recall", "roc_auc", "pr_auc"):
            value = best_threshold_row.get(key)
            if isinstance(value, float):
                lines.append(f"- **{key}**: {value:.4f}")
            elif value is not None:
                lines.append(f"- **{key}**: {value}")
    if charts:
        lines.extend(["", "## Charts", ""])
        for chart_name, chart_path in charts.items():
            lines.append(f"- **{chart_name}**: {chart_path}")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(report_path)


def export_accident_post_eval_bundle(
    *,
    output_dir: str | Path,
    split: str,
    threshold: float,
    metrics_payload: dict[str, Any],
    sweep_payload: dict[str, Any],
    predictions: list[dict[str, Any]],
) -> dict[str, Any]:
    target_dir = ensure_dir(output_dir)
    assets_dir = ensure_dir(target_dir / "assets")

    metrics_path = target_dir / f"{split}_metrics.json"
    predictions_path = target_dir / f"{split}_predictions.jsonl"
    sweep_path = target_dir / f"{split}_threshold_sweep.json"
    write_json(metrics_path, metrics_payload)
    write_json(sweep_path, sweep_payload)
    write_jsonl(predictions_path, predictions)

    charts: dict[str, str] = {}
    confusion_matrix = metrics_payload.get("metadata", {}).get("confusion_matrix", [])
    confusion_chart = write_confusion_matrix_plot(confusion_matrix, assets_dir / f"{split}_confusion_matrix.png")
    sweep_chart = write_threshold_sweep_plot(sweep_payload, assets_dir / f"{split}_threshold_sweep.png")
    if confusion_chart:
        charts["confusion_matrix"] = confusion_chart
    if sweep_chart:
        charts["threshold_sweep"] = sweep_chart

    report_path = write_post_eval_report(
        output_dir=target_dir,
        split=split,
        threshold=threshold,
        metrics=metrics_payload.get("metrics", {}),
        best_threshold_row=sweep_payload.get("best"),
        charts=charts,
    )
    return {
        "metrics_path": str(metrics_path),
        "predictions_path": str(predictions_path),
        "threshold_sweep_path": str(sweep_path),
        "report_path": report_path,
        "charts": charts,
        "output_dir": str(target_dir),
    }
