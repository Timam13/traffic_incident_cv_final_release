from __future__ import annotations

from typing import Iterable

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from traffic_incident_cv.domain.schemas import MetricBundle


def _to_numpy(values: Iterable) -> np.ndarray:
    return np.asarray(list(values))


def _safe_metric(fn, *args, **kwargs) -> float:
    try:
        return float(fn(*args, **kwargs))
    except ValueError:
        return float("nan")


def classification_metrics(
    y_true: Iterable,
    y_pred: Iterable,
    y_score: Iterable | None = None,
    average: str = "binary",
) -> MetricBundle:
    yt = _to_numpy(y_true)
    yp = _to_numpy(y_pred)
    metrics = {
        "accuracy": float(accuracy_score(yt, yp)),
        "precision": float(precision_score(yt, yp, average=average, zero_division=0)),
        "recall": float(recall_score(yt, yp, average=average, zero_division=0)),
        "f1": float(f1_score(yt, yp, average=average, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(yt, yp)),
    }
    metadata = {"confusion_matrix": confusion_matrix(yt, yp).tolist()}
    if y_score is not None:
        ys = _to_numpy(y_score).astype(float)
        if ys.ndim > 1 and ys.shape[1] > 1:
            ys = ys[:, 1]
        ys = ys.reshape(-1)
        metrics["roc_auc"] = _safe_metric(roc_auc_score, yt, ys)
        metrics["pr_auc"] = _safe_metric(average_precision_score, yt, ys)
    return MetricBundle(metrics=metrics, metadata=metadata)


def multiclass_metrics(y_true: Iterable, y_pred: Iterable, y_score: Iterable | None = None) -> MetricBundle:
    yt = _to_numpy(y_true)
    yp = _to_numpy(y_pred)
    metrics = {
        "accuracy": float(accuracy_score(yt, yp)),
        "macro_f1": float(f1_score(yt, yp, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(yt, yp)),
    }
    metadata = {"confusion_matrix": confusion_matrix(yt, yp).tolist()}
    if y_score is not None:
        ys = _to_numpy(y_score).astype(float)
        metrics["roc_auc_ovr"] = _safe_metric(roc_auc_score, yt, ys, multi_class="ovr", average="macro")
    return MetricBundle(metrics=metrics, metadata=metadata)
