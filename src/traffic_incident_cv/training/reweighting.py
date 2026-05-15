from __future__ import annotations

from collections import Counter
from typing import Iterable


def normalize_targets(targets: Iterable[int | None]) -> list[int]:
    return [int(target) for target in targets if target is not None]


def count_targets(targets: Iterable[int | None]) -> dict[int, int]:
    normalized = normalize_targets(targets)
    return dict(sorted(Counter(normalized).items()))


def balanced_class_weights(targets: Iterable[int | None], num_classes: int) -> list[float]:
    counts = count_targets(targets)
    total = sum(counts.values())
    if total <= 0 or num_classes <= 0:
        return [0.0] * max(num_classes, 0)
    weights: list[float] = []
    for class_index in range(num_classes):
        count = counts.get(class_index, 0)
        if count <= 0:
            weights.append(0.0)
        else:
            weights.append(total / (num_classes * count))
    return weights


def weighted_sample_weights(targets: Iterable[int | None], num_classes: int) -> list[float]:
    normalized = list(targets)
    class_weights = balanced_class_weights(normalized, num_classes=num_classes)
    sample_weights: list[float] = []
    for target in normalized:
        if target is None:
            sample_weights.append(0.0)
        else:
            sample_weights.append(class_weights[int(target)])
    return sample_weights
