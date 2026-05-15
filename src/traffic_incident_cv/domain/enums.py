from __future__ import annotations

from enum import Enum


class Split(str, Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class TaskType(str, Enum):
    ACCIDENT = "accident"
    SEVERITY = "severity"
    ANOMALY = "anomaly"


class BinaryLabel(str, Enum):
    ACCIDENT = "accident"
    NO_ACCIDENT = "no_accident"


class SeverityLabel(str, Enum):
    MINOR = "minor"
    MODERATE = "moderate"
    MAJOR = "major"


class ModelFamily(str, Enum):
    CONVNEXT_BASELINE = "convnext_baseline"
    R3D_BASELINE = "r3d_baseline"
    YOLO_CONVNEXT_HYBRID = "yolo_convnext_hybrid"
    ANOMALY_BASELINE = "anomaly_baseline"
