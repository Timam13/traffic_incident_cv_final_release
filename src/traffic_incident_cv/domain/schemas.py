from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from traffic_incident_cv.domain.enums import BinaryLabel, Split, TaskType


@dataclass(slots=True)
class ClipRecord:
    clip_id: str
    video_path: str
    dataset: str
    split: Split | str
    task: TaskType | str
    binary_label: BinaryLabel | str | None = None
    severity_label: str | None = None
    camera_id: str | None = None
    fps: float | None = None
    duration_sec: float | None = None
    clip_start_sec: float | None = None
    clip_end_sec: float | None = None
    clip_start_frame: int | None = None
    clip_end_frame: int | None = None
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "clip_id": self.clip_id,
            "video_path": self.video_path,
            "dataset": self.dataset,
            "split": str(self.split.value if hasattr(self.split, 'value') else self.split),
            "task": str(self.task.value if hasattr(self.task, 'value') else self.task),
            "binary_label": self.binary_label.value if hasattr(self.binary_label, 'value') else self.binary_label,
            "severity_label": self.severity_label,
            "camera_id": self.camera_id,
            "fps": self.fps,
            "duration_sec": self.duration_sec,
            "clip_start_sec": self.clip_start_sec,
            "clip_end_sec": self.clip_end_sec,
            "clip_start_frame": self.clip_start_frame,
            "clip_end_frame": self.clip_end_frame,
            "tags": list(self.tags),
        }

    def validate(self) -> None:
        if not self.clip_id:
            raise ValueError("clip_id must be non-empty")
        if not self.video_path:
            raise ValueError("video_path must be non-empty")
        if not self.dataset:
            raise ValueError("dataset must be non-empty")
        if not self.split:
            raise ValueError("split must be non-empty")
        if not self.task:
            raise ValueError("task must be non-empty")

    @property
    def path(self) -> Path:
        return Path(self.video_path)


@dataclass(slots=True)
class ExperimentConfig:
    experiment_id: str
    task: str
    manifest_paths: dict[str, str]
    model: dict[str, Any]
    training: dict[str, Any]
    runtime: dict[str, Any]
    data: dict[str, Any]
    project: dict[str, Any]
    evaluation: dict[str, Any]
    optimization: dict[str, Any] = field(default_factory=dict)
    reporting: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class PredictionRecord:
    clip_id: str
    task: str
    predicted_label: str | None
    score: float | None
    probabilities: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class MetricBundle:
    metrics: dict[str, float]
    per_class: dict[str, dict[str, float]] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
