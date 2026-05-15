from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Iterable, Protocol, Sequence

from traffic_incident_cv.domain.schemas import ClipRecord, MetricBundle, PredictionRecord


class ManifestBuilderInterface(Protocol):
    def build(self) -> list[ClipRecord]:
        ...

    def save(self, path: str) -> None:
        ...


class ClipDatasetInterface(Protocol):
    def __len__(self) -> int:
        ...

    def __getitem__(self, index: int) -> dict[str, Any]:
        ...


class ClipSamplerInterface(Protocol):
    def sample(self, *, total_frames: int, num_frames: int, stride: int) -> list[int]:
        ...


class FeatureExtractorInterface(Protocol):
    def forward_features(self, batch: Any) -> Any:
        ...


class TemporalAggregatorInterface(Protocol):
    def aggregate(self, frame_embeddings: Any) -> Any:
        ...


class ObjectDetectorInterface(Protocol):
    def detect(self, frames: Any) -> list[dict[str, Any]]:
        ...


class ClassifierHeadInterface(Protocol):
    def classify(self, embedding: Any) -> Any:
        ...


class AnomalyScorerInterface(Protocol):
    def score(self, inputs: Any) -> Any:
        ...


class PipelineInterface(ABC):
    @abstractmethod
    def fit(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def validate(self) -> MetricBundle:
        raise NotImplementedError

    @abstractmethod
    def predict(self, items: Sequence[ClipRecord]) -> list[PredictionRecord]:
        raise NotImplementedError

    @abstractmethod
    def export(self, output_dir: str) -> None:
        raise NotImplementedError


class TrainerInterface(ABC):
    @abstractmethod
    def run_epoch(self, split: str) -> dict[str, float]:
        raise NotImplementedError

    @abstractmethod
    def save_checkpoint(self, output_path: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def load_checkpoint(self, checkpoint_path: str) -> None:
        raise NotImplementedError


class EvaluatorInterface(ABC):
    @abstractmethod
    def compute_metrics(self, y_true: Iterable[Any], y_pred: Iterable[Any], y_score: Any | None = None) -> MetricBundle:
        raise NotImplementedError

    @abstractmethod
    def render_report(self, metrics: MetricBundle, output_dir: str) -> None:
        raise NotImplementedError
