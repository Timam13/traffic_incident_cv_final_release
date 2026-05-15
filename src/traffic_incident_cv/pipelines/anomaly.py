from __future__ import annotations

from dataclasses import dataclass

from traffic_incident_cv.domain.schemas import ClipRecord, MetricBundle, PredictionRecord
from traffic_incident_cv.interfaces.base import PipelineInterface


@dataclass
class AnomalyDetectionPipeline(PipelineInterface):
    config: dict

    def fit(self) -> None:
        raise NotImplementedError("Train anomaly baseline or scoring model.")

    def validate(self) -> MetricBundle:
        raise NotImplementedError("Compute anomaly metrics.")

    def predict(self, items: list[ClipRecord]) -> list[PredictionRecord]:
        raise NotImplementedError("Implement anomaly inference.")

    def export(self, output_dir: str) -> None:
        raise NotImplementedError("Export anomaly artifacts.")
