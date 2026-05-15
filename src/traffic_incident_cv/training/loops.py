from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from traffic_incident_cv.interfaces.base import TrainerInterface
from traffic_incident_cv.utils.io import ensure_dir, write_json

logger = logging.getLogger(__name__)


@dataclass
class TrainerState:
    epoch: int = 0
    best_metric: float | None = None
    best_epoch: int | None = None
    last_metrics: dict[str, float] = field(default_factory=dict)
    history: list[dict[str, float]] = field(default_factory=list)


@dataclass
class GenericTrainer(TrainerInterface):
    model: Any
    config: dict
    output_dir: str
    epoch_runner: Callable[[str, int], dict[str, float]] | None = None
    on_epoch_end: Callable[[TrainerState, dict[str, float]], None] | None = None
    selection_metric: str = "val_f1"
    maximize_metric: bool = True
    state: TrainerState = field(default_factory=TrainerState)

    def run_epoch(self, split: str) -> dict[str, float]:
        logger.info(
            "Running %s epoch=%s for model=%s",
            split,
            self.state.epoch,
            getattr(self.model, "name", type(self.model).__name__),
        )
        if self.epoch_runner is not None:
            metrics = self.epoch_runner(split, self.state.epoch)
        else:
            metrics = {f"{split}_loss": 0.0}
        return metrics

    def run(self, epochs: int) -> TrainerState:
        self.bootstrap_output_dir()
        for epoch in range(1, max(epochs, 1) + 1):
            self.state.epoch = epoch
            train_metrics = self.run_epoch("train")
            val_metrics = self.run_epoch("val")
            merged_metrics = {"epoch": float(epoch), **train_metrics, **val_metrics}
            self.state.last_metrics = merged_metrics
            self.state.history.append(merged_metrics)
            self._update_best(merged_metrics)
            if self.on_epoch_end is not None:
                self.on_epoch_end(self.state, merged_metrics)
            logger.info("Epoch %s metrics=%s", epoch, merged_metrics)
            self._persist_progress()
            print(
                json.dumps(
                    {
                        "event": "epoch_end",
                        "epoch": epoch,
                        "best_metric": self.state.best_metric,
                        "best_epoch": self.state.best_epoch,
                        "metrics": merged_metrics,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
        return self.state

    def save_checkpoint(self, output_path: str) -> None:
        payload = {
            "epoch": self.state.epoch,
            "best_metric": self.state.best_metric,
            "best_epoch": self.state.best_epoch,
            "last_metrics": self.state.last_metrics,
            "history": self.state.history,
            "model": self.model.summary() if hasattr(self.model, "summary") else {"name": type(self.model).__name__},
            "note": "Lightweight trainer metadata checkpoint. Neural network weights are stored separately in model_best.pt or model_last.pt.",
        }
        write_json(output_path, payload)

    def load_checkpoint(self, checkpoint_path: str) -> None:
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.state.epoch = int(payload.get("epoch", 0))
        best_metric = payload.get("best_metric")
        self.state.best_metric = float(best_metric) if best_metric is not None else None
        best_epoch = payload.get("best_epoch")
        self.state.best_epoch = int(best_epoch) if best_epoch is not None else None
        self.state.last_metrics = {key: float(value) for key, value in payload.get("last_metrics", {}).items()}
        self.state.history = [
            {metric_name: float(metric_value) for metric_name, metric_value in item.items()}
            for item in payload.get("history", [])
        ]
        logger.info("Loaded trainer checkpoint from %s", path)

    def bootstrap_output_dir(self) -> Path:
        return ensure_dir(self.output_dir)

    def _persist_progress(self) -> None:
        output_dir = self.bootstrap_output_dir()
        write_json(
            output_dir / "last_metrics.json",
            {
                "epoch": self.state.epoch,
                "best_metric": self.state.best_metric,
                "best_epoch": self.state.best_epoch,
                "metrics": self.state.last_metrics,
            },
        )
        write_json(
            output_dir / "history.json",
            {
                "history": self.state.history,
            },
        )

    def _update_best(self, metrics: dict[str, float]) -> None:
        candidate = self._extract_selection_metric(metrics)
        if candidate is None:
            return
        if self.state.best_metric is None:
            self.state.best_metric = candidate
            self.state.best_epoch = self.state.epoch
            return
        is_better = candidate > self.state.best_metric if self.maximize_metric else candidate < self.state.best_metric
        if is_better:
            self.state.best_metric = candidate
            self.state.best_epoch = self.state.epoch

    def _extract_selection_metric(self, metrics: dict[str, float]) -> float | None:
        if self.selection_metric in metrics:
            return float(metrics[self.selection_metric])
        val_metric = f"val_{self.selection_metric}"
        if val_metric in metrics:
            return float(metrics[val_metric])
        return None
