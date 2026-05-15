from __future__ import annotations

from typing import Any, Callable

from traffic_incident_cv.domain.schemas import ClipRecord, PredictionRecord
from traffic_incident_cv.evaluation.metrics import multiclass_metrics
from traffic_incident_cv.pipelines.accident import AccidentDetectionPipeline

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


SEVERITY_TARGETS = {
    "minor": 0,
    "moderate": 1,
    "major": 2,
}


class SeverityClassificationPipeline(AccidentDetectionPipeline):
    def predict(self, items: list[ClipRecord]) -> list[PredictionRecord]:
        return [
            PredictionRecord(
                clip_id=item.clip_id,
                task="severity",
                predicted_label=None,
                score=None,
                metadata={
                    "status": "use_saved_run_inference_for_predictions",
                    "video_path": item.video_path,
                },
            )
            for item in items
        ]

    def _build_epoch_runner(
        self,
        exp: Any,
        model: Any,
        output_dir: Any,
        selection_metric: str,
    ) -> Callable[[str, int], dict[str, float]] | None:
        context = self._build_torch_context(exp, model, output_dir, selection_metric)
        if context is None:
            self.torch_context = None
            return None
        self.torch_context = context

        def run_epoch(split: str, epoch_index: int) -> dict[str, float]:
            loader = context.train_loader if split == "train" else context.val_loader
            if loader is None or len(loader.dataset) == 0:
                return {f"{split}_loss": 0.0}

            is_train = split == "train"
            model.train(mode=is_train)
            total_loss = 0.0
            total_items = 0
            y_true: list[int] = []
            y_pred: list[int] = []
            y_score: list[list[float]] = []

            for batch in loader:
                frames = batch["frames"].to(context.device, non_blocking=True)
                targets = batch["targets"].to(context.device, non_blocking=True)
                if is_train:
                    context.optimizer.zero_grad(set_to_none=True)
                with torch.autocast(
                    device_type=context.device.type,
                    enabled=context.amp_enabled and context.device.type == "cuda",
                ):
                    logits = model(frames)
                    loss = context.criterion(logits, targets)
                if is_train:
                    context.scaler.scale(loss).backward()
                    context.scaler.step(context.optimizer)
                    context.scaler.update()
                probabilities = torch.softmax(logits.detach(), dim=1)
                predictions = probabilities.argmax(dim=1)
                batch_size = int(targets.shape[0])
                total_loss += float(loss.detach().item()) * batch_size
                total_items += batch_size
                y_true.extend(targets.detach().cpu().tolist())
                y_pred.extend(predictions.detach().cpu().tolist())
                y_score.extend(probabilities.detach().cpu().tolist())

            mean_loss = total_loss / max(total_items, 1)
            bundle = multiclass_metrics(y_true, y_pred, y_score=y_score)
            metrics = {f"{split}_loss": mean_loss}
            metrics.update({f"{split}_{name}": value for name, value in bundle.metrics.items()})
            if not is_train and context.scheduler is not None:
                context.scheduler.step()
            return metrics

        return run_epoch

    @staticmethod
    def _record_target(record: ClipRecord) -> int | None:
        return SEVERITY_TARGETS.get(record.severity_label) if record.severity_label is not None else None

    @staticmethod
    def _resolve_selection_metric(task: str, configured_metric: str | None) -> str:
        if configured_metric:
            return configured_metric
        return "macro_f1"
