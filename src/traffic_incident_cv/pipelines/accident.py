from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from traffic_incident_cv.config import to_experiment_config
from traffic_incident_cv.data.clip_sampling import RandomWindowClipSampler, TailFocusedSampler, UniformClipSampler
from traffic_incident_cv.data.manifest import load_manifest, save_manifest, summarize_manifest
from traffic_incident_cv.data.transforms import (
    Compose,
    build_domain_randomization_transforms,
    build_motion_preprocessing_transforms,
    build_spatial_preprocessing_transforms,
)
from traffic_incident_cv.data.video_dataset import ManifestClipDataset
from traffic_incident_cv.domain.schemas import ClipRecord, MetricBundle, PredictionRecord
from traffic_incident_cv.evaluation.metrics import classification_metrics
from traffic_incident_cv.evaluation.reports import write_markdown_report
from traffic_incident_cv.interfaces.base import PipelineInterface
from traffic_incident_cv.models.encoders.convnext_stub import ConvNeXtClipClassifier
from traffic_incident_cv.models.fusion.hybrid import HybridYoloConvNeXtModel
from traffic_incident_cv.models.video.r3d_stub import R3DClipClassifier
from traffic_incident_cv.training.loops import GenericTrainer
from traffic_incident_cv.training.reweighting import balanced_class_weights, count_targets, weighted_sample_weights
from traffic_incident_cv.utils.io import ensure_dir, write_json
from traffic_incident_cv.utils.seed import set_seed

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None

try:
    import torch
    import torch.nn.functional as F
    from torch import nn
    from torch.utils.data import DataLoader, WeightedRandomSampler
except Exception:  # pragma: no cover
    torch = None
    F = None
    nn = None
    DataLoader = None
    WeightedRandomSampler = None


@dataclass
class _TorchTrainingContext:
    device: Any
    train_loader: Any
    val_loader: Any
    optimizer: Any
    criterion: Any
    scaler: Any
    scheduler: Any
    task: str
    amp_enabled: bool
    selection_metric: str
    last_checkpoint_path: Path
    reweighting: dict[str, Any]


@dataclass(frozen=True)
class _ClipResizeTransform:
    input_size: int

    def __call__(self, clip: Any) -> Any:
        if torch is None or F is None:
            return clip
        if isinstance(clip, torch.Tensor):
            tensor = clip.float()
        else:
            tensor = torch.from_numpy(np.asarray(clip)).float()
        if tensor.ndim != 4:
            raise ValueError(f"Expected clip tensor with shape [C, T, H, W], got {tuple(tensor.shape)}")
        frames = tensor.permute(1, 0, 2, 3)
        resized = F.interpolate(frames, size=(self.input_size, self.input_size), mode="bilinear", align_corners=False)
        return resized.permute(1, 0, 2, 3).contiguous()


@dataclass
class AccidentDetectionPipeline(PipelineInterface):
    config: dict
    trainer: GenericTrainer | None = field(default=None, init=False, repr=False)
    latest_metrics: MetricBundle | None = field(default=None, init=False, repr=False)
    latest_output_dir: Path | None = field(default=None, init=False, repr=False)
    torch_context: _TorchTrainingContext | None = field(default=None, init=False, repr=False)

    def fit(self) -> None:
        exp = to_experiment_config(self.config)
        if not self._allow_scaffold_fallback(exp.runtime):
            self._assert_real_training_readiness(exp)
        seed = self._resolve_seed(exp.project, exp.runtime)
        if seed is not None:
            set_seed(seed, deterministic=self._resolve_deterministic(exp.project, exp.runtime))
        output_root = Path(exp.project.get("output_dir", "outputs"))
        output_dir = ensure_dir(output_root / exp.experiment_id)
        manifest_summary = self._collect_manifest_summary(exp.manifest_paths)
        write_json(output_dir / "config.json", exp.to_dict())
        write_json(output_dir / "manifest_summary.json", manifest_summary)

        model = self._build_model(exp.model)
        selection_metric = self._resolve_selection_metric(exp.task, exp.reporting.get("save_best_metric"))
        epoch_runner = self._build_epoch_runner(exp, model, output_dir, selection_metric)

        def _persist_torch_snapshot(target_path: Path, *, epoch_index: int, metrics_payload: dict[str, float]) -> None:
            if self.torch_context is None or torch is None:
                return
            torch.save(
                {
                    "epoch": epoch_index,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": self.torch_context.optimizer.state_dict(),
                    "last_metrics": metrics_payload,
                    "config": exp.to_dict(),
                },
                target_path,
            )

        def _on_epoch_end(state: Any, metrics_payload: dict[str, float]) -> None:
            if self.torch_context is None or torch is None:
                return
            current_epoch = int(metrics_payload.get("epoch", 0) or 0)
            if state.best_epoch == current_epoch:
                _persist_torch_snapshot(
                    output_dir / "model_best.pt",
                    epoch_index=current_epoch,
                    metrics_payload=metrics_payload,
                )

        trainer = GenericTrainer(
            model=model,
            config=self.config,
            output_dir=str(output_dir),
            epoch_runner=epoch_runner,
            on_epoch_end=_on_epoch_end,
            selection_metric=selection_metric,
            maximize_metric=self._should_maximize_metric(selection_metric),
        )
        trainer.run(int(exp.optimization.get("epochs", 1) or 1))
        trainer.save_checkpoint(output_dir / "checkpoint.json")

        if self.torch_context is not None and torch is not None:
            _persist_torch_snapshot(
                self.torch_context.last_checkpoint_path,
                epoch_index=trainer.state.epoch,
                metrics_payload=trainer.state.last_metrics,
            )

        report_metrics = dict(trainer.state.last_metrics)
        report_epoch = trainer.state.epoch
        if trainer.state.best_epoch is not None:
            best_index = int(trainer.state.best_epoch) - 1
            if 0 <= best_index < len(trainer.state.history):
                report_metrics = dict(trainer.state.history[best_index])
                report_epoch = int(trainer.state.best_epoch)

        metrics = MetricBundle(
            metrics=report_metrics,
            metadata={
                "best_metric": trainer.state.best_metric,
                "best_epoch": trainer.state.best_epoch,
                "reported_metrics_epoch": report_epoch,
                "history_length": len(trainer.state.history),
                "last_epoch_metrics": dict(trainer.state.last_metrics),
                "manifest_summary": manifest_summary,
                "training_mode": "torch" if self.torch_context is not None else "scaffold",
                "reweighting": self.torch_context.reweighting if self.torch_context is not None else {},
            },
        )
        write_json(output_dir / "metrics.json", metrics.to_dict())
        write_markdown_report(metrics, str(output_dir), title=f"{exp.experiment_id} report")
        self.trainer = trainer
        self.latest_metrics = metrics
        self.latest_output_dir = output_dir

    def validate(self) -> MetricBundle:
        if self.latest_metrics is None:
            raise RuntimeError("fit() must run before validate().")
        return self.latest_metrics

    def predict(self, items: list[ClipRecord]) -> list[PredictionRecord]:
        return [
            PredictionRecord(
                clip_id=item.clip_id,
                task="accident",
                predicted_label=None,
                score=None,
                metadata={
                    "status": "use_saved_run_inference_for_predictions",
                    "video_path": item.video_path,
                },
            )
            for item in items
        ]

    def export(self, output_dir: str) -> None:
        if self.latest_metrics is None:
            raise RuntimeError("fit() must run before export().")
        export_dir = ensure_dir(output_dir)
        write_json(export_dir / "metrics.json", self.latest_metrics.to_dict())
        if self.trainer is not None:
            self.trainer.save_checkpoint(export_dir / "checkpoint.json")

    def _build_model(self, model_cfg: dict[str, Any]) -> Any:
        family = str(model_cfg.get("family", "convnext_baseline"))
        if family == "convnext_baseline":
            return ConvNeXtClipClassifier(
                name=family,
                num_classes=int(model_cfg.get("num_classes", 2)),
                encoder_name=model_cfg.get("encoder_name", "convnext_tiny"),
                temporal_pooling=model_cfg.get("temporal_pooling", "mean"),
                pretrained=bool(model_cfg.get("pretrained", True)),
                allow_encoder_fallback=bool(model_cfg.get("allow_encoder_fallback", False)),
            )
        if family == "r3d_baseline":
            return R3DClipClassifier(
                name=family,
                num_classes=int(model_cfg.get("num_classes", 2)),
                encoder_name=model_cfg.get("encoder_name", "r3d_18"),
                pretrained=bool(model_cfg.get("pretrained", True)),
                allow_encoder_fallback=bool(model_cfg.get("allow_encoder_fallback", False)),
            )
        if family == "yolo_convnext_hybrid":
            return HybridYoloConvNeXtModel(
                name=family,
                num_classes=int(model_cfg.get("num_classes", 2)),
                fusion_mode=str(model_cfg.get("fusion_mode", "concat")),
                use_global_branch=bool(model_cfg.get("use_global_branch", True)),
                use_local_branch=bool(model_cfg.get("use_local_branch", True)),
                encoder_name=str(model_cfg.get("encoder_name", "convnext_tiny")),
                pretrained=bool(model_cfg.get("pretrained", True)),
                local_crop_top=float(model_cfg.get("local_crop_top", 0.35) or 0.35),
                local_crop_left=float(model_cfg.get("local_crop_left", 0.10) or 0.10),
                local_crop_bottom=float(model_cfg.get("local_crop_bottom", 1.0) or 1.0),
                local_crop_right=float(model_cfg.get("local_crop_right", 0.90) or 0.90),
                global_logit_scale=float(model_cfg.get("global_logit_scale", 1.0) or 1.0),
                local_logit_scale=float(model_cfg.get("local_logit_scale", 1.0) or 1.0),
                use_detector_guidance=bool(model_cfg.get("use_detector_guidance", False)),
                detector_weights=str(model_cfg.get("detector_weights", "yolo11n.pt")),
                detector_conf_threshold=float(model_cfg.get("detector_conf_threshold", 0.25) or 0.25),
                detector_iou_threshold=float(model_cfg.get("detector_iou_threshold", 0.45) or 0.45),
                detector_device=model_cfg.get("detector_device"),
                detector_max_det=int(model_cfg.get("detector_max_det", 20) or 20),
                detector_target_class_ids=tuple(model_cfg.get("detector_target_class_ids", (2, 3, 5, 7)) or ()),
                detector_num_sampled_frames=int(model_cfg.get("detector_num_sampled_frames", 2) or 2),
                detector_expand_ratio=float(model_cfg.get("detector_expand_ratio", 0.12) or 0.12),
                detector_min_crop_ratio=float(model_cfg.get("detector_min_crop_ratio", 0.25) or 0.25),
                detector_require_backend=bool(model_cfg.get("detector_require_backend", False)),
                detector_flow_mode=str(model_cfg.get("detector_flow_mode", "union_roi")),
                detector_object_pooling=str(model_cfg.get("detector_object_pooling", "score_weighted_mean")),
                detector_max_objects_per_sample=int(model_cfg.get("detector_max_objects_per_sample", 3) or 3),
            )
        raise ValueError(f"Unsupported model family={family!r}")

    def _build_epoch_runner(
        self,
        exp: Any,
        model: Any,
        output_dir: Path,
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
            y_score: list[float] = []

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
                y_score.extend(probabilities[:, 1].detach().cpu().tolist())

            mean_loss = total_loss / max(total_items, 1)
            bundle = classification_metrics(y_true, y_pred, y_score=y_score, average="binary")
            metrics = {f"{split}_loss": mean_loss}
            metrics.update({f"{split}_{name}": value for name, value in bundle.metrics.items()})
            if not is_train and context.scheduler is not None:
                context.scheduler.step()
            return metrics

        return run_epoch

    def _build_torch_context(
        self,
        exp: Any,
        model: Any,
        output_dir: Path,
        selection_metric: str,
    ) -> _TorchTrainingContext | None:
        if torch is None or nn is None or DataLoader is None or F is None or np is None:
            return None
        train_manifest_path = exp.manifest_paths.get("train")
        val_manifest_path = exp.manifest_paths.get("val")
        if not train_manifest_path or not val_manifest_path:
            return None

        resolved_train_manifest, train_records = self._prepare_accessible_manifest(
            train_manifest_path,
            output_dir,
            "train",
        )
        resolved_val_manifest, val_records = self._prepare_accessible_manifest(
            val_manifest_path,
            output_dir,
            "val",
        )
        if resolved_train_manifest is None or resolved_val_manifest is None:
            return None
        train_targets = {self._record_target(record) for record in train_records}
        train_targets.discard(None)
        if len(train_targets) < 2:
            return None

        train_transform = self._build_clip_transform(
            int(exp.data.get("input_size", 224) or 224),
            mode="train",
            data_config=exp.data,
        )
        eval_transform = self._build_clip_transform(
            int(exp.data.get("input_size", 224) or 224),
            mode="eval",
            data_config=exp.data,
        )
        train_clip_sampler_mode = str(exp.data.get("train_clip_sampler", "random") or "random")
        eval_clip_sampler_mode = str(exp.data.get("eval_clip_sampler", "uniform") or "uniform")
        dataset_kwargs = {
            "num_frames": int(exp.data.get("clip_num_frames", 16) or 16),
            "stride": int(exp.data.get("clip_stride", 2) or 2),
            "return_metadata_only": False,
            "channel_first": True,
            "normalize": True,
            "as_torch_tensor": True,
        }
        train_dataset = ManifestClipDataset(
            manifest_path=str(resolved_train_manifest),
            sampler=self._resolve_clip_sampler(train_clip_sampler_mode),
            transform=train_transform,
            **dataset_kwargs,
        )
        val_dataset = ManifestClipDataset(
            manifest_path=str(resolved_val_manifest),
            sampler=self._resolve_clip_sampler(eval_clip_sampler_mode),
            transform=eval_transform,
            **dataset_kwargs,
        )
        if len(train_dataset) == 0 or len(val_dataset) == 0:
            return None

        batch_size = int(exp.data.get("batch_size", 8) or 8)
        num_workers = int(exp.runtime.get("num_workers", 0) or 0)
        seed = self._resolve_seed(exp.project, exp.runtime)
        sampler_generator = None
        train_loader_generator = None
        eval_loader_generator = None
        worker_init_fn = self._seed_worker if num_workers > 0 else None
        if seed is not None:
            sampler_generator = torch.Generator()
            sampler_generator.manual_seed(seed)
            train_loader_generator = torch.Generator()
            train_loader_generator.manual_seed(seed + 1)
            eval_loader_generator = torch.Generator()
            eval_loader_generator.manual_seed(seed + 2)
        train_targets = [self._record_target(record) for record in train_records]
        target_histogram = count_targets(train_targets)
        num_classes = int(exp.model.get("num_classes", 2) or 2)
        sampler_mode = str(exp.data.get("train_sampler", "shuffle") or "shuffle").lower()
        train_sampler = None
        if sampler_mode == "weighted" and WeightedRandomSampler is not None:
            sample_weights = weighted_sample_weights(train_targets, num_classes=num_classes)
            if any(weight > 0 for weight in sample_weights):
                train_sampler = WeightedRandomSampler(
                    weights=torch.as_tensor(sample_weights, dtype=torch.double),
                    num_samples=len(sample_weights),
                    replacement=True,
                    generator=sampler_generator,
                )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=train_sampler is None,
            sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=self._collate_batch,
            worker_init_fn=worker_init_fn,
            generator=train_loader_generator,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=self._collate_batch,
            worker_init_fn=worker_init_fn,
            generator=eval_loader_generator,
        )

        device_name = str(exp.runtime.get("device", "cuda"))
        use_cuda = torch.cuda.is_available() and device_name.startswith("cuda")
        device = torch.device("cuda" if use_cuda else "cpu")
        model.build()
        model.to(device)

        class_weight_mode = str(exp.training.get("class_weighting", "none") or "none").lower()
        class_weight_tensor = None
        class_weight_values: list[float] | None = None
        if class_weight_mode == "balanced":
            class_weight_values = balanced_class_weights(train_targets, num_classes=num_classes)
            if any(weight > 0 for weight in class_weight_values):
                class_weight_tensor = torch.as_tensor(class_weight_values, dtype=torch.float32, device=device)
        criterion = nn.CrossEntropyLoss(
            weight=class_weight_tensor,
            label_smoothing=float(exp.training.get("label_smoothing", 0.0) or 0.0),
        )
        optimizer_name = str(exp.optimization.get("optimizer", "adamw")).lower()
        learning_rate = float(exp.optimization.get("lr", 3e-4) or 3e-4)
        weight_decay = float(exp.optimization.get("weight_decay", 1e-4) or 1e-4)
        if optimizer_name == "sgd":
            optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
        else:
            optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = None
        if str(exp.optimization.get("scheduler", "")).lower() == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max(int(exp.optimization.get("epochs", 1) or 1), 1),
            )
        amp_enabled = bool(exp.runtime.get("amp", True)) and device.type == "cuda"
        scaler = torch.amp.GradScaler(device=device.type, enabled=amp_enabled)
        return _TorchTrainingContext(
            device=device,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            criterion=criterion,
            scaler=scaler,
            scheduler=scheduler,
            task=exp.task,
            amp_enabled=amp_enabled,
            selection_metric=selection_metric,
            last_checkpoint_path=output_dir / "model_last.pt",
            reweighting={
                "sampler_mode": sampler_mode,
                "class_weighting": class_weight_mode,
                "train_target_histogram": target_histogram,
                "class_weights": class_weight_values or [],
                "train_clip_sampler": train_clip_sampler_mode,
                "eval_clip_sampler": eval_clip_sampler_mode,
                "seed": seed,
            },
        )

    @staticmethod
    def _allow_scaffold_fallback(runtime_cfg: dict[str, Any]) -> bool:
        return bool(runtime_cfg.get("allow_scaffold_fallback", False))

    def _assert_real_training_readiness(self, exp: Any) -> None:
        issues: list[str] = []
        manifest_records: dict[str, list[ClipRecord]] = {}
        manifest_paths = {
            split: exp.manifest_paths.get(split)
            for split in ("train", "val")
        }
        for split, manifest_path in manifest_paths.items():
            if not manifest_path:
                issues.append(f"experiment.manifest_paths.{split} is not configured")
                continue
            path = Path(manifest_path)
            if not path.exists():
                issues.append(
                    f"{split} manifest not found: {path}. Build manifests with python scripts/prepare_data.py or restore data/manifests/."
                )
                continue
            manifest_records[split] = load_manifest(path)

        if torch is None or nn is None or DataLoader is None or F is None or np is None:
            issues.append(
                "Training dependencies are unavailable. Install requirements-train.txt before launching a real fit()."
            )

        train_manifest_path = manifest_paths.get("train")
        if train_manifest_path and "train" in manifest_records:
            accessible_train_records = [
                record
                for record in manifest_records["train"]
                if ManifestClipDataset.resolve_path_for_record(train_manifest_path, record).exists()
            ]
            if not accessible_train_records:
                issues.append(
                    "No accessible clips were found for the train split. Check dataset mounts, manifest paths, or TRAFFIC_INCIDENT_VOLUME_ROOT."
                )
            distinct_targets = {self._record_target(record) for record in accessible_train_records}
            distinct_targets.discard(None)
            if accessible_train_records and len(distinct_targets) < 2:
                issues.append(
                    f"Accessible train clips cover only {len(distinct_targets)} distinct target class(es). Need at least 2 classes for a meaningful fit()."
                )

        val_manifest_path = manifest_paths.get("val")
        if val_manifest_path and "val" in manifest_records:
            accessible_val_records = [
                record
                for record in manifest_records["val"]
                if ManifestClipDataset.resolve_path_for_record(val_manifest_path, record).exists()
            ]
            if not accessible_val_records:
                issues.append(
                    "No accessible clips were found for the val split. Check dataset mounts, manifest paths, or TRAFFIC_INCIDENT_VOLUME_ROOT."
                )

        if not issues:
            return

        details = "\n".join(f"- {issue}" for issue in issues)
        raise RuntimeError(
            "Unable to start real training because the dataset preflight failed.\n"
            f"{details}\n"
            "Set runtime.allow_scaffold_fallback=true only for scaffold/unit-test runs that intentionally skip real video access."
        )

    @staticmethod
    def _resolve_seed(project_cfg: dict[str, Any], runtime_cfg: dict[str, Any]) -> int | None:
        seed = runtime_cfg.get("seed", project_cfg.get("seed"))
        if seed is None:
            return None
        return int(seed)

    @staticmethod
    def _resolve_deterministic(project_cfg: dict[str, Any], runtime_cfg: dict[str, Any]) -> bool:
        return bool(runtime_cfg.get("deterministic", project_cfg.get("deterministic", True)))

    @staticmethod
    def _seed_worker(_worker_id: int) -> None:
        if torch is None or np is None:
            return
        worker_seed = torch.initial_seed() % (2**32)
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    @staticmethod
    def _resolve_clip_sampler(mode: str | None) -> Any:
        sampler_mode = str(mode or "uniform").lower()
        if sampler_mode == "random":
            return RandomWindowClipSampler()
        if sampler_mode == "tail":
            return TailFocusedSampler()
        if sampler_mode == "uniform":
            return UniformClipSampler()
        raise ValueError(f"Unsupported clip sampler={mode!r}")

    @staticmethod
    def _prepare_accessible_manifest(
        manifest_path: str,
        output_dir: Path,
        split: str,
    ) -> tuple[Path | None, list[ClipRecord]]:
        records = load_manifest(manifest_path)
        accessible_records: list[ClipRecord] = []
        for record in records:
            resolved_path = ManifestClipDataset.resolve_path_for_record(manifest_path, record)
            if not resolved_path.exists():
                continue
            accessible_records.append(
                ClipRecord(
                    clip_id=record.clip_id,
                    video_path=str(resolved_path),
                    dataset=record.dataset,
                    split=record.split,
                    task=record.task,
                    binary_label=record.binary_label,
                    severity_label=record.severity_label,
                    camera_id=record.camera_id,
                    fps=record.fps,
                    duration_sec=record.duration_sec,
                    clip_start_sec=record.clip_start_sec,
                    clip_end_sec=record.clip_end_sec,
                    clip_start_frame=record.clip_start_frame,
                    clip_end_frame=record.clip_end_frame,
                    tags=list(record.tags),
                )
            )
        if not accessible_records:
            return None, []
        resolved_manifest_path = output_dir / f"resolved_{split}.jsonl"
        save_manifest(resolved_manifest_path, accessible_records)
        return resolved_manifest_path, accessible_records

    @staticmethod
    def _build_clip_transform(
        input_size: int,
        *,
        mode: str = "eval",
        data_config: dict[str, Any] | None = None,
    ) -> Callable[[Any], Any]:
        transforms: list[Callable[[Any], Any]] = []
        transforms.extend(build_spatial_preprocessing_transforms(dict((data_config or {}).get("spatial_crop", {}) or {})))
        transforms.append(_ClipResizeTransform(input_size=input_size))
        transforms.extend(build_motion_preprocessing_transforms(dict((data_config or {}).get("frame_difference", {}) or {})))
        if mode == "train":
            augmentation_config = dict((data_config or {}).get("train_augmentations", {}) or {})
            transforms.extend(build_domain_randomization_transforms(augmentation_config))
        if len(transforms) == 1:
            return transforms[0]
        return Compose(transforms=transforms)

    @staticmethod
    def _collate_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
        if torch is None:
            raise RuntimeError("torch is required for collating training batches")
        frames = torch.stack([item["frames"] for item in batch])
        targets = torch.as_tensor([item["target"] for item in batch], dtype=torch.long)
        return {
            "frames": frames,
            "targets": targets,
            "clip_ids": [item["clip_id"] for item in batch],
        }

    @staticmethod
    def _record_target(record: ClipRecord) -> int | None:
        label = record.binary_label.value if hasattr(record.binary_label, "value") else record.binary_label
        if label == "no_accident":
            return 0
        if label == "accident":
            return 1
        return None

    @staticmethod
    def _collect_manifest_summary(manifest_paths: dict[str, str]) -> dict[str, dict[str, Any]]:
        summary: dict[str, dict[str, Any]] = {}
        for split, manifest_path in manifest_paths.items():
            path = Path(manifest_path)
            records = load_manifest(path)
            summary[split] = {
                "manifest_path": str(path),
                "count": len(records),
                "labels": summarize_manifest(records),
            }
        return summary

    @staticmethod
    def _resolve_selection_metric(task: str, configured_metric: str | None) -> str:
        if configured_metric:
            lowered = configured_metric.lower()
            if task == "accident" and lowered == "macro_f1":
                return "f1"
            return configured_metric
        return "f1" if task == "accident" else "macro_f1"

    @staticmethod
    def _should_maximize_metric(metric_name: str) -> bool:
        lowered = metric_name.lower()
        return "loss" not in lowered and "error" not in lowered
