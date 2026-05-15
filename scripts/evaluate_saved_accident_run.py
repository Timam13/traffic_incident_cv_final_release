from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from _bootstrap import ensure_src_path

ensure_src_path()

from traffic_incident_cv.config import to_experiment_config
from traffic_incident_cv.data.clip_sampling import UniformClipSampler
from traffic_incident_cv.data.video_dataset import ManifestClipDataset
from traffic_incident_cv.domain.schemas import ExperimentConfig
from traffic_incident_cv.evaluation.accident_post_eval import (
    build_prediction_rows,
    export_accident_post_eval_bundle,
    threshold_sweep,
)
from traffic_incident_cv.evaluation.metrics import classification_metrics
from traffic_incident_cv.pipelines.accident import AccidentDetectionPipeline

try:
    import torch
    from torch.utils.data import DataLoader
except Exception:  # pragma: no cover
    torch = None
    DataLoader = None


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _to_exp(config: dict[str, Any]) -> ExperimentConfig:
    if "experiment_id" in config:
        return ExperimentConfig(
            experiment_id=config["experiment_id"],
            task=config["task"],
            manifest_paths=dict(config.get("manifest_paths", {})),
            model=dict(config.get("model", {})),
            training=dict(config.get("training", {})),
            runtime=dict(config.get("runtime", {})),
            data=dict(config.get("data", {})),
            project=dict(config.get("project", {})),
            evaluation=dict(config.get("evaluation", {})),
            optimization=dict(config.get("optimization", {})),
            reporting=dict(config.get("reporting", {})),
        )
    return to_experiment_config(config)


def _resolve_device(device_name: str) -> Any:
    if torch is None:
        raise RuntimeError("torch is required for saved-run evaluation")
    requested = (device_name or "cuda").lower()
    if requested.startswith("cuda") and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _resolve_manifest_path(run_dir: Path, manifest_path: str, output_dir_name: str) -> Path:
    path = Path(manifest_path)
    if path.is_absolute() and path.exists():
        return path

    candidates: list[Path] = []
    if path.is_absolute():
        candidates.append(path)
    else:
        candidates.extend(
            [
                Path.cwd() / path,
                run_dir / path,
                run_dir.parent / path,
            ]
        )
        for parent in run_dir.parents:
            if parent.name == output_dir_name:
                candidates.insert(0, parent.parent / path)
                break
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve() if candidates else path.resolve()


def _load_model(run_dir: Path, config: dict[str, Any], device: Any) -> Any:
    best_checkpoint_path = run_dir / "model_best.pt"
    checkpoint_path = best_checkpoint_path if best_checkpoint_path.exists() else run_dir / "model_last.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)

    pipeline = AccidentDetectionPipeline(config=config)
    exp = _to_exp(config)
    model = pipeline._build_model(exp.model)
    model.build()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict")
    if not state_dict:
        raise RuntimeError(f"Checkpoint {checkpoint_path} does not contain model_state_dict")
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def _build_loader(run_dir: Path, config: dict[str, Any], split: str) -> tuple[Any, list[Any], Path]:
    if DataLoader is None:
        raise RuntimeError("torch DataLoader is required for saved-run evaluation")
    exp = _to_exp(config)
    manifest_path = exp.manifest_paths.get(split)
    if not manifest_path:
        raise KeyError(f"Split {split!r} is not configured for this experiment")
    resolved_manifest_source = _resolve_manifest_path(
        run_dir,
        manifest_path,
        str(exp.project.get("output_dir", "outputs") or "outputs"),
    )

    pipeline = AccidentDetectionPipeline(config=config)
    resolved_manifest_path, records = pipeline._prepare_accessible_manifest(
        str(resolved_manifest_source),
        run_dir,
        split,
    )
    if resolved_manifest_path is None or not records:
        raise RuntimeError(f"No accessible records found for split={split}")

    dataset = ManifestClipDataset(
        manifest_path=str(resolved_manifest_path),
        sampler=UniformClipSampler(),
        num_frames=int(exp.data.get("clip_num_frames", 16) or 16),
        stride=int(exp.data.get("clip_stride", 2) or 2),
        return_metadata_only=False,
        transform=pipeline._build_clip_transform(
            int(exp.data.get("input_size", 224) or 224),
            mode="eval",
            data_config=exp.data,
        ),
        channel_first=True,
        normalize=True,
        as_torch_tensor=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(exp.data.get("batch_size", 8) or 8),
        shuffle=False,
        num_workers=int(exp.runtime.get("num_workers", 0) or 0),
        pin_memory=True,
        collate_fn=pipeline._collate_batch,
    )
    return loader, records, resolved_manifest_path


def evaluate_saved_run(
    *,
    run_dir: str | Path,
    split: str = "test",
    threshold: float = 0.5,
    device_name: str = "cuda",
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    if torch is None:
        raise RuntimeError("torch is required for saved-run evaluation")

    run_path = Path(run_dir).resolve()
    config = _read_json(run_path / "config.json")
    device = _resolve_device(device_name)
    loader, records, resolved_manifest_path = _build_loader(run_path, config, split)
    model = _load_model(run_path, config, device)

    y_true: list[int] = []
    y_score: list[float] = []
    clip_ids: list[str] = []

    with torch.inference_mode():
        for batch in loader:
            frames = batch["frames"].to(device, non_blocking=True)
            logits = model(frames)
            probabilities = torch.softmax(logits, dim=1)
            clip_ids.extend(batch["clip_ids"])
            y_score.extend(probabilities[:, 1].detach().cpu().tolist())
            y_true.extend(batch["targets"].detach().cpu().tolist())

    y_pred = [1 if score >= threshold else 0 for score in y_score]
    bundle = classification_metrics(y_true, y_pred, y_score=y_score, average="binary")
    sweep_payload = threshold_sweep(y_true, y_score)
    predictions = build_prediction_rows(
        clip_ids=clip_ids,
        y_true=y_true,
        y_score=y_score,
        threshold=threshold,
    )

    metrics_payload = {
        "metrics": bundle.metrics,
        "metadata": {
            **bundle.metadata,
            "run_dir": str(run_path),
            "split": split,
            "threshold": threshold,
            "device": str(device),
            "resolved_manifest_path": str(resolved_manifest_path),
            "record_count": len(records),
            "positive_count": int(sum(1 for item in y_true if int(item) == 1)),
            "negative_count": int(sum(1 for item in y_true if int(item) == 0)),
            "best_threshold_from_sweep": sweep_payload.get("best", {}).get("threshold"),
        },
    }
    export_payload = export_accident_post_eval_bundle(
        output_dir=output_dir or (run_path / f"post_eval_{split}"),
        split=split,
        threshold=threshold,
        metrics_payload=metrics_payload,
        sweep_payload=sweep_payload,
        predictions=predictions,
    )
    return {
        "run_dir": str(run_path),
        "split": split,
        "threshold": threshold,
        "metrics": bundle.metrics,
        "metadata": metrics_payload["metadata"],
        **export_payload,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate saved accident run on a requested split.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = evaluate_saved_run(
        run_dir=args.run_dir,
        split=args.split,
        threshold=args.threshold,
        device_name=args.device,
        output_dir=args.output_dir,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
