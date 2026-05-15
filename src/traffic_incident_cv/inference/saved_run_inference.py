from __future__ import annotations

import json
import math
import os
import shutil
import statistics
import uuid
from collections import Counter
from html import escape
from pathlib import Path
from typing import Any

from traffic_incident_cv.config import to_experiment_config
from traffic_incident_cv.data.manifest import load_manifest, save_manifest
from traffic_incident_cv.data.video_dataset import ManifestClipDataset, default_video_reader_factory
from traffic_incident_cv.domain.schemas import ClipRecord, ExperimentConfig
from traffic_incident_cv.pipelines.accident import AccidentDetectionPipeline
from traffic_incident_cv.pipelines.severity import SEVERITY_TARGETS, SeverityClassificationPipeline
from traffic_incident_cv.utils.io import ensure_dir, write_json, write_jsonl

try:
    import torch
    from torch.utils.data import DataLoader
except Exception:  # pragma: no cover
    torch = None
    DataLoader = None


ACCIDENT_LABELS = ["no_accident", "accident"]
CURATED_UI_MANIFEST_CANDIDATES = (
    ("demo_data", "ui_examples.json"),
    ("outputs", "ui_test_dataset_bridge_v1_8fps_160", "ui_examples.json"),
)


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _relocate_project_path(raw_path: str | Path | None, project_root: str | Path) -> Path | None:
    if raw_path in (None, ""):
        return None
    project_root = Path(project_root).resolve()
    candidate = Path(raw_path)
    if candidate.exists():
        return candidate.resolve()
    parts = candidate.parts
    for start_index in range(len(parts)):
        suffix = parts[start_index:]
        if not suffix:
            continue
        relocated = project_root.joinpath(*suffix)
        if relocated.exists():
            return relocated.resolve()
    return None


def _find_curated_ui_manifest_path(project_root: str | Path) -> Path | None:
    project_root = Path(project_root).resolve()
    for rel_parts in CURATED_UI_MANIFEST_CANDIDATES:
        curated_manifest = project_root.joinpath(*rel_parts)
        if curated_manifest.exists():
            return curated_manifest
    return None


def _load_curated_ui_manifest(project_root: str | Path) -> dict[str, Any] | None:
    curated_manifest = _find_curated_ui_manifest_path(project_root)
    if curated_manifest is None:
        return None
    try:
        return json.loads(curated_manifest.read_text(encoding="utf-8"))
    except Exception:
        return None


def _to_exp(config: dict[str, Any]) -> Any:
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


def _run_dir_is_usable(path: Path) -> bool:
    return path.is_dir() and (path / "config.json").exists() and (
        (path / "model_best.pt").exists() or (path / "model_last.pt").exists()
    )


def _run_dir_selection_score(path: Path) -> tuple[float, float]:
    metrics_path = path / "metrics.json"
    if not metrics_path.exists():
        return (float("-inf"), float("-inf"))
    try:
        payload = _read_json(metrics_path)
    except Exception:
        return (float("-inf"), float("-inf"))
    metadata = payload.get("metadata") or {}
    metrics = payload.get("metrics") or {}
    best_metric = _safe_float(metadata.get("best_metric"), default=float("-inf"))
    val_macro_f1 = _safe_float(metrics.get("val_macro_f1"), default=float("-inf"))
    return (best_metric, val_macro_f1)


def discover_default_run_dir(project_root: str | Path, task: str) -> Path | None:
    project_root = Path(project_root).resolve()
    env_name = "TRAFFIC_INCIDENT_ACCIDENT_RUN_DIR" if task == "accident" else "TRAFFIC_INCIDENT_SEVERITY_RUN_DIR"

    candidates: list[Path] = []

    env_path = os.getenv(env_name)
    if env_path:
        candidates.append(Path(env_path))

    curated_payload = _load_curated_ui_manifest(project_root)
    if curated_payload:
        curated_key = "accident_run_dir" if task == "accident" else "severity_run_dir"
        curated_candidate = _relocate_project_path(curated_payload.get(curated_key), project_root)
        if curated_candidate is not None:
            candidates.append(curated_candidate)

    if task == "accident":
        candidates.extend(
            [
                project_root / "models" / "hybrid",
                project_root / "models" / "accident_convnext_d02",
                project_root / "models" / "accident_a03r_no_harmonization",
            ]
        )
    elif task == "severity":
        candidates.extend(
            [
                project_root / "models" / "severity",
                project_root / "models" / "severity_b01_baseline",
            ]
        )
    else:
        raise ValueError(f"Unsupported task={task!r}")

    usable_candidates: list[Path] = []
    for candidate in candidates:
        candidate = candidate.resolve()
        if _run_dir_is_usable(candidate):
            usable_candidates.append(candidate)

    if not usable_candidates:
        return None

    if task == "severity":
        usable_candidates.sort(key=_run_dir_selection_score, reverse=True)

    return usable_candidates[0]


def discover_demo_examples(project_root: str | Path) -> list[dict[str, str]]:
    project_root = Path(project_root).resolve()
    curated_manifest = _find_curated_ui_manifest_path(project_root) or (
        project_root / "outputs" / "ui_test_dataset_bridge_v1_8fps_160" / "ui_examples.json"
    )
    curated_payload = _load_curated_ui_manifest(project_root)
    if curated_payload is not None:
        try:
            examples = []
            for item in curated_payload.get("examples", []):
                video_path = _relocate_project_path(item.get("video_path"), project_root)
                if video_path is None:
                    continue
                examples.append(
                    {
                        "title": str(item.get("title") or item.get("case_id") or video_path.stem),
                        "video_path": str(video_path),
                        "source_manifest": str(curated_manifest),
                    }
                )
            if examples:
                return examples
        except Exception:
            pass
    manifest_candidates = [
        (
            "Accident example",
            project_root / "data" / "manifests" / "balanced_grouped_windows_gap96" / "balanced_grouped_windows_test.jsonl",
        ),
        (
            "Severity example",
            project_root / "data" / "manifests" / "balanced_accident_severity_test.jsonl",
        ),
    ]
    examples: list[dict[str, str]] = []
    seen_video_paths: set[str] = set()
    for title, manifest_path in manifest_candidates:
        if not manifest_path.exists():
            continue
        for record in load_manifest(manifest_path):
            resolved = ManifestClipDataset.resolve_path_for_record(manifest_path, record)
            if not resolved.exists():
                continue
            key = str(resolved.resolve())
            if key in seen_video_paths:
                continue
            seen_video_paths.add(key)
            examples.append(
                {
                    "title": title,
                    "video_path": key,
                    "source_manifest": str(manifest_path.resolve()),
                }
            )
            break
    return examples


def _resolve_device(device_name: str) -> Any:
    if torch is None:
        raise RuntimeError("torch is required for inference")
    requested = (device_name or "cuda").lower()
    if requested.startswith("cuda") and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _label_names(task: str) -> list[str]:
    if task == "accident":
        return ACCIDENT_LABELS
    return [label for label, _ in sorted(SEVERITY_TARGETS.items(), key=lambda item: item[1])]


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def _pipeline_for_task(task: str, config: dict[str, Any]) -> Any:
    if task == "accident":
        return AccidentDetectionPipeline(config=config)
    if task == "severity":
        return SeverityClassificationPipeline(config=config)
    raise ValueError(f"Unsupported task={task!r}")


def _load_model(run_dir: Path, task: str, config: dict[str, Any], device: Any) -> Any:
    checkpoint_path = run_dir / "model_best.pt"
    if not checkpoint_path.exists():
        checkpoint_path = run_dir / "model_last.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)
    pipeline = _pipeline_for_task(task, config)
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


def _probe_video(video_path: str | Path) -> tuple[int, float, float]:
    reader = default_video_reader_factory(str(video_path))
    try:
        total_frames = len(reader)
    finally:
        reader.close()
    if total_frames <= 0:
        raise ValueError(f"Video contains no readable frames: {video_path}")
    capture_fps = None
    try:
        import cv2

        capture = cv2.VideoCapture(str(video_path))
        try:
            capture_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        finally:
            capture.release()
    except Exception:  # pragma: no cover
        capture_fps = None
    fps = capture_fps if capture_fps and capture_fps > 0 else 0.0
    duration_sec = float(total_frames / fps) if fps and fps > 0 else 0.0
    return total_frames, fps, duration_sec


def _estimate_window_span_frames(config: dict[str, Any], run_dir: Path, *, minimum: int) -> int:
    exp = _to_exp(config)
    manifest_candidates = [
        exp.manifest_paths.get("train"),
        exp.manifest_paths.get("val"),
        exp.manifest_paths.get("test"),
    ]
    window_lengths: list[int] = []
    for manifest_path in manifest_candidates:
        if not manifest_path:
            continue
        candidate = Path(manifest_path)
        if not candidate.is_absolute():
            candidate = (run_dir / manifest_path).resolve() if (run_dir / manifest_path).exists() else (Path.cwd() / manifest_path).resolve()
        if not candidate.exists():
            continue
        for record in load_manifest(candidate)[:64]:
            if record.clip_start_frame is not None and record.clip_end_frame is not None:
                window_lengths.append(int(record.clip_end_frame) - int(record.clip_start_frame) + 1)
            elif record.clip_start_sec is not None and record.clip_end_sec is not None and record.fps:
                span = max(int(round((float(record.clip_end_sec) - float(record.clip_start_sec)) * float(record.fps))), 1)
                window_lengths.append(span)
        if window_lengths:
            break
    if window_lengths:
        return max(int(statistics.median(window_lengths)), minimum)
    return minimum


def build_sliding_windows(total_frames: int, window_span_frames: int, step_frames: int) -> list[tuple[int, int]]:
    if total_frames <= 0:
        return []
    span = max(int(window_span_frames), 1)
    step = max(int(step_frames), 1)
    if total_frames <= span:
        return [(0, total_frames - 1)]
    windows: list[tuple[int, int]] = []
    start = 0
    while start + span <= total_frames:
        windows.append((start, start + span - 1))
        start += step
    tail_start = max(total_frames - span, 0)
    tail_window = (tail_start, total_frames - 1)
    if not windows or windows[-1] != tail_window:
        windows.append(tail_window)
    deduped: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for window in windows:
        if window in seen:
            continue
        seen.add(window)
        deduped.append(window)
    return deduped


def merge_positive_segments(
    window_rows: list[dict[str, Any]],
    *,
    positive_threshold: float,
    merge_gap_sec: float = 0.0,
) -> list[dict[str, Any]]:
    positive_rows = [
        row
        for row in window_rows
        if float(row.get("accident_score", 0.0) or 0.0) >= positive_threshold
    ]
    positive_rows.sort(key=lambda item: float(item.get("start_sec", 0.0) or 0.0))
    segments: list[dict[str, Any]] = []
    for row in positive_rows:
        severity_label = row.get("severity_label")
        if not segments:
            segments.append(
                {
                    "segment_id": "segment_0001",
                    "start_sec": row["start_sec"],
                    "end_sec": row["end_sec"],
                    "start_frame": row["start_frame"],
                    "end_frame": row["end_frame"],
                    "window_count": 1,
                    "max_accident_score": row["accident_score"],
                    "severity_votes": Counter([severity_label]) if severity_label else Counter(),
                }
            )
            continue
        current = segments[-1]
        if float(row["start_sec"]) <= float(current["end_sec"]) + merge_gap_sec:
            current["end_sec"] = max(float(current["end_sec"]), float(row["end_sec"]))
            current["end_frame"] = max(int(current["end_frame"]), int(row["end_frame"]))
            current["window_count"] = int(current["window_count"]) + 1
            current["max_accident_score"] = max(float(current["max_accident_score"]), float(row["accident_score"]))
            if severity_label:
                current["severity_votes"][severity_label] += 1
            continue
        segment_index = len(segments) + 1
        segments.append(
            {
                "segment_id": f"segment_{segment_index:04d}",
                "start_sec": row["start_sec"],
                "end_sec": row["end_sec"],
                "start_frame": row["start_frame"],
                "end_frame": row["end_frame"],
                "window_count": 1,
                "max_accident_score": row["accident_score"],
                "severity_votes": Counter([severity_label]) if severity_label else Counter(),
            }
        )
    for segment in segments:
        votes = segment.pop("severity_votes")
        segment["severity_label"] = votes.most_common(1)[0][0] if votes else None
        segment["severity_vote_counts"] = dict(votes)
    return segments


def _create_clip_records_for_windows(
    *,
    task: str,
    video_path: str | Path,
    fps: float,
    windows: list[tuple[int, int]],
) -> list[ClipRecord]:
    records: list[ClipRecord] = []
    for index, (start_frame, end_frame) in enumerate(windows):
        start_sec = round(start_frame / fps, 4) if fps and fps > 0 else None
        end_sec = round((end_frame + 1) / fps, 4) if fps and fps > 0 else None
        payload = {
            "clip_id": f"demo_{task}_{index:04d}",
            "video_path": str(Path(video_path).resolve()),
            "dataset": "demo_upload",
            "split": "demo",
            "task": task,
            "camera_id": "demo",
            "fps": fps or None,
            "duration_sec": round((end_frame - start_frame + 1) / fps, 4) if fps and fps > 0 else None,
            "clip_start_sec": start_sec,
            "clip_end_sec": end_sec,
            "clip_start_frame": start_frame,
            "clip_end_frame": end_frame,
            "tags": ["source:demo_ui"],
        }
        if task == "accident":
            payload["binary_label"] = "no_accident"
        else:
            payload["severity_label"] = "minor"
        records.append(ClipRecord(**payload))
    return records


def _build_loader_for_records(
    *,
    run_dir: Path,
    task: str,
    config: dict[str, Any],
    records: list[ClipRecord],
    temp_dir: Path,
) -> tuple[Any, Path]:
    if DataLoader is None:
        raise RuntimeError("torch DataLoader is required for inference")
    exp = _to_exp(config)
    pipeline = _pipeline_for_task(task, config)
    manifest_path = temp_dir / f"demo_{task}_manifest.jsonl"
    save_manifest(manifest_path, records)
    eval_sampler_mode = str(exp.data.get("eval_clip_sampler", "uniform") or "uniform")
    dataset = ManifestClipDataset(
        manifest_path=str(manifest_path),
        sampler=pipeline._resolve_clip_sampler(eval_sampler_mode),
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
        num_workers=0,
        pin_memory=True,
        collate_fn=pipeline._collate_batch,
    )
    return loader, manifest_path


def infer_records_for_run(
    *,
    run_dir: str | Path,
    task: str,
    records: list[ClipRecord],
    device_name: str = "cuda",
    accident_threshold: float = 0.5,
    temp_dir: str | Path | None = None,
) -> dict[str, Any]:
    if torch is None:
        raise RuntimeError("torch is required for inference")
    if not records:
        raise ValueError("No records provided for inference")
    run_path = Path(run_dir).resolve()
    config = _read_json(run_path / "config.json")
    labels = _label_names(task)
    device = _resolve_device(device_name)
    work_dir = ensure_dir(temp_dir or (run_path / "_demo_temp" / uuid.uuid4().hex))
    loader, manifest_path = _build_loader_for_records(
        run_dir=run_path,
        task=task,
        config=config,
        records=records,
        temp_dir=work_dir,
    )
    model = _load_model(run_path, task, config, device)
    predictions: list[dict[str, Any]] = []
    record_lookup = {record.clip_id: record for record in records}
    with torch.inference_mode():
        for batch in loader:
            frames = batch["frames"].to(device, non_blocking=True)
            logits = model(frames)
            probabilities = torch.softmax(logits, dim=1).detach().cpu().tolist()
            for clip_id, probability_vector in zip(batch["clip_ids"], probabilities, strict=True):
                record = record_lookup[clip_id]
                best_index = max(range(len(probability_vector)), key=lambda index: probability_vector[index])
                label = labels[best_index]
                row = {
                    "clip_id": clip_id,
                    "video_path": record.video_path,
                    "start_frame": record.clip_start_frame,
                    "end_frame": record.clip_end_frame,
                    "start_sec": record.clip_start_sec,
                    "end_sec": record.clip_end_sec,
                    "probabilities": {
                        label_name: round(float(probability_vector[index]), 6)
                        for index, label_name in enumerate(labels)
                    },
                }
                if task == "accident":
                    accident_score = float(probability_vector[1])
                    row["accident_score"] = round(accident_score, 6)
                    row["accident_label"] = "accident" if accident_score >= accident_threshold else "no_accident"
                else:
                    row["severity_score"] = round(float(probability_vector[best_index]), 6)
                    row["severity_label"] = label
                predictions.append(row)
    if manifest_path.exists():
        manifest_path.unlink()
    return {
        "run_dir": str(run_path),
        "task": task,
        "device": str(device),
        "records": predictions,
        "labels": labels,
    }


def infer_windows_for_run(
    *,
    run_dir: str | Path,
    task: str,
    video_path: str | Path,
    device_name: str = "cuda",
    accident_threshold: float = 0.5,
    window_span_frames: int | None = None,
    step_frames: int | None = None,
    max_windows: int | None = None,
    temp_dir: str | Path | None = None,
) -> dict[str, Any]:
    run_path = Path(run_dir).resolve()
    config = _read_json(run_path / "config.json")
    exp = _to_exp(config)
    total_frames, fps, duration_sec = _probe_video(video_path)
    minimum_span = max(int(exp.data.get("clip_num_frames", 16) or 16) * max(int(exp.data.get("clip_stride", 2) or 2), 1), 1)
    inferred_span = _estimate_window_span_frames(config, run_path, minimum=minimum_span)
    effective_span = int(window_span_frames or inferred_span)
    effective_step = int(step_frames or max(effective_span // 2, 1))
    windows = build_sliding_windows(total_frames, effective_span, effective_step)
    if max_windows is not None:
        windows = windows[: max(int(max_windows), 1)]
    records = _create_clip_records_for_windows(
        task=task,
        video_path=video_path,
        fps=fps,
        windows=windows,
    )
    payload = infer_records_for_run(
        run_dir=run_path,
        task=task,
        records=records,
        device_name=device_name,
        accident_threshold=accident_threshold,
        temp_dir=temp_dir,
    )
    payload["video"] = {
        "path": str(Path(video_path).resolve()),
        "frame_count": total_frames,
        "fps": fps,
        "duration_sec": duration_sec,
    }
    payload["windowing"] = {
        "window_span_frames": effective_span,
        "step_frames": effective_step,
        "window_count": len(records),
    }
    return payload


def assess_video_cascade(
    *,
    video_path: str | Path,
    accident_run_dir: str | Path,
    severity_run_dir: str | Path | None = None,
    device_name: str = "cuda",
    accident_threshold: float = 0.5,
    window_span_frames: int | None = None,
    step_frames: int | None = None,
    max_windows: int | None = None,
    temp_dir: str | Path | None = None,
) -> dict[str, Any]:
    accident_payload = infer_windows_for_run(
        run_dir=accident_run_dir,
        task="accident",
        video_path=video_path,
        device_name=device_name,
        accident_threshold=accident_threshold,
        window_span_frames=window_span_frames,
        step_frames=step_frames,
        max_windows=max_windows,
        temp_dir=temp_dir,
    )
    accident_rows = accident_payload["records"]
    severity_rows_by_clip_id: dict[str, dict[str, Any]] = {}
    severity_payload: dict[str, Any] | None = None
    if severity_run_dir:
        positive_windows = [
            (row["start_frame"], row["end_frame"])
            for row in accident_rows
            if row["accident_label"] == "accident"
        ]
        if positive_windows:
            total_frames = int(accident_payload["video"]["frame_count"])
            fps = float(accident_payload["video"]["fps"] or 0.0)
            severity_records = _create_clip_records_for_windows(
                task="severity",
                video_path=video_path,
                fps=fps,
                windows=positive_windows,
            )
            severity_payload = infer_records_for_run(
                run_dir=severity_run_dir,
                task="severity",
                records=severity_records,
                device_name=device_name,
                accident_threshold=accident_threshold,
                temp_dir=temp_dir,
            )
            severity_rows_by_clip_id = {row["clip_id"]: row for row in severity_payload["records"]}
            # The severity clip ids are re-generated from positive windows in the same order.
            severity_iter = iter(severity_payload["records"])
            for row in accident_rows:
                if row["accident_label"] != "accident":
                    row["severity_label"] = None
                    row["severity_probabilities"] = {}
                    continue
                severity_row = next(severity_iter, None)
                if severity_row is None:
                    row["severity_label"] = None
                    row["severity_probabilities"] = {}
                    continue
                row["severity_label"] = severity_row["severity_label"]
                row["severity_probabilities"] = severity_row["probabilities"]
        else:
            for row in accident_rows:
                row["severity_label"] = None
                row["severity_probabilities"] = {}
    else:
        for row in accident_rows:
            row["severity_label"] = None
            row["severity_probabilities"] = {}

    segments = merge_positive_segments(accident_rows, positive_threshold=accident_threshold)
    positive_rows = [row for row in accident_rows if row["accident_label"] == "accident"]
    severity_counter = Counter(row["severity_label"] for row in positive_rows if row.get("severity_label"))
    mean_score = statistics.mean(_safe_float(row.get("accident_score")) for row in accident_rows) if accident_rows else 0.0
    median_score = statistics.median(_safe_float(row.get("accident_score")) for row in accident_rows) if accident_rows else 0.0
    positive_coverage_sec = sum(
        max(_safe_float(segment.get("end_sec")) - _safe_float(segment.get("start_sec")), 0.0)
        for segment in segments
    )
    first_positive_sec = min((_safe_float(segment.get("start_sec")) for segment in segments), default=None)
    summary = {
        "video_path": str(Path(video_path).resolve()),
        "frame_count": accident_payload["video"]["frame_count"],
        "fps": accident_payload["video"]["fps"],
        "duration_sec": accident_payload["video"]["duration_sec"],
        "window_count": len(accident_rows),
        "positive_window_count": len(positive_rows),
        "positive_window_ratio": round(len(positive_rows) / max(len(accident_rows), 1), 4),
        "segment_count": len(segments),
        "positive_coverage_sec": round(positive_coverage_sec, 4),
        "first_positive_sec": None if first_positive_sec is None else round(first_positive_sec, 4),
        "max_accident_score": max((float(row["accident_score"]) for row in accident_rows), default=0.0),
        "mean_accident_score": round(mean_score, 6),
        "median_accident_score": round(median_score, 6),
        "overall_accident_label": "accident" if positive_rows else "no_accident",
        "dominant_severity_label": severity_counter.most_common(1)[0][0] if severity_counter else None,
        "accident_threshold": accident_threshold,
    }
    return {
        "summary": summary,
        "accident": accident_payload,
        "severity": severity_payload,
        "window_predictions": accident_rows,
        "segments": segments,
        "severity_vote_counts": dict(severity_counter),
    }


def _format_float(value: Any, digits: int = 4) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def _render_window_timeline_svg(
    window_rows: list[dict[str, Any]],
    *,
    segments: list[dict[str, Any]],
    threshold: float,
    duration_sec: float,
    width: int = 1120,
    height: int = 270,
) -> str:
    if not window_rows:
        return "<div class='empty-viz'>Not enough data to build the timeline.</div>"
    chart_left = 72
    chart_top = 26
    chart_width = width - 120
    chart_height = 164
    score_base_y = chart_top + chart_height
    duration = max(duration_sec, max(_safe_float(row.get("end_sec")) for row in window_rows), 1.0)

    def x_pos(sec: float) -> float:
        return chart_left + (_safe_float(sec) / duration) * chart_width

    def y_pos(score: float) -> float:
        bounded = max(0.0, min(_safe_float(score), 1.0))
        return score_base_y - bounded * chart_height

    grid = []
    labels = []
    for tick in range(6):
        ratio = tick / 5
        y = chart_top + chart_height * ratio
        value = 1.0 - ratio
        grid.append(
            f"<line x1='{chart_left}' y1='{y:.2f}' x2='{chart_left + chart_width}' y2='{y:.2f}' stroke='rgba(31,26,21,0.10)' stroke-width='1' />"
        )
        labels.append(
            f"<text x='{chart_left - 12}' y='{y + 4:.2f}' text-anchor='end' font-size='11' fill='#6f665c'>{value:.1f}</text>"
        )
    for tick in range(7):
        sec = duration * tick / 6
        x = x_pos(sec)
        grid.append(
            f"<line x1='{x:.2f}' y1='{chart_top}' x2='{x:.2f}' y2='{score_base_y}' stroke='rgba(31,26,21,0.08)' stroke-width='1' />"
        )
        labels.append(
            f"<text x='{x:.2f}' y='{score_base_y + 18:.2f}' text-anchor='middle' font-size='11' fill='#6f665c'>{sec:.1f}s</text>"
        )
    threshold_y = y_pos(threshold)
    threshold_line = (
        f"<line x1='{chart_left}' y1='{threshold_y:.2f}' x2='{chart_left + chart_width}' y2='{threshold_y:.2f}' stroke='#a03c22' stroke-width='2' stroke-dasharray='6 6' />"
        f"<text x='{chart_left + chart_width - 4:.2f}' y='{threshold_y - 8:.2f}' text-anchor='end' font-size='11' fill='#a03c22'>threshold {threshold:.2f}</text>"
    )
    line_points = []
    area_points = [f"{chart_left},{score_base_y}"]
    dots = []
    for row in window_rows:
        mid_sec = (_safe_float(row.get("start_sec")) + _safe_float(row.get("end_sec"))) / 2.0
        score = _safe_float(row.get("accident_score"))
        x = x_pos(mid_sec)
        y = y_pos(score)
        line_points.append(f"{x:.2f},{y:.2f}")
        area_points.append(f"{x:.2f},{y:.2f}")
        fill = "#a03c22" if score >= threshold else "#0d6b55"
        dots.append(f"<circle cx='{x:.2f}' cy='{y:.2f}' r='5' fill='{fill}' stroke='white' stroke-width='2' />")
    area_points.append(f"{chart_left + chart_width},{score_base_y}")

    band_y = score_base_y + 32
    bands = []
    band_labels = []
    for segment in segments:
        start_x = x_pos(_safe_float(segment.get("start_sec")))
        end_x = x_pos(_safe_float(segment.get("end_sec")))
        band_width = max(end_x - start_x, 4.0)
        severity = str(segment.get("severity_label") or "detected")
        color = {"major": "#a03c22", "moderate": "#c17e17", "minor": "#0d6b55"}.get(severity, "#0d6b55")
        bands.append(
            f"<rect x='{start_x:.2f}' y='{band_y:.2f}' width='{band_width:.2f}' height='16' rx='8' fill='{color}' fill-opacity='0.92' />"
        )
        band_labels.append(
            f"<text x='{start_x + band_width / 2:.2f}' y='{band_y + 12:.2f}' text-anchor='middle' font-size='10' fill='white'>{escape(severity)}</text>"
        )

    return f"""
    <svg viewBox="0 0 {width} {height}" role="img" aria-label="Score timeline by window">
      <defs>
        <linearGradient id="scoreGradient" x1="0" x2="0" y1="0" y2="1">
          <stop offset="0%" stop-color="#0d6b55" stop-opacity="0.34"></stop>
          <stop offset="100%" stop-color="#0d6b55" stop-opacity="0.03"></stop>
        </linearGradient>
      </defs>
      <rect x="0" y="0" width="{width}" height="{height}" rx="24" fill="rgba(255,255,255,0.72)"></rect>
      {''.join(grid)}
      {''.join(labels)}
      {threshold_line}
      <polygon points="{' '.join(area_points)}" fill="url(#scoreGradient)"></polygon>
      <polyline fill="none" stroke="#0d6b55" stroke-width="3" points="{' '.join(line_points)}"></polyline>
      {''.join(dots)}
      {''.join(bands)}
      {''.join(band_labels)}
      <text x="{chart_left}" y="{band_y + 38:.2f}" font-size="11" fill="#6f665c">Annotation segments</text>
    </svg>
    """


def _render_severity_votes_svg(vote_counts: dict[str, int], *, width: int = 420, height: int = 210) -> str:
    if not vote_counts:
        return "<div class='empty-viz'>No positive windows were found, so the severity chart is unavailable.</div>"
    labels = [label for label, _ in sorted(vote_counts.items(), key=lambda item: (-item[1], item[0]))]
    max_value = max(vote_counts.values())
    chart_left = 96
    chart_top = 30
    bar_height = 28
    bar_gap = 18
    bars = []
    for index, label in enumerate(labels):
        value = int(vote_counts[label])
        width_ratio = value / max_value if max_value else 0.0
        bar_width = width_ratio * (width - chart_left - 36)
        y = chart_top + index * (bar_height + bar_gap)
        color = {"major": "#a03c22", "moderate": "#c17e17", "minor": "#0d6b55"}.get(label, "#0d6b55")
        bars.append(
            f"<text x='18' y='{y + 19:.2f}' font-size='13' fill='#6f665c'>{escape(label)}</text>"
            f"<rect x='{chart_left}' y='{y:.2f}' width='{bar_width:.2f}' height='{bar_height}' rx='14' fill='{color}' />"
            f"<text x='{chart_left + bar_width + 10:.2f}' y='{y + 19:.2f}' font-size='13' fill='#1c1814'>{value}</text>"
        )
    return f"""
    <svg viewBox="0 0 {width} {height}" role="img" aria-label="Severity vote counts">
      <rect x="0" y="0" width="{width}" height="{height}" rx="22" fill="rgba(255,255,255,0.72)"></rect>
      <text x="18" y="22" font-size="12" fill="#6f665c" style="letter-spacing:0.14em;text-transform:uppercase;">Severity votes</text>
      {''.join(bars)}
    </svg>
    """


def _top_windows(window_rows: list[dict[str, Any]], *, limit: int = 5) -> list[dict[str, Any]]:
    ordered = sorted(window_rows, key=lambda item: _safe_float(item.get("accident_score")), reverse=True)
    return ordered[:limit]


def _render_assessment_html(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    window_rows = payload["window_predictions"]
    segment_rows = payload["segments"]
    severity_votes = payload.get("severity_vote_counts", {})
    video_path = escape(str(summary["video_path"]))
    cards = [
        ("Overall accident label", summary["overall_accident_label"]),
        ("Max accident score", _format_float(summary["max_accident_score"])),
        ("Positive windows", str(summary["positive_window_count"])),
        ("Dominant severity", summary.get("dominant_severity_label") or "-"),
        ("Mean score", _format_float(summary.get("mean_accident_score"))),
        ("Median score", _format_float(summary.get("median_accident_score"))),
        ("Segments", str(summary.get("segment_count", 0))),
        ("First positive, sec", _format_float(summary.get("first_positive_sec"))),
        ("Coverage, sec", _format_float(summary.get("positive_coverage_sec"))),
        ("Duration, sec", _format_float(summary["duration_sec"])),
        ("FPS", _format_float(summary["fps"], digits=2)),
    ]
    card_html = "".join(
        f"<div class='card'><div class='label'>{escape(label)}</div><div class='value'>{escape(value)}</div></div>"
        for label, value in cards
    )
    segment_html = "".join(
        (
            "<tr>"
            f"<td>{escape(item['segment_id'])}</td>"
            f"<td>{_format_float(item['start_sec'])}</td>"
            f"<td>{_format_float(item['end_sec'])}</td>"
            f"<td>{escape(str(item.get('severity_label') or '—'))}</td>"
            f"<td>{_format_float(item.get('max_accident_score'))}</td>"
            f"<td>{escape(str(item.get('window_count')))}</td>"
            "</tr>"
        )
        for item in segment_rows
    ) or "<tr><td colspan='6'>No segments detected</td></tr>"
    window_html = "".join(
        (
            "<tr>"
            f"<td>{escape(item['clip_id'])}</td>"
            f"<td>{_format_float(item['start_sec'])}</td>"
            f"<td>{_format_float(item['end_sec'])}</td>"
            f"<td>{_format_float(item.get('accident_score'))}</td>"
            f"<td>{escape(str(item.get('accident_label') or '—'))}</td>"
            f"<td>{escape(str(item.get('severity_label') or '—'))}</td>"
            "</tr>"
        )
        for item in window_rows
    )
    timeline_svg = _render_window_timeline_svg(
        window_rows,
        segments=segment_rows,
        threshold=_safe_float(summary.get("accident_threshold"), 0.5),
        duration_sec=_safe_float(summary.get("duration_sec"), 0.0),
    )
    severity_svg = _render_severity_votes_svg(severity_votes)
    top_windows_html = "".join(
        (
            "<tr>"
            f"<td>{escape(item['clip_id'])}</td>"
            f"<td>{_format_float(item.get('start_sec'))}</td>"
            f"<td>{_format_float(item.get('end_sec'))}</td>"
            f"<td>{_format_float(item.get('accident_score'))}</td>"
            f"<td>{escape(str(item.get('severity_label') or '—'))}</td>"
            "</tr>"
        )
        for item in _top_windows(window_rows)
    ) or "<tr><td colspan='5'>No windows available</td></tr>"
    raw_json = escape(json.dumps(payload, ensure_ascii=False, indent=2))
    return f"""<!doctype html>
    <html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Traffic Incident CV Demo Report</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f3eee5;
      --panel: rgba(255,255,255,0.82);
      --line: rgba(31, 28, 26, 0.12);
      --ink: #1c1814;
      --muted: #6f665c;
      --accent: #0d6b55;
      --warn: #9a3b1f;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Segoe UI", "Inter", sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(13,107,85,0.10), transparent 32%),
        radial-gradient(circle at top right, rgba(154,59,31,0.10), transparent 28%),
        linear-gradient(180deg, #f8f4ed 0%, var(--bg) 100%);
    }}
    .wrap {{ max-width: 1240px; margin: 0 auto; padding: 28px; }}
    .hero, .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 28px;
      box-shadow: 0 16px 42px rgba(44, 35, 24, 0.10);
      backdrop-filter: blur(18px);
    }}
    .hero {{ padding: 28px; margin-bottom: 20px; }}
    .hero h1 {{ margin: 0 0 10px; font-size: 42px; line-height: 1.05; }}
    .hero p {{ margin: 6px 0; color: var(--muted); }}
    .cards {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 14px;
      margin-top: 18px;
    }}
    .card {{
      border: 1px solid var(--line);
      border-radius: 22px;
      padding: 16px 18px;
      background: rgba(255,255,255,0.72);
    }}
    .label {{
      font-size: 12px;
      letter-spacing: 0.14em;
      text-transform: uppercase;
      color: var(--muted);
      margin-bottom: 10px;
    }}
    .value {{
      font-size: 28px;
      font-weight: 700;
      line-height: 1.1;
    }}
    .panel {{ padding: 22px; margin-bottom: 18px; }}
    .viz-grid {{
      display: grid;
      grid-template-columns: minmax(0, 2fr) minmax(320px, 1fr);
      gap: 18px;
      align-items: start;
    }}
    .viz-card {{
      border: 1px solid var(--line);
      border-radius: 24px;
      background: rgba(255,255,255,0.70);
      padding: 14px;
    }}
    .viz-card svg {{
      display: block;
      width: 100%;
      height: auto;
    }}
    .empty-viz {{
      padding: 18px;
      color: var(--muted);
      font-size: 14px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 14px;
    }}
    th, td {{
      padding: 12px 10px;
      border-bottom: 1px solid var(--line);
      text-align: left;
      vertical-align: top;
    }}
    th {{
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.12em;
      color: var(--muted);
    }}
    .code {{
      overflow: auto;
      white-space: pre-wrap;
      font-family: ui-monospace, SFMono-Regular, Consolas, monospace;
      font-size: 12px;
      background: rgba(28,24,20,0.06);
      border-radius: 18px;
      padding: 16px;
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <section class="hero">
      <div class="label">Demo report</div>
      <h1>Video assessment and auto-annotation</h1>
      <p>{video_path}</p>
      <p>This report is built from the current cascade: accident detection first, then severity on positive windows.</p>
      <div class="cards">{card_html}</div>
    </section>
    <section class="panel">
      <div class="label">Visual summary</div>
      <div class="viz-grid">
        <div class="viz-card">{timeline_svg}</div>
        <div class="viz-card">{severity_svg}</div>
      </div>
    </section>
    <section class="panel">
      <div class="label">Annotation segments</div>
      <table>
        <thead>
          <tr>
            <th>Segment</th>
            <th>Start, sec</th>
            <th>End, sec</th>
            <th>Severity</th>
            <th>Max score</th>
            <th>Windows</th>
          </tr>
        </thead>
        <tbody>{segment_html}</tbody>
      </table>
    </section>
    <section class="panel">
      <div class="label">Top-scoring windows</div>
      <table>
        <thead>
          <tr>
            <th>Window</th>
            <th>Start, sec</th>
            <th>End, sec</th>
            <th>Accident score</th>
            <th>Severity</th>
          </tr>
        </thead>
        <tbody>{top_windows_html}</tbody>
      </table>
    </section>
    <section class="panel">
      <div class="label">Window predictions</div>
      <table>
        <thead>
          <tr>
            <th>Window</th>
            <th>Start, sec</th>
            <th>End, sec</th>
            <th>Accident score</th>
            <th>Accident label</th>
            <th>Severity</th>
          </tr>
        </thead>
        <tbody>{window_html}</tbody>
      </table>
    </section>
    <section class="panel">
      <div class="label">Raw JSON</div>
      <div class="code">{raw_json}</div>
    </section>
  </div>
</body>
</html>
"""


def write_assessment_bundle(
    *,
    output_dir: str | Path,
    payload: dict[str, Any],
    uploaded_video_path: str | Path | None = None,
) -> dict[str, str]:
    output_dir = ensure_dir(output_dir)
    assessment_path = output_dir / "assessment.json"
    annotations_path = output_dir / "annotations.jsonl"
    windows_path = output_dir / "window_predictions.jsonl"
    report_path = output_dir / "report.html"
    write_json(assessment_path, payload)
    write_jsonl(annotations_path, payload["segments"])
    write_jsonl(windows_path, payload["window_predictions"])
    report_path.write_text(_render_assessment_html(payload), encoding="utf-8")
    copied_video_path = None
    if uploaded_video_path is not None:
        source = Path(uploaded_video_path)
        target = output_dir / source.name
        if source.resolve() != target.resolve():
            shutil.copy2(source, target)
        copied_video_path = str(target.resolve())
    return {
        "assessment_json": str(assessment_path.resolve()),
        "annotations_jsonl": str(annotations_path.resolve()),
        "window_predictions_jsonl": str(windows_path.resolve()),
        "report_html": str(report_path.resolve()),
        "copied_video_path": copied_video_path,
    }
