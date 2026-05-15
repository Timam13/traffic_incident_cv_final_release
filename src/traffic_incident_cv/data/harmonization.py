from __future__ import annotations

import json
import math
import os
import shutil
import statistics
import subprocess
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from traffic_incident_cv.data.manifest import load_manifest, save_manifest
from traffic_incident_cv.data.video_dataset import ManifestClipDataset
from traffic_incident_cv.domain.schemas import ClipRecord
from traffic_incident_cv.utils.io import ensure_dir, write_json

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None


@dataclass(slots=True)
class HarmonizationProfile:
    name: str
    target_fps: float
    target_width: int
    target_height: int
    video_codec: str = "libx264"
    pix_fmt: str = "yuv420p"
    preset: str = "veryfast"
    crf: int = 28
    keep_audio: bool = False
    resize_mode: str = "pad"
    scale_flags: str = "lanczos"

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "HarmonizationProfile":
        return cls(
            name=str(payload["name"]),
            target_fps=float(payload["target_fps"]),
            target_width=int(payload["target_width"]),
            target_height=int(payload["target_height"]),
            video_codec=str(payload.get("video_codec", "libx264")),
            pix_fmt=str(payload.get("pix_fmt", "yuv420p")),
            preset=str(payload.get("preset", "veryfast")),
            crf=int(payload.get("crf", 28)),
            keep_audio=bool(payload.get("keep_audio", False)),
            resize_mode=str(payload.get("resize_mode", "pad")),
            scale_flags=str(payload.get("scale_flags", "lanczos")),
        )


@dataclass(slots=True)
class VideoProbeStats:
    clip_id: str
    dataset: str
    split: str
    source_path: str
    relative_path: str
    binary_label: str | None
    severity_label: str | None
    camera_id: str | None
    tags: list[str]
    fps: float | None
    duration_sec: float
    width: int
    height: int
    frame_count: int | None
    codec_name: str
    pix_fmt: str
    format_name: str
    size_bytes: int
    bitrate_bps: float | None
    luminance_mean: float | None = None
    luminance_std: float | None = None
    laplacian_var: float | None = None
    colorfulness: float | None = None

    @property
    def pixel_rate(self) -> float | None:
        if self.fps is None or self.width <= 0 or self.height <= 0:
            return None
        return float(self.fps) * float(self.width) * float(self.height)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class HarmonizationJob:
    clip_id: str
    dataset: str
    split: str
    source_path: str
    target_path: str
    source_fps: float | None
    source_width: int
    source_height: int
    target_fps: float
    target_width: int
    target_height: int
    relative_path: str
    needs_fps_change: bool
    needs_resize: bool
    needs_reencode: bool
    pixel_rate_reduction: float | None
    ffmpeg_args: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def resolve_ffprobe_path(explicit_path: str | None = None) -> str:
    candidates = [
        explicit_path,
        os.getenv("TRAFFIC_INCIDENT_FFPROBE"),
        shutil.which("ffprobe"),
    ]
    for candidate in candidates:
        if candidate:
            return str(candidate)
    raise FileNotFoundError("ffprobe was not found. Set TRAFFIC_INCIDENT_FFPROBE or install ffprobe.")


def resolve_ffmpeg_path(explicit_path: str | None = None) -> str:
    candidates = [
        explicit_path,
        os.getenv("TRAFFIC_INCIDENT_FFMPEG"),
        shutil.which("ffmpeg"),
    ]
    for candidate in candidates:
        if candidate:
            return str(candidate)
    raise FileNotFoundError("ffmpeg was not found. Set TRAFFIC_INCIDENT_FFMPEG or install ffmpeg.")


def _parse_rate(value: str | None) -> float | None:
    if not value or value == "0/0":
        return None
    if "/" in value:
        numerator, denominator = value.split("/", 1)
        denominator_value = float(denominator)
        return float(numerator) / denominator_value if denominator_value else None
    return float(value)


def probe_video_metadata(source_path: Path, *, ffprobe_path: str | None = None) -> dict[str, Any]:
    ffprobe = resolve_ffprobe_path(ffprobe_path)
    command = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,r_frame_rate,avg_frame_rate,codec_name,pix_fmt,nb_frames,duration",
        "-show_entries",
        "format=size,duration,bit_rate,format_name",
        "-of",
        "json",
        str(source_path),
    ]
    payload = json.loads(subprocess.run(command, capture_output=True, text=True, check=True).stdout)
    stream = (payload.get("streams") or [{}])[0]
    fmt = payload.get("format") or {}
    fps = _parse_rate(stream.get("avg_frame_rate")) or _parse_rate(stream.get("r_frame_rate"))
    duration_sec = float(stream.get("duration") or fmt.get("duration") or 0.0)
    frame_count_raw = stream.get("nb_frames")
    frame_count = int(frame_count_raw) if frame_count_raw not in {None, "N/A"} else None
    size_bytes = int(float(fmt.get("size") or source_path.stat().st_size))
    bitrate_bps_raw = fmt.get("bit_rate")
    bitrate_bps = float(bitrate_bps_raw) if bitrate_bps_raw not in {None, "N/A"} else None
    return {
        "fps": fps,
        "duration_sec": duration_sec,
        "width": int(stream.get("width") or 0),
        "height": int(stream.get("height") or 0),
        "frame_count": frame_count,
        "codec_name": str(stream.get("codec_name") or "unknown"),
        "pix_fmt": str(stream.get("pix_fmt") or "unknown"),
        "format_name": str(fmt.get("format_name") or "unknown"),
        "size_bytes": size_bytes,
        "bitrate_bps": bitrate_bps,
    }


def _colorfulness(frame_rgb: np.ndarray) -> float:
    frame = frame_rgb.astype(np.float32)
    red = frame[:, :, 0]
    green = frame[:, :, 1]
    blue = frame[:, :, 2]
    rg = np.abs(red - green)
    yb = np.abs(0.5 * (red + green) - blue)
    return float(
        math.sqrt(float(rg.std()) ** 2 + float(yb.std()) ** 2)
        + 0.3 * math.sqrt(float(rg.mean()) ** 2 + float(yb.mean()) ** 2)
    )


def sample_visual_signature(
    source_path: Path,
    *,
    sample_points: tuple[float, ...] = (0.1, 0.5, 0.9),
) -> dict[str, float | None]:
    if cv2 is None:  # pragma: no cover
        return {
            "luminance_mean": None,
            "luminance_std": None,
            "laplacian_var": None,
            "colorfulness": None,
        }
    capture = cv2.VideoCapture(str(source_path))
    if not capture.isOpened():
        return {
            "luminance_mean": None,
            "luminance_std": None,
            "laplacian_var": None,
            "colorfulness": None,
        }
    try:
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if frame_count <= 0:
            return {
                "luminance_mean": None,
                "luminance_std": None,
                "laplacian_var": None,
                "colorfulness": None,
            }
        frame_indices = sorted(
            {
                min(max(int((frame_count - 1) * ratio), 0), max(frame_count - 1, 0))
                for ratio in sample_points
            }
        )
        luminance_mean_values: list[float] = []
        luminance_std_values: list[float] = []
        laplacian_values: list[float] = []
        colorfulness_values: list[float] = []
        for frame_index in frame_indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
            ok, frame_bgr = capture.read()
            if not ok or frame_bgr is None:
                continue
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            luminance_mean_values.append(float(frame_gray.mean()))
            luminance_std_values.append(float(frame_gray.std()))
            laplacian_values.append(float(cv2.Laplacian(frame_gray, cv2.CV_64F).var()))
            colorfulness_values.append(_colorfulness(frame_rgb))
        if not luminance_mean_values:
            return {
                "luminance_mean": None,
                "luminance_std": None,
                "laplacian_var": None,
                "colorfulness": None,
            }
        return {
            "luminance_mean": float(statistics.fmean(luminance_mean_values)),
            "luminance_std": float(statistics.fmean(luminance_std_values)),
            "laplacian_var": float(statistics.fmean(laplacian_values)),
            "colorfulness": float(statistics.fmean(colorfulness_values)),
        }
    finally:
        capture.release()


def scan_video_records(
    records: Iterable[ClipRecord],
    *,
    manifest_path: str | Path | None = None,
    dataset_roots: dict[str, str | Path] | None = None,
    ffprobe_path: str | None = None,
    sample_frames: bool = True,
) -> list[VideoProbeStats]:
    scanned: list[VideoProbeStats] = []
    for record in records:
        resolved_path = _resolve_source_path(
            record,
            manifest_path=manifest_path,
            dataset_roots=dataset_roots,
        )
        relative_path = _normalize_relative_path(record.video_path)
        if dataset_roots and record.dataset in dataset_roots:
            dataset_root = Path(dataset_roots[record.dataset]).resolve()
            try:
                relative_path = str(resolved_path.resolve().relative_to(dataset_root).as_posix())
            except ValueError:
                relative_path = _normalize_relative_path(record.video_path)
        metadata = probe_video_metadata(resolved_path, ffprobe_path=ffprobe_path)
        visual = sample_visual_signature(resolved_path) if sample_frames else {}
        scanned.append(
            VideoProbeStats(
                clip_id=record.clip_id,
                dataset=record.dataset,
                split=str(record.split),
                source_path=str(resolved_path),
                relative_path=relative_path,
                binary_label=record.binary_label.value if hasattr(record.binary_label, "value") else record.binary_label,
                severity_label=record.severity_label,
                camera_id=record.camera_id,
                tags=list(record.tags),
                fps=metadata["fps"],
                duration_sec=metadata["duration_sec"],
                width=metadata["width"],
                height=metadata["height"],
                frame_count=metadata["frame_count"],
                codec_name=metadata["codec_name"],
                pix_fmt=metadata["pix_fmt"],
                format_name=metadata["format_name"],
                size_bytes=metadata["size_bytes"],
                bitrate_bps=metadata["bitrate_bps"],
                luminance_mean=visual.get("luminance_mean"),
                luminance_std=visual.get("luminance_std"),
                laplacian_var=visual.get("laplacian_var"),
                colorfulness=visual.get("colorfulness"),
            )
        )
    return scanned


def _normalize_relative_path(path: str) -> str:
    return str(Path(path.replace("\\", "/")).as_posix()).lstrip("./")


def _relative_path_within_dataset_root(path: str, dataset_root: Path) -> Path:
    raw_path = Path(path.replace("\\", "/"))
    clean_parts = [part for part in raw_path.parts if part not in {"", ".", ".."}]
    if dataset_root.name in clean_parts:
        dataset_index = clean_parts.index(dataset_root.name)
        suffix_parts = clean_parts[dataset_index + 1 :]
        return Path(*suffix_parts) if suffix_parts else Path(raw_path.name)
    return Path(*clean_parts) if clean_parts else Path(raw_path.name)


def _resolve_source_path(
    record: ClipRecord,
    *,
    manifest_path: str | Path | None = None,
    dataset_roots: dict[str, str | Path] | None = None,
) -> Path:
    resolved_path = (
        ManifestClipDataset.resolve_path_for_record(manifest_path, record)
        if manifest_path is not None
        else Path(record.video_path).resolve()
    )
    if resolved_path.exists():
        return resolved_path
    if dataset_roots and record.dataset in dataset_roots:
        dataset_root = Path(dataset_roots[record.dataset]).resolve()
        candidate = (dataset_root / _relative_path_within_dataset_root(record.video_path, dataset_root)).resolve()
        if candidate.exists():
            return candidate
    return resolved_path


def _safe_quantile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    if len(values) == 1:
        return float(values[0])
    return float(np.quantile(np.asarray(values, dtype=float), quantile))


def _series_summary(values: Iterable[float | None]) -> dict[str, float | None]:
    clean = [float(value) for value in values if value is not None and not math.isnan(float(value))]
    if not clean:
        return {"min": None, "p10": None, "p50": None, "mean": None, "p90": None, "max": None}
    return {
        "min": float(min(clean)),
        "p10": _safe_quantile(clean, 0.10),
        "p50": _safe_quantile(clean, 0.50),
        "mean": float(statistics.fmean(clean)),
        "p90": _safe_quantile(clean, 0.90),
        "max": float(max(clean)),
    }


def summarize_video_stats(stats: Iterable[VideoProbeStats]) -> dict[str, Any]:
    stats_list = list(stats)
    fps_counter = Counter(round(item.fps or 0.0, 2) for item in stats_list if item.fps is not None)
    resolution_counter = Counter((item.width, item.height) for item in stats_list)
    codec_counter = Counter(item.codec_name for item in stats_list)
    pix_fmt_counter = Counter(item.pix_fmt for item in stats_list)
    format_counter = Counter(item.format_name for item in stats_list)
    split_counter = Counter(item.split for item in stats_list)
    binary_counter = Counter(item.binary_label for item in stats_list if item.binary_label is not None)
    severity_counter = Counter(item.severity_label for item in stats_list if item.severity_label is not None)
    total_size_bytes = sum(item.size_bytes for item in stats_list)
    total_duration_sec = sum(item.duration_sec for item in stats_list)
    return {
        "count": len(stats_list),
        "total_size_bytes": total_size_bytes,
        "total_size_gb": total_size_bytes / 1024**3,
        "total_duration_sec": total_duration_sec,
        "total_duration_h": total_duration_sec / 3600.0,
        "split_counts": dict(sorted(split_counter.items())),
        "binary_label_counts": dict(sorted(binary_counter.items())),
        "severity_label_counts": dict(sorted(severity_counter.items())),
        "fps_distribution": dict(sorted(fps_counter.items())),
        "resolution_distribution": {f"{width}x{height}": count for (width, height), count in resolution_counter.most_common()},
        "codec_distribution": dict(sorted(codec_counter.items())),
        "pix_fmt_distribution": dict(sorted(pix_fmt_counter.items())),
        "format_distribution": dict(sorted(format_counter.items())),
        "duration_summary": _series_summary(item.duration_sec for item in stats_list),
        "size_mb_summary": _series_summary(item.size_bytes / 1024**2 for item in stats_list),
        "bitrate_mbps_summary": _series_summary(
            (item.bitrate_bps / 1e6) if item.bitrate_bps is not None else None for item in stats_list
        ),
        "luminance_mean_summary": _series_summary(item.luminance_mean for item in stats_list),
        "luminance_std_summary": _series_summary(item.luminance_std for item in stats_list),
        "laplacian_var_summary": _series_summary(item.laplacian_var for item in stats_list),
        "colorfulness_summary": _series_summary(item.colorfulness for item in stats_list),
        "pixel_rate_mpps_summary": _series_summary(
            (item.pixel_rate / 1e6) if item.pixel_rate is not None else None for item in stats_list
        ),
    }


def summarize_accident_task(records: Iterable[ClipRecord]) -> dict[str, Any]:
    split_counts: dict[str, Counter[str]] = defaultdict(Counter)
    dataset_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for record in records:
        label = str(record.binary_label.value if hasattr(record.binary_label, "value") else record.binary_label)
        split_counts[str(record.split)][label] += 1
        dataset_counts[str(record.split)][str(record.dataset)] += 1
    imbalance_summary: dict[str, dict[str, float | int]] = {}
    for split, counts in split_counts.items():
        positives = int(counts.get("accident", 0))
        negatives = int(counts.get("no_accident", 0))
        ratio = (positives / negatives) if negatives else math.inf
        imbalance_summary[split] = {
            "accident": positives,
            "no_accident": negatives,
            "positive_to_negative_ratio": ratio,
        }
    return {
        "split_label_counts": {split: dict(counts) for split, counts in sorted(split_counts.items())},
        "split_dataset_counts": {split: dict(counts) for split, counts in sorted(dataset_counts.items())},
        "imbalance_summary": imbalance_summary,
    }


def summarize_severity_task(records: Iterable[ClipRecord]) -> dict[str, Any]:
    split_counts: dict[str, Counter[str]] = defaultdict(Counter)
    augmented_counts: dict[str, int] = defaultdict(int)
    for record in records:
        if record.severity_label is not None:
            split_counts[str(record.split)][record.severity_label] += 1
        if "augmented_source" in record.tags:
            augmented_counts[str(record.split)] += 1
    return {
        "split_label_counts": {split: dict(counts) for split, counts in sorted(split_counts.items())},
        "split_augmented_counts": dict(sorted(augmented_counts.items())),
    }


def build_ffmpeg_filter(profile: HarmonizationProfile) -> str:
    fps_filter = f"fps={profile.target_fps:g}"
    scale_filter = (
        f"scale={profile.target_width}:{profile.target_height}:"
        f"force_original_aspect_ratio=decrease:flags={profile.scale_flags}"
    )
    if profile.resize_mode == "pad":
        layout_filter = (
            f"pad={profile.target_width}:{profile.target_height}:"
            f"(ow-iw)/2:(oh-ih)/2:black"
        )
    else:
        layout_filter = f"crop={profile.target_width}:{profile.target_height}"
    return ",".join([fps_filter, scale_filter, layout_filter, f"format={profile.pix_fmt}"])


def build_ffmpeg_command(
    source_path: Path,
    target_path: Path,
    profile: HarmonizationProfile,
    *,
    ffmpeg_path: str | None = None,
    overwrite: bool = False,
) -> list[str]:
    ffmpeg = resolve_ffmpeg_path(ffmpeg_path)
    command = [ffmpeg, "-hide_banner", "-loglevel", "error"]
    command.append("-y" if overwrite else "-n")
    command.extend(["-i", str(source_path)])
    if not profile.keep_audio:
        command.append("-an")
    command.extend(
        [
            "-vf",
            build_ffmpeg_filter(profile),
            "-c:v",
            profile.video_codec,
            "-preset",
            profile.preset,
            "-crf",
            str(profile.crf),
            "-pix_fmt",
            profile.pix_fmt,
            str(target_path),
        ]
    )
    return command


def build_harmonization_jobs(
    stats: Iterable[VideoProbeStats],
    *,
    output_root: str | Path,
    profile: HarmonizationProfile,
    overwrite: bool = False,
    ffmpeg_path: str | None = None,
) -> list[HarmonizationJob]:
    root = Path(output_root)
    jobs: list[HarmonizationJob] = []
    for item in stats:
        target_relative = Path(item.dataset) / Path(item.relative_path).with_suffix(".mp4")
        target_path = (root / profile.name / target_relative).resolve()
        source_pixel_rate = item.pixel_rate
        target_pixel_rate = profile.target_fps * profile.target_width * profile.target_height
        pixel_rate_reduction = (
            (source_pixel_rate / target_pixel_rate)
            if source_pixel_rate is not None and target_pixel_rate > 0
            else None
        )
        job = HarmonizationJob(
            clip_id=item.clip_id,
            dataset=item.dataset,
            split=item.split,
            source_path=item.source_path,
            target_path=str(target_path),
            source_fps=item.fps,
            source_width=item.width,
            source_height=item.height,
            target_fps=profile.target_fps,
            target_width=profile.target_width,
            target_height=profile.target_height,
            relative_path=str(target_relative.as_posix()),
            needs_fps_change=(item.fps is None) or abs(float(item.fps) - profile.target_fps) > 0.05,
            needs_resize=(item.width != profile.target_width) or (item.height != profile.target_height),
            needs_reencode=True,
            pixel_rate_reduction=pixel_rate_reduction,
            ffmpeg_args=build_ffmpeg_command(
                Path(item.source_path),
                target_path,
                profile,
                ffmpeg_path=ffmpeg_path,
                overwrite=overwrite,
            ),
        )
        jobs.append(job)
    return jobs


def summarize_harmonization_jobs(jobs: Iterable[HarmonizationJob]) -> dict[str, Any]:
    job_list = list(jobs)
    pixel_reductions = [
        float(job.pixel_rate_reduction)
        for job in job_list
        if job.pixel_rate_reduction is not None and not math.isnan(float(job.pixel_rate_reduction))
    ]
    return {
        "count": len(job_list),
        "fps_change_count": sum(1 for job in job_list if job.needs_fps_change),
        "resize_count": sum(1 for job in job_list if job.needs_resize),
        "reencode_count": sum(1 for job in job_list if job.needs_reencode),
        "pixel_rate_reduction_summary": _series_summary(pixel_reductions),
    }


def rewrite_manifest_for_harmonization(
    source_manifest_path: str | Path,
    *,
    output_root: str | Path,
    output_manifest_path: str | Path,
    profile: HarmonizationProfile,
    dataset_roots: dict[str, str | Path] | None = None,
) -> list[ClipRecord]:
    records = load_manifest(source_manifest_path)
    rewritten: list[ClipRecord] = []
    for record in records:
        source_relative = _normalize_relative_path(record.video_path)
        resolved_source_path = _resolve_source_path(
            record,
            manifest_path=source_manifest_path,
            dataset_roots=dataset_roots,
        )
        if dataset_roots and record.dataset in dataset_roots:
            dataset_root = Path(dataset_roots[record.dataset]).resolve()
            try:
                source_relative = str(resolved_source_path.resolve().relative_to(dataset_root).as_posix())
            except ValueError:
                source_relative = _normalize_relative_path(record.video_path)
        target_path = (Path(output_root) / profile.name / record.dataset / Path(source_relative)).with_suffix(".mp4").resolve()
        tags = list(record.tags)
        harmonization_tags = [
            f"harmonized_profile:{profile.name}",
            f"harmonized_fps:{profile.target_fps:g}",
            f"harmonized_size:{profile.target_width}x{profile.target_height}",
        ]
        for tag in harmonization_tags:
            if tag not in tags:
                tags.append(tag)
        rewritten.append(
            ClipRecord(
                clip_id=record.clip_id,
                video_path=str(target_path),
                dataset=record.dataset,
                split=record.split,
                task=record.task,
                binary_label=record.binary_label,
                severity_label=record.severity_label,
                camera_id=record.camera_id,
                fps=profile.target_fps,
                duration_sec=record.duration_sec,
                tags=tags,
            )
        )
    save_manifest(output_manifest_path, rewritten)
    return rewritten


def execute_harmonization_jobs(
    jobs: Iterable[HarmonizationJob],
    *,
    limit: int | None = None,
    progress_path: str | Path | None = None,
    fail_fast: bool = False,
) -> dict[str, Any]:
    job_list = list(jobs)
    executed = 0
    skipped = 0
    failed = 0
    failures: list[dict[str, Any]] = []
    started_at = time.strftime("%Y-%m-%dT%H:%M:%S")
    progress_target = Path(progress_path).resolve() if progress_path is not None else None

    def write_progress(*, status: str, current_index: int, current_job: HarmonizationJob | None = None) -> None:
        if progress_target is None:
            return
        payload: dict[str, Any] = {
            "status": status,
            "started_at": started_at,
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "total_jobs": len(job_list),
            "processed_jobs": current_index,
            "executed_jobs": executed,
            "skipped_jobs": skipped,
            "failed_jobs": failed,
            "remaining_jobs": max(len(job_list) - current_index, 0),
            "failures": failures,
        }
        if current_job is not None:
            payload["current_job"] = {
                "clip_id": current_job.clip_id,
                "dataset": current_job.dataset,
                "split": current_job.split,
                "relative_path": current_job.relative_path,
                "target_path": current_job.target_path,
            }
        write_json(progress_target, payload)

    write_progress(status="running", current_index=0)
    for index, job in enumerate(job_list, start=1):
        target_path = Path(job.target_path)
        ensure_dir(target_path.parent)
        if target_path.exists():
            skipped += 1
            print(f"[{index}/{len(job_list)}] SKIP {job.dataset} {job.relative_path}", flush=True)
            write_progress(status="running", current_index=index, current_job=job)
            continue
        try:
            subprocess.run(job.ffmpeg_args, check=True)
            executed += 1
            print(f"[{index}/{len(job_list)}] DONE {job.dataset} {job.relative_path}", flush=True)
        except subprocess.CalledProcessError as exc:
            failed += 1
            failure = {
                "clip_id": job.clip_id,
                "dataset": job.dataset,
                "relative_path": job.relative_path,
                "return_code": int(exc.returncode),
            }
            failures.append(failure)
            print(
                f"[{index}/{len(job_list)}] FAIL {job.dataset} {job.relative_path} rc={exc.returncode}",
                flush=True,
            )
            write_progress(status="failed" if fail_fast else "running", current_index=index, current_job=job)
            if fail_fast:
                raise
        else:
            write_progress(status="running", current_index=index, current_job=job)
        if limit is not None and executed >= limit:
            write_progress(status="stopped_at_limit", current_index=index, current_job=job)
            return {
                "executed": executed,
                "skipped": skipped,
                "failed": failed,
                "failures": failures,
                "status": "stopped_at_limit",
            }
    final_status = "completed_with_failures" if failed else "completed"
    write_progress(status=final_status, current_index=len(job_list))
    return {
        "executed": executed,
        "skipped": skipped,
        "failed": failed,
        "failures": failures,
        "status": final_status,
    }


def build_domain_shift_summary(
    *,
    ai_city_stats: Iterable[VideoProbeStats],
    balanced_stats: Iterable[VideoProbeStats],
    profile: HarmonizationProfile,
) -> dict[str, Any]:
    ai_summary = summarize_video_stats(ai_city_stats)
    balanced_summary = summarize_video_stats(balanced_stats)
    ai_fps_median = ai_summary["duration_summary"]
    return {
        "ai_city_vs_balanced": {
            "fps_modes": {
                "ai_city": ai_summary["fps_distribution"],
                "balanced_accident": balanced_summary["fps_distribution"],
            },
            "resolution_modes": {
                "ai_city": dict(list(ai_summary["resolution_distribution"].items())[:5]),
                "balanced_accident": dict(list(balanced_summary["resolution_distribution"].items())[:5]),
            },
            "duration_median_sec": {
                "ai_city": ai_summary["duration_summary"]["p50"],
                "balanced_accident": balanced_summary["duration_summary"]["p50"],
            },
            "bitrate_median_mbps": {
                "ai_city": ai_summary["bitrate_mbps_summary"]["p50"],
                "balanced_accident": balanced_summary["bitrate_mbps_summary"]["p50"],
            },
            "luminance_mean_median": {
                "ai_city": ai_summary["luminance_mean_summary"]["p50"],
                "balanced_accident": balanced_summary["luminance_mean_summary"]["p50"],
            },
            "laplacian_var_median": {
                "ai_city": ai_summary["laplacian_var_summary"]["p50"],
                "balanced_accident": balanced_summary["laplacian_var_summary"]["p50"],
            },
            "recommended_profile": {
                "name": profile.name,
                "target_fps": profile.target_fps,
                "target_size": [profile.target_width, profile.target_height],
                "codec": profile.video_codec,
                "pix_fmt": profile.pix_fmt,
                "preset": profile.preset,
                "crf": profile.crf,
            },
        }
    }


def plot_harmonization_report(
    *,
    ai_city_stats: Iterable[VideoProbeStats],
    balanced_stats: Iterable[VideoProbeStats],
    output_dir: str | Path,
) -> dict[str, str]:
    if plt is None:  # pragma: no cover
        return {}
    ensure_dir(output_dir)
    chart_paths: dict[str, str] = {}
    ai_list = list(ai_city_stats)
    balanced_list = list(balanced_stats)

    figure, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes[0, 0].hist(
        [[item.fps for item in ai_list if item.fps is not None], [item.fps for item in balanced_list if item.fps is not None]],
        bins=12,
        label=["AI City", "Balanced"],
        alpha=0.8,
    )
    axes[0, 0].set_title("FPS Distribution")
    axes[0, 0].legend()

    axes[0, 1].hist(
        [[item.duration_sec for item in ai_list], [item.duration_sec for item in balanced_list]],
        bins=20,
        label=["AI City", "Balanced"],
        alpha=0.8,
    )
    axes[0, 1].set_title("Duration Distribution (sec)")
    axes[0, 1].legend()

    axes[1, 0].scatter(
        [item.width for item in ai_list],
        [item.height for item in ai_list],
        label="AI City",
        alpha=0.8,
    )
    axes[1, 0].scatter(
        [item.width for item in balanced_list],
        [item.height for item in balanced_list],
        label="Balanced",
        alpha=0.4,
    )
    axes[1, 0].set_title("Resolution Scatter")
    axes[1, 0].set_xlabel("width")
    axes[1, 0].set_ylabel("height")
    axes[1, 0].legend()

    axes[1, 1].scatter(
        [item.luminance_mean for item in ai_list if item.luminance_mean is not None],
        [item.laplacian_var for item in ai_list if item.luminance_mean is not None and item.laplacian_var is not None],
        label="AI City",
        alpha=0.8,
    )
    axes[1, 1].scatter(
        [item.luminance_mean for item in balanced_list if item.luminance_mean is not None],
        [item.laplacian_var for item in balanced_list if item.luminance_mean is not None and item.laplacian_var is not None],
        label="Balanced",
        alpha=0.35,
    )
    axes[1, 1].set_title("Visual Signature")
    axes[1, 1].set_xlabel("mean luminance")
    axes[1, 1].set_ylabel("laplacian variance")
    axes[1, 1].legend()

    figure.tight_layout()
    summary_chart_path = Path(output_dir) / "input_domain_summary.png"
    figure.savefig(summary_chart_path, dpi=160)
    plt.close(figure)
    chart_paths["input_domain_summary"] = str(summary_chart_path)
    return chart_paths


def write_harmonization_report(
    *,
    output_dir: str | Path,
    ai_city_summary: dict[str, Any],
    balanced_summary: dict[str, Any],
    accident_summary: dict[str, Any],
    severity_summary: dict[str, Any],
    domain_shift_summary: dict[str, Any],
    harmonization_job_summary: dict[str, Any],
    chart_paths: dict[str, str],
) -> dict[str, str]:
    report_dir = ensure_dir(output_dir)
    markdown_path = report_dir / "input_harmonization_report.md"
    json_path = report_dir / "input_harmonization_report.json"
    payload = {
        "ai_city": ai_city_summary,
        "balanced_accident": balanced_summary,
        "accident_task": accident_summary,
        "severity_task": severity_summary,
        "domain_shift": domain_shift_summary,
        "harmonization_plan": harmonization_job_summary,
        "charts": chart_paths,
    }
    write_json(json_path, payload)
    def fmt(value: Any, digits: int = 2) -> str:
        if value is None:
            return "n/a"
        return f"{float(value):.{digits}f}"
    markdown = f"""# Input Harmonization Report

## Executive Summary

- AI City and Balanced Accident are strongly mismatched by fps, duration, resolution, and visual texture.
- The merged accident task is heavily imbalanced: negatives come almost entirely from AI City and positives from Balanced Accident.
- Recommended default harmonization profile: `{domain_shift_summary['ai_city_vs_balanced']['recommended_profile']['name']}` at `{domain_shift_summary['ai_city_vs_balanced']['recommended_profile']['target_fps']} fps` and `{domain_shift_summary['ai_city_vs_balanced']['recommended_profile']['target_size'][0]}x{domain_shift_summary['ai_city_vs_balanced']['recommended_profile']['target_size'][1]}`.

## Current Composition

- AI City videos: `{ai_city_summary['count']}` files, `{fmt(ai_city_summary['total_size_gb'])} GB`, `{fmt(ai_city_summary['total_duration_h'])} h`
- Balanced Accident videos: `{balanced_summary['count']}` files, `{fmt(balanced_summary['total_size_gb'])} GB`, `{fmt(balanced_summary['total_duration_h'])} h`
- Accident train imbalance: `{accident_summary['imbalance_summary']['train']['accident']}` accident vs `{accident_summary['imbalance_summary']['train']['no_accident']}` no_accident
- Accident val imbalance: `{accident_summary['imbalance_summary']['val']['accident']}` accident vs `{accident_summary['imbalance_summary']['val']['no_accident']}` no_accident

## Domain Shift

- AI City fps distribution: `{ai_city_summary['fps_distribution']}`
- Balanced fps distribution: `{balanced_summary['fps_distribution']}`
- AI City median duration: `{fmt(ai_city_summary['duration_summary']['p50'])} sec`
- Balanced median duration: `{fmt(balanced_summary['duration_summary']['p50'])} sec`
- AI City median laplacian variance: `{fmt(ai_city_summary['laplacian_var_summary']['p50'])}`
- Balanced median laplacian variance: `{fmt(balanced_summary['laplacian_var_summary']['p50'])}`

## Harmonization Plan

- Planned jobs: `{harmonization_job_summary['count']}`
- Files requiring fps conversion: `{harmonization_job_summary['fps_change_count']}`
- Files requiring spatial resize/pad: `{harmonization_job_summary['resize_count']}`
- Median pixel-rate reduction: `{fmt(harmonization_job_summary['pixel_rate_reduction_summary']['p50'])}x`

## Charts

{chr(10).join(f'- `{name}`: `{path}`' for name, path in chart_paths.items())}
"""
    markdown_path.write_text(markdown, encoding="utf-8")
    return {
        "markdown_path": str(markdown_path),
        "json_path": str(json_path),
    }
