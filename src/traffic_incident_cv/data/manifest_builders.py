from __future__ import annotations

import csv
import math
import re
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Iterable

from traffic_incident_cv.data.manifest import load_manifest
from traffic_incident_cv.domain.schemas import ClipRecord


VIDEO_EXTENSIONS_DEFAULT = {".mp4", ".avi", ".mov", ".mkv"}


@dataclass(slots=True)
class ManifestArtifact:
    name: str
    records: list[ClipRecord]
    output_path: str | None = None


def _stable_records(records: Iterable[ClipRecord]) -> list[ClipRecord]:
    return sorted(records, key=lambda record: (str(record.split), record.dataset, record.clip_id))


def iter_video_files(
    root_dir: Path,
    video_extensions: set[str],
    include_substrings: list[str] | None = None,
    exclude_substrings: list[str] | None = None,
) -> Iterable[Path]:
    if not root_dir.exists():
        raise FileNotFoundError(root_dir)
    include = [item.lower() for item in (include_substrings or [])]
    exclude = [item.lower() for item in (exclude_substrings or [])]
    for path in sorted(root_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in video_extensions:
            continue
        lowered = str(path).lower()
        if include and not all(token in lowered for token in include):
            continue
        if exclude and any(token in lowered for token in exclude):
            continue
        yield path


def infer_split_from_path(path: Path, split_patterns: dict[str, list[str]]) -> str | None:
    lowered_parts = [part.lower() for part in path.parts]
    for split_name, candidates in split_patterns.items():
        for candidate in candidates:
            if candidate.lower() in lowered_parts:
                return split_name
    return None


def split_records(records: Iterable[ClipRecord]) -> dict[str, list[ClipRecord]]:
    grouped: dict[str, list[ClipRecord]] = {}
    for record in records:
        grouped.setdefault(str(record.split), []).append(record)
    return {split: _stable_records(items) for split, items in grouped.items()}


def summarize_records_by_split(records: Iterable[ClipRecord]) -> dict[str, int]:
    return {split: len(items) for split, items in split_records(records).items()}


def _normalize_camera_id(raw_id: str | int) -> str:
    match = re.search(r"(\d+)", str(raw_id))
    if match is None:
        raise ValueError(f"Unable to normalize camera id from {raw_id!r}")
    numeric = int(match.group(1))
    return f"cam_{numeric}"


def extract_camera_id(path: Path, camera_regex: str | None) -> str | None:
    if not camera_regex:
        return None
    regex = re.compile(camera_regex, flags=re.IGNORECASE)
    for candidate in (path.stem, path.name):
        match = regex.search(candidate)
        if match:
            group = match.group(1) if match.groups() else match.group(0)
            return _normalize_camera_id(group)
    return None


def _invert_group_assignments(group_assignments: dict[str, list[str]]) -> dict[str, str]:
    group_to_split: dict[str, str] = {}
    for split, group_ids in group_assignments.items():
        for group_id in group_ids:
            normalized = _normalize_camera_id(group_id)
            if normalized in group_to_split:
                raise ValueError(f"Group {normalized!r} is assigned to multiple splits.")
            group_to_split[normalized] = split
    return group_to_split


def assign_split_by_group(group_id: str | None, group_assignments: dict[str, list[str]] | None) -> str | None:
    if not group_assignments:
        return None
    if group_id is None:
        raise ValueError("Cannot assign split by group without a group id.")
    group_to_split = _invert_group_assignments(group_assignments)
    if group_id not in group_to_split:
        raise ValueError(f"Group {group_id!r} is missing from split assignments.")
    return group_to_split[group_id]


def load_video_stats(stats_path: str | Path | None) -> dict[str, dict[str, float]]:
    if stats_path is None:
        return {}
    path = Path(stats_path)
    if not path.exists():
        raise FileNotFoundError(path)
    stats: dict[str, dict[str, float]] = {}
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            filename = row["vid_name"]
            fps = float(Fraction(row["fps"]))
            frame_num = int(row["frame_num"])
            stats[filename] = {
                "fps": fps,
                "frame_num": frame_num,
                "duration_sec": frame_num / fps if fps else 0.0,
            }
    return stats


def infer_condition_tags(path: Path, tags_by_suffix: dict[str, str] | None) -> list[str]:
    if not tags_by_suffix:
        return []
    stem = path.stem.lower()
    tags: list[str] = []
    for suffix, tag in tags_by_suffix.items():
        normalized_suffix = suffix.lower().lstrip("_")
        if stem.endswith(f"_{normalized_suffix}"):
            tags.append(tag)
    return tags


def _merge_unique_tags(*tag_groups: Iterable[str]) -> list[str]:
    merged: list[str] = []
    for tag_group in tag_groups:
        for tag in tag_group:
            if tag not in merged:
                merged.append(tag)
    return merged


def _resolve_split(path: Path, dataset: dict) -> str:
    split = assign_split_by_group(
        group_id=extract_camera_id(path, dataset.get("camera_regex")),
        group_assignments=dataset.get("split_assignments"),
    )
    if split is not None:
        return split
    inferred = infer_split_from_path(path, dataset.get("split_patterns", {}))
    if inferred is None:
        raise ValueError(f"Unable to infer split for {path}")
    return inferred


def build_ai_city_records(cfg: dict) -> list[ClipRecord]:
    dataset = cfg["dataset"]
    root = Path(dataset["root_dir"])
    video_extensions = {suffix.lower() for suffix in dataset.get("video_extensions", VIDEO_EXTENSIONS_DEFAULT)}
    stats = load_video_stats(dataset.get("stats_path"))
    records: list[ClipRecord] = []
    for video_path in iter_video_files(
        root,
        video_extensions=video_extensions,
        include_substrings=dataset.get("include_substrings"),
        exclude_substrings=dataset.get("exclude_substrings"),
    ):
        split = _resolve_split(video_path, dataset)
        camera_id = extract_camera_id(video_path, dataset.get("camera_regex"))
        video_stats = stats.get(video_path.name, {})
        tags = _merge_unique_tags(
            dataset.get("tags", []),
            infer_condition_tags(video_path, dataset.get("tags_by_suffix")),
        )
        records.append(
            ClipRecord(
                clip_id=f"{dataset['name']}_{split}_{video_path.stem}",
                video_path=str(video_path),
                dataset=dataset["name"],
                split=split,
                task=dataset.get("task", "accident"),
                binary_label=dataset["class_policy"]["binary_label"],
                severity_label=dataset["class_policy"].get("severity_label"),
                camera_id=camera_id,
                fps=video_stats.get("fps"),
                duration_sec=video_stats.get("duration_sec"),
                tags=tags,
            )
        )
    return _stable_records(records)


def build_balanced_records(cfg: dict) -> tuple[list[ClipRecord], list[ClipRecord]]:
    dataset = cfg["dataset"]
    root = Path(dataset["root_dir"])
    video_extensions = {suffix.lower() for suffix in dataset.get("video_extensions", VIDEO_EXTENSIONS_DEFAULT)}
    severity_dirs = {value.lower() for value in dataset.get("severity_dirs", ["minor", "moderate", "major"])}
    augmented_prefixes = tuple(prefix.lower() for prefix in dataset.get("augmented_prefixes", ["aug"]))
    real_tags = dataset.get("real_tags", ["real_video"])
    augmented_tags = dataset.get("augmented_tags", ["augmented_source"])
    common_tags = dataset.get("tags", [])
    binary_records: list[ClipRecord] = []
    severity_records: list[ClipRecord] = []
    for video_path in iter_video_files(
        root,
        video_extensions=video_extensions,
        include_substrings=dataset.get("include_substrings"),
        exclude_substrings=dataset.get("exclude_substrings"),
    ):
        split = infer_split_from_path(video_path, dataset["split_patterns"])
        if split is None:
            raise ValueError(f"Unable to infer split for {video_path}")
        severity_label = next((part.lower() for part in video_path.parts if part.lower() in severity_dirs), None)
        if severity_label is None:
            raise ValueError(f"Unable to infer severity label for {video_path}")
        is_augmented = video_path.name.lower().startswith(augmented_prefixes)
        tags = _merge_unique_tags(
            common_tags,
            augmented_tags if is_augmented else real_tags,
        )
        clip_id = f"{dataset['name']}_{split}_{severity_label}_{video_path.stem}"
        binary_records.append(
            ClipRecord(
                clip_id=clip_id,
                video_path=str(video_path),
                dataset=dataset["name"],
                split=split,
                task="accident",
                binary_label="accident",
                severity_label=severity_label,
                tags=tags,
            )
        )
        severity_records.append(
            ClipRecord(
                clip_id=clip_id,
                video_path=str(video_path),
                dataset=dataset["name"],
                split=split,
                task="severity",
                binary_label="accident",
                severity_label=severity_label,
                tags=tags,
            )
        )
    return _stable_records(binary_records), _stable_records(severity_records)


def merge_manifest_records(cfg: dict, manifest_cache: dict[str, list[ClipRecord]] | None = None) -> list[ClipRecord]:
    dataset = cfg["dataset"]
    records: list[ClipRecord] = []
    seen_clip_ids: set[str] = set()
    for manifest_path in dataset.get("input_manifests", []):
        source_records = manifest_cache.get(manifest_path) if manifest_cache is not None else None
        if source_records is None:
            source_records = load_manifest(manifest_path)
        for record in source_records:
            if record.clip_id in seen_clip_ids:
                raise ValueError(f"Duplicate clip_id during manifest merge: {record.clip_id}")
            seen_clip_ids.add(record.clip_id)
            records.append(record)
    return _stable_records(records)


def _resolve_manifest_records(
    manifest_path: str,
    *,
    manifest_cache: dict[str, list[ClipRecord]] | None = None,
) -> list[ClipRecord]:
    source_records = manifest_cache.get(manifest_path) if manifest_cache is not None else None
    if source_records is None:
        source_records = load_manifest(manifest_path)
    return list(source_records)


def _merge_tags(*groups: Iterable[str]) -> list[str]:
    return _merge_unique_tags(*groups)


def _candidate_windows(
    record: ClipRecord,
    *,
    clip_duration_sec: float,
    clip_stride_sec: float,
) -> list[tuple[float, float, int, int]]:
    duration_sec = float(record.duration_sec or 0.0)
    fps = float(record.fps or 0.0)
    if duration_sec <= 0.0:
        raise ValueError(f"Negative clip generation requires duration_sec for {record.clip_id}")
    if fps <= 0.0:
        raise ValueError(f"Negative clip generation requires fps for {record.clip_id}")
    clip_duration = min(float(clip_duration_sec), duration_sec)
    max_start = max(duration_sec - clip_duration, 0.0)
    starts: list[float] = []
    if max_start <= 0.0:
        starts = [0.0]
    else:
        current = 0.0
        while current <= max_start + 1e-6:
            starts.append(round(current, 4))
            current += float(clip_stride_sec)
        if abs(starts[-1] - max_start) > 1e-6:
            starts.append(round(max_start, 4))
    windows: list[tuple[float, float, int, int]] = []
    for start_sec in starts:
        end_sec = min(start_sec + clip_duration, duration_sec)
        start_frame = max(int(round(start_sec * fps)), 0)
        end_frame = max(int(round(end_sec * fps)) - 1, start_frame)
        windows.append((start_sec, end_sec, start_frame, end_frame))
    return windows


def _evenly_spaced_indices(total: int, count: int) -> list[int]:
    if count <= 0 or total <= 0:
        return []
    if count >= total:
        return list(range(total))
    return [min(((2 * index + 1) * total) // (2 * count), total - 1) for index in range(count)]


def _allocate_candidate_quotas(
    capacities: list[tuple[ClipRecord, list[tuple[float, float, int, int]]]],
    target_count: int,
) -> dict[str, int]:
    total_capacity = sum(len(windows) for _, windows in capacities)
    if target_count >= total_capacity:
        return {record.clip_id: len(windows) for record, windows in capacities}
    quotas: dict[str, int] = {}
    remainders: list[tuple[float, int, str]] = []
    assigned = 0
    for record, windows in capacities:
        capacity = len(windows)
        exact = (target_count * capacity) / total_capacity if total_capacity else 0.0
        base = min(int(math.floor(exact)), capacity)
        quotas[record.clip_id] = base
        assigned += base
        remainders.append((exact - base, capacity, record.clip_id))
    remaining = max(target_count - assigned, 0)
    ordered = sorted(remainders, key=lambda item: (-item[0], -item[1], item[2]))
    while remaining > 0:
        progressed = False
        for _, capacity, clip_id in ordered:
            if quotas[clip_id] >= capacity:
                continue
            quotas[clip_id] += 1
            remaining -= 1
            progressed = True
            if remaining == 0:
                break
        if not progressed:
            break
    return quotas


def build_accident_clip_bundle_records(
    cfg: dict,
    *,
    manifest_cache: dict[str, list[ClipRecord]] | None = None,
) -> list[ClipRecord]:
    dataset = cfg["dataset"]
    input_manifests = list(dataset.get("input_manifests", []))
    negative_manifest_path = dataset.get("negative_source_manifest") or (input_manifests[0] if input_manifests else None)
    positive_manifest_path = dataset.get("positive_source_manifest") or (
        input_manifests[1] if len(input_manifests) > 1 else None
    )
    if negative_manifest_path is None or positive_manifest_path is None:
        raise ValueError("accident_clip_bundle requires negative_source_manifest and positive_source_manifest")

    negative_source_records = _resolve_manifest_records(negative_manifest_path, manifest_cache=manifest_cache)
    positive_records = _resolve_manifest_records(positive_manifest_path, manifest_cache=manifest_cache)
    policy = dataset.get("negative_policy", {})
    clip_duration_sec = float(policy.get("clip_duration_sec", 8.0))
    clip_stride_sec = float(policy.get("clip_stride_sec", 4.0))
    target_ratio = float(policy.get("target_negative_to_positive_ratio", 1.0))
    policy_name = str(policy.get("name", "balanced_neg_v1"))
    negative_tags = list(policy.get("negative_tags", ["temporal_clip", "clip_role:no_accident_window"]))
    positive_tags = list(policy.get("positive_tags", ["clip_role:accident_clip"]))

    positive_by_split: dict[str, list[ClipRecord]] = split_records(positive_records)
    negative_source_by_split: dict[str, list[ClipRecord]] = split_records(negative_source_records)
    negatives: list[ClipRecord] = []

    for split, positive_split_records in positive_by_split.items():
        target_negative_count = int(round(len(positive_split_records) * target_ratio))
        source_records = negative_source_by_split.get(split, [])
        capacities = [(record, _candidate_windows(record, clip_duration_sec=clip_duration_sec, clip_stride_sec=clip_stride_sec)) for record in source_records]
        quotas = _allocate_candidate_quotas(capacities, target_negative_count)
        for record, windows in capacities:
            quota = quotas.get(record.clip_id, 0)
            for window_index in _evenly_spaced_indices(len(windows), quota):
                start_sec, end_sec, start_frame, end_frame = windows[window_index]
                negatives.append(
                    ClipRecord(
                        clip_id=f"{record.clip_id}__neg_{start_frame:06d}_{end_frame:06d}",
                        video_path=record.video_path,
                        dataset=record.dataset,
                        split=record.split,
                        task="accident",
                        binary_label="no_accident",
                        severity_label=None,
                        camera_id=record.camera_id,
                        fps=record.fps,
                        duration_sec=round(end_sec - start_sec, 4),
                        clip_start_sec=start_sec,
                        clip_end_sec=end_sec,
                        clip_start_frame=start_frame,
                        clip_end_frame=end_frame,
                        tags=_merge_tags(
                            record.tags,
                            negative_tags,
                            [f"clip_policy:{policy_name}"],
                        ),
                    )
                )

    positives: list[ClipRecord] = []
    for record in positive_records:
        positives.append(
            ClipRecord(
                clip_id=record.clip_id,
                video_path=record.video_path,
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
                tags=_merge_tags(record.tags, positive_tags),
            )
        )
    return _stable_records([*positives, *negatives])


def build_artifacts_from_config(
    cfg: dict,
    manifest_cache: dict[str, list[ClipRecord]] | None = None,
) -> list[ManifestArtifact]:
    dataset = cfg["dataset"]
    builder = dataset.get("builder", dataset["name"])
    if builder == "ai_city":
        records = build_ai_city_records(cfg)
        split_outputs = dataset.get("output_manifests", {})
        artifacts = [ManifestArtifact(name="all", records=records, output_path=dataset.get("output_manifest"))]
        artifacts.extend(
            ManifestArtifact(name=split, records=split_records(records).get(split, []), output_path=output_path)
            for split, output_path in split_outputs.items()
        )
        return artifacts
    if builder == "balanced_accident":
        binary_records, severity_records = build_balanced_records(cfg)
        binary_split = split_records(binary_records)
        severity_split = split_records(severity_records)
        artifacts = [
            ManifestArtifact(name="binary_all", records=binary_records, output_path=dataset.get("output_manifest_binary")),
            ManifestArtifact(name="severity_all", records=severity_records, output_path=dataset.get("output_manifest_severity")),
        ]
        artifacts.extend(
            ManifestArtifact(name=f"binary_{split}", records=binary_split.get(split, []), output_path=output_path)
            for split, output_path in dataset.get("output_manifests_binary", {}).items()
        )
        artifacts.extend(
            ManifestArtifact(name=f"severity_{split}", records=severity_split.get(split, []), output_path=output_path)
            for split, output_path in dataset.get("output_manifests_severity", {}).items()
        )
        return artifacts
    if builder == "merged_manifest":
        records = merge_manifest_records(cfg, manifest_cache=manifest_cache)
        split_outputs = dataset.get("output_manifests", {})
        artifacts = [ManifestArtifact(name="all", records=records, output_path=dataset.get("output_manifest"))]
        artifacts.extend(
            ManifestArtifact(name=split, records=split_records(records).get(split, []), output_path=output_path)
            for split, output_path in split_outputs.items()
        )
        return artifacts
    if builder == "accident_clip_bundle":
        records = build_accident_clip_bundle_records(cfg, manifest_cache=manifest_cache)
        split_outputs = dataset.get("output_manifests", {})
        artifacts = [ManifestArtifact(name="all", records=records, output_path=dataset.get("output_manifest"))]
        artifacts.extend(
            ManifestArtifact(name=split, records=split_records(records).get(split, []), output_path=output_path)
            for split, output_path in split_outputs.items()
        )
        return artifacts
    raise ValueError(f"Unsupported dataset builder: {builder}")


def summarize_artifacts(artifacts: Iterable[ManifestArtifact]) -> dict[str, dict[str, int]]:
    summary: dict[str, dict[str, int]] = {}
    for artifact in artifacts:
        summary[artifact.name] = {
            "count": len(artifact.records),
            "by_split": summarize_records_by_split(artifact.records),
        }
    return summary
