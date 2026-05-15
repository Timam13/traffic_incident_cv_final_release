from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from traffic_incident_cv.data.manifest import summarize_manifest
from traffic_incident_cv.data.manifest_builders import (
    build_ai_city_records,
    build_artifacts_from_config,
    build_balanced_records,
    extract_camera_id,
    infer_split_from_path,
    iter_video_files,
    load_video_stats,
)
from traffic_incident_cv.data.split_policy import assert_no_camera_leakage


@dataclass(slots=True)
class AuditResult:
    dataset: str
    ready: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _compare_expected(
    *,
    actual: Any,
    expected: Any,
    label: str,
    errors: list[str],
) -> None:
    if expected is None:
        return
    if actual != expected:
        errors.append(f"{label}: expected {expected!r}, got {actual!r}")


def _normalize_split_counts(counts: dict[str, int], split_names: list[str]) -> dict[str, int]:
    return {split: int(counts.get(split, 0)) for split in split_names}


def _normalize_nested_counts(
    counts: dict[str, dict[str, int]],
    split_names: list[str],
    class_names: list[str],
) -> dict[str, dict[str, int]]:
    normalized: dict[str, dict[str, int]] = {}
    for split in split_names:
        normalized[split] = {class_name: int(counts.get(split, {}).get(class_name, 0)) for class_name in class_names}
    return normalized


def _collect_video_paths(dataset: dict) -> list[Path]:
    root = Path(dataset["root_dir"])
    video_extensions = {suffix.lower() for suffix in dataset.get("video_extensions", {".mp4", ".avi", ".mov", ".mkv"})}
    return list(
        iter_video_files(
            root,
            video_extensions=video_extensions,
            include_substrings=dataset.get("include_substrings"),
            exclude_substrings=dataset.get("exclude_substrings"),
        )
    )


def audit_ai_city_dataset(cfg: dict) -> AuditResult:
    dataset = cfg["dataset"]
    errors: list[str] = []
    warnings: list[str] = []
    expectations = dataset.get("expectations", {})
    split_names = list(dataset.get("split_patterns", {}).keys())
    try:
        video_paths = _collect_video_paths(dataset)
    except FileNotFoundError as exc:
        errors.append(f"Missing dataset root: {exc}")
        video_paths = []
    try:
        stats = load_video_stats(dataset.get("stats_path"))
    except FileNotFoundError as exc:
        errors.append(f"Missing stats file: {exc}")
        stats = {}
    split_counts: dict[str, int] = defaultdict(int)
    camera_ids: set[str] = set()
    condition_counts: dict[str, int] = defaultdict(int)
    for video_path in video_paths:
        split = infer_split_from_path(video_path, dataset.get("split_patterns", {}))
        if split is not None:
            split_counts[split] += 1
        camera_id = extract_camera_id(video_path, dataset.get("camera_regex"))
        if camera_id is not None:
            camera_ids.add(camera_id)
        stem = video_path.stem.lower()
        for suffix, tag in dataset.get("tags_by_suffix", {}).items():
            normalized_suffix = suffix.lower().lstrip("_")
            if stem.endswith(f"_{normalized_suffix}"):
                condition_counts[tag] += 1

    video_names = {path.name for path in video_paths}
    stats_names = set(stats)
    missing_stats = sorted(video_names - stats_names)
    extra_stats = sorted(stats_names - video_names)
    if missing_stats:
        errors.append(f"Missing stats rows for {len(missing_stats)} AI City videos.")
    if extra_stats:
        warnings.append(f"Stats file contains {len(extra_stats)} extra rows not found on disk.")

    records = []
    if not errors:
        try:
            records = build_ai_city_records(cfg)
            assert_no_camera_leakage(records)
        except Exception as exc:
            errors.append(str(exc))
            records = []

    _compare_expected(actual=len(video_paths), expected=expectations.get("total_videos"), label="AI City total_videos", errors=errors)
    normalized_split_counts = _normalize_split_counts(split_counts, split_names) if split_names else dict(sorted(split_counts.items()))
    _compare_expected(actual=normalized_split_counts, expected=expectations.get("split_counts"), label="AI City split_counts", errors=errors)
    _compare_expected(actual=len(camera_ids), expected=expectations.get("unique_cameras"), label="AI City unique_cameras", errors=errors)
    _compare_expected(actual=len(stats), expected=expectations.get("stats_rows"), label="AI City stats_rows", errors=errors)
    assigned_split_counter: dict[str, int] = defaultdict(int)
    for record in records:
        assigned_split_counter[str(record.split)] += 1
    if records:
        assigned_split_counts = _normalize_split_counts(assigned_split_counter, split_names)
    else:
        assigned_split_counts = _normalize_split_counts({}, split_names)
    _compare_expected(
        actual=assigned_split_counts,
        expected=expectations.get("manifest_split_counts"),
        label="AI City manifest_split_counts",
        errors=errors,
    )

    summary = {
        "root_dir": dataset["root_dir"],
        "total_videos": len(video_paths),
        "split_counts": normalized_split_counts,
        "manifest_split_counts": assigned_split_counts,
        "unique_cameras": len(camera_ids),
        "stats_rows": len(stats),
        "missing_stats_rows": missing_stats,
        "extra_stats_rows": extra_stats,
        "condition_counts": dict(sorted(condition_counts.items())),
        "manifest_count": len(records),
        "label_summary": summarize_manifest(records) if records else {},
    }
    return AuditResult(dataset=dataset["name"], ready=not errors, errors=errors, warnings=warnings, summary=summary)


def audit_balanced_accident_dataset(cfg: dict) -> AuditResult:
    dataset = cfg["dataset"]
    errors: list[str] = []
    warnings: list[str] = []
    expectations = dataset.get("expectations", {})
    split_names = list(dataset.get("split_patterns", {}).keys())
    try:
        video_paths = _collect_video_paths(dataset)
    except FileNotFoundError as exc:
        errors.append(f"Missing dataset root: {exc}")
        video_paths = []
    split_counts: dict[str, int] = defaultdict(int)
    class_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    augmented_counts: dict[str, int] = defaultdict(int)
    augmented_class_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    real_counts: dict[str, int] = defaultdict(int)

    severity_dirs = {value.lower() for value in dataset.get("severity_dirs", ["minor", "moderate", "major"])}
    class_names = sorted(severity_dirs)
    augmented_prefixes = tuple(prefix.lower() for prefix in dataset.get("augmented_prefixes", ["aug"]))
    for video_path in video_paths:
        split = infer_split_from_path(video_path, dataset.get("split_patterns", {}))
        if split is None:
            errors.append(f"Unable to infer split for {video_path}")
            continue
        severity_label = next((part.lower() for part in video_path.parts if part.lower() in severity_dirs), None)
        if severity_label is None:
            errors.append(f"Unable to infer severity label for {video_path}")
            continue
        split_counts[split] += 1
        class_counts[split][severity_label] += 1
        is_augmented = video_path.name.lower().startswith(augmented_prefixes)
        if is_augmented:
            augmented_counts[split] += 1
            augmented_class_counts[split][severity_label] += 1
        else:
            real_counts[split] += 1

    binary_records = []
    severity_records = []
    if not errors:
        try:
            binary_records, severity_records = build_balanced_records(cfg)
        except Exception as exc:
            errors.append(str(exc))
            binary_records = []
            severity_records = []

    _compare_expected(actual=len(video_paths), expected=expectations.get("total_videos"), label="Balanced total_videos", errors=errors)
    normalized_split_counts = _normalize_split_counts(split_counts, split_names) if split_names else dict(sorted(split_counts.items()))
    normalized_class_counts = _normalize_nested_counts(class_counts, split_names, class_names)
    normalized_augmented_counts = _normalize_split_counts(augmented_counts, split_names)
    normalized_real_counts = _normalize_split_counts(real_counts, split_names)
    normalized_augmented_class_counts = _normalize_nested_counts(augmented_class_counts, split_names, class_names)
    _compare_expected(actual=normalized_split_counts, expected=expectations.get("split_counts"), label="Balanced split_counts", errors=errors)
    _compare_expected(
        actual=normalized_class_counts,
        expected=expectations.get("class_counts"),
        label="Balanced class_counts",
        errors=errors,
    )
    _compare_expected(actual=normalized_augmented_counts, expected=expectations.get("augmented_counts"), label="Balanced augmented_counts", errors=errors)
    _compare_expected(actual=normalized_real_counts, expected=expectations.get("real_counts"), label="Balanced real_counts", errors=errors)
    _compare_expected(
        actual=normalized_augmented_class_counts,
        expected=expectations.get("augmented_class_counts"),
        label="Balanced augmented_class_counts",
        errors=errors,
    )

    summary = {
        "root_dir": dataset["root_dir"],
        "total_videos": len(video_paths),
        "split_counts": normalized_split_counts,
        "class_counts": normalized_class_counts,
        "augmented_counts": normalized_augmented_counts,
        "real_counts": normalized_real_counts,
        "augmented_class_counts": normalized_augmented_class_counts,
        "binary_manifest_count": len(binary_records),
        "severity_manifest_count": len(severity_records),
    }
    return AuditResult(dataset=dataset["name"], ready=not errors, errors=errors, warnings=warnings, summary=summary)


def audit_manifest_bundle(
    cfg: dict,
    manifest_cache: dict[str, list[Any]] | None = None,
) -> AuditResult:
    dataset = cfg["dataset"]
    errors: list[str] = []
    warnings: list[str] = []
    expectations = dataset.get("expectations", {})
    manifest_paths = dataset.get("input_manifests", [])
    missing_manifests = [
        path
        for path in manifest_paths
        if not Path(path).exists() and (manifest_cache is None or path not in manifest_cache)
    ]
    if missing_manifests:
        errors.append(f"Missing input manifests: {missing_manifests}")
        artifacts = []
    else:
        try:
            artifacts = build_artifacts_from_config(cfg, manifest_cache=manifest_cache)  # type: ignore[arg-type]
        except Exception as exc:
            errors.append(str(exc))
            artifacts = []

    all_records = next((artifact.records for artifact in artifacts if artifact.name == "all"), [])
    split_counts: dict[str, int] = defaultdict(int)
    for record in all_records:
        split_counts[str(record.split)] += 1
    split_names = list(dataset.get("output_manifests", {}).keys())
    normalized_split_counts = _normalize_split_counts(split_counts, split_names) if split_names else dict(sorted(split_counts.items()))
    _compare_expected(actual=len(all_records), expected=expectations.get("total_records"), label="Merged manifest total_records", errors=errors)
    _compare_expected(actual=normalized_split_counts, expected=expectations.get("split_counts"), label="Merged manifest split_counts", errors=errors)

    summary = {
        "input_manifests": manifest_paths,
        "input_manifest_exists": {
            path: Path(path).exists() or (manifest_cache is not None and path in manifest_cache)
            for path in manifest_paths
        },
        "artifact_counts": {artifact.name: len(artifact.records) for artifact in artifacts},
        "split_counts": normalized_split_counts,
        "label_summary": summarize_manifest(all_records) if all_records else {},
    }
    return AuditResult(dataset=dataset["name"], ready=not errors, errors=errors, warnings=warnings, summary=summary)


def audit_dataset_from_config(
    cfg: dict,
    manifest_cache: dict[str, list[Any]] | None = None,
) -> AuditResult:
    builder = cfg["dataset"].get("builder", cfg["dataset"]["name"])
    if builder == "ai_city":
        return audit_ai_city_dataset(cfg)
    if builder == "balanced_accident":
        return audit_balanced_accident_dataset(cfg)
    if builder in {"merged_manifest", "accident_clip_bundle"}:
        return audit_manifest_bundle(cfg, manifest_cache=manifest_cache)
    raise ValueError(f"Unsupported audit builder: {builder}")
