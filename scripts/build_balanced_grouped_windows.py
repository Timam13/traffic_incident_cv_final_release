from __future__ import annotations

import argparse
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import cv2

from _bootstrap import ensure_src_path

ensure_src_path()

from traffic_incident_cv.data.manifest import load_manifest, save_manifest, summarize_manifest
from traffic_incident_cv.domain.schemas import ClipRecord
from traffic_incident_cv.utils.io import ensure_dir, write_json


AUGMENTED_PREFIX_RE = re.compile(r"^aug_\d+_", flags=re.IGNORECASE)


def _normalize_base_event_id(record: ClipRecord) -> str:
    stem = Path(record.video_path).stem.lower()
    return AUGMENTED_PREFIX_RE.sub("", stem)


def _is_augmented(record: ClipRecord) -> bool:
    return "augmented_source" in set(record.tags)


def _merge_tags(*groups: Iterable[str]) -> list[str]:
    merged: list[str] = []
    for group in groups:
        for tag in group:
            if tag not in merged:
                merged.append(tag)
    return merged


def _resolve_group_severity(records: list[ClipRecord]) -> str | None:
    severities = {str(record.severity_label).lower() for record in records if record.severity_label}
    if len(severities) != 1:
        return None
    return next(iter(severities))


def _split_counts(total: int, train_ratio: float, val_ratio: float) -> tuple[int, int, int]:
    train_count = int(total * train_ratio)
    val_count = int(total * val_ratio)
    assigned = train_count + val_count
    if assigned > total:
        overflow = assigned - total
        val_count = max(val_count - overflow, 0)
        assigned = train_count + val_count
    test_count = max(total - assigned, 0)
    return train_count, val_count, test_count


def _assign_group_splits(
    severity_by_group: dict[str, str],
    *,
    seed: int,
    train_ratio: float,
    val_ratio: float,
) -> dict[str, str]:
    by_severity: dict[str, list[str]] = defaultdict(list)
    for group_id, severity in severity_by_group.items():
        by_severity[severity].append(group_id)

    assignments: dict[str, str] = {}
    for severity, group_ids in sorted(by_severity.items()):
        ordered = sorted(group_ids)
        random.Random(f"{seed}:{severity}").shuffle(ordered)
        train_count, val_count, _ = _split_counts(len(ordered), train_ratio=train_ratio, val_ratio=val_ratio)
        for index, group_id in enumerate(ordered):
            if index < train_count:
                assignments[group_id] = "train"
            elif index < train_count + val_count:
                assignments[group_id] = "val"
            else:
                assignments[group_id] = "test"
    return assignments


def _probe_video(video_path: Path) -> tuple[int, float]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Unable to open video: {video_path}")
    try:
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    finally:
        capture.release()
    if frame_count <= 0:
        raise ValueError(f"Video has no readable frames: {video_path}")
    return frame_count, fps


def _window_record(
    *,
    record: ClipRecord,
    assigned_split: str,
    base_event_id: str,
    binary_label: str,
    start_frame: int,
    end_frame: int,
    fps: float | None,
    extra_tags: list[str],
) -> ClipRecord:
    clip_duration_frames = max(end_frame - start_frame + 1, 1)
    start_sec = round(start_frame / fps, 4) if fps and fps > 0 else None
    end_sec = round((end_frame + 1) / fps, 4) if fps and fps > 0 else None
    duration_sec = round(clip_duration_frames / fps, 4) if fps and fps > 0 else None
    window_suffix = f"{start_frame:06d}_{end_frame:06d}"
    clip_role = "pos" if binary_label == "accident" else "neg"
    return ClipRecord(
        clip_id=f"{record.clip_id}__{clip_role}_{window_suffix}",
        video_path=record.video_path,
        dataset=record.dataset,
        split=assigned_split,
        task="accident",
        binary_label=binary_label,
        severity_label=record.severity_label if binary_label == "accident" else None,
        camera_id=record.camera_id,
        fps=fps,
        duration_sec=duration_sec,
        clip_start_sec=start_sec,
        clip_end_sec=end_sec,
        clip_start_frame=start_frame,
        clip_end_frame=end_frame,
        tags=_merge_tags(
            record.tags,
            [
                "manifest:balanced_grouped_windows_v1",
                f"base_event_id:{base_event_id}",
                f"grouped_split:{assigned_split}",
            ],
            extra_tags,
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build grouped symmetric Balanced-window manifests.")
    parser.add_argument(
        "--input-manifest",
        default="outputs/harmonized_manifests/bridge_v1_8fps_160/balanced_accident_binary_all_bridge_v1_8fps_160.jsonl",
    )
    parser.add_argument("--output-dir", default="data/manifests/balanced_grouped_windows_gap96")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--sample-span-frames", type=int, default=32)
    parser.add_argument("--min-total-frames", type=int, default=64)
    parser.add_argument("--train-real-only", action="store_true")
    parser.add_argument("--eval-real-only", action="store_true", default=True)
    args = parser.parse_args()

    records = load_manifest(args.input_manifest)
    grouped_records: dict[str, list[ClipRecord]] = defaultdict(list)
    for record in records:
        grouped_records[_normalize_base_event_id(record)].append(record)

    severity_by_group: dict[str, str] = {}
    severity_collisions: dict[str, list[str]] = {}
    for group_id, group_items in grouped_records.items():
        severity = _resolve_group_severity(group_items)
        if severity is None:
            labels = sorted({str(item.severity_label).lower() for item in group_items if item.severity_label})
            severity_collisions[group_id] = labels
            continue
        severity_by_group[group_id] = severity

    assignments = _assign_group_splits(
        severity_by_group,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )

    output_dir = ensure_dir(args.output_dir)
    frame_cache: dict[str, tuple[int, float]] = {}
    by_split: dict[str, list[ClipRecord]] = {"train": [], "val": [], "test": []}
    skipped = Counter()

    for base_event_id, group_items in grouped_records.items():
        assigned_split = assignments.get(base_event_id)
        if assigned_split is None:
            skipped["severity_collision_group"] += 1
            continue
        for record in group_items:
            is_augmented = _is_augmented(record)
            if assigned_split == "train" and args.train_real_only and is_augmented:
                skipped["train_augmented_filtered"] += 1
                continue
            if assigned_split in {"val", "test"} and args.eval_real_only and is_augmented:
                skipped[f"{assigned_split}_augmented_filtered"] += 1
                continue

            cached = frame_cache.get(record.video_path)
            if cached is None:
                frame_cache[record.video_path] = _probe_video(Path(record.video_path))
                cached = frame_cache[record.video_path]
            total_frames, probed_fps = cached
            fps = float(record.fps) if record.fps else probed_fps
            if total_frames < max(args.min_total_frames, args.sample_span_frames):
                skipped["too_short"] += 1
                continue

            negative_start = 0
            negative_end = args.sample_span_frames - 1
            positive_end = total_frames - 1
            positive_start = max(total_frames - args.sample_span_frames, 0)
            if negative_end >= positive_start:
                skipped["overlapping_windows"] += 1
                continue

            by_split[assigned_split].append(
                _window_record(
                    record=record,
                    assigned_split=assigned_split,
                    base_event_id=base_event_id,
                    binary_label="no_accident",
                    start_frame=negative_start,
                    end_frame=negative_end,
                    fps=fps,
                    extra_tags=[
                        "window_type:head",
                        "clip_role:no_accident_window",
                        "negative_source:same_balanced_video",
                    ],
                )
            )
            by_split[assigned_split].append(
                _window_record(
                    record=record,
                    assigned_split=assigned_split,
                    base_event_id=base_event_id,
                    binary_label="accident",
                    start_frame=positive_start,
                    end_frame=positive_end,
                    fps=fps,
                    extra_tags=[
                        "window_type:tail",
                        "clip_role:accident_window",
                    ],
                )
            )

    all_records = [*by_split["train"], *by_split["val"], *by_split["test"]]
    output_paths = {
        "train": output_dir / "balanced_grouped_windows_train.jsonl",
        "val": output_dir / "balanced_grouped_windows_val.jsonl",
        "test": output_dir / "balanced_grouped_windows_test.jsonl",
        "all": output_dir / "balanced_grouped_windows_all.jsonl",
    }
    for split_name, output_path in output_paths.items():
        split_records = all_records if split_name == "all" else by_split[split_name]
        save_manifest(output_path, split_records)

    summary = {
        "seed": args.seed,
        "input_manifest": str(Path(args.input_manifest).resolve()),
        "output_paths": {key: str(value.resolve()) for key, value in output_paths.items()},
        "config": {
            "train_ratio": args.train_ratio,
            "val_ratio": args.val_ratio,
            "sample_span_frames": args.sample_span_frames,
            "min_total_frames": args.min_total_frames,
            "train_real_only": args.train_real_only,
            "eval_real_only": args.eval_real_only,
        },
        "group_counts": {
            "total_groups": len(grouped_records),
            "usable_groups": len(assignments),
            "severity_collision_groups": len(severity_collisions),
        },
        "severity_collision_examples": dict(list(sorted(severity_collisions.items()))[:25]),
        "skipped": dict(skipped),
        "split_counts": {split: len(records) for split, records in by_split.items()},
        "split_label_counts": {
            split: summarize_manifest(records)
            for split, records in by_split.items()
        },
        "split_augmented_counts": {
            split: {
                "augmented": sum("augmented_source" in set(record.tags) for record in records),
                "real": sum("real_video" in set(record.tags) for record in records),
            }
            for split, records in by_split.items()
        },
        "assignment_counts": dict(Counter(assignments.values())),
    }
    write_json(output_dir / "balanced_grouped_windows_summary.json", summary)


if __name__ == "__main__":
    main()
