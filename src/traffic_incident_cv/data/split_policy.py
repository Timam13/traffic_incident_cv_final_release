from __future__ import annotations

from collections import defaultdict

from traffic_incident_cv.domain.schemas import ClipRecord


def assert_no_camera_leakage(records: list[ClipRecord]) -> None:
    camera_to_splits: dict[str, set[str]] = defaultdict(set)
    for record in records:
        if record.camera_id:
            camera_to_splits[record.camera_id].add(str(record.split))
    bad = {
        camera: sorted(splits)
        for camera, splits in camera_to_splits.items()
        if len(splits) > 1
    }
    if bad:
        raise ValueError(f"Camera leakage detected: {bad}")
