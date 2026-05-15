from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from traffic_incident_cv.domain.schemas import ClipRecord
from traffic_incident_cv.utils.io import write_jsonl


def load_manifest(path: str | Path) -> list[ClipRecord]:
    records: list[ClipRecord] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            payload = json.loads(line)
            record = ClipRecord(**payload)
            record.validate()
            records.append(record)
    return records


def save_manifest(path: str | Path, records: Iterable[ClipRecord]) -> None:
    write_jsonl(path, records)


def summarize_manifest(records: list[ClipRecord]) -> dict[str, int]:
    summary: dict[str, int] = {}
    for record in records:
        key = f"{record.task}:{record.binary_label or record.severity_label or 'none'}"
        summary[key] = summary.get(key, 0) + 1
    return summary
