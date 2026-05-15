from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Iterable

from traffic_incident_cv.domain.schemas import ClipRecord, PredictionRecord


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_jsonl(path: str | Path, records: Iterable[ClipRecord | PredictionRecord | dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            if hasattr(record, "to_dict"):
                payload = record.to_dict()  # type: ignore[attr-defined]
            elif is_dataclass(record):
                payload = asdict(record)
            elif hasattr(record, "__dict__"):
                payload = record.__dict__
            else:
                payload = record
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
