from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Protocol

import numpy as np

from traffic_incident_cv.data.manifest import load_manifest
from traffic_incident_cv.interfaces.base import ClipDatasetInterface, ClipSamplerInterface

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

try:
    from decord import VideoReader as DecordVideoReaderImpl
except Exception:  # pragma: no cover
    DecordVideoReaderImpl = None

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


BINARY_LABEL_TO_INDEX = {
    "no_accident": 0,
    "accident": 1,
}

SEVERITY_LABEL_TO_INDEX = {
    "minor": 0,
    "moderate": 1,
    "major": 2,
}


class VideoReaderInterface(Protocol):
    def __len__(self) -> int:
        ...

    def read_frame(self, index: int) -> np.ndarray:
        ...

    def close(self) -> None:
        ...


class OpenCvVideoReader:
    def __init__(self, video_path: str) -> None:
        if cv2 is None:  # pragma: no cover
            raise RuntimeError("opencv-python is not installed. Install project optional dependency 'train'.")
        self.video_path = video_path
        self._capture = cv2.VideoCapture(video_path)
        if not self._capture.isOpened():
            raise ValueError(f"Unable to open video: {video_path}")
        self._frame_count = int(self._capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self._next_frame_index = 0

    def __len__(self) -> int:
        return self._frame_count

    def read_frame(self, index: int) -> np.ndarray:
        if index < 0:
            raise IndexError(index)
        if index < self._next_frame_index:
            if not self._capture.set(cv2.CAP_PROP_POS_FRAMES, int(index)):
                raise IndexError(index)
            self._next_frame_index = int(index)
        while self._next_frame_index < index:
            ok, _ = self._capture.read()
            if not ok:
                raise IndexError(index)
            self._next_frame_index += 1
        ok, frame = self._capture.read()
        if not ok or frame is None:
            raise IndexError(index)
        self._next_frame_index += 1
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def close(self) -> None:
        self._capture.release()


class DecordVideoReader:
    def __init__(self, video_path: str) -> None:
        if DecordVideoReaderImpl is None:  # pragma: no cover
            raise RuntimeError("decord is not installed. Install project optional dependency 'train'.")
        self.video_path = video_path
        self._reader = DecordVideoReaderImpl(video_path)

    def __len__(self) -> int:
        return len(self._reader)

    def read_frame(self, index: int) -> np.ndarray:
        if index < 0:
            raise IndexError(index)
        try:
            return self._reader[index].asnumpy()
        except Exception as exc:
            raise IndexError(index) from exc

    def close(self) -> None:
        return None


def default_video_reader_factory(video_path: str) -> VideoReaderInterface:
    if DecordVideoReaderImpl is not None:
        return DecordVideoReader(video_path)
    return OpenCvVideoReader(video_path)


def _to_channel_first(clip: np.ndarray) -> np.ndarray:
    return np.transpose(clip, (3, 0, 1, 2))


def _normalize_clip_dtype(clip: np.ndarray) -> np.ndarray:
    if clip.dtype == np.uint8:
        return clip.astype(np.float32) / 255.0
    return clip.astype(np.float32)


@dataclass
class ManifestClipDataset(ClipDatasetInterface):
    manifest_path: str
    sampler: ClipSamplerInterface
    num_frames: int
    stride: int
    return_metadata_only: bool = True
    transform: Callable[[Any], Any] | None = None
    channel_first: bool = True
    normalize: bool = True
    as_torch_tensor: bool = False
    video_reader_factory: Callable[[str], VideoReaderInterface] = default_video_reader_factory

    def __post_init__(self) -> None:
        self.manifest_path = str(Path(self.manifest_path))
        self.records = load_manifest(self.manifest_path)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        sample = {
            "clip_id": record.clip_id,
            "video_path": record.video_path,
            "task": record.task,
            "binary_label": record.binary_label,
            "severity_label": record.severity_label,
            "target": self._resolve_target(record),
            "metadata": record.to_dict(),
        }
        if self.return_metadata_only:
            sample["frames"] = None
            sample["frame_indices"] = []
            return sample

        video_path = self.resolve_path_for_record(self.manifest_path, record)
        if not video_path.exists():
            raise FileNotFoundError(video_path)

        reader = self.video_reader_factory(str(video_path))
        try:
            total_frames = len(reader)
            if total_frames <= 0:
                raise ValueError(f"Video contains no frames: {video_path}")
            window_start, window_end = self._resolve_frame_window(record, total_frames=total_frames)
            window_length = max(window_end - window_start + 1, 1)
            frame_indices = self.sampler.sample(
                total_frames=window_length,
                num_frames=self.num_frames,
                stride=self.stride,
            )
            if not frame_indices:
                raise ValueError(f"Sampler returned no frame indices for {video_path}")
            offset_frame_indices = [min(window_start + frame_index, window_end) for frame_index in frame_indices]
            frames = [reader.read_frame(frame_index) for frame_index in offset_frame_indices]
        finally:
            reader.close()

        clip = np.stack(frames, axis=0)
        if self.normalize:
            clip = _normalize_clip_dtype(clip)
        if self.channel_first:
            clip = _to_channel_first(clip)
        processed: Any = clip
        if self.transform is not None:
            processed = self.transform(processed)
        if self.as_torch_tensor:
            if torch is None:
                raise RuntimeError("torch is not installed. Install project optional dependency 'train'.")
            if not isinstance(processed, torch.Tensor):
                processed = torch.from_numpy(np.asarray(processed))
        sample["frames"] = processed
        sample["frame_indices"] = offset_frame_indices
        sample["clip_frame_window"] = [window_start, window_end]
        sample["resolved_video_path"] = str(video_path)
        return sample

    @staticmethod
    def _resolve_target(record: Any) -> int | None:
        task = str(record.task)
        if task == "accident":
            label = record.binary_label.value if hasattr(record.binary_label, "value") else record.binary_label
            return BINARY_LABEL_TO_INDEX.get(label) if label is not None else None
        if task == "severity":
            return SEVERITY_LABEL_TO_INDEX.get(record.severity_label) if record.severity_label is not None else None
        return None

    @classmethod
    def resolve_path_for_record(cls, manifest_path: str | Path, record: Any) -> Path:
        dataset_name = getattr(record, "dataset", None)
        video_path = getattr(record, "video_path", None)
        return cls.resolve_video_path(manifest_path, str(video_path), str(dataset_name) if dataset_name is not None else None)

    @classmethod
    def resolve_video_path(cls, manifest_path: str | Path, video_path: str, dataset: str | None = None) -> Path:
        normalized = video_path.replace("\\", "/")
        raw_path = Path(normalized)
        candidates: list[Path] = []
        manifest_dir = Path(manifest_path).resolve().parent
        if raw_path.is_absolute():
            candidates.append(raw_path)
        else:
            candidates.append((manifest_dir / normalized).resolve())
            sanitized = cls._sanitize_relative_path(normalized)
            if sanitized is not None:
                volume_root = os.getenv("TRAFFIC_INCIDENT_VOLUME_ROOT")
                if volume_root:
                    candidates.append((Path(volume_root) / sanitized).resolve())
                candidates.extend(cls._ancestor_root_candidates(manifest_dir, sanitized))
                candidates.extend(cls._dataset_root_candidates(sanitized, normalized, dataset))
        seen: set[str] = set()
        for candidate in candidates:
            candidate_key = str(candidate)
            if candidate_key in seen:
                continue
            seen.add(candidate_key)
            if candidate.exists():
                return candidate
        if candidates:
            return candidates[0]
        return raw_path

    @staticmethod
    def _resolve_frame_window(record: Any, *, total_frames: int) -> tuple[int, int]:
        start_frame = getattr(record, "clip_start_frame", None)
        end_frame = getattr(record, "clip_end_frame", None)
        if start_frame is None or end_frame is None:
            start_sec = getattr(record, "clip_start_sec", None)
            end_sec = getattr(record, "clip_end_sec", None)
            if start_sec is not None or end_sec is not None:
                fps = getattr(record, "fps", None)
                if fps is None:
                    duration_sec = getattr(record, "duration_sec", None)
                    if duration_sec:
                        fps = total_frames / float(duration_sec)
                if fps is not None and fps > 0:
                    if start_sec is not None:
                        start_frame = int(round(float(start_sec) * float(fps)))
                    if end_sec is not None:
                        end_frame = max(int(round(float(end_sec) * float(fps))) - 1, 0)
        start = int(start_frame) if start_frame is not None else 0
        end = int(end_frame) if end_frame is not None else total_frames - 1
        start = max(0, min(start, total_frames - 1))
        end = max(start, min(end, total_frames - 1))
        return start, end

    @staticmethod
    def _sanitize_relative_path(video_path: str) -> Path | None:
        parts = [part for part in PurePosixPath(video_path).parts if part not in {"", ".", ".."}]
        if not parts:
            return None
        return Path(*parts)

    @staticmethod
    def _ancestor_root_candidates(manifest_dir: Path, sanitized: Path, max_depth: int = 6) -> list[Path]:
        candidates: list[Path] = []
        current = manifest_dir.resolve()
        for _ in range(max_depth):
            candidates.append((current / sanitized).resolve())
            if current.parent == current:
                break
            current = current.parent
        return candidates

    @staticmethod
    def _dataset_root_candidates(sanitized: Path, normalized: str, dataset: str | None) -> list[Path]:
        dataset_key = (dataset or "").lower()
        candidates: list[Path] = []
        mappings = [
            ("TRAFFIC_INCIDENT_AI_CITY_ROOT", "AIC21_Track1_Vehicle_Counting", {"ai_city", "accident_binary"}),
            ("TRAFFIC_INCIDENT_BALANCED_ROOT", "Balanced Accident Video Dataset", {"balanced_accident", "accident_binary"}),
        ]
        for env_name, marker, supported_datasets in mappings:
            root = os.getenv(env_name)
            if not root:
                continue
            marker_prefix = f"{marker}/"
            if marker_prefix in normalized:
                suffix = normalized.split(marker_prefix, 1)[1]
                candidates.append((Path(root) / Path(suffix)).resolve())
                continue
            if dataset_key in supported_datasets:
                candidates.append((Path(root) / sanitized).resolve())
        return candidates
