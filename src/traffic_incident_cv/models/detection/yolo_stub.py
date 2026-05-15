from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

try:
    from ultralytics import YOLO as UltralyticsYOLO
except Exception:  # pragma: no cover
    UltralyticsYOLO = None


@dataclass
class YoloDetectorAdapter:
    weights: str = "yolo11n.pt"
    conf_threshold: float = 0.25
    iou_threshold: float = 0.45
    device: str | None = None
    max_det: int = 20
    target_class_ids: tuple[int, ...] = (2, 3, 5, 7)
    num_sampled_frames: int = 2
    expand_ratio: float = 0.12
    min_crop_ratio: float = 0.25
    require_backend: bool = False
    _model: Any = field(default=None, init=False, repr=False)

    def build(self) -> None:
        if self._model is not None:
            return
        if UltralyticsYOLO is None:
            if self.require_backend:
                raise RuntimeError(
                    "ultralytics YOLO backend is unavailable. Install project optional dependency 'train'."
                )
            return
        try:
            self._model = UltralyticsYOLO(self.weights)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to initialize ultralytics YOLO from weights={self.weights!r}. "
                "Ensure the weights file exists locally or that direct download is available."
            ) from exc

    def detect(self, frames: Any) -> list[dict[str, Any]]:
        return self.detect_batch_rois(frames)

    def detect_batch_rois(self, frames: Any) -> list[dict[str, Any]]:
        clip = self._to_clip_batch(frames)
        batch_size, _, time_steps, height, width = clip.shape
        if time_steps <= 0:
            return [self._empty_roi(sample_index, width, height) for sample_index in range(batch_size)]
        self.build()
        if self._model is None:
            return [self._empty_roi(sample_index, width, height) for sample_index in range(batch_size)]

        frame_indices = self._select_frame_indices(time_steps)
        detections: list[dict[str, Any]] = []
        for sample_index in range(batch_size):
            images = [self._to_hwc_uint8(clip[sample_index, :, frame_index]) for frame_index in frame_indices]
            results = self._model.predict(
                source=images,
                conf=float(self.conf_threshold),
                iou=float(self.iou_threshold),
                device=self._resolve_device(),
                max_det=int(self.max_det),
                verbose=False,
            )
            boxes: list[tuple[float, float, float, float]] = []
            scores: list[float] = []
            objects: list[dict[str, Any]] = []
            for result in results:
                result_boxes = getattr(result, "boxes", None)
                if result_boxes is None:
                    continue
                xyxy_rows = getattr(result_boxes, "xyxy", None)
                conf_rows = getattr(result_boxes, "conf", None)
                cls_rows = getattr(result_boxes, "cls", None)
                if xyxy_rows is None or conf_rows is None or cls_rows is None:
                    continue
                xyxy_np = self._to_numpy(xyxy_rows)
                conf_np = self._to_numpy(conf_rows)
                cls_np = self._to_numpy(cls_rows)
                for coords, score, class_id in zip(xyxy_np, conf_np, cls_np):
                    if self.target_class_ids and int(class_id) not in self.target_class_ids:
                        continue
                    raw_box = tuple(float(value) for value in coords.tolist())
                    expanded_box = self._expand_box(raw_box, width=width, height=height)
                    boxes.append(raw_box)
                    scores.append(float(score))
                    objects.append(
                        {
                            "class_id": int(class_id),
                            "score": float(score),
                            "raw_xyxy": tuple(int(round(value)) for value in raw_box),
                            "roi_xyxy": expanded_box,
                            "used_detector": True,
                        }
                    )
            roi = self._merge_boxes(boxes, width=width, height=height)
            if roi is None:
                detections.append(self._empty_roi(sample_index, width, height))
                continue
            objects.sort(key=lambda item: float(item["score"]), reverse=True)
            detections.append(
                {
                    "sample_index": sample_index,
                    "roi_xyxy": roi,
                    "mean_score": float(sum(scores) / len(scores)) if scores else None,
                    "max_score": max(scores) if scores else None,
                    "box_count": len(boxes),
                    "frame_indices": frame_indices,
                    "used_detector": True,
                    "objects": objects,
                }
            )
        return detections

    def _resolve_device(self) -> str:
        if self.device:
            return self.device
        if torch is not None and torch.cuda.is_available():
            return "cuda:0"
        return "cpu"

    def _select_frame_indices(self, total_frames: int) -> list[int]:
        count = max(1, min(int(self.num_sampled_frames), total_frames))
        if count == 1:
            return [total_frames // 2]
        stride = (total_frames - 1) / float(count - 1)
        return sorted({int(round(index * stride)) for index in range(count)})

    def _merge_boxes(
        self,
        boxes: list[tuple[float, float, float, float]],
        *,
        width: int,
        height: int,
    ) -> tuple[int, int, int, int] | None:
        if not boxes:
            return None
        x1 = min(item[0] for item in boxes)
        y1 = min(item[1] for item in boxes)
        x2 = max(item[2] for item in boxes)
        y2 = max(item[3] for item in boxes)
        box_w = max(x2 - x1, 1.0)
        box_h = max(y2 - y1, 1.0)
        pad_w = box_w * float(self.expand_ratio)
        pad_h = box_h * float(self.expand_ratio)
        min_w = width * float(self.min_crop_ratio)
        min_h = height * float(self.min_crop_ratio)
        expanded_w = max(box_w + 2.0 * pad_w, min_w)
        expanded_h = max(box_h + 2.0 * pad_h, min_h)
        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0
        left = max(0.0, center_x - expanded_w / 2.0)
        top = max(0.0, center_y - expanded_h / 2.0)
        right = min(float(width), center_x + expanded_w / 2.0)
        bottom = min(float(height), center_y + expanded_h / 2.0)
        left = min(left, max(0.0, right - min_w))
        top = min(top, max(0.0, bottom - min_h))
        return (
            int(round(left)),
            int(round(top)),
            max(int(round(right)), int(round(left)) + 1),
            max(int(round(bottom)), int(round(top)) + 1),
        )

    def _expand_box(
        self,
        box: tuple[float, float, float, float],
        *,
        width: int,
        height: int,
    ) -> tuple[int, int, int, int]:
        x1, y1, x2, y2 = box
        box_w = max(x2 - x1, 1.0)
        box_h = max(y2 - y1, 1.0)
        pad_w = box_w * float(self.expand_ratio)
        pad_h = box_h * float(self.expand_ratio)
        min_w = width * float(self.min_crop_ratio)
        min_h = height * float(self.min_crop_ratio)
        expanded_w = max(box_w + 2.0 * pad_w, min_w)
        expanded_h = max(box_h + 2.0 * pad_h, min_h)
        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0
        left = max(0.0, center_x - expanded_w / 2.0)
        top = max(0.0, center_y - expanded_h / 2.0)
        right = min(float(width), center_x + expanded_w / 2.0)
        bottom = min(float(height), center_y + expanded_h / 2.0)
        left = min(left, max(0.0, right - min_w))
        top = min(top, max(0.0, bottom - min_h))
        return (
            int(round(left)),
            int(round(top)),
            max(int(round(right)), int(round(left)) + 1),
            max(int(round(bottom)), int(round(top)) + 1),
        )

    @staticmethod
    def _empty_roi(sample_index: int, width: int, height: int) -> dict[str, Any]:
        return {
            "sample_index": sample_index,
            "roi_xyxy": (0, 0, width, height),
            "mean_score": None,
            "max_score": None,
            "box_count": 0,
            "frame_indices": [],
            "used_detector": False,
            "objects": [],
        }

    @staticmethod
    def _to_clip_batch(frames: Any) -> Any:
        if torch is None:
            raise RuntimeError("torch is not installed. Install project optional dependency 'train'.")
        if not isinstance(frames, torch.Tensor):
            clip = torch.as_tensor(frames, dtype=torch.float32)
        else:
            clip = frames.detach().float()
        if clip.ndim == 4:
            clip = clip.unsqueeze(0)
        if clip.ndim != 5:
            raise ValueError(f"Expected clip tensor with shape [B, C, T, H, W], got {tuple(clip.shape)}")
        return clip.cpu()

    @staticmethod
    def _to_hwc_uint8(frame: Any) -> np.ndarray:
        if torch is not None and isinstance(frame, torch.Tensor):
            array = frame.detach().cpu().numpy()
        else:
            array = np.asarray(frame)
        if array.ndim != 3:
            raise ValueError(f"Expected frame tensor with shape [C, H, W], got {tuple(array.shape)}")
        hwc = np.transpose(array, (1, 2, 0))
        hwc = np.clip(hwc, 0.0, 1.0)
        return (hwc * 255.0).astype(np.uint8)

    @staticmethod
    def _to_numpy(value: Any) -> np.ndarray:
        if hasattr(value, "detach"):
            return value.detach().cpu().numpy()
        return np.asarray(value)
