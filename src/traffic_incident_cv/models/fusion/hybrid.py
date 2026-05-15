from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from traffic_incident_cv.models.base import BaseModelStub
from traffic_incident_cv.models.detection.yolo_stub import YoloDetectorAdapter
from traffic_incident_cv.models.encoders.convnext_stub import ConvNeXtClipClassifier

try:
    import torch
    from torch import nn
    import torch.nn.functional as F
except Exception:  # pragma: no cover
    torch = None
    nn = None
    F = None


TorchModuleBase = nn.Module if nn is not None else object


@dataclass(eq=False)
class HybridYoloConvNeXtModel(BaseModelStub, TorchModuleBase):
    detector: YoloDetectorAdapter | None = None
    global_encoder: ConvNeXtClipClassifier | None = field(init=False, repr=False)
    local_encoder: ConvNeXtClipClassifier | None = field(init=False, repr=False)
    fusion_mode: str = "concat"
    use_global_branch: bool = True
    use_local_branch: bool = True
    encoder_name: str = "convnext_tiny"
    pretrained: bool = True
    local_crop_top: float = 0.35
    local_crop_left: float = 0.10
    local_crop_bottom: float = 1.0
    local_crop_right: float = 0.90
    global_logit_scale: float = 1.0
    local_logit_scale: float = 1.0
    use_detector_guidance: bool = False
    detector_weights: str = "yolo11n.pt"
    detector_conf_threshold: float = 0.25
    detector_iou_threshold: float = 0.45
    detector_device: str | None = None
    detector_max_det: int = 20
    detector_target_class_ids: tuple[int, ...] = (2, 3, 5, 7)
    detector_num_sampled_frames: int = 2
    detector_expand_ratio: float = 0.12
    detector_min_crop_ratio: float = 0.25
    detector_require_backend: bool = False
    detector_flow_mode: str = "union_roi"
    detector_object_pooling: str = "score_weighted_mean"
    detector_max_objects_per_sample: int = 3
    fusion_head: Any = field(init=False, repr=False)
    feature_dim: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if nn is not None:
            nn.Module.__init__(self)
        self.global_encoder = None
        self.local_encoder = None
        self.fusion_head = None

    def build(self) -> None:
        if self.detector is None:
            self.detector = YoloDetectorAdapter(
                weights=self.detector_weights,
                conf_threshold=self.detector_conf_threshold,
                iou_threshold=self.detector_iou_threshold,
                device=self.detector_device,
                max_det=self.detector_max_det,
                target_class_ids=self.detector_target_class_ids,
                num_sampled_frames=self.detector_num_sampled_frames,
                expand_ratio=self.detector_expand_ratio,
                min_crop_ratio=self.detector_min_crop_ratio,
                require_backend=self.detector_require_backend,
            )
        if self.use_global_branch and self.global_encoder is None:
            self.global_encoder = ConvNeXtClipClassifier(
                name="global_convnext",
                num_classes=self.num_classes,
                encoder_name=self.encoder_name,
                pretrained=self.pretrained,
            )
        if self.use_local_branch and self.local_encoder is None:
            self.local_encoder = ConvNeXtClipClassifier(
                name="local_convnext",
                num_classes=self.num_classes,
                encoder_name=self.encoder_name,
                pretrained=self.pretrained,
            )
        if self.global_encoder is not None:
            self.global_encoder.build()
        if self.local_encoder is not None:
            self.local_encoder.build()
        if self.fusion_head is None and self.fusion_mode == "concat":
            if nn is None:
                raise RuntimeError("torch is not installed. Install project optional dependency 'train'.")
            dims = 0
            if self.use_global_branch and self.global_encoder is not None:
                dims += int(self.global_encoder.feature_dim)
            if self.use_local_branch and self.local_encoder is not None:
                dims += int(self.local_encoder.feature_dim)
            if dims <= 0:
                raise ValueError("Hybrid model requires at least one active branch.")
            self.feature_dim = dims
            self.fusion_head = nn.Linear(dims, self.num_classes)

    def forward(self, clip: Any) -> Any:
        self.build()
        if torch is None:
            raise RuntimeError("torch is not installed. Install project optional dependency 'train'.")
        if not isinstance(clip, torch.Tensor):
            clip = torch.as_tensor(clip, dtype=torch.float32)
        else:
            clip = clip.float()
        if clip.ndim == 4:
            clip = clip.unsqueeze(0)
        if clip.ndim != 5:
            raise ValueError(f"Expected clip tensor with shape [B, C, T, H, W], got {tuple(clip.shape)}")

        global_embedding = None
        local_embedding = None
        global_logits = None
        local_logits = None

        if self.use_global_branch and self.global_encoder is not None:
            global_embedding = self.global_encoder.encode_clip(clip)
            global_logits = self.global_encoder.classify_embedding(global_embedding) * float(self.global_logit_scale)

        if self.use_local_branch and self.local_encoder is not None:
            if self.use_detector_guidance and self.detector_flow_mode == "object_pool":
                local_embedding, local_logits = self._forward_local_object_pool(clip)
            else:
                local_clip = self._crop_clip(clip)
                local_embedding = self.local_encoder.encode_clip(local_clip)
                local_logits = self.local_encoder.classify_embedding(local_embedding) * float(self.local_logit_scale)

        if self.fusion_mode == "concat":
            embeddings = [item for item in (global_embedding, local_embedding) if item is not None]
            if not embeddings:
                raise ValueError("Hybrid concat fusion requires at least one active branch.")
            fused = torch.cat(embeddings, dim=1) if len(embeddings) > 1 else embeddings[0]
            return self.fusion_head(fused)
        if self.fusion_mode == "mean_logits":
            logits = [item for item in (global_logits, local_logits) if item is not None]
            if not logits:
                raise ValueError("Hybrid mean_logits fusion requires at least one active branch.")
            if len(logits) == 1:
                return logits[0]
            return torch.stack(logits, dim=0).mean(dim=0)
        raise ValueError(f"Unsupported fusion_mode={self.fusion_mode!r}")

    def summary(self) -> dict[str, Any]:
        payload = super().summary()
        payload.update(
            {
                "fusion_mode": self.fusion_mode,
                "use_global_branch": self.use_global_branch,
                "use_local_branch": self.use_local_branch,
                "encoder_name": self.encoder_name,
                "pretrained": self.pretrained,
                "use_detector_guidance": self.use_detector_guidance,
                "local_crop": {
                    "top": self.local_crop_top,
                    "left": self.local_crop_left,
                    "bottom": self.local_crop_bottom,
                    "right": self.local_crop_right,
                },
                "detector": {
                    "weights": self.detector_weights,
                    "conf_threshold": self.detector_conf_threshold,
                    "iou_threshold": self.detector_iou_threshold,
                    "device": self.detector_device,
                    "max_det": self.detector_max_det,
                    "target_class_ids": list(self.detector_target_class_ids),
                    "num_sampled_frames": self.detector_num_sampled_frames,
                    "expand_ratio": self.detector_expand_ratio,
                    "min_crop_ratio": self.detector_min_crop_ratio,
                    "require_backend": self.detector_require_backend,
                    "flow_mode": self.detector_flow_mode,
                    "object_pooling": self.detector_object_pooling,
                    "max_objects_per_sample": self.detector_max_objects_per_sample,
                },
            }
        )
        return payload

    def _crop_clip(self, clip: Any) -> Any:
        if self.use_detector_guidance and self.detector_flow_mode == "topk_union":
            return self._crop_clip_with_topk_union(clip)
        if self.use_detector_guidance:
            return self._crop_clip_with_detector(clip)
        return self._crop_clip_without_detector(clip)

    def _crop_clip_without_detector(self, clip: Any) -> Any:
        _, _, _, height, width = clip.shape
        top = max(0, min(height - 1, int(height * float(self.local_crop_top))))
        left = max(0, min(width - 1, int(width * float(self.local_crop_left))))
        bottom = max(top + 1, min(height, int(height * float(self.local_crop_bottom))))
        right = max(left + 1, min(width, int(width * float(self.local_crop_right))))
        return clip[:, :, :, top:bottom, left:right]

    def _crop_clip_with_detector(self, clip: Any) -> Any:
        if self.detector is None:
            raise RuntimeError("Detector guidance requested but detector is not initialized.")
        if torch is None or F is None:
            raise RuntimeError("torch is not installed. Install project optional dependency 'train'.")
        batch_size, _, _, height, width = clip.shape
        rois = self.detector.detect_batch_rois(clip)
        if len(rois) != batch_size:
            raise ValueError(f"Detector returned {len(rois)} roi entries for batch size {batch_size}.")
        cropped_clips: list[Any] = []
        for sample_index, roi in enumerate(rois):
            left, top, right, bottom = roi["roi_xyxy"]
            sample_clip = clip[sample_index : sample_index + 1, :, :, top:bottom, left:right]
            resized = self._resize_sample_clip(sample_clip, height=height, width=width)
            cropped_clips.append(resized)
        return torch.cat(cropped_clips, dim=0)

    def _crop_clip_with_topk_union(self, clip: Any) -> Any:
        if self.detector is None:
            raise RuntimeError("Detector guidance requested but detector is not initialized.")
        if torch is None or F is None:
            raise RuntimeError("torch is not installed. Install project optional dependency 'train'.")
        batch_size, _, _, height, width = clip.shape
        detections = self.detector.detect_batch_rois(clip)
        if len(detections) != batch_size:
            raise ValueError(f"Detector returned {len(detections)} roi entries for batch size {batch_size}.")
        cropped_clips: list[Any] = []
        for sample_index, detection in enumerate(detections):
            sample_clip = clip[sample_index : sample_index + 1]
            roi = self._resolve_topk_union_roi(detection, width=width, height=height)
            left, top, right, bottom = roi
            crop = sample_clip[:, :, :, top:bottom, left:right]
            cropped_clips.append(self._resize_sample_clip(crop, height=height, width=width))
        return torch.cat(cropped_clips, dim=0)

    def _forward_local_object_pool(self, clip: Any) -> tuple[Any, Any]:
        if self.detector is None or self.local_encoder is None:
            raise RuntimeError("Object-driven detector flow requires initialized detector and local encoder.")
        if torch is None:
            raise RuntimeError("torch is not installed. Install project optional dependency 'train'.")
        batch_size, _, _, height, width = clip.shape
        detections = self.detector.detect_batch_rois(clip)
        if len(detections) != batch_size:
            raise ValueError(f"Detector returned {len(detections)} entries for batch size {batch_size}.")

        pooled_embeddings: list[Any] = []
        pooled_logits: list[Any] = []
        for sample_index, detection in enumerate(detections):
            sample_clip = clip[sample_index : sample_index + 1]
            object_crops = self._extract_object_crops(
                sample_clip,
                detection=detection,
                height=height,
                width=width,
            )
            embeddings = self.local_encoder.encode_clip(object_crops)
            logits = self.local_encoder.classify_embedding(embeddings) * float(self.local_logit_scale)
            pooled_embeddings.append(self._pool_object_tensor(embeddings, detection=detection))
            pooled_logits.append(self._pool_object_tensor(logits, detection=detection))
        return torch.cat(pooled_embeddings, dim=0), torch.cat(pooled_logits, dim=0)

    def _extract_object_crops(
        self,
        sample_clip: Any,
        *,
        detection: dict[str, Any],
        height: int,
        width: int,
    ) -> Any:
        if torch is None:
            raise RuntimeError("torch is not installed. Install project optional dependency 'train'.")
        objects = list(detection.get("objects", []) or [])
        if objects:
            max_objects = max(1, int(self.detector_max_objects_per_sample))
            selected = sorted(objects, key=lambda item: float(item.get("score", 0.0)), reverse=True)[:max_objects]
            object_crops = []
            for item in selected:
                left, top, right, bottom = item["roi_xyxy"]
                crop = sample_clip[:, :, :, top:bottom, left:right]
                object_crops.append(self._resize_sample_clip(crop, height=height, width=width))
            return torch.cat(object_crops, dim=0)

        if detection.get("used_detector"):
            left, top, right, bottom = detection["roi_xyxy"]
            crop = sample_clip[:, :, :, top:bottom, left:right]
        else:
            crop = self._crop_clip_without_detector(sample_clip)
        return self._resize_sample_clip(crop, height=height, width=width)

    def _pool_object_tensor(self, tensor: Any, *, detection: dict[str, Any]) -> Any:
        if torch is None:
            raise RuntimeError("torch is not installed. Install project optional dependency 'train'.")
        if tensor.ndim != 2:
            raise ValueError(f"Expected pooled tensor with shape [N, D], got {tuple(tensor.shape)}")
        if tensor.shape[0] == 1:
            return tensor
        pooling = str(self.detector_object_pooling or "score_weighted_mean").lower()
        if pooling == "max":
            return tensor.max(dim=0, keepdim=True).values
        if pooling == "mean":
            return tensor.mean(dim=0, keepdim=True)
        if pooling != "score_weighted_mean":
            raise ValueError(f"Unsupported detector_object_pooling={self.detector_object_pooling!r}")
        objects = list(detection.get("objects", []) or [])
        scores = [float(item.get("score", 0.0)) for item in objects[: tensor.shape[0]]]
        if len(scores) != tensor.shape[0] or sum(scores) <= 0.0:
            return tensor.mean(dim=0, keepdim=True)
        weights = torch.as_tensor(scores, dtype=tensor.dtype, device=tensor.device).unsqueeze(1)
        weights = weights / weights.sum()
        return (tensor * weights).sum(dim=0, keepdim=True)

    def _resolve_topk_union_roi(
        self,
        detection: dict[str, Any],
        *,
        width: int,
        height: int,
    ) -> tuple[int, int, int, int]:
        objects = list(detection.get("objects", []) or [])
        if not objects:
            return tuple(detection.get("roi_xyxy", (0, 0, width, height)))
        max_objects = max(1, int(self.detector_max_objects_per_sample))
        selected = sorted(objects, key=lambda item: float(item.get("score", 0.0)), reverse=True)[:max_objects]
        left = min(int(item["roi_xyxy"][0]) for item in selected)
        top = min(int(item["roi_xyxy"][1]) for item in selected)
        right = max(int(item["roi_xyxy"][2]) for item in selected)
        bottom = max(int(item["roi_xyxy"][3]) for item in selected)
        left = max(0, min(left, width - 1))
        top = max(0, min(top, height - 1))
        right = max(left + 1, min(right, width))
        bottom = max(top + 1, min(bottom, height))
        return left, top, right, bottom

    @staticmethod
    def _resize_sample_clip(sample_clip: Any, *, height: int, width: int) -> Any:
        if torch is None or F is None:
            raise RuntimeError("torch is not installed. Install project optional dependency 'train'.")
        _, channels, time_steps, _, _ = sample_clip.shape
        frames = sample_clip.permute(0, 2, 1, 3, 4).reshape(time_steps, channels, sample_clip.shape[-2], sample_clip.shape[-1])
        resized = F.interpolate(frames, size=(height, width), mode="bilinear", align_corners=False)
        return resized.reshape(1, time_steps, channels, height, width).permute(0, 2, 1, 3, 4).contiguous()
