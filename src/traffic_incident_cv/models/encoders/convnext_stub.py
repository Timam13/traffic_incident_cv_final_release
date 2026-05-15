from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from traffic_incident_cv.models.base import BaseModelStub

try:
    import torch
    from torch import nn
except Exception:  # pragma: no cover
    torch = None
    nn = None

try:
    from torchvision import models as tv_models
except Exception:  # pragma: no cover
    tv_models = None


TorchModuleBase = nn.Module if nn is not None else object


if nn is not None:

    class _FallbackFrameEncoder(nn.Module):
        def __init__(self, in_channels: int = 3, feature_dim: int = 128) -> None:
            super().__init__()
            self.layers = nn.Sequential(
                nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, feature_dim, kernel_size=3, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1)),
            )

        def forward(self, x: Any) -> Any:
            return self.layers(x).flatten(1)


    class _TorchvisionConvNeXtEncoder(nn.Module):
        def __init__(self, backbone: Any) -> None:
            super().__init__()
            self.features = backbone.features
            self.avgpool = backbone.avgpool

        def forward(self, x: Any) -> Any:
            x = self.features(x)
            x = self.avgpool(x)
            return torch.flatten(x, 1)

else:  # pragma: no cover

    class _FallbackFrameEncoder:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("torch is not installed. Install project optional dependency 'train'.")


    class _TorchvisionConvNeXtEncoder:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("torch is not installed. Install project optional dependency 'train'.")


@dataclass(eq=False)
class ConvNeXtClipClassifier(BaseModelStub, TorchModuleBase):
    encoder_name: str = "convnext_tiny"
    temporal_pooling: str = "mean"
    pretrained: bool = True
    allow_encoder_fallback: bool = False
    encoder: Any = field(init=False, repr=False)
    classifier_head: Any = field(init=False, repr=False)
    feature_dim: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if nn is not None:
            nn.Module.__init__(self)
        self.encoder = None
        self.classifier_head = None

    def build(self) -> None:
        if self.encoder is not None:
            return
        if torch is None or nn is None:
            raise RuntimeError("torch is not installed. Install project optional dependency 'train'.")
        self.encoder, self.feature_dim = self._build_frame_encoder()
        self.classifier_head = nn.Linear(self.feature_dim, self.num_classes)

    def forward(self, clip: Any) -> Any:
        clip_embedding = self.encode_clip(clip)
        return self.classify_embedding(clip_embedding)

    def encode_clip(self, clip: Any) -> Any:
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
        batch_size, channels, time_steps, height, width = clip.shape
        frame_batch = clip.permute(0, 2, 1, 3, 4).reshape(batch_size * time_steps, channels, height, width)
        frame_embeddings = self.encoder(frame_batch).reshape(batch_size, time_steps, self.feature_dim)
        return self._pool_temporal(frame_embeddings)

    def classify_embedding(self, clip_embedding: Any) -> Any:
        self.build()
        return self.classifier_head(clip_embedding)

    def summary(self) -> dict[str, Any]:
        payload = super().summary()
        payload.update(
            {
                "encoder_name": self.encoder_name,
                "temporal_pooling": self.temporal_pooling,
                "pretrained": self.pretrained,
                "allow_encoder_fallback": self.allow_encoder_fallback,
                "feature_dim": self.feature_dim or None,
                "backend": "torchvision" if tv_models is not None else "fallback_conv",
            }
        )
        return payload

    def _build_frame_encoder(self) -> tuple[Any, int]:
        if tv_models is not None and hasattr(tv_models, self.encoder_name):
            backbone_factory = getattr(tv_models, self.encoder_name)
            backbone = self._instantiate_torchvision_backbone(backbone_factory)
            feature_dim = int(backbone.classifier[-1].in_features)
            return _TorchvisionConvNeXtEncoder(backbone), feature_dim
        if not self.allow_encoder_fallback:
            raise RuntimeError(
                "torchvision ConvNeXt backend is unavailable. "
                "Install a compatible torchvision build or set model.allow_encoder_fallback=true "
                "only for explicit fallback/scaffold experiments."
            )
        return _FallbackFrameEncoder(), 128

    def _instantiate_torchvision_backbone(self, backbone_factory: Any) -> Any:
        kwargs: dict[str, Any] = {}
        if self.pretrained:
            try:
                from torchvision.models import get_model_weights

                kwargs["weights"] = get_model_weights(backbone_factory).DEFAULT
            except Exception:
                kwargs["pretrained"] = True
        else:
            kwargs["weights"] = None
        try:
            return backbone_factory(**kwargs)
        except TypeError:
            kwargs.pop("weights", None)
            kwargs["pretrained"] = self.pretrained
            return backbone_factory(**kwargs)

    def _pool_temporal(self, frame_embeddings: Any) -> Any:
        if self.temporal_pooling == "mean":
            return frame_embeddings.mean(dim=1)
        if self.temporal_pooling == "max":
            return frame_embeddings.max(dim=1).values
        raise ValueError(f"Unsupported temporal_pooling={self.temporal_pooling!r}")
