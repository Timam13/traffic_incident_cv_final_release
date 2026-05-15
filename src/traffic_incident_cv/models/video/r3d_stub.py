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
    from torchvision.models import get_model_weights
    from torchvision.models import video as tv_video_models
except Exception:  # pragma: no cover
    get_model_weights = None
    tv_video_models = None


TorchModuleBase = nn.Module if nn is not None else object


if nn is not None:

    class _FallbackVideoEncoder(nn.Module):
        def __init__(self, in_channels: int = 3, feature_dim: int = 128) -> None:
            super().__init__()
            self.layers = nn.Sequential(
                nn.Conv3d(in_channels, 32, kernel_size=3, stride=(1, 2, 2), padding=1),
                nn.ReLU(inplace=True),
                nn.Conv3d(32, 64, kernel_size=3, stride=(1, 2, 2), padding=1),
                nn.ReLU(inplace=True),
                nn.Conv3d(64, feature_dim, kernel_size=3, stride=(2, 2, 2), padding=1),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool3d((1, 1, 1)),
            )

        def forward(self, clip: Any) -> Any:
            return self.layers(clip).flatten(1)


    class _TorchvisionVideoEncoder(nn.Module):
        def __init__(self, backbone: Any) -> None:
            super().__init__()
            self.features = nn.Sequential(*list(backbone.children())[:-1])

        def forward(self, clip: Any) -> Any:
            return self.features(clip).flatten(1)

else:  # pragma: no cover

    class _FallbackVideoEncoder:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("torch is not installed. Install project optional dependency 'train'.")


    class _TorchvisionVideoEncoder:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("torch is not installed. Install project optional dependency 'train'.")


@dataclass(eq=False)
class R3DClipClassifier(BaseModelStub, TorchModuleBase):
    encoder_name: str = "r3d_18"
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
        self.encoder, self.feature_dim = self._build_video_encoder()
        self.classifier_head = nn.Linear(self.feature_dim, self.num_classes)

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
        clip_embedding = self.encoder(clip)
        return self.classifier_head(clip_embedding)

    def summary(self) -> dict[str, Any]:
        payload = super().summary()
        payload.update(
            {
                "encoder_name": self.encoder_name,
                "pretrained": self.pretrained,
                "allow_encoder_fallback": self.allow_encoder_fallback,
                "feature_dim": self.feature_dim or None,
                "backend": "torchvision_video" if tv_video_models is not None else "fallback_conv3d",
            }
        )
        return payload

    def _build_video_encoder(self) -> tuple[Any, int]:
        if tv_video_models is not None and hasattr(tv_video_models, self.encoder_name):
            backbone_factory = getattr(tv_video_models, self.encoder_name)
            backbone = self._instantiate_torchvision_backbone(backbone_factory)
            feature_dim = int(backbone.fc.in_features)
            return _TorchvisionVideoEncoder(backbone), feature_dim
        if not self.allow_encoder_fallback:
            raise RuntimeError(
                "torchvision video backend is unavailable. "
                "Install a compatible torchvision build or set model.allow_encoder_fallback=true "
                "only for explicit fallback/scaffold experiments."
            )
        return _FallbackVideoEncoder(), 128

    def _instantiate_torchvision_backbone(self, backbone_factory: Any) -> Any:
        kwargs: dict[str, Any] = {}
        if self.pretrained:
            try:
                if get_model_weights is not None:
                    kwargs["weights"] = get_model_weights(backbone_factory).DEFAULT
                else:
                    kwargs["pretrained"] = True
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