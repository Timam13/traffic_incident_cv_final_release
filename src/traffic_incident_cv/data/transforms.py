from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None

try:
    import torch
    import torch.nn.functional as F
except Exception:  # pragma: no cover
    torch = None
    F = None


@dataclass
class IdentityTransform:
    def __call__(self, x: Any) -> Any:
        return x


@dataclass
class Compose:
    transforms: list

    def __call__(self, x: Any) -> Any:
        for transform in self.transforms:
            x = transform(x)
        return x


def _is_torch_tensor(x: Any) -> bool:
    return torch is not None and isinstance(x, torch.Tensor)


def _as_clip_array(x: Any) -> tuple[Any, str]:
    if _is_torch_tensor(x):
        return x.float(), 'torch'
    if np is None:
        raise RuntimeError('numpy is required for clip transforms when torch tensors are unavailable')
    return np.asarray(x, dtype=np.float32), 'numpy'


def _restore_clip(x: Any, kind: str) -> Any:
    if kind == 'torch':
        return x.contiguous()
    if np is None:
        return x
    return np.asarray(x, dtype=np.float32)


def _clip_shape(x: Any) -> tuple[int, int, int, int]:
    if len(getattr(x, 'shape', ())) != 4:
        raise ValueError(f'Expected clip tensor with shape [C, T, H, W], got {tuple(getattr(x, "shape", ())) }')
    return tuple(int(v) for v in x.shape)


@dataclass(frozen=True)
class ClipRandomHorizontalFlip:
    probability: float = 0.0

    def __call__(self, clip: Any) -> Any:
        if self.probability <= 0 or random.random() >= self.probability:
            return clip
        data, kind = _as_clip_array(clip)
        _clip_shape(data)
        if kind == 'torch':
            return _restore_clip(torch.flip(data, dims=(-1,)), kind)
        return _restore_clip(np.flip(data, axis=-1).copy(), kind)


@dataclass(frozen=True)
class ClipRandomGrayscale:
    probability: float = 0.0

    def __call__(self, clip: Any) -> Any:
        if self.probability <= 0 or random.random() >= self.probability:
            return clip
        data, kind = _as_clip_array(clip)
        channels, _, _, _ = _clip_shape(data)
        if channels < 3:
            return clip
        if kind == 'torch':
            grayscale = data[:3].mean(dim=0, keepdim=True)
            data = data.clone()
            data[:3] = grayscale.repeat(3, 1, 1, 1)
            return _restore_clip(data.clamp(0.0, 1.0), kind)
        grayscale = data[:3].mean(axis=0, keepdims=True)
        data = data.copy()
        data[:3] = np.repeat(grayscale, 3, axis=0)
        return _restore_clip(np.clip(data, 0.0, 1.0), kind)


@dataclass(frozen=True)
class ClipRandomBrightnessContrast:
    probability: float = 0.0
    brightness_delta: float = 0.0
    contrast_low: float = 1.0
    contrast_high: float = 1.0

    def __call__(self, clip: Any) -> Any:
        if self.probability <= 0 or random.random() >= self.probability:
            return clip
        data, kind = _as_clip_array(clip)
        _clip_shape(data)
        contrast = random.uniform(self.contrast_low, self.contrast_high)
        brightness = random.uniform(-self.brightness_delta, self.brightness_delta)
        if kind == 'torch':
            return _restore_clip((data * contrast + brightness).clamp(0.0, 1.0), kind)
        return _restore_clip(np.clip(data * contrast + brightness, 0.0, 1.0), kind)


@dataclass(frozen=True)
class ClipRandomGaussianNoise:
    probability: float = 0.0
    std_low: float = 0.0
    std_high: float = 0.0

    def __call__(self, clip: Any) -> Any:
        if self.probability <= 0 or random.random() >= self.probability:
            return clip
        data, kind = _as_clip_array(clip)
        _clip_shape(data)
        std = random.uniform(self.std_low, self.std_high)
        if std <= 0:
            return clip
        if kind == 'torch':
            noise = torch.randn_like(data) * std
            return _restore_clip((data + noise).clamp(0.0, 1.0), kind)
        noise = np.random.normal(loc=0.0, scale=std, size=data.shape).astype(np.float32)
        return _restore_clip(np.clip(data + noise, 0.0, 1.0), kind)


@dataclass(frozen=True)
class ClipRandomBlur:
    probability: float = 0.0
    kernel_sizes: tuple[int, ...] = (3,)

    def __call__(self, clip: Any) -> Any:
        if self.probability <= 0 or random.random() >= self.probability:
            return clip
        data, kind = _as_clip_array(clip)
        _clip_shape(data)
        kernel = max(int(random.choice(self.kernel_sizes or (3,))), 1)
        if kernel % 2 == 0:
            kernel += 1
        if kernel <= 1:
            return clip
        if kind == 'torch' and F is not None:
            frames = data.permute(1, 0, 2, 3)
            blurred = F.avg_pool2d(frames, kernel_size=kernel, stride=1, padding=kernel // 2)
            return _restore_clip(blurred.permute(1, 0, 2, 3).clamp(0.0, 1.0), kind)
        return clip


@dataclass(frozen=True)
class ClipRandomDownsampleUpsample:
    probability: float = 0.0
    scale_min: float = 1.0
    scale_max: float = 1.0

    def __call__(self, clip: Any) -> Any:
        if self.probability <= 0 or random.random() >= self.probability:
            return clip
        data, kind = _as_clip_array(clip)
        _, _, height, width = _clip_shape(data)
        scale = random.uniform(self.scale_min, self.scale_max)
        if scale >= 0.999:
            return clip
        down_h = max(1, int(height * scale))
        down_w = max(1, int(width * scale))
        if kind == 'torch' and F is not None:
            frames = data.permute(1, 0, 2, 3)
            down = F.interpolate(frames, size=(down_h, down_w), mode='bilinear', align_corners=False)
            up = F.interpolate(down, size=(height, width), mode='bilinear', align_corners=False)
            return _restore_clip(up.permute(1, 0, 2, 3).clamp(0.0, 1.0), kind)
        return clip


@dataclass(frozen=True)
class ClipRandomOcclusion:
    probability: float = 0.0
    scale_min: float = 0.0
    scale_max: float = 0.0
    fill_value: float = 0.0

    def __call__(self, clip: Any) -> Any:
        if self.probability <= 0 or random.random() >= self.probability:
            return clip
        data, kind = _as_clip_array(clip)
        _, _, height, width = _clip_shape(data)
        scale = random.uniform(self.scale_min, self.scale_max)
        if scale <= 0:
            return clip
        occ_h = max(1, int(height * scale))
        occ_w = max(1, int(width * scale))
        top = random.randint(0, max(height - occ_h, 0))
        left = random.randint(0, max(width - occ_w, 0))
        if kind == 'torch':
            patched = data.clone()
            patched[:, :, top:top + occ_h, left:left + occ_w] = float(self.fill_value)
            return _restore_clip(patched.clamp(0.0, 1.0), kind)
        patched = data.copy()
        patched[:, :, top:top + occ_h, left:left + occ_w] = float(self.fill_value)
        return _restore_clip(np.clip(patched, 0.0, 1.0), kind)


@dataclass(frozen=True)
class ClipFrameDifference:
    mode: str = 'previous'
    absolute: bool = True
    grayscale: bool = False
    gain: float = 1.0

    def __call__(self, clip: Any) -> Any:
        data, kind = _as_clip_array(clip)
        channels, time_steps, _, _ = _clip_shape(data)
        if time_steps <= 1:
            return clip
        if self.grayscale and channels >= 3:
            if kind == 'torch':
                grayscale = data[:3].mean(dim=0, keepdim=True)
                data = data.clone()
                data[:3] = grayscale.repeat(3, 1, 1, 1)
            else:
                grayscale = data[:3].mean(axis=0, keepdims=True)
                data = data.copy()
                data[:3] = np.repeat(grayscale, 3, axis=0)
        if kind == 'torch':
            diff = torch.zeros_like(data)
            if self.mode == 'first':
                diff[:, 1:] = data[:, 1:] - data[:, :1]
            elif self.mode == 'previous':
                diff[:, 1:] = data[:, 1:] - data[:, :-1]
            else:
                raise ValueError(f'Unsupported frame_difference mode={self.mode!r}')
            if self.absolute:
                diff = diff.abs()
            if self.gain != 1.0:
                diff = diff * float(self.gain)
            return _restore_clip(diff.clamp(0.0, 1.0), kind)
        diff = np.zeros_like(data, dtype=np.float32)
        if self.mode == 'first':
            diff[:, 1:] = data[:, 1:] - data[:, :1]
        elif self.mode == 'previous':
            diff[:, 1:] = data[:, 1:] - data[:, :-1]
        else:
            raise ValueError(f'Unsupported frame_difference mode={self.mode!r}')
        if self.absolute:
            diff = np.abs(diff)
        if self.gain != 1.0:
            diff = diff * float(self.gain)
        return _restore_clip(np.clip(diff, 0.0, 1.0), kind)


@dataclass(frozen=True)
class ClipSpatialCrop:
    top: float = 0.0
    left: float = 0.0
    bottom: float = 1.0
    right: float = 1.0

    def __call__(self, clip: Any) -> Any:
        data, kind = _as_clip_array(clip)
        _, _, height, width = _clip_shape(data)
        top = max(0, min(height - 1, int(height * float(self.top))))
        left = max(0, min(width - 1, int(width * float(self.left))))
        bottom = max(top + 1, min(height, int(height * float(self.bottom))))
        right = max(left + 1, min(width, int(width * float(self.right))))
        if kind == 'torch':
            return _restore_clip(data[:, :, top:bottom, left:right], kind)
        return _restore_clip(np.asarray(data[:, :, top:bottom, left:right], dtype=np.float32), kind)


def build_spatial_preprocessing_transforms(config: dict[str, Any] | None) -> list[Any]:
    if not config:
        return []
    if config.get('enabled') is False:
        return []
    return [
        ClipSpatialCrop(
            top=float(config.get('top', 0.0) or 0.0),
            left=float(config.get('left', 0.0) or 0.0),
            bottom=float(config.get('bottom', 1.0) or 1.0),
            right=float(config.get('right', 1.0) or 1.0),
        )
    ]


def build_motion_preprocessing_transforms(config: dict[str, Any] | None) -> list[Any]:
    if not config:
        return []
    if config.get('enabled') is False:
        return []
    return [
        ClipFrameDifference(
            mode=str(config.get('mode', 'previous') or 'previous'),
            absolute=bool(config.get('absolute', True)),
            grayscale=bool(config.get('grayscale', False)),
            gain=float(config.get('gain', 1.0) or 1.0),
        )
    ]


def build_domain_randomization_transforms(config: dict[str, Any] | None) -> list[Any]:
    if not config:
        return []
    if config.get('enabled') is False:
        return []
    kernel_sizes = tuple(int(item) for item in config.get('blur_kernel_sizes', [3, 5]))
    transforms: list[Any] = []
    horizontal_flip_prob = float(config.get('horizontal_flip_prob', 0.0) or 0.0)
    if horizontal_flip_prob > 0:
        transforms.append(ClipRandomHorizontalFlip(probability=horizontal_flip_prob))
    brightness_contrast_prob = float(config.get('brightness_contrast_prob', 0.0) or 0.0)
    if brightness_contrast_prob > 0:
        transforms.append(
            ClipRandomBrightnessContrast(
                probability=brightness_contrast_prob,
                brightness_delta=float(config.get('brightness_delta', 0.0) or 0.0),
                contrast_low=float(config.get('contrast_low', 1.0) or 1.0),
                contrast_high=float(config.get('contrast_high', 1.0) or 1.0),
            )
        )
    grayscale_prob = float(config.get('grayscale_prob', 0.0) or 0.0)
    if grayscale_prob > 0:
        transforms.append(ClipRandomGrayscale(probability=grayscale_prob))
    blur_prob = float(config.get('blur_prob', 0.0) or 0.0)
    if blur_prob > 0:
        transforms.append(ClipRandomBlur(probability=blur_prob, kernel_sizes=kernel_sizes))
    downsample_prob = float(config.get('downsample_prob', 0.0) or 0.0)
    if downsample_prob > 0:
        transforms.append(
            ClipRandomDownsampleUpsample(
                probability=downsample_prob,
                scale_min=float(config.get('downsample_scale_min', 1.0) or 1.0),
                scale_max=float(config.get('downsample_scale_max', 1.0) or 1.0),
            )
        )
    gaussian_noise_prob = float(config.get('gaussian_noise_prob', 0.0) or 0.0)
    if gaussian_noise_prob > 0:
        transforms.append(
            ClipRandomGaussianNoise(
                probability=gaussian_noise_prob,
                std_low=float(config.get('gaussian_noise_std_low', 0.0) or 0.0),
                std_high=float(config.get('gaussian_noise_std_high', 0.0) or 0.0),
            )
        )
    occlusion_prob = float(config.get('occlusion_prob', 0.0) or 0.0)
    if occlusion_prob > 0:
        transforms.append(
            ClipRandomOcclusion(
                probability=occlusion_prob,
                scale_min=float(config.get('occlusion_scale_min', 0.0) or 0.0),
                scale_max=float(config.get('occlusion_scale_max', 0.0) or 0.0),
                fill_value=float(config.get('occlusion_fill_value', 0.0) or 0.0),
            )
        )
    return transforms
