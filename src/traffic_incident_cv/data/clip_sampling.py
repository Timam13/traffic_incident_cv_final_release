from __future__ import annotations

import random
from dataclasses import dataclass

from traffic_incident_cv.interfaces.base import ClipSamplerInterface


@dataclass
class UniformClipSampler(ClipSamplerInterface):
    def sample(self, *, total_frames: int, num_frames: int, stride: int) -> list[int]:
        if total_frames <= 0:
            return []
        needed = num_frames * stride
        if total_frames <= needed:
            return [min(i * stride, total_frames - 1) for i in range(num_frames)]
        start = max((total_frames - needed) // 2, 0)
        return [start + i * stride for i in range(num_frames)]


@dataclass
class RandomWindowClipSampler(ClipSamplerInterface):
    def sample(self, *, total_frames: int, num_frames: int, stride: int) -> list[int]:
        if total_frames <= 0:
            return []
        needed = num_frames * stride
        if total_frames <= needed:
            return [min(i * stride, total_frames - 1) for i in range(num_frames)]
        max_start = max(total_frames - needed, 0)
        start = random.randint(0, max_start)
        return [start + i * stride for i in range(num_frames)]


@dataclass
class TailFocusedSampler(ClipSamplerInterface):
    focus_ratio: float = 0.7

    def sample(self, *, total_frames: int, num_frames: int, stride: int) -> list[int]:
        if total_frames <= 0:
            return []
        end = total_frames - 1
        start = max(int(total_frames * self.focus_ratio) - num_frames * stride, 0)
        return [min(start + i * stride, end) for i in range(num_frames)]
