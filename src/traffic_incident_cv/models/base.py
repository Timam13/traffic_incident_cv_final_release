from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(eq=False)
class BaseModelStub:
    name: str
    num_classes: int
    __hash__ = object.__hash__

    def summary(self) -> dict[str, Any]:
        return {"name": self.name, "num_classes": self.num_classes}
