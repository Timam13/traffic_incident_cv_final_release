from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class Registry:
    name: str
    _items: dict[str, Callable[..., Any]] = field(default_factory=dict)

    def register(self, key: str, factory: Callable[..., Any]) -> None:
        if key in self._items:
            raise KeyError(f"{key!r} already registered in {self.name}")
        self._items[key] = factory

    def get(self, key: str) -> Callable[..., Any]:
        if key not in self._items:
            available = ", ".join(sorted(self._items)) or "<empty>"
            raise KeyError(f"{key!r} is not registered in {self.name}. Available: {available}")
        return self._items[key]

    def keys(self) -> list[str]:
        return sorted(self._items)
