"""The common source interface for hardware capture and replay."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator

from .frames import ThermalFrame


class FrameSource(ABC):
    """A deterministically closeable sequence of validated thermal frames."""

    @abstractmethod
    def __iter__(self) -> Iterator[ThermalFrame]:
        """Yield frames in source order until closed or exhausted."""

    @abstractmethod
    def close(self) -> None:
        """Release source resources. This operation is idempotent."""

    def __enter__(self) -> FrameSource:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
