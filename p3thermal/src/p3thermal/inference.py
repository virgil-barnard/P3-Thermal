"""Loopback client for the isolated Torch learned-preview sidecar."""

from __future__ import annotations

import time
from urllib.request import Request, urlopen

import numpy as np
from numpy.typing import NDArray

from .frames import ThermalFrame

_SCALES = (2, 3, 4)


class RemoteLearnedPreview:
    """Use the isolated GPU sidecar without adding Torch to camera runtime."""

    def __init__(self, url: str) -> None:
        self._url = url.rstrip("/")
        self.device = "sidecar"

    def render(self, frame: ThermalFrame, scale: int) -> tuple[NDArray[np.uint16], int]:
        if scale not in _SCALES:
            raise ValueError("scale must be one of 2, 3, or 4")
        started_ns = time.monotonic_ns()
        request = Request(
            f"{self._url}/api/preview?scale={scale}",
            data=frame.thermal_raw.astype("<u2", copy=False).tobytes(),
            headers={
                "Content-Type": "application/octet-stream",
                "X-P3-Width": str(frame.thermal_raw.shape[1]),
                "X-P3-Height": str(frame.thermal_raw.shape[0]),
            },
            method="POST",
        )
        with urlopen(request, timeout=5) as response:  # noqa: S310
            width = int(response.headers["X-P3-Width"])
            height = int(response.headers["X-P3-Height"])
            body = response.read()
        values = np.frombuffer(body, dtype="<u2").reshape(height, width).copy()
        values.setflags(write=False)
        return values, time.monotonic_ns() - started_ns
