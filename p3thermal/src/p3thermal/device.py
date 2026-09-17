"""The single-owner P3 camera source."""

from __future__ import annotations

import time
from collections.abc import Iterator
from typing import Protocol

import usb.core

from .frames import CameraEvent, CameraState, DeviceProfile, FrameQuality, ThermalFrame
from .protocol import FRAME_HEIGHT, FRAME_WIDTH, FrameAssembler, decode_frame
from .source import FrameSource
from .transport import P3Transport

P3_PROFILE = DeviceProfile(
    identifier="p3-provisional-256x192",
    vendor_id=0x3474,
    product_id=0x45A2,
    thermal_width=FRAME_WIDTH,
    thermal_height=FRAME_HEIGHT,
    raw_frame_size=197_632,
)


class StreamingTransport(Protocol):
    """The exclusive USB operations required by a streaming camera."""

    def start_stream(self) -> object: ...

    def read_stream_chunk(self) -> bytes: ...

    def close(self) -> None: ...


class Camera(FrameSource):
    """One P3 owner that delivers validated immutable frames in stream order."""

    def __init__(
        self,
        transport: StreamingTransport,
        profile: DeviceProfile = P3_PROFILE,
        max_consecutive_timeouts: int = 3,
    ) -> None:
        if max_consecutive_timeouts < 1:
            raise ValueError("max_consecutive_timeouts must be positive")
        self._transport = transport
        self._profile = profile
        self._assembler = FrameAssembler()
        self._max_consecutive_timeouts = max_consecutive_timeouts
        self._state = CameraState.READY
        self._events: list[CameraEvent] = []
        self._sequence = 0
        self._stream_epoch = 0

    @classmethod
    def open(cls) -> Camera:
        """Open the directly connected P3 with the qualified USB transport."""
        return cls(P3Transport.open())

    @property
    def profile(self) -> DeviceProfile:
        """The immutable profile associated with every delivered frame."""
        return self._profile

    @property
    def state(self) -> CameraState:
        """Current camera lifecycle state."""
        return self._state

    @property
    def events(self) -> tuple[CameraEvent, ...]:
        """Immutable snapshot of acquisition diagnostics."""
        return tuple(self._events)

    def __iter__(self) -> Iterator[ThermalFrame]:
        if self._state is CameraState.CLOSED:
            raise RuntimeError("camera is closed")
        if self._state is not CameraState.READY:
            raise RuntimeError("camera already has an active stream")
        self._transport.start_stream()
        self._state = CameraState.STREAMING
        self._stream_epoch += 1
        consecutive_timeouts = 0
        while self._state is CameraState.STREAMING:
            try:
                chunks = self._assembler.feed(self._transport.read_stream_chunk())
            except usb.core.USBTimeoutError:
                consecutive_timeouts += 1
                self._event("read_timeout", str(consecutive_timeouts))
                if consecutive_timeouts == self._max_consecutive_timeouts:
                    self._state = CameraState.FAULTED
                    self._event("fault", "consecutive read timeout limit reached")
                    raise RuntimeError("P3 stream timed out repeatedly") from None
                continue
            consecutive_timeouts = 0
            for raw_bytes in chunks:
                decoded = decode_frame(raw_bytes)
                yield ThermalFrame(
                    sequence=self._sequence,
                    stream_epoch=self._stream_epoch,
                    received_monotonic_ns=time.monotonic_ns(),
                    profile=self._profile,
                    thermal_raw=decoded.thermal_raw,
                    brightness=decoded.brightness,
                    raw_bytes=raw_bytes,
                    quality=FrameQuality.VALID,
                    device_counter=decoded.start_marker.counter_3,
                )
                self._sequence += 1

    def close(self) -> None:
        """Stop this source and release its exclusive transport ownership."""
        if self._state is not CameraState.CLOSED:
            self._state = CameraState.CLOSED
            self._transport.close()

    def _event(self, kind: str, detail: str) -> None:
        self._events.append(CameraEvent(time.monotonic_ns(), kind, detail))
