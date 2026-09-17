"""Immutable data shared by acquisition, recording, and processing."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Flag, StrEnum, auto

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class DeviceProfile:
    """Verified properties of one supported camera and stream layout."""

    identifier: str
    vendor_id: int
    product_id: int
    thermal_width: int
    thermal_height: int
    raw_frame_size: int
    firmware_version: str | None = None


class FrameQuality(Flag):
    """Known qualifications of a delivered frame."""

    VALID = 0
    COUNTER_DISCONTINUITY = auto()
    CONTROL_TRANSITION = auto()
    SHUTTER_CORRECTION = auto()


class CameraState(StrEnum):
    """Observable lifecycle state of a physical camera owner."""

    READY = "ready"
    STREAMING = "streaming"
    FAULTED = "faulted"
    CLOSED = "closed"


@dataclass(frozen=True, slots=True)
class CameraEvent:
    """A timestamped acquisition event retained for diagnostics."""

    monotonic_ns: int
    kind: str
    detail: str


@dataclass(frozen=True, slots=True)
class ThermalFrame:
    """One validated native frame with timing and stream provenance.

    Pixel arrays are made read-only so one consumer cannot alter another
    consumer's measurements or a recording's numerical source data.
    """

    sequence: int
    stream_epoch: int
    received_monotonic_ns: int
    profile: DeviceProfile
    thermal_raw: NDArray[np.uint16]
    brightness: NDArray[np.uint8] | None
    raw_bytes: bytes
    quality: FrameQuality = FrameQuality.VALID
    device_counter: int | None = None

    def __post_init__(self) -> None:
        expected_shape = (self.profile.thermal_height, self.profile.thermal_width)
        if self.sequence < 0 or self.stream_epoch < 0:
            raise ValueError("sequence and stream_epoch must be non-negative")
        if self.received_monotonic_ns < 0:
            raise ValueError("received_monotonic_ns must be non-negative")
        if (
            self.thermal_raw.dtype != np.uint16
            or self.thermal_raw.shape != expected_shape
        ):
            raise ValueError(
                f"thermal_raw must be a uint16 array shaped {expected_shape}"
            )
        if self.brightness is not None:
            if (
                self.brightness.dtype != np.uint8
                or self.brightness.shape != expected_shape
            ):
                raise ValueError(
                    f"brightness must be a uint8 array shaped {expected_shape}"
                )
            self.brightness.setflags(write=False)
        self.thermal_raw.setflags(write=False)
