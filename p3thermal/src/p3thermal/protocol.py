"""P3 byte framing and native image-layout decoding."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

MARKER_SIZE = 12
PIXEL_DATA_SIZE = 197_632
FRAME_SIZE = MARKER_SIZE + PIXEL_DATA_SIZE + MARKER_SIZE
FRAME_WIDTH = 256
FRAME_HEIGHT = 192
FRAME_ROWS = FRAME_HEIGHT * 2 + 2

READ_MODEL = bytes.fromhex("0101810001000000000000001e0000004f90")
READ_FIRMWARE_VERSION = bytes.fromhex("0101810002000000000000000c0000001f63")
READ_PART_NUMBER = bytes.fromhex("01018100060000000000000040000000654f")
READ_SERIAL = bytes.fromhex("01018100070000000000000040000000104c")
READ_HARDWARE_VERSION = bytes.fromhex("010181000a00000000000000400000001959")
START_STREAM = bytes.fromhex("012f81000000000000000000010000004930")

IDENTIFICATION_REGISTERS = (
    ("model", READ_MODEL, 30),
    ("firmware_version", READ_FIRMWARE_VERSION, 12),
    ("part_number", READ_PART_NUMBER, 64),
    ("serial", READ_SERIAL, 64),
    ("hardware_version", READ_HARDWARE_VERSION, 64),
)


@dataclass(frozen=True, slots=True)
class FrameMarker:
    """One validated P3 frame marker."""

    sync: int
    counter_1: int
    counter_2: int
    counter_3: int

    @property
    def is_start(self) -> bool:
        return self.sync in (0x8C, 0x8D)


@dataclass(frozen=True, slots=True)
class NativeFrame:
    """Validated raw P3 frame and its native planes."""

    raw_bytes: bytes
    start_marker: FrameMarker
    end_marker: FrameMarker
    brightness: NDArray[np.uint8]
    metadata: bytes
    thermal_raw: NDArray[np.uint16]


def parse_marker(data: bytes) -> FrameMarker:
    """Parse one 12-byte marker without assigning it a stream role."""
    if len(data) != MARKER_SIZE or data[0] != MARKER_SIZE:
        raise ValueError("invalid P3 marker length")
    return FrameMarker(
        sync=data[1],
        counter_1=int.from_bytes(data[2:6], "little"),
        counter_2=int.from_bytes(data[6:10], "little"),
        counter_3=int.from_bytes(data[10:12], "little"),
    )


def validate_frame(data: bytes) -> tuple[FrameMarker, FrameMarker]:
    """Validate P3 frame structure and return its paired markers."""
    if len(data) != FRAME_SIZE:
        raise ValueError(f"P3 frame must be {FRAME_SIZE} bytes")
    start = parse_marker(data[:MARKER_SIZE])
    end = parse_marker(data[-MARKER_SIZE:])
    if not start.is_start:
        raise ValueError("P3 frame start marker has invalid sync")
    if end.sync != start.sync + 2:
        raise ValueError("P3 frame markers have mismatched sync parity")
    if end.counter_1 != start.counter_1:
        raise ValueError("P3 frame markers have mismatched counter 1")
    return start, end


def decode_frame(data: bytes) -> NativeFrame:
    """Decode a structurally validated P3 frame into immutable-native inputs."""
    start, end = validate_frame(data)
    pixels = np.frombuffer(
        data, dtype="<u2", count=FRAME_ROWS * FRAME_WIDTH, offset=MARKER_SIZE
    )
    full = pixels.reshape((FRAME_ROWS, FRAME_WIDTH))
    brightness = full[:FRAME_HEIGHT].astype(np.uint8)
    metadata = bytes(full[FRAME_HEIGHT : FRAME_HEIGHT + 2])
    thermal = full[FRAME_HEIGHT + 2 :].copy()
    brightness.setflags(write=False)
    thermal.setflags(write=False)
    return NativeFrame(data, start, end, brightness, metadata, thermal)


class FrameAssembler:
    """Resynchronize arbitrary USB chunks into validated P3 frames."""

    def __init__(self) -> None:
        self._buffer = bytearray()

    def feed(self, chunk: bytes) -> list[bytes]:
        """Consume a chunk and return every complete, structurally valid frame."""
        self._buffer.extend(chunk)
        frames: list[bytes] = []
        cursor = 0
        incomplete_start: int | None = None
        while cursor <= len(self._buffer) - MARKER_SIZE:
            if not _is_start_marker(self._buffer, cursor):
                cursor += 1
                continue
            end = cursor + FRAME_SIZE
            if end > len(self._buffer):
                incomplete_start = cursor
                cursor += 1
                continue
            candidate = bytes(self._buffer[cursor:end])
            try:
                validate_frame(candidate)
            except ValueError:
                cursor += 1
                continue
            frames.append(candidate)
            del self._buffer[:end]
            cursor = 0
            incomplete_start = None
        if incomplete_start is not None:
            del self._buffer[:incomplete_start]
        elif len(self._buffer) > MARKER_SIZE - 1:
            del self._buffer[: -(MARKER_SIZE - 1)]
        return frames


def _is_start_marker(data: bytearray, offset: int) -> bool:
    return data[offset] == MARKER_SIZE and data[offset + 1] in (0x8C, 0x8D)
