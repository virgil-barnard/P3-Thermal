from collections.abc import Iterator

import numpy as np
import pytest
import usb.core

from p3thermal import Camera, CameraState, FrameQuality
from p3thermal.protocol import FRAME_HEIGHT, FRAME_WIDTH


def native_frame(counter: int = 7, sync: int = 0x8C) -> bytes:
    def marker(counter_1: int, counter_3: int, marker_sync: int) -> bytes:
        return (
            b"\x0c"
            + bytes([marker_sync])
            + counter_1.to_bytes(4, "little")
            + (3).to_bytes(4, "little")
            + counter_3.to_bytes(2, "little")
        )

    pixels = np.zeros((FRAME_HEIGHT * 2 + 2, FRAME_WIDTH), dtype="<u2")
    return marker(counter, 40, sync) + pixels.tobytes() + marker(counter, 80, sync + 2)


class FakeTransport:
    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks: Iterator[bytes] = iter(chunks)
        self.started = False
        self.closed = False

    def start_stream(self) -> None:
        self.started = True

    def read_stream_chunk(self) -> bytes:
        return next(self._chunks)

    def close(self) -> None:
        self.closed = True


def test_camera_yields_immutable_frame_and_closes_transport() -> None:
    transport = FakeTransport([native_frame()])
    camera = Camera(transport)

    frame = next(iter(camera))
    camera.close()

    assert transport.started
    assert transport.closed
    assert frame.sequence == 0
    assert frame.stream_epoch == 1
    assert frame.quality == FrameQuality.VALID
    assert frame.device_counter == 40
    assert not frame.thermal_raw.flags.writeable


def test_camera_preserves_counter_without_unqualified_continuity_inference() -> None:
    first = native_frame()
    second = native_frame(counter=8, sync=0x8D)
    transport = FakeTransport([first + second])
    camera = Camera(transport)
    frames = iter(camera)

    next(frames)
    second_frame = next(frames)
    camera.close()

    assert second_frame.quality == FrameQuality.VALID
    assert second_frame.device_counter == 40


def test_camera_faults_after_bounded_read_timeouts() -> None:
    class TimeoutTransport(FakeTransport):
        def __init__(self) -> None:
            super().__init__([])

        def read_stream_chunk(self) -> bytes:
            raise usb.core.USBTimeoutError("late")

    camera = Camera(TimeoutTransport(), max_consecutive_timeouts=2)

    with pytest.raises(RuntimeError, match="timed out"):
        next(iter(camera))

    assert camera.state is CameraState.FAULTED
    assert [event.kind for event in camera.events] == [
        "read_timeout",
        "read_timeout",
        "fault",
    ]
    camera.close()
