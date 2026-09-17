import numpy as np
import pytest

from p3thermal.protocol import (
    FRAME_HEIGHT,
    FRAME_SIZE,
    FRAME_WIDTH,
    FrameAssembler,
    decode_frame,
    validate_frame,
)


def marker(sync: int, counter_1: int, counter_2: int, counter_3: int) -> bytes:
    return (
        b"\x0c"
        + bytes([sync])
        + b"".join(
            value.to_bytes(size, "little")
            for value, size in ((counter_1, 4), (counter_2, 4), (counter_3, 2))
        )
    )


def native_frame(counter: int = 7, sync: int = 0x8C) -> bytes:
    pixels = np.zeros((FRAME_HEIGHT * 2 + 2, FRAME_WIDTH), dtype="<u2")
    pixels[0, 0] = 0x12AB
    pixels[FRAME_HEIGHT, 0] = 0xCAFE
    pixels[FRAME_HEIGHT + 2, 0] = 0x1234
    return (
        marker(sync, counter, 3, 40)
        + pixels.tobytes()
        + marker(sync + 2, counter, 4, 80)
    )


def test_assembler_reassembles_fragmented_frame_after_stale_bytes() -> None:
    frame = native_frame()
    assembler = FrameAssembler()

    assert assembler.feed(b"stale\x0c\x8c") == []
    assert assembler.feed(frame[:1000]) == []
    assert assembler.feed(frame[1000:120_000]) == []
    assert assembler.feed(frame[120_000:]) == [frame]


def test_assembler_skips_invalid_candidate_and_finds_later_frame() -> None:
    invalid = bytearray(native_frame())
    invalid[-10] ^= 1
    frame = native_frame(counter=8, sync=0x8D)

    assert FrameAssembler().feed(bytes(invalid) + frame) == [frame]


def test_decode_frame_exposes_native_planes() -> None:
    decoded = decode_frame(native_frame())

    assert decoded.start_marker.counter_1 == 7
    assert decoded.end_marker.counter_3 == 80
    assert decoded.brightness.dtype == np.uint8
    assert decoded.brightness[0, 0] == 0xAB
    assert not decoded.brightness.flags.writeable
    assert decoded.metadata[:2] == b"\xfe\xca"
    assert decoded.thermal_raw.dtype == np.uint16
    assert decoded.thermal_raw[0, 0] == 0x1234
    assert not decoded.thermal_raw.flags.writeable


def test_invalid_marker_counter_is_rejected() -> None:
    frame = bytearray(native_frame())
    frame[-10] ^= 1

    with pytest.raises(ValueError, match="counter 1"):
        validate_frame(bytes(frame))


def test_frame_size_is_native_layout_plus_markers() -> None:
    assert len(native_frame()) == FRAME_SIZE
