from pathlib import Path

import numpy as np
import pytest

from p3thermal import Camera, FrameQuality, Recorder, Replay
from p3thermal.protocol import FRAME_HEIGHT, FRAME_WIDTH
from p3thermal.recording import RecordingError


def _native_frame() -> bytes:
    def marker(sync: int, counter_3: int) -> bytes:
        return (
            b"\x0c"
            + bytes([sync])
            + (7).to_bytes(4, "little")
            + (3).to_bytes(4, "little")
            + counter_3.to_bytes(2, "little")
        )

    pixels = np.arange(FRAME_HEIGHT * 2 + 2, dtype="<u2").repeat(FRAME_WIDTH)
    return marker(0x8C, 40) + pixels.tobytes() + marker(0x8E, 80)


class _Transport:
    def __init__(self, frames: list[bytes]) -> None:
        self._frames = iter(frames)

    def start_stream(self) -> None:
        pass

    def read_stream_chunk(self) -> bytes:
        return next(self._frames)

    def close(self) -> None:
        pass


def _frames() -> tuple[object, object]:
    source = Camera(_Transport([_native_frame(), _native_frame()]))
    iterator = iter(source)
    first, second = next(iterator), next(iterator)
    source.close()
    return first, second


def test_recording_round_trips_losslessly_and_exports(tmp_path: Path) -> None:
    first, second = _frames()
    session = tmp_path / "session"
    with Recorder(session, first.profile) as recorder:
        recorder.write(first)
        recorder.write(second)
        assert recorder.frames_written == 2

    replay = Replay(session)
    frames = list(replay)
    export = replay.export_thermal(tmp_path / "thermal.npz")

    assert [frame.raw_bytes for frame in frames] == [first.raw_bytes, second.raw_bytes]
    assert frames[1].quality == FrameQuality.VALID
    assert np.array_equal(np.load(export)["thermal_raw"][0], first.thermal_raw)


def test_replay_recovers_valid_prefix_of_truncated_final_record(tmp_path: Path) -> None:
    first, second = _frames()
    session = tmp_path / "session"
    with Recorder(session, first.profile) as recorder:
        recorder.write(first)
        recorder.write(second)
    chunk = session / "frames-000001.bin"
    with chunk.open("r+b") as stream:
        stream.seek(-10, 2)
        stream.truncate()

    frames = list(Replay(session))

    assert [frame.sequence for frame in frames] == [first.sequence]


def test_replay_rejects_checksum_mismatch(tmp_path: Path) -> None:
    first, _ = _frames()
    session = tmp_path / "session"
    with Recorder(session, first.profile) as recorder:
        recorder.write(first)
    chunk = session / "frames-000001.bin"
    with chunk.open("r+b") as stream:
        stream.seek(-1, 2)
        stream.write(b"x")

    with pytest.raises(RecordingError, match="checksum"):
        list(Replay(session))
