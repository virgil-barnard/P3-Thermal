"""Lossless, append-only P3 sessions and deterministic replay."""

from __future__ import annotations

import hashlib
import json
import struct
from collections.abc import Iterator
from dataclasses import asdict
from pathlib import Path
from typing import BinaryIO

import numpy as np

from .frames import DeviceProfile, FrameQuality, ThermalFrame
from .protocol import FRAME_SIZE, decode_frame
from .source import FrameSource

SCHEMA_VERSION = 1
FRAME_MAGIC = b"P3F1"
FRAME_HEADER = struct.Struct("<4sQQQIiI32s")
MAX_RAW_FRAME_BYTES = FRAME_SIZE


class RecordingError(RuntimeError):
    """A session cannot be written or replayed safely."""


class Recorder:
    """Append validated frames to a recoverable session directory.

    Each record carries its own SHA-256 hash. A partially written final record is
    ignored by replay, preserving the valid prefix after power loss or a stopped
    process.
    """

    def __init__(self, directory: str | Path, profile: DeviceProfile) -> None:
        self.directory = Path(directory)
        self.profile = profile
        self._stream: BinaryIO | None = None
        self._index: BinaryIO | None = None
        self._written = 0

    def __enter__(self) -> Recorder:
        self.directory.mkdir(parents=True, exist_ok=False)
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "complete": False,
            "profile": asdict(self.profile),
            "frame_format": "P3F1: header followed by validated native bytes",
        }
        _write_json(self.directory / "manifest.json", manifest)
        self._stream = (self.directory / "frames-000001.bin").open("xb")
        self._index = (self.directory / "index.jsonl").open("x", encoding="utf-8")
        return self

    @property
    def frames_written(self) -> int:
        return self._written

    def write(self, frame: ThermalFrame) -> None:
        """Write one validated frame and a rebuildable seek index entry."""
        if self._stream is None or self._index is None:
            raise RecordingError("recorder is not open")
        if len(frame.raw_bytes) > MAX_RAW_FRAME_BYTES:
            raise RecordingError("raw frame exceeds the supported session limit")
        decoded = decode_frame(frame.raw_bytes)
        if not np.array_equal(decoded.thermal_raw, frame.thermal_raw):
            raise RecordingError("frame raw bytes do not match its thermal plane")
        offset = self._stream.tell()
        digest = hashlib.sha256(frame.raw_bytes).digest()
        device_counter = -1 if frame.device_counter is None else frame.device_counter
        header = FRAME_HEADER.pack(
            FRAME_MAGIC,
            frame.sequence,
            frame.stream_epoch,
            frame.received_monotonic_ns,
            frame.quality.value,
            device_counter,
            len(frame.raw_bytes),
            digest,
        )
        self._stream.write(header)
        self._stream.write(frame.raw_bytes)
        self._index.write(
            json.dumps(
                {
                    "sequence": frame.sequence,
                    "received_monotonic_ns": frame.received_monotonic_ns,
                    "offset": offset,
                },
                sort_keys=True,
            )
            + "\n"
        )
        self._written += 1

    def close(self) -> None:
        """Flush the valid prefix and mark a normally closed session complete."""
        if self._stream is None or self._index is None:
            return
        self._stream.close()
        self._index.close()
        self._stream = None
        self._index = None
        manifest_path = self.directory / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["complete"] = True
        manifest["frames_written"] = self._written
        _write_json(manifest_path, manifest)

    def __exit__(self, *_: object) -> None:
        self.close()


class Replay(FrameSource):
    """Replay every intact record in a session in original source order."""

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        manifest_path = self.directory / "manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.profile = DeviceProfile(**manifest["profile"])
        except (KeyError, OSError, TypeError, json.JSONDecodeError) as exc:
            raise RecordingError("invalid recording manifest") from exc
        if manifest.get("schema_version") != SCHEMA_VERSION:
            raise RecordingError("unsupported recording schema version")
        self._closed = False

    def __iter__(self) -> Iterator[ThermalFrame]:
        if self._closed:
            raise RecordingError("replay is closed")
        for path in sorted(self.directory.glob("frames-*.bin")):
            with path.open("rb") as stream:
                while not self._closed:
                    record = _read_record(stream)
                    if record is None:
                        break
                    sequence, epoch, received, quality, counter, raw_bytes = record
                    decoded = decode_frame(raw_bytes)
                    yield ThermalFrame(
                        sequence=sequence,
                        stream_epoch=epoch,
                        received_monotonic_ns=received,
                        profile=self.profile,
                        thermal_raw=decoded.thermal_raw,
                        brightness=decoded.brightness,
                        raw_bytes=raw_bytes,
                        quality=FrameQuality(quality),
                        device_counter=None if counter == -1 else counter,
                    )

    def export_thermal(self, path: str | Path) -> Path:
        """Write replayed native thermal planes as a NumPy archive."""
        output = Path(path)
        frames = list(self)
        np.savez_compressed(
            output,
            thermal_raw=np.stack([frame.thermal_raw for frame in frames]),
            sequence=np.array([frame.sequence for frame in frames], dtype=np.uint64),
            received_monotonic_ns=np.array(
                [frame.received_monotonic_ns for frame in frames], dtype=np.uint64
            ),
        )
        return output

    def close(self) -> None:
        self._closed = True


def _read_record(
    stream: BinaryIO,
) -> tuple[int, int, int, int, int, bytes] | None:
    header = stream.read(FRAME_HEADER.size)
    if not header:
        return None
    if len(header) != FRAME_HEADER.size:
        return None
    unpacked = FRAME_HEADER.unpack(header)
    magic, sequence, epoch, received, quality, counter, size, digest = unpacked
    if magic != FRAME_MAGIC or size > MAX_RAW_FRAME_BYTES:
        raise RecordingError("invalid frame record header")
    raw_bytes = stream.read(size)
    if len(raw_bytes) != size:
        return None
    if hashlib.sha256(raw_bytes).digest() != digest:
        raise RecordingError("frame record checksum mismatch")
    return sequence, epoch, received, quality, counter, raw_bytes


def _write_json(path: Path, value: object) -> None:
    contents = json.dumps(value, indent=2, sort_keys=True) + "\n"
    path.write_text(contents, encoding="utf-8")
