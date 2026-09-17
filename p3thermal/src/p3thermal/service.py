"""Loopback-only live service with a latest-frame viewer handoff."""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from .frames import ThermalFrame
from .inference import RemoteLearnedPreview
from .recording import Recorder
from .source import FrameSource


@dataclass(frozen=True, slots=True)
class PreviewFrame:
    """A derived preview tied to the native frame that produced it."""

    sequence: int
    scale: int
    values: Any
    inference_ns: int


class LiveService:
    """Own a source worker; previews always read the most recent frame only."""

    def __init__(
        self,
        source: FrameSource,
        preview: RemoteLearnedPreview | None = None,
    ) -> None:
        self._source = source
        self._lock = threading.Lock()
        self._latest: ThermalFrame | None = None
        self._frames_received = 0
        self._error: str | None = None
        self._closed = False
        self._recorder: Recorder | None = None
        self._preview_runtime = preview
        self._preview: PreviewFrame | None = None
        self._preview_scale = 2
        self._preview_generation = 0
        self._preview_frames_completed = 0
        self._preview_frames_skipped = 0
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._preview_ready = threading.Condition(self._lock)
        self._preview_worker = threading.Thread(target=self._run_preview, daemon=True)

    def start(self) -> None:
        self._worker.start()
        if self._preview_runtime is not None:
            self._preview_worker.start()

    def close(self) -> None:
        with self._preview_ready:
            self._closed = True
            self._preview_ready.notify_all()
        self._source.close()
        self._worker.join(timeout=3)
        if self._preview_runtime is not None:
            self._preview_worker.join(timeout=3)
        self.stop_recording()

    def status(self) -> dict[str, object]:
        with self._lock:
            frame = self._latest
            return {
                "frames_received": self._frames_received,
                "latest_sequence": None if frame is None else frame.sequence,
                "quality": None if frame is None else frame.quality.name,
                "error": self._error,
                "recording": self._recorder is not None,
                "frames_written": 0
                if self._recorder is None
                else self._recorder.frames_written,
                "temperature_interpretation": "unqualified",
                "learned_preview": {
                    "available": self._preview_runtime is not None,
                    "device": None
                    if self._preview_runtime is None
                    else self._preview_runtime.device,
                    "scale": self._preview_scale,
                    "latest_sequence": None
                    if self._preview is None
                    else self._preview.sequence,
                    "inference_ms": None
                    if self._preview is None
                    else round(self._preview.inference_ns / 1_000_000, 2),
                    "frames_completed": self._preview_frames_completed,
                    "frames_skipped": self._preview_frames_skipped,
                },
            }

    def latest_frame(self) -> ThermalFrame | None:
        with self._lock:
            return self._latest

    def latest_preview(self) -> PreviewFrame | None:
        with self._lock:
            return self._preview

    def set_preview_scale(self, scale: int) -> None:
        if self._preview_runtime is None:
            raise RuntimeError("learned preview is not configured")
        if scale not in (2, 3, 4):
            raise ValueError("scale must be one of 2, 3, or 4")
        with self._preview_ready:
            self._preview_scale = scale
            self._preview_generation += 1
            self._preview = None
            self._preview_ready.notify_all()

    def start_recording(self, directory: str | Path) -> None:
        with self._lock:
            if self._recorder is not None:
                raise RuntimeError("recording is already active")
            if self._latest is None:
                raise RuntimeError("wait for the first validated frame")
            recorder = Recorder(directory, self._latest.profile)
            self._recorder = recorder.__enter__()

    def stop_recording(self) -> None:
        with self._lock:
            recorder, self._recorder = self._recorder, None
        if recorder is not None:
            recorder.close()

    def _run(self) -> None:
        try:
            for frame in self._source:
                with self._lock:
                    self._latest = frame
                    self._frames_received += 1
                    if self._recorder is not None:
                        self._recorder.write(frame)
                    self._preview_ready.notify_all()
        except Exception as exc:
            with self._lock:
                self._error = f"{type(exc).__name__}: {exc}"

    def _run_preview(self) -> None:
        assert self._preview_runtime is not None
        completed_sequence = -1
        completed_generation = -1
        while True:
            with self._preview_ready:
                while (
                    self._error is None
                    and not self._closed
                    and (
                        self._latest is None
                        or (
                            self._latest.sequence == completed_sequence
                            and self._preview_generation == completed_generation
                        )
                    )
                ):
                    self._preview_ready.wait()
                if self._error is not None or self._closed:
                    return
                frame = self._latest
                scale = self._preview_scale
                generation = self._preview_generation
            assert frame is not None
            try:
                values, inference_ns = self._preview_runtime.render(frame, scale)
            except Exception as exc:
                with self._lock:
                    self._error = f"learned preview {type(exc).__name__}: {exc}"
                return
            with self._preview_ready:
                if frame.sequence > completed_sequence + 1:
                    self._preview_frames_skipped += frame.sequence - completed_sequence - 1
                completed_sequence = frame.sequence
                completed_generation = generation
                if scale == self._preview_scale:
                    self._preview = PreviewFrame(
                        frame.sequence, scale, values, inference_ns
                    )
                self._preview_frames_completed += 1


def serve(
    source: FrameSource,
    host: str = "127.0.0.1",
    port: int = 8765,
    preview: RemoteLearnedPreview | None = None,
) -> None:
    """Run the local viewer until interrupted."""
    service = LiveService(source, preview)
    service.start()
    server = ThreadingHTTPServer((host, port), _handler(service))
    try:
        server.serve_forever()
    finally:
        server.server_close()
        service.close()


def _handler(service: LiveService) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            path = urlsplit(self.path).path
            if path == "/":
                self._send_bytes("text/html; charset=utf-8", _PAGE.encode())
            elif path == "/api/status":
                self._send_json(service.status())
            elif path == "/api/frame":
                frame = service.latest_frame()
                if frame is None:
                    self.send_error(HTTPStatus.NO_CONTENT)
                    return
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("Content-Length", str(frame.thermal_raw.nbytes))
                self.send_header("X-P3-Width", str(frame.thermal_raw.shape[1]))
                self.send_header("X-P3-Height", str(frame.thermal_raw.shape[0]))
                self.end_headers()
                self.wfile.write(frame.thermal_raw.astype("<u2", copy=False).tobytes())
            elif path == "/api/preview":
                preview = service.latest_preview()
                if preview is None:
                    self.send_error(HTTPStatus.NO_CONTENT)
                    return
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("Content-Length", str(preview.values.nbytes))
                self.send_header("X-P3-Width", str(preview.values.shape[1]))
                self.send_header("X-P3-Height", str(preview.values.shape[0]))
                self.send_header("X-P3-Source-Sequence", str(preview.sequence))
                self.send_header("X-P3-Scale", str(preview.scale))
                self.end_headers()
                self.wfile.write(preview.values.astype("<u2", copy=False).tobytes())
            else:
                self.send_error(HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:  # noqa: N802
            path = urlsplit(self.path).path
            if path == "/api/recording/stop":
                service.stop_recording()
                self._send_json(service.status())
                return
            if path == "/api/preview":
                try:
                    size = int(self.headers.get("Content-Length", "0"))
                    value: dict[str, Any] = json.loads(self.rfile.read(size))
                    service.set_preview_scale(value["scale"])
                except (KeyError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
                    self.send_error(HTTPStatus.BAD_REQUEST, str(exc))
                    return
                self._send_json(service.status())
                return
            if path != "/api/recording/start":
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                value: dict[str, Any] = json.loads(self.rfile.read(size))
                service.start_recording(value["directory"])
            except (
                KeyError,
                OSError,
                RuntimeError,
                ValueError,
                json.JSONDecodeError,
            ) as exc:
                self.send_error(HTTPStatus.BAD_REQUEST, str(exc))
                return
            self._send_json(service.status())

        def log_message(self, _format: str, *_args: object) -> None:
            pass

        def _send_json(self, value: object) -> None:
            self._send_bytes("application/json", json.dumps(value).encode())

        def _send_bytes(self, content_type: str, body: bytes) -> None:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return Handler


_PAGE = files("p3thermal.ui").joinpath("index.html").read_text(encoding="utf-8")
