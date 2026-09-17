import time
from collections.abc import Iterator
from http.server import ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from urllib.request import Request, urlopen

import numpy as np

from p3thermal.frames import DeviceProfile, ThermalFrame
from p3thermal.service import LiveService, _handler
from p3thermal.source import FrameSource


class Source(FrameSource):
    def __init__(self) -> None:
        self.closed = False
        profile = DeviceProfile("test", 1, 2, 2, 2, 24)
        self.frame = ThermalFrame(
            0,
            1,
            1,
            profile,
            np.array([[1, 2], [3, 4]], dtype=np.uint16),
            np.zeros((2, 2), dtype=np.uint8),
            b"raw",
        )

    def __iter__(self) -> Iterator[ThermalFrame]:
        yield self.frame
        while not self.closed:
            time.sleep(0.001)

    def close(self) -> None:
        self.closed = True


class Preview:
    device = "test"

    def render(self, frame: ThermalFrame, scale: int) -> tuple[np.ndarray, int]:
        values = np.repeat(np.repeat(frame.thermal_raw, scale, axis=0), scale, axis=1)
        values.setflags(write=False)
        return values, 1_500_000


class FlakyPreview(Preview):
    def __init__(self) -> None:
        self.failed = False

    def render(self, frame: ThermalFrame, scale: int) -> tuple[np.ndarray, int]:
        if not self.failed:
            self.failed = True
            raise ConnectionError("sidecar restarted")
        return super().render(frame, scale)


def test_live_service_keeps_latest_frame_and_can_start_recording(
    tmp_path: Path,
) -> None:
    service = LiveService(Source())
    service.start()
    for _ in range(100):
        if service.latest_frame() is not None:
            break
        time.sleep(0.005)
    service.start_recording(tmp_path / "session")

    assert service.status()["recording"] is True
    assert service.latest_frame() is not None

    service.close()
    assert service.status()["recording"] is False


def test_http_endpoints_expose_status_and_native_little_endian_frame() -> None:
    service = LiveService(Source())
    service.start()
    for _ in range(100):
        if service.latest_frame() is not None:
            break
        time.sleep(0.005)
    server = ThreadingHTTPServer(("127.0.0.1", 0), _handler(service))
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_port}"
    try:
        with urlopen(base_url) as response:
            assert b"Digital zoom" in response.read()
        with urlopen(f"{base_url}/api/status") as response:
            assert b'"frames_received": 1' in response.read()
        with urlopen(f"{base_url}/api/frame") as response:
            assert response.headers["X-P3-Width"] == "2"
            expected = np.array([[1, 2], [3, 4]], dtype="<u2").tobytes()
            assert response.read() == expected
    finally:
        server.shutdown()
        server.server_close()
        service.close()


def test_http_learned_preview_is_derived_and_scale_selectable() -> None:
    service = LiveService(Source(), Preview())
    service.start()
    server = ThreadingHTTPServer(("127.0.0.1", 0), _handler(service))
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_port}"
    try:
        for _ in range(100):
            if service.latest_preview() is not None:
                break
            time.sleep(0.005)
        else:
            raise AssertionError("learned preview was not rendered")
        with urlopen(f"{base_url}/api/preview") as response:
            assert response.headers["X-P3-Scale"] == "2"
            assert response.headers["X-P3-Source-Sequence"] == "0"
            assert len(response.read()) == 4 * 4 * 2
        request = Request(
            f"{base_url}/api/preview",
            data=b'{"scale": 3}',
            method="POST",
        )
        with urlopen(request) as response:
            assert b'"scale": 3' in response.read()
        for _ in range(100):
            preview = service.latest_preview()
            if preview is not None and preview.scale == 3:
                break
            time.sleep(0.005)
        else:
            raise AssertionError("new learned preview scale was not rendered")
        with urlopen(f"{base_url}/api/preview") as response:
            assert len(response.read()) == 6 * 6 * 2
    finally:
        server.shutdown()
        server.server_close()
        service.close()


def test_learned_preview_retries_after_a_transient_sidecar_error() -> None:
    service = LiveService(Source(), FlakyPreview())
    service.start()
    try:
        for _ in range(300):
            if service.latest_preview() is not None:
                break
            time.sleep(0.01)
        else:
            raise AssertionError("learned preview did not recover")
        assert service.status()["error"] is None
        assert service.status()["learned_preview"]["error"] is None
    finally:
        service.close()
