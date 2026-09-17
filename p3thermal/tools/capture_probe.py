"""Capture one native P3 transfer and leave the device in its idle state."""

from __future__ import annotations

import time

import usb.core

from p3thermal.protocol import READ_MODEL, FrameAssembler
from p3thermal.transport import P3Transport


def main() -> None:
    with P3Transport.open() as transport:
        model_response = transport.command(READ_MODEL, 30)
        model = model_response.response.split(b"\0", 1)[0].decode("ascii", "replace")
        first, final = transport.start_stream()
        assembler = FrameAssembler()
        deadline = time.monotonic() + 10
        reads = 0
        frame = None
        while time.monotonic() < deadline and frame is None:
            try:
                frames = assembler.feed(transport.read_stream_chunk())
                reads += 1
            except usb.core.USBTimeoutError:
                continue
            if frames:
                frame = frames[0]
        if frame is None:
            raise RuntimeError(f"no validated frame after {reads} USB reads")
        print(f"model: {model}")
        first_ack = f"{first.status_after_write.hex()}/{first.response.hex()}"
        print(f"start responses: {first_ack} ", end="")
        print(f"{final.status_after_write.hex()}/{final.response.hex()}")
        print(
            f"frame bytes: {len(frame)} USB reads: {reads} marker: {frame[:12].hex()}"
        )


if __name__ == "__main__":
    main()
