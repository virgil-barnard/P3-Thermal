"""Read-only device qualification reports."""

from __future__ import annotations

import platform
import sys
from importlib.metadata import version
from typing import Protocol

from .protocol import IDENTIFICATION_REGISTERS


class InspectableTransport(Protocol):
    """The read-only subset of transport required for a P3 profile report."""

    def command(self, command: bytes, response_length: int) -> object: ...

    def descriptor_report(self) -> dict[str, object]: ...


def build_inspection_report(transport: InspectableTransport) -> dict[str, object]:
    """Read identifiers and descriptor data without changing streaming state."""
    identity = {}
    for name, command, response_length in IDENTIFICATION_REGISTERS:
        response = transport.command(command, response_length)
        identity[name] = response.response.split(b"\0", 1)[0].decode("ascii", "replace")
    return {
        "schema_version": 1,
        "software": {
            "p3thermal": version("p3thermal"),
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "pyusb": version("pyusb"),
            "numpy": version("numpy"),
        },
        "identification": identity,
        "usb": transport.descriptor_report(),
    }
