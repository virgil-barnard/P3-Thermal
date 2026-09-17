"""Native USB ownership and bounded P3 control/stream operations."""

from __future__ import annotations

import time
from dataclasses import dataclass

import usb.core
import usb.util

from .protocol import START_STREAM

VID = 0x3474
PID = 0x45A2
CONTROL_INTERFACE = 0
STREAM_INTERFACE = 1
STREAM_ALT_SETTING = 1
STREAM_ENDPOINT = 0x81


@dataclass(frozen=True, slots=True)
class CommandResponse:
    """Status bytes observed around one P3 vendor command."""

    status_after_write: bytes
    response: bytes
    status_after_read: bytes


class P3Transport:
    """One owned P3 USB device. Call close when finished."""

    def __init__(self, device: usb.core.Device) -> None:
        self._device = device
        self._claimed_interfaces: list[int] = []
        self._streaming = False

    @classmethod
    def open(cls) -> P3Transport:
        """Find, configure, and claim the qualified P3 interfaces."""
        device = usb.core.find(idVendor=VID, idProduct=PID, backend=_backend())
        if device is None:
            raise RuntimeError("P3 (3474:45a2) was not found")
        transport = cls(device)
        try:
            device.set_configuration()
            usb.util.claim_interface(device, CONTROL_INTERFACE)
            transport._claimed_interfaces.append(CONTROL_INTERFACE)
            usb.util.claim_interface(device, STREAM_INTERFACE)
            transport._claimed_interfaces.append(STREAM_INTERFACE)
            return transport
        except BaseException:
            transport.close()
            raise

    def command(self, command: bytes, response_length: int) -> CommandResponse:
        """Send one serialized vendor command and collect its acknowledgements."""
        self._require_open()
        self._device.ctrl_transfer(0x41, 0x20, 0, 0, command, timeout=1_000)
        status_after_write = bytes(
            self._device.ctrl_transfer(0xC1, 0x22, 0, 0, 1, timeout=1_000)
        )
        response = bytes(
            self._device.ctrl_transfer(0xC1, 0x21, 0, 0, response_length, timeout=1_000)
        )
        status_after_read = bytes(
            self._device.ctrl_transfer(0xC1, 0x22, 0, 0, 1, timeout=1_000)
        )
        return CommandResponse(status_after_write, response, status_after_read)

    def start_stream(self) -> tuple[CommandResponse, CommandResponse]:
        """Run the qualified P3 stream transition sequence."""
        self._require_open()
        first = self.command(START_STREAM, 1)
        time.sleep(1)
        self._device.set_interface_altsetting(
            interface=STREAM_INTERFACE, alternate_setting=STREAM_ALT_SETTING
        )
        self._streaming = True
        self._device.ctrl_transfer(0x40, 0xEE, 0, 1, None, timeout=1_000)
        time.sleep(2)
        return first, self.command(START_STREAM, 1)

    def read_stream_chunk(self, size: int = 65_536, timeout_ms: int = 1_000) -> bytes:
        """Read one bounded bulk chunk; its boundary has no framing meaning."""
        self._require_open()
        if not self._streaming:
            raise RuntimeError("P3 stream is not active")
        return bytes(self._device.read(STREAM_ENDPOINT, size, timeout=timeout_ms))

    def descriptor_report(self) -> dict[str, object]:
        """Return USB descriptors needed to qualify this host/device profile."""
        self._require_open()
        interfaces: list[dict[str, object]] = []
        for configuration in self._device:
            for interface in configuration:
                endpoints = [
                    {
                        "address": endpoint.bEndpointAddress,
                        "attributes": endpoint.bmAttributes,
                        "max_packet_size": endpoint.wMaxPacketSize,
                    }
                    for endpoint in interface
                ]
                interfaces.append(
                    {
                        "configuration": configuration.bConfigurationValue,
                        "number": interface.bInterfaceNumber,
                        "alternate_setting": interface.bAlternateSetting,
                        "class": interface.bInterfaceClass,
                        "subclass": interface.bInterfaceSubClass,
                        "protocol": interface.bInterfaceProtocol,
                        "endpoints": endpoints,
                    }
                )
        return {
            "vendor_id": self._device.idVendor,
            "product_id": self._device.idProduct,
            "device_version_bcd": self._device.bcdDevice,
            "usb_version_bcd": self._device.bcdUSB,
            "interfaces": interfaces,
        }

    def close(self) -> None:
        """Restore the idle alternate setting and release claimed interfaces."""
        if self._streaming:
            try:
                self._device.set_interface_altsetting(
                    interface=STREAM_INTERFACE, alternate_setting=0
                )
            finally:
                self._streaming = False
        while self._claimed_interfaces:
            interface = self._claimed_interfaces.pop()
            usb.util.release_interface(self._device, interface)
        usb.util.dispose_resources(self._device)

    def _require_open(self) -> None:
        if not self._claimed_interfaces:
            raise RuntimeError("P3 transport is closed")

    def __enter__(self) -> P3Transport:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def _backend() -> object | None:
    """Use the packaged libusb DLL on Windows; use PyUSB's default elsewhere."""
    try:
        import libusb_package
        import usb.backend.libusb1
    except ImportError:
        return None
    return usb.backend.libusb1.get_backend(find_library=libusb_package.find_library)
