from dataclasses import dataclass

from p3thermal.diagnostics import build_inspection_report


@dataclass
class FakeResponse:
    response: bytes


class FakeTransport:
    def command(self, command: bytes, response_length: int) -> FakeResponse:
        values = {30: b"P3\0", 12: b"00.00.02.17\0", 64: b"P30-TEST\0"}
        return FakeResponse(values[response_length])

    def descriptor_report(self) -> dict[str, object]:
        return {"vendor_id": 0x3474, "product_id": 0x45A2, "interfaces": []}


def test_inspection_report_collects_read_only_identification() -> None:
    report = build_inspection_report(FakeTransport())

    assert report["schema_version"] == 1
    assert report["identification"] == {
        "model": "P3",
        "firmware_version": "00.00.02.17",
        "part_number": "P30-TEST",
        "serial": "P30-TEST",
        "hardware_version": "P30-TEST",
    }
    assert report["usb"]["vendor_id"] == 0x3474
