import numpy as np
import pytest

from p3thermal import DeviceProfile, ThermalFrame


@pytest.fixture
def profile() -> DeviceProfile:
    return DeviceProfile(
        identifier="p3-provisional",
        vendor_id=0x3474,
        product_id=0x45A2,
        thermal_width=256,
        thermal_height=192,
        raw_frame_size=197632,
    )


def test_frame_arrays_are_read_only(profile: DeviceProfile) -> None:
    thermal = np.zeros((192, 256), dtype=np.uint16)
    brightness = np.zeros((192, 256), dtype=np.uint8)

    frame = ThermalFrame(0, 0, 1, profile, thermal, brightness, b"frame")

    assert not frame.thermal_raw.flags.writeable
    assert frame.brightness is not None
    assert not frame.brightness.flags.writeable
    with pytest.raises(ValueError):
        frame.thermal_raw[0, 0] = 1


def test_frame_rejects_wrong_shape(profile: DeviceProfile) -> None:
    with pytest.raises(ValueError, match="thermal_raw"):
        ThermalFrame(0, 0, 1, profile, np.zeros((1, 1), dtype=np.uint16), None, b"")
