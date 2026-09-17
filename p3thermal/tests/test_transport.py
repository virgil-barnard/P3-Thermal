import pytest
import usb.core
import usb.util

from p3thermal.transport import P3Transport


class FakeDevice:
    def set_configuration(self) -> None:
        pass


def test_open_releases_interface_claimed_before_later_claim_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    device = FakeDevice()
    released: list[int] = []
    disposed: list[FakeDevice] = []

    monkeypatch.setattr("p3thermal.transport._backend", lambda: None)
    monkeypatch.setattr(usb.core, "find", lambda **_: device)

    def claim(_: FakeDevice, interface: int) -> None:
        if interface == 1:
            raise RuntimeError("interface 1 busy")

    monkeypatch.setattr(usb.util, "claim_interface", claim)
    monkeypatch.setattr(usb.util, "release_interface", lambda _, i: released.append(i))
    monkeypatch.setattr(usb.util, "dispose_resources", lambda d: disposed.append(d))

    with pytest.raises(RuntimeError, match="interface 1 busy"):
        P3Transport.open()

    assert released == [0]
    assert disposed == [device]
