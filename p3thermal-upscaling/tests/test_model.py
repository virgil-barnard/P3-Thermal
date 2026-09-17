import warnings

import pytest
import torch

from p3thermal_upscaling.model import MultiScaleUpsampler, RawDnUpsampler


@pytest.mark.parametrize("scale", [2, 3, 4])
def test_raw_dn_model_selects_requested_output_shape(scale):
    output = RawDnUpsampler(MultiScaleUpsampler(channels=8))(torch.ones(1, 1, 24, 32), scale)
    assert output.shape == (1, 1, 24 * scale, 32 * scale)


def test_model_rejects_unsupported_scale():
    with pytest.raises(ValueError, match="2, 3, or 4"):
        MultiScaleUpsampler(channels=8)(torch.ones(1, 1, 24, 32), 8)


def test_raw_dn_model_is_torchscript_exportable():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        exported = torch.jit.script(RawDnUpsampler(MultiScaleUpsampler(channels=8)))
    assert exported(torch.ones(1, 1, 24, 32), 3).shape == (1, 1, 72, 96)
