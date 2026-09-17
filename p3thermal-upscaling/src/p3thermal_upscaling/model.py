"""Small shared-trunk upsampler whose output units remain thermal DN."""

from __future__ import annotations

from torch import Tensor, nn
from torch.nn import functional as F


class MultiScaleUpsampler(nn.Module):
    """Predict a residual over bicubic interpolation for one supported scale."""

    def __init__(self, channels: int = 32) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.head2 = nn.Conv2d(channels, 4, 3, padding=1)
        self.head3 = nn.Conv2d(channels, 9, 3, padding=1)
        self.head4 = nn.Conv2d(channels, 16, 3, padding=1)

    def forward(self, image: Tensor, scale: int) -> Tensor:
        base = F.interpolate(
            image,
            size=(image.shape[-2] * scale, image.shape[-1] * scale),
            mode="bicubic",
            align_corners=False,
        )
        features = self.features(image)
        if scale == 2:
            residual = F.pixel_shuffle(self.head2(features), 2)
        elif scale == 3:
            residual = F.pixel_shuffle(self.head3(features), 3)
        elif scale == 4:
            residual = F.pixel_shuffle(self.head4(features), 4)
        else:
            raise ValueError("scale must be one of 2, 3, or 4")
        return base + residual


class RawDnUpsampler(nn.Module):
    """Embed per-frame DN normalization so exported inference matches training."""

    def __init__(self, model: MultiScaleUpsampler) -> None:
        super().__init__()
        self.model = model

    def forward(self, raw: Tensor, scale: int) -> Tensor:
        center = raw.mean(dim=(2, 3), keepdim=True)
        spread = raw.std(dim=(2, 3), keepdim=True).clamp_min(16.0)
        normalized = (raw - center) / spread
        return self.model(normalized, scale) * spread + center
