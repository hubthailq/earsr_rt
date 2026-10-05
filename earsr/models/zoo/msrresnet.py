"""MSRResNet (SRResNet không batch-norm), 16 khối, 64 kênh; cỡ tương đương
EDSR-baseline. Tên tham số theo trọng số ``msrresnet_x4_psnr.pth`` của KAIR."""
from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F


class ResidualBlockNoBN(nn.Module):
    def __init__(self, nc: int = 64):
        super().__init__()
        self.conv1 = nn.Conv2d(nc, nc, 3, 1, 1)
        self.conv2 = nn.Conv2d(nc, nc, 3, 1, 1)

    def forward(self, x):
        return x + self.conv2(F.relu(self.conv1(x), inplace=False))


class MSRResNet(nn.Module):
    def __init__(self, in_nc: int = 3, out_nc: int = 3, nc: int = 64, nb: int = 16, upscale: int = 4):
        super().__init__()
        if upscale != 4:
            raise ValueError("chỉ có trọng số công bố cho ×4")
        self.upscale = upscale
        self.conv_first = nn.Conv2d(in_nc, nc, 3, 1, 1)
        self.recon_trunk = nn.Sequential(*[ResidualBlockNoBN(nc) for _ in range(nb)])
        self.upconv1 = nn.Conv2d(nc, nc * 4, 3, 1, 1)
        self.upconv2 = nn.Conv2d(nc, nc * 4, 3, 1, 1)
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.HRconv = nn.Conv2d(nc, nc, 3, 1, 1)
        self.conv_last = nn.Conv2d(nc, out_nc, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.1, inplace=False)

    def forward(self, x):
        fea = self.lrelu(self.conv_first(x))
        out = self.recon_trunk(fea)
        out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
        out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
        out = self.conv_last(self.lrelu(self.HRconv(out)))
        return out + F.interpolate(x, scale_factor=self.upscale, mode="bilinear", align_corners=False)
