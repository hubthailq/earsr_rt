"""Các loss ngoài L1: cảm nhận (VGG19), GAN, LDL. Dùng cho các mốc "cùng thân,
khác cách huấn luyện" (mục 1.8) và cho N3.

CHƯA HUẤN LUYỆN THẬT với các loss này. Trọng số loss là giá trị của Real-ESRGAN
và LDL, chưa dò cho ảnh tai.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

# tên tầng của VGG19 (torchvision.features) -> chỉ số; lấy đặc trưng TRƯỚC kích hoạt như ESRGAN
VGG19_LAYERS = {"conv1_2": 2, "conv2_2": 7, "conv3_4": 16, "conv4_4": 25, "conv5_4": 34}
REALESRGAN_WEIGHTS = {"conv1_2": 0.1, "conv2_2": 0.1, "conv3_4": 1.0, "conv4_4": 1.0, "conv5_4": 1.0}


class PerceptualLoss(nn.Module):
    """L1 giữa đặc trưng VGG19 của ảnh SR và ảnh đáp án (ảnh trong [0, 1]).

    ``pretrained=True`` tải trọng số ImageNet của torchvision (cần mạng ở lần đầu).
    ``pretrained=False`` chỉ dùng cho kiểm thử.
    """

    def __init__(self, layer_weights: dict | None = None, pretrained: bool = True):
        super().__init__()
        import torchvision

        self.layer_weights = dict(layer_weights or REALESRGAN_WEIGHTS)
        last = max(VGG19_LAYERS[k] for k in self.layer_weights)
        w = torchvision.models.VGG19_Weights.IMAGENET1K_V1 if pretrained else None
        vgg = torchvision.models.vgg19(weights=w).features[: last + 1]
        for m in vgg:  # ReLU inplace sẽ ghi đè đặc trưng vừa lấy
            if isinstance(m, nn.ReLU):
                m.inplace = False
        self.vgg = vgg.eval()
        for p in self.vgg.parameters():
            p.requires_grad_(False)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self._idx = {v: k for k, v in VGG19_LAYERS.items() if k in self.layer_weights}
        self.eval()

    def train(self, mode: bool = True):  # luôn ở chế độ eval
        return super().train(False)

    def features(self, x: torch.Tensor) -> dict:
        x = (x - self.mean) / self.std
        out = {}
        for i, m in enumerate(self.vgg):
            x = m(x)
            if i in self._idx:
                out[self._idx[i]] = x
        return out

    def forward(self, sr: torch.Tensor, hr: torch.Tensor) -> torch.Tensor:
        fs, fh = self.features(sr), self.features(hr.detach())
        return sum(w * F.l1_loss(fs[k], fh[k]) for k, w in self.layer_weights.items())


class UNetDiscriminatorSN(nn.Module):
    """Bộ phân biệt U-Net có chuẩn hóa phổ, như Real-ESRGAN. Nhận ảnh cỡ bất kỳ
    chia hết cho 8; trả bản đồ logit cùng cỡ ảnh vào."""

    def __init__(self, num_in_ch: int = 3, num_feat: int = 64):
        super().__init__()
        n, sn = num_feat, spectral_norm
        self.conv0 = nn.Conv2d(num_in_ch, n, 3, 1, 1)
        self.conv1 = sn(nn.Conv2d(n, n * 2, 4, 2, 1, bias=False))
        self.conv2 = sn(nn.Conv2d(n * 2, n * 4, 4, 2, 1, bias=False))
        self.conv3 = sn(nn.Conv2d(n * 4, n * 8, 4, 2, 1, bias=False))
        self.conv4 = sn(nn.Conv2d(n * 8, n * 4, 3, 1, 1, bias=False))
        self.conv5 = sn(nn.Conv2d(n * 4, n * 2, 3, 1, 1, bias=False))
        self.conv6 = sn(nn.Conv2d(n * 2, n, 3, 1, 1, bias=False))
        self.conv7 = sn(nn.Conv2d(n, n, 3, 1, 1, bias=False))
        self.conv8 = sn(nn.Conv2d(n, n, 3, 1, 1, bias=False))
        self.conv9 = nn.Conv2d(n, 1, 3, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = lambda t: F.leaky_relu(t, 0.2)
        up = lambda t: F.interpolate(t, scale_factor=2, mode="bilinear", align_corners=False)
        x0 = a(self.conv0(x))
        x1 = a(self.conv1(x0))
        x2 = a(self.conv2(x1))
        x3 = a(self.conv3(x2))
        x4 = a(self.conv4(up(x3))) + x2
        x5 = a(self.conv5(up(x4))) + x1
        x6 = a(self.conv6(up(x5))) + x0
        return self.conv9(a(self.conv8(a(self.conv7(x6)))))


def gan_loss(logits: torch.Tensor, real: bool) -> torch.Tensor:
    """GAN thường (BCE với logit)."""
    t = torch.ones_like(logits) if real else torch.zeros_like(logits)
    return F.binary_cross_entropy_with_logits(logits, t)


def _local_var(residual: torch.Tensor, ksize: int) -> torch.Tensor:
    pad = (ksize - 1) // 2
    r = F.pad(residual, [pad] * 4, mode="reflect")
    u = r.unfold(2, ksize, 1).unfold(3, ksize, 1)
    return u.var(dim=(-1, -2), unbiased=True)


@torch.no_grad()
def ldl_artifact_map(hr: torch.Tensor, sr: torch.Tensor, sr_ema: torch.Tensor, ksize: int = 7) -> torch.Tensor:
    """Bản đồ artifact của LDL (Liang và cộng sự, CVPR 2022).

    Trọng số = (phương sai toàn patch của phần dư)^(1/5) × phương sai cục bộ
    ``ksize``×``ksize`` của phần dư; đặt về 0 ở nơi mô hình hiện tại sai ít hơn
    mô hình EMA (nơi đó coi là chi tiết đúng, không phạt).
    """
    res_ema = (hr - sr_ema).abs().sum(1, keepdim=True)
    res_sr = (hr - sr).abs().sum(1, keepdim=True)
    patch_w = res_sr.var(dim=(-1, -2, -3), keepdim=True) ** (1 / 5)
    w = patch_w * _local_var(res_sr, ksize)
    w[res_sr < res_ema] = 0
    return w


def ldl_loss(sr: torch.Tensor, hr: torch.Tensor, sr_ema: torch.Tensor, ksize: int = 7) -> torch.Tensor:
    w = ldl_artifact_map(hr, sr.detach(), sr_ema.detach(), ksize)
    return F.l1_loss(w * sr, w * hr)
