"""Hai bộ dò điểm mốc tai, cố ý khác họ kiến trúc (mục 1.9 của kế hoạch: bộ dò
dùng để đo phải khác bộ dò sinh nhãn huấn luyện cho N1).

HeatmapNet : mã hóa, giải mã kiểu U-Net; ra K bản đồ nhiệt ở 1/2 độ phân giải;
             toạ độ lấy bằng soft-argmax.
RegressNet : chuỗi tích chập bước 2, gộp toàn cục, hồi quy thẳng 2K toạ độ.

Cả hai nhận ảnh RGB [0, 1] cỡ cố định ``input_hw`` và có ``predict(x)`` trả về
toạ độ (B, K, 2) dạng (x, y) theo điểm ảnh của ảnh vào.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _cbr(c_in: int, c_out: int, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(c_in, c_out, 3, stride, 1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True))


def soft_argmax(hm: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
    """(B, K, h, w) -> (B, K, 2) toạ độ (x, y) theo điểm ảnh của bản đồ."""
    b, k, h, w = hm.shape
    p = F.softmax(hm.reshape(b, k, -1) * beta, dim=-1).reshape(b, k, h, w)
    xs = torch.arange(w, device=hm.device, dtype=hm.dtype)
    ys = torch.arange(h, device=hm.device, dtype=hm.dtype)
    return torch.stack([(p.sum(2) * xs).sum(-1), (p.sum(3) * ys).sum(-1)], -1)


class HeatmapNet(nn.Module):
    arch = "heatmap"

    def __init__(self, n_points: int = 55, width: int = 32, input_hw: tuple[int, int] = (136, 96)):
        super().__init__()
        if input_hw[0] % 8 or input_hw[1] % 8:
            raise ValueError("input_hw phải chia hết cho 8")
        w = width
        self.n_points, self.input_hw = n_points, tuple(input_hw)
        self.e0 = nn.Sequential(_cbr(3, w), _cbr(w, w))
        self.e1 = nn.Sequential(_cbr(w, 2 * w, 2), _cbr(2 * w, 2 * w))
        self.e2 = nn.Sequential(_cbr(2 * w, 4 * w, 2), _cbr(4 * w, 4 * w))
        self.e3 = nn.Sequential(_cbr(4 * w, 8 * w, 2), _cbr(8 * w, 8 * w))
        self.d2 = nn.Sequential(_cbr(12 * w, 4 * w), _cbr(4 * w, 4 * w))
        self.d1 = nn.Sequential(_cbr(6 * w, 2 * w), _cbr(2 * w, 2 * w))
        self.out = nn.Conv2d(2 * w, n_points, 1)
        self.beta = nn.Parameter(torch.tensor(10.0))   # độ nhọn của soft-argmax, học được

    def heatmaps(self, x: torch.Tensor) -> torch.Tensor:
        up = lambda t: F.interpolate(t, scale_factor=2, mode="bilinear", align_corners=False)
        f0 = self.e0(x)
        f1 = self.e1(f0)
        f2 = self.e2(f1)
        f3 = self.e3(f2)
        y = self.d2(torch.cat([up(f3), f2], 1))
        y = self.d1(torch.cat([up(y), f1], 1))
        return self.out(y)                              # ở 1/2 độ phân giải

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.predict(x)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        c = soft_argmax(self.heatmaps(x), self.beta.clamp(1.0, 100.0))
        return (c + 0.5) * 2.0 - 0.5                    # về toạ độ ảnh vào (tâm điểm ảnh)


class RegressNet(nn.Module):
    arch = "regress"

    def __init__(self, n_points: int = 55, width: int = 32, input_hw: tuple[int, int] = (136, 96)):
        super().__init__()
        w = width
        self.n_points, self.input_hw = n_points, tuple(input_hw)
        self.body = nn.Sequential(_cbr(3, w, 2), _cbr(w, w), _cbr(w, 2 * w, 2), _cbr(2 * w, 2 * w),
                                  _cbr(2 * w, 4 * w, 2), _cbr(4 * w, 4 * w), _cbr(4 * w, 8 * w, 2),
                                  _cbr(8 * w, 8 * w))
        h, ww = input_hw[0] // 16, input_hw[1] // 16
        self.pool = nn.AdaptiveAvgPool2d((max(1, h // 2), max(1, ww // 2)))
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(8 * w * max(1, h // 2) * max(1, ww // 2), 512),
                                nn.ReLU(inplace=True), nn.Linear(512, 2 * n_points))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.predict(x)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        o = self.fc(self.pool(self.body(x))).reshape(x.shape[0], self.n_points, 2)
        h, w = self.input_hw
        scale = torch.tensor([w, h], device=x.device, dtype=o.dtype)
        return (o + 0.5) * scale - 0.5                  # đầu ra 0 ứng với tâm ảnh


ARCHS = {"heatmap": HeatmapNet, "regress": RegressNet}


def build_detector(arch: str, n_points: int = 55, width: int = 32, input_hw=(136, 96)) -> nn.Module:
    if arch not in ARCHS:
        raise ValueError(f"arch phải thuộc {list(ARCHS)}")
    return ARCHS[arch](n_points=n_points, width=width, input_hw=tuple(input_hw))
