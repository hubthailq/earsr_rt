"""EDSR-baseline (Lim và cộng sự, 2017): 16 khối residual, 64 kênh, không batch-norm.
Tên tham số theo trọng số chính thức ``edsr_baseline_x4-6b446fab.pt`` của tác giả
(kho EDSR-PyTorch). Mạng làm việc trên dải 0..255."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

RGB_MEAN = (0.4488, 0.4371, 0.4040)


class MeanShift(nn.Module):
    """Trừ (sign = -1) hoặc cộng (sign = +1) trung bình RGB của DIV2K.

    Bản gốc là một ``nn.Conv2d`` 1×1 bị đóng băng. Ở đây ``weight`` và ``bias`` là
    buffer: khóa trong state_dict vẫn như bản gốc nên nạp chặt được, nhưng chúng
    không phải tham số, nên không mục tiêu huấn luyện nào mở khóa nhầm."""

    def __init__(self, rgb_range: float, sign: int):
        super().__init__()
        self.register_buffer("weight", torch.eye(3).view(3, 3, 1, 1))
        self.register_buffer("bias", sign * rgb_range * torch.tensor(RGB_MEAN))

    def forward(self, x):
        return F.conv2d(x, self.weight, self.bias)


class ResBlock(nn.Module):
    def __init__(self, nf: int):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(nf, nf, 3, 1, 1), nn.ReLU(inplace=False), nn.Conv2d(nf, nf, 3, 1, 1))

    def forward(self, x):
        return x + self.body(x)


class EDSR(nn.Module):
    def __init__(self, nf: int = 64, nb: int = 16, upscale: int = 4, rgb_range: float = 255.0):
        super().__init__()
        if upscale != 4:
            raise ValueError("kho chỉ có trọng số EDSR-baseline cho ×4")
        self.sub_mean = MeanShift(rgb_range, -1)
        self.add_mean = MeanShift(rgb_range, +1)
        self.head = nn.Sequential(nn.Conv2d(3, nf, 3, 1, 1))
        self.body = nn.Sequential(*[ResBlock(nf) for _ in range(nb)], nn.Conv2d(nf, nf, 3, 1, 1))
        up = nn.Sequential(nn.Conv2d(nf, nf * 4, 3, 1, 1), nn.PixelShuffle(2),
                           nn.Conv2d(nf, nf * 4, 3, 1, 1), nn.PixelShuffle(2))
        self.tail = nn.Sequential(up, nn.Conv2d(nf, 3, 3, 1, 1))

    def forward(self, x):
        x = self.head(self.sub_mean(x))
        return self.add_mean(self.tail(x + self.body(x)))
