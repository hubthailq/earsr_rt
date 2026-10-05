"""N3: hai đầu ra và cổng điểm ảnh; N1: đầu phụ cấu trúc (chỉ lúc huấn luyện).

    thân -> đặc trưng F
            |- đầu trung thực  f_S = up_S(F)
            |- đầu kết cấu     f_T = up_T(F)
            `- cổng            G = sigmoid(g([F_up, |f_T - f_S|]) + bias)
    x_hat = G * f_S + (1 - G) * f_T

``gate_bias`` dịch cổng lúc suy luận: giá trị lớn cho bản thiên trung thực (F),
giá trị nhỏ cho bản thiên sắc nét (P), từ cùng một bộ trọng số.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..span import SPAN


class PixelGate(nn.Module):
    """Cổng theo điểm ảnh ở độ phân giải ra. Nhận đặc trưng LR và |f_T - f_S|."""

    def __init__(self, feat_ch: int, scale: int, hidden: int = 16):
        super().__init__()
        self.scale = scale
        self.reduce = nn.Conv2d(feat_ch, hidden, 1)
        self.body = nn.Sequential(nn.Conv2d(hidden + 1, hidden, 3, padding=1), nn.SiLU(),
                                  nn.Conv2d(hidden, 1, 3, padding=1))

    def forward(self, feat: torch.Tensor, f_s: torch.Tensor, f_t: torch.Tensor, bias: float = 0.0) -> torch.Tensor:
        up = F.interpolate(self.reduce(feat), scale_factor=self.scale, mode="nearest")
        diff = (f_t - f_s).abs().mean(1, keepdim=True)
        return torch.sigmoid(self.body(torch.cat([up, diff], 1)) + bias)


class TwoHeadGated(nn.Module):
    """Thân SPAN với hai đầu ra và một cổng.

    ``forward(x, gate=None)``:
        gate=None      : dùng cổng học được, dịch bởi ``gate_bias``.
        gate=1.0 / 0.0 : ép cổng; 1 cho đúng f_S, 0 cho đúng f_T (kiểm thử dựa vào đây).
    """

    def __init__(self, backbone: SPAN, hidden: int = 16):
        super().__init__()
        self.backbone = backbone
        fc, s = backbone.feature_channels, backbone.upscale
        self.head_t = nn.Sequential(nn.Conv2d(fc, 3 * s * s, 3, padding=1, padding_mode=backbone.padding_mode),
                                    nn.PixelShuffle(s))
        self.gate = PixelGate(fc, s, hidden)
        self.gate_bias = 0.0

    def forward(self, x: torch.Tensor, gate: float | None = None, return_all: bool = False):
        feat = self.backbone.forward_features(x)
        f_s = self.backbone.upsampler(feat)
        f_t = self.head_t(feat)
        if gate is None:
            g = self.gate(feat, f_s, f_t, self.gate_bias)
        else:
            g = torch.full_like(f_s[:, :1], float(gate))
        out = g * f_s + (1 - g) * f_t
        return (out, f_s, f_t, g) if return_all else out


def gate_label(f_s: torch.Tensor, f_t: torch.Tensor, hr: torch.Tensor, tau: float) -> torch.Tensor:
    """Nhãn cổng có biên τ: 1 (giữ đầu trung thực) ở nơi đầu kết cấu sai hơn đầu
    trung thực quá τ, ngược lại 0 (cho phép thêm chi tiết). Sai số là L1 trung bình theo kênh."""
    e_s = (f_s - hr).abs().mean(1, keepdim=True)
    e_t = (f_t - hr).abs().mean(1, keepdim=True)
    return (e_t > e_s + tau).float()


class AuxStructureHead(nn.Module):
    """N1: dự đoán bản đồ nhiệt điểm mốc (hoặc đường viền) từ đặc trưng của thân.
    Chỉ dùng lúc huấn luyện; gỡ đi không làm đổi đầu ra SR."""

    def __init__(self, feat_ch: int, n_maps: int, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(feat_ch, hidden, 3, padding=1), nn.SiLU(), nn.Conv2d(hidden, n_maps, 1))

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.net(feat)
