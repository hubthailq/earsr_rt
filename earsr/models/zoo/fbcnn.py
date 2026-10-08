"""FBCNN (Jiang và cộng sự, ICCV 2021): mạng khử vết nén JPEG, tự đoán mức nén của ảnh vào.

Mã rút gọn từ ``models/network_fbcnn.py`` của kho https://github.com/jiaxi-jiang/FBCNN (giấy phép Apache-2.0),
giữ nguyên tên tham số để nạp chặt trọng số ``fbcnn_color.pth`` của tác giả. Chỉ giữ cấu hình của bản màu:
bốn tầng 64, 128, 256, 512 kênh, bốn khối residual mỗi tầng, thu nhỏ bằng tích chập bước 2, phóng bằng tích chập
chuyển vị. Mạng nhận và trả ảnh RGB trong [0, 1], cùng cỡ; cỡ bất kỳ (mạng tự đệm tới bội số của 8).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _seq(*mods: nn.Module) -> nn.Module:
    return mods[0] if len(mods) == 1 else nn.Sequential(*mods)


class ResBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.res = nn.Sequential(nn.Conv2d(ch, ch, 3, 1, 1), nn.ReLU(inplace=True), nn.Conv2d(ch, ch, 3, 1, 1))

    def forward(self, x):
        return x + self.res(x)


class QFAttention(nn.Module):
    """Khối residual có nhánh được co giãn và dịch theo mức nén (gamma, beta)."""

    def __init__(self, ch: int):
        super().__init__()
        self.res = nn.Sequential(nn.Conv2d(ch, ch, 3, 1, 1), nn.ReLU(inplace=True), nn.Conv2d(ch, ch, 3, 1, 1))

    def forward(self, x, gamma, beta):
        return x + gamma[..., None, None] * self.res(x) + beta[..., None, None]


class FBCNN(nn.Module):
    def __init__(self, in_nc: int = 3, out_nc: int = 3, nc=(64, 128, 256, 512), nb: int = 4):
        super().__init__()
        self.nb = nb
        self.m_head = nn.Conv2d(in_nc, nc[0], 3, 1, 1)
        self.m_down1 = _seq(*[ResBlock(nc[0]) for _ in range(nb)], nn.Conv2d(nc[0], nc[1], 2, 2, 0))
        self.m_down2 = _seq(*[ResBlock(nc[1]) for _ in range(nb)], nn.Conv2d(nc[1], nc[2], 2, 2, 0))
        self.m_down3 = _seq(*[ResBlock(nc[2]) for _ in range(nb)], nn.Conv2d(nc[2], nc[3], 2, 2, 0))
        self.m_body_encoder = _seq(*[ResBlock(nc[3]) for _ in range(nb)])
        self.m_body_decoder = _seq(*[ResBlock(nc[3]) for _ in range(nb)])
        self.m_up3 = nn.ModuleList([nn.ConvTranspose2d(nc[3], nc[2], 2, 2, 0), *[QFAttention(nc[2]) for _ in range(nb)]])
        self.m_up2 = nn.ModuleList([nn.ConvTranspose2d(nc[2], nc[1], 2, 2, 0), *[QFAttention(nc[1]) for _ in range(nb)]])
        self.m_up1 = nn.ModuleList([nn.ConvTranspose2d(nc[1], nc[0], 2, 2, 0), *[QFAttention(nc[0]) for _ in range(nb)]])
        self.m_tail = nn.Conv2d(nc[0], out_nc, 3, 1, 1)
        self.qf_pred = _seq(*[ResBlock(nc[3]) for _ in range(nb)], nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
                            nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, 1), nn.Sigmoid())
        self.qf_embed = _seq(nn.Linear(1, 512), nn.ReLU(), nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, 512), nn.ReLU())
        self.to_gamma_3 = _seq(nn.Linear(512, nc[2]), nn.Sigmoid())
        self.to_beta_3 = _seq(nn.Linear(512, nc[2]), nn.Tanh())
        self.to_gamma_2 = _seq(nn.Linear(512, nc[1]), nn.Sigmoid())
        self.to_beta_2 = _seq(nn.Linear(512, nc[1]), nn.Tanh())
        self.to_gamma_1 = _seq(nn.Linear(512, nc[0]), nn.Sigmoid())
        self.to_beta_1 = _seq(nn.Linear(512, nc[0]), nn.Tanh())

    def forward(self, x, qf_input=None):
        """Trả (ảnh đã khử nén, mức nén đoán được trong [0, 1]). ``qf_input``: ép mức nén thay cho giá trị đoán."""
        h, w = x.shape[-2:]
        x = F.pad(x, (0, math.ceil(w / 8) * 8 - w, 0, math.ceil(h / 8) * 8 - h), mode="replicate")
        x1 = self.m_head(x)
        x2 = self.m_down1(x1)
        x3 = self.m_down2(x2)
        x4 = self.m_down3(x3)
        x = self.m_body_encoder(x4)
        qf = self.qf_pred(x)
        x = self.m_body_decoder(x)
        e = self.qf_embed(qf if qf_input is None else qf_input)
        x = x + x4
        for up, skip, g, b in ((self.m_up3, x3, self.to_gamma_3(e), self.to_beta_3(e)),
                               (self.m_up2, x2, self.to_gamma_2(e), self.to_beta_2(e)),
                               (self.m_up1, x1, self.to_gamma_1(e), self.to_beta_1(e))):
            x = up[0](x)
            for blk in list(up)[1:]:
                x = blk(x, g, b)
            x = x + skip
        return self.m_tail(x)[..., :h, :w], qf


class FBCNNBlind(nn.Module):
    """FBCNN ở chế độ mù (mạng tự đoán mức nén); chỉ trả ảnh."""

    def __init__(self):
        super().__init__()
        self.net = FBCNN()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)[0]
