"""SPAN (Swift Parameter-free Attention Network) với hai thứ có thể cấu hình
mà bản gốc không có: kiểu đệm viền và số khối.

Bản gốc: Wan và cộng sự, CVPRW 2024, https://github.com/hongyuanyu/SPAN.
Tên tham số được giữ đúng như bản gốc, nên nạp được trọng số công bố (cả dạng
huấn luyện có nhánh ``sk``/``conv`` lẫn dạng deploy chỉ có ``eval_conv``).

Công thức của một khối SPAB, theo bài gốc:
    H = c3(SiLU(c2(SiLU(c1(x)))))
    out = (H + x) * (sigmoid(H) - 0.5)
tức là cộng phần dư trước rồi mới nhân với attention. Thứ tự ngược lại không
gây lỗi chạy nhưng cho kết quả sai (tests/test_span.py kiểm công thức này).

Kiểu đệm viền (``padding_mode``):
    'zeros'     như bản gốc
    'replicate' lặp điểm ảnh viền
    'reflect'   phản chiếu
Kiểu đệm áp cho mọi tích chập 3×3 của mạng, ở cả nhánh huấn luyện lẫn nhánh đã
gộp, nên hai nhánh luôn cho cùng kết quả (tests/test_span.py).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

PADDING_MODES = ("zeros", "replicate", "reflect")
_FPAD = {"zeros": "constant", "replicate": "replicate", "reflect": "reflect"}


class Conv3XC(nn.Module):
    """Tích chập 3×3 có tham số hóa lại.

    Lúc huấn luyện: 1×1 -> 3×3 -> 1×1 (rộng gấp ``gain`` lần) cộng một nhánh tắt
    1×1. Lúc suy luận: gộp thành một tích chập 3×3 (``eval_conv``).
    ``deploy=True`` chỉ tạo ``eval_conv`` (dùng cho trọng số đã gộp).
    """

    def __init__(self, c_in: int, c_out: int, gain: int = 2, bias: bool = True,
                 padding_mode: str = "zeros", deploy: bool = False):
        super().__init__()
        if padding_mode not in PADDING_MODES:
            raise ValueError(f"padding_mode phải thuộc {PADDING_MODES}")
        self.padding_mode = padding_mode
        self.deploy = deploy
        if not deploy:
            self.sk = nn.Conv2d(c_in, c_out, 1, bias=bias)
            self.conv = nn.Sequential(
                nn.Conv2d(c_in, c_in * gain, 1, bias=bias),
                nn.Conv2d(c_in * gain, c_out * gain, 3, padding=0, bias=bias),
                nn.Conv2d(c_out * gain, c_out, 1, bias=bias),
            )
        self.eval_conv = nn.Conv2d(c_in, c_out, 3, padding=1, bias=bias, padding_mode=padding_mode)
        self.eval_conv.weight.requires_grad = False
        if bias:
            self.eval_conv.bias.requires_grad = False
        if not deploy:
            self.update_params()

    @torch.no_grad()
    def update_params(self) -> None:
        """Gộp ba tích chập và nhánh tắt vào ``eval_conv``."""
        if self.deploy:
            return
        w1, b1 = self.conv[0].weight, self.conv[0].bias
        w2, b2 = self.conv[1].weight, self.conv[1].bias
        w3, b3 = self.conv[2].weight, self.conv[2].bias
        w = F.conv2d(w1.flip(2, 3).permute(1, 0, 2, 3), w2, padding=2).flip(2, 3).permute(1, 0, 2, 3)
        b = (w2 * b1.reshape(1, -1, 1, 1)).sum((1, 2, 3)) + b2
        wc = F.conv2d(w.flip(2, 3).permute(1, 0, 2, 3), w3, padding=0).flip(2, 3).permute(1, 0, 2, 3)
        bc = (w3 * b.reshape(1, -1, 1, 1)).sum((1, 2, 3)) + b3
        wc = wc + F.pad(self.sk.weight, [1, 1, 1, 1])
        bc = bc + self.sk.bias
        self.eval_conv.weight.copy_(wc)
        self.eval_conv.bias.copy_(bc)

    def train(self, mode: bool = True):
        super().train(mode)
        if not mode:
            self.update_params()  # vào chế độ eval thì gộp lại theo trọng số mới nhất
        return self

    def switch_to_deploy(self) -> None:
        if self.deploy:
            return
        self.update_params()
        del self.sk
        del self.conv
        self.deploy = True

    def unfreeze_deploy(self) -> None:
        """Cho phép tinh chỉnh một mạng ở dạng đã gộp (trọng số công bố chỉ có
        ``eval_conv``): bật gradient cho ``eval_conv``. Chỉ có nghĩa khi deploy=True."""
        if not self.deploy:
            raise RuntimeError("unfreeze_deploy chỉ dùng cho dạng đã gộp")
        for q in self.eval_conv.parameters():
            q.requires_grad_(True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and not self.deploy:
            xp = F.pad(x, (1, 1, 1, 1), mode=_FPAD[self.padding_mode])
            return self.conv(xp) + self.sk(x)
        return self.eval_conv(x)


class SPAB(nn.Module):
    def __init__(self, ch: int, padding_mode: str = "zeros", deploy: bool = False):
        super().__init__()
        self.c1_r = Conv3XC(ch, ch, padding_mode=padding_mode, deploy=deploy)
        self.c2_r = Conv3XC(ch, ch, padding_mode=padding_mode, deploy=deploy)
        self.c3_r = Conv3XC(ch, ch, padding_mode=padding_mode, deploy=deploy)
        self.act1 = nn.SiLU(inplace=False)

    def forward(self, x: torch.Tensor):
        # Bản gốc dùng SiLU(inplace=True) lên chính out1, nên giá trị thứ hai trả
        # về là out1 SAU kích hoạt. Ở đây viết tường minh để không phụ thuộc inplace.
        a1 = self.act1(self.c1_r(x))
        a2 = self.act1(self.c2_r(a1))
        out3 = self.c3_r(a2)
        sim_att = torch.sigmoid(out3) - 0.5
        out = (out3 + x) * sim_att  # cộng phần dư trước, nhân attention sau
        return out, a1


class SPAN(nn.Module):
    """SPAN với ``feature_channels`` kênh và ``n_blocks`` khối (bản gốc: 48 và 6).

    Vào và ra: tensor RGB trong [0, 1].
    """

    def __init__(self, num_in_ch: int = 3, num_out_ch: int = 3, feature_channels: int = 48,
                 upscale: int = 4, n_blocks: int = 6, padding_mode: str = "zeros",
                 img_range: float = 1.0, rgb_mean=(0.4488, 0.4371, 0.4040), deploy: bool = False):
        super().__init__()
        if n_blocks < 2:
            raise ValueError("n_blocks phải từ 2 trở lên")
        self.n_blocks = n_blocks
        self.upscale = upscale
        self.padding_mode = padding_mode
        self.feature_channels = feature_channels
        # Dải giá trị là một phần của trọng số (mạng học với ảnh vào nhân img_range), nên nó nằm trong
        # state_dict: checkpoint nào cũng mang theo, và nạp lại thì giá trị trong checkpoint thắng giá trị lúc dựng.
        self.register_buffer("img_range", torch.tensor(float(img_range)))
        self.register_buffer("mean", torch.tensor(rgb_mean).view(1, 3, 1, 1), persistent=False)
        fc = feature_channels
        self.conv_1 = Conv3XC(num_in_ch, fc, padding_mode=padding_mode, deploy=deploy)
        for i in range(1, n_blocks + 1):
            setattr(self, f"block_{i}", SPAB(fc, padding_mode=padding_mode, deploy=deploy))
        self.conv_cat = nn.Conv2d(fc * 4, fc, 1, bias=True)
        self.conv_2 = Conv3XC(fc, fc, padding_mode=padding_mode, deploy=deploy)
        self.upsampler = nn.Sequential(
            nn.Conv2d(fc, num_out_ch * upscale * upscale, 3, padding=1, padding_mode=padding_mode),
            nn.PixelShuffle(upscale),
        )

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys,
                              error_msgs):
        # Trọng số công bố không có khóa img_range: giữ giá trị lúc dựng, và vẫn nạp chặt được
        # (cùng cách PyTorch xử lý num_batches_tracked của BatchNorm).
        key = prefix + "img_range"
        if key not in state_dict:
            state_dict[key] = self.img_range.detach().clone()
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys,
                                      error_msgs)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Đặc trưng ngay trước tầng phóng (dùng cho các đầu tùy chọn)."""
        x = (x - self.mean) * self.img_range
        feat = self.conv_1(x)
        out = feat
        out_b1 = None
        last_inner = None
        for i in range(1, self.n_blocks + 1):
            out, inner = getattr(self, f"block_{i}")(out)
            if i == 1:
                out_b1 = out
            last_inner = inner
        out_final = self.conv_2(out)
        return self.conv_cat(torch.cat([feat, out_final, out_b1, last_inner], 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.upsampler(self.forward_features(x))

    def switch_to_deploy(self) -> "SPAN":
        for m in self.modules():
            if isinstance(m, Conv3XC):
                m.switch_to_deploy()
        return self


def is_deploy_state_dict(sd: dict) -> bool:
    """True nếu trọng số chỉ có ``eval_conv`` (đã gộp nhánh)."""
    return not any(".sk." in k or ".conv.0." in k for k in sd)


def switch_all_to_deploy(model: nn.Module) -> nn.Module:
    """Gộp mọi ``Conv3XC`` trong ``model`` (kể cả khi SPAN nằm trong một lớp bọc)."""
    for m in model.modules():
        if isinstance(m, Conv3XC):
            m.switch_to_deploy()
    return model
