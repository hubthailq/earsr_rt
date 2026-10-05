"""Phép thử ngữ cảnh (T6 i, v): thiếu ngữ cảnh quanh ảnh vào làm sai số tăng bao nhiêu.

Ý tưởng. Lấy một khung ảnh LR đủ rộng và một cửa sổ ở giữa. Cho mạng chạy trên
cửa sổ kèm một dải ngữ cảnh thật rộng m điểm ảnh ở mỗi cạnh, rồi chỉ giữ phần
đầu ra ứng với cửa sổ. Với m = 0 mạng chỉ thấy cửa sổ và phải đệm ở viền, đúng
như tình huống của benchmark. Vì cùng một cửa sổ được chấm ở mọi m, chênh lệch
sai số tại cùng điểm ảnh là do thiếu ngữ cảnh, không do nội dung ở viền khác
nội dung ở giữa.

Thêm hai điều kiện không cần huấn luyện để tách hai nguyên nhân:
- ``rep16`` / ``ref16``: m = 0, nhưng trước khi vào mạng, cửa sổ được đệm 16 px
  bằng lặp viền hoặc phản chiếu, rồi cắt bỏ ở đầu ra. Cách này đẩy lớp đệm số 0
  của mạng ra xa cửa sổ mà không thêm thông tin thật.
  Nếu sai số về gần mức có ngữ cảnh thật: phần mất là do đệm số 0.
  Nếu sai số giữ nguyên như m = 0: phần mất là do thiếu thông tin, không kiểu
  đệm nào bù được.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from ..data.resize import imresize
from ..io import to_tensor, to_uint8
from .metrics import rgb_to_y

RING_EDGES = (0, 1, 2, 3, 4, 8)  # vành theo khoảng cách tới mép cửa sổ (px LR): 0,1,2,3,[4..7],[8..]


@dataclass(frozen=True)
class ContextConfig:
    win_hw: tuple[int, int] = (51, 36)       # cửa sổ LR (H, W): bằng ảnh vào của ô chính
    margins: tuple[int, ...] = (0, 4, 8, 16, 24)
    prepad: tuple[tuple[str, int], ...] = (("replicate", 16), ("reflect", 16))
    scale: int = 4

    @property
    def max_margin(self) -> int:
        return max(self.margins)

    @property
    def frame_lr_hw(self) -> tuple[int, int]:
        m = self.max_margin
        return self.win_hw[0] + 2 * m, self.win_hw[1] + 2 * m


def make_context_frame(img: np.ndarray, cfg: ContextConfig) -> tuple[np.ndarray, ContextConfig]:
    """Ảnh gốc uint8 -> khung HR đúng bằng ``scale × frame_lr_hw``.

    Ảnh nằm ngang thì cửa sổ được xoay ngang theo. Ảnh được thu nhỏ vừa đủ để
    khung nằm trọn trong ảnh, rồi cắt giữa. Trả về khung và cấu hình đã xoay.
    """
    h, w = img.shape[:2]
    if w > h:
        cfg = ContextConfig((cfg.win_hw[1], cfg.win_hw[0]), cfg.margins, cfg.prepad, cfg.scale)
    fh, fw = cfg.frame_lr_hw[0] * cfg.scale, cfg.frame_lr_hw[1] * cfg.scale
    r = max(fh / h, fw / w)
    if r > 1.0:
        raise ValueError(f"ảnh {h}×{w} nhỏ hơn khung cần {fh}×{fw}")
    nh, nw = max(fh, int(round(h * r))), max(fw, int(round(w * r)))
    small = imresize(img, out_size=(nh, nw))
    t, l = (nh - fh) // 2, (nw - fw) // 2
    return small[t:t + fh, l:l + fw], cfg


def ring_index(win_hw: tuple[int, int], scale: int) -> np.ndarray:
    """Chỉ số vành của từng điểm ảnh HR trong cửa sổ, theo RING_EDGES."""
    h, w = win_hw
    yy, xx = np.mgrid[0:h, 0:w]
    d = np.minimum(np.minimum(yy, h - 1 - yy), np.minimum(xx, w - 1 - xx))
    idx = np.searchsorted(np.array(RING_EDGES), d, side="right") - 1
    return np.kron(idx, np.ones((scale, scale), dtype=np.int64))


@torch.no_grad()
def run_context_test(predict, frame_hr: np.ndarray, cfg: ContextConfig) -> list[dict]:
    """Chạy mọi điều kiện cho một khung. ``predict``: hàm lr uint8 -> sr uint8.

    Trả về danh sách dòng: cond, psnr_y, mse_y, mse_ring0..5.
    """
    s = cfg.scale
    M = cfg.max_margin
    wh, ww = cfg.win_hw
    lr_full = imresize(frame_hr, out_size=(frame_hr.shape[0] // s, frame_hr.shape[1] // s))
    hr_win = frame_hr[M * s:(M + wh) * s, M * s:(M + ww) * s]
    hy = rgb_to_y(hr_win)
    rings = ring_index(cfg.win_hw, s)
    rows = []

    def score(sr_win: np.ndarray, cond: str) -> None:
        e2 = (rgb_to_y(sr_win) - hy) ** 2
        row = {"cond": cond, "mse_y": float(e2.mean()),
               "psnr_y": float(10 * np.log10(255.0 ** 2 / max(e2.mean(), 1e-12)))}
        for k in range(len(RING_EDGES)):
            sel = rings == k
            row[f"mse_ring{k}"] = float(e2[sel].mean()) if sel.any() else float("nan")
        rows.append(row)

    for m in cfg.margins:
        x = lr_full[M - m:M + wh + m, M - m:M + ww + m]
        sr = predict(np.ascontiguousarray(x))
        score(sr[m * s:(m + wh) * s, m * s:(m + ww) * s], f"m{m}")
    win = np.ascontiguousarray(lr_full[M:M + wh, M:M + ww])
    for mode, p in cfg.prepad:
        xt = F.pad(to_tensor(win), (p, p, p, p), mode=mode)
        sr = predict(to_uint8(xt))
        score(sr[p * s:(p + wh) * s, p * s:(p + ww) * s], f"{mode[:3]}{p}")
    return rows
