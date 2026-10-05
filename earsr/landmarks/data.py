"""Dữ liệu huấn luyện bộ dò điểm mốc.

Mỗi mẫu: cắt một khung quanh tai từ ảnh gốc (hộp bao điểm mốc, nới ngẫu nhiên,
ép về tỉ lệ khung của ảnh vào), xoay nhẹ, lật ngang, đổi sáng, làm mờ ngẫu
nhiên; rồi đưa về cỡ ``input_hw``. Khung nới ngẫu nhiên để bộ dò chịu được các
kiểu khung khác nhau (AMI: tai chiếm gần hết ảnh; ảnh ngoài thực tế: lỏng hơn).

Phép cắt, xoay, co dùng một ``cv2.warpAffine`` (nội suy song tuyến tính). Đây là
ngoại lệ có chủ ý của quy tắc "một hàm thu phóng": nó chỉ dùng cho ảnh huấn
luyện bộ dò, không bao giờ cho ảnh đáp án hay ảnh LR của benchmark.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from ..io import imread_rgb


def crop_matrix(pts: np.ndarray, input_hw: tuple[int, int], margin: tuple[float, float, float, float],
                angle_deg: float = 0.0, flip: bool = False) -> np.ndarray:
    """Ma trận affine 2×3 đưa ảnh gốc về ảnh vào ``input_hw``.

    ``margin`` = (trái, trên, phải, dưới): phần nới thêm quanh hộp bao điểm mốc,
    tính theo tỉ lệ của bề rộng và bề cao hộp. Hộp sau khi nới được mở tiếp cho
    đúng tỉ lệ khung của ảnh vào (không làm méo tai).
    """
    h_in, w_in = input_hw
    x0, y0 = pts.min(0)
    x1, y1 = pts.max(0)
    bw, bh = max(x1 - x0, 1.0), max(y1 - y0, 1.0)
    x0, x1 = x0 - margin[0] * bw, x1 + margin[2] * bw
    y0, y1 = y0 - margin[1] * bh, y1 + margin[3] * bh
    cx, cy, bw, bh = (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0
    if bw / bh < w_in / h_in:
        bw = bh * w_in / h_in
    else:
        bh = bw * h_in / w_in
    s = w_in / bw
    a = np.deg2rad(angle_deg)
    c, sn = np.cos(a) * s, np.sin(a) * s
    m = np.array([[c, -sn, 0.0], [sn, c, 0.0]], dtype=np.float64)
    # tâm hộp về tâm ảnh vào (quy ước tâm điểm ảnh)
    m[:, 2] = np.array([(w_in - 1) / 2, (h_in - 1) / 2]) - m[:, :2] @ np.array([cx, cy])
    if flip:
        m = np.array([[-1.0, 0.0, w_in - 1.0], [0.0, 1.0, 0.0]]) @ np.vstack([m, [0, 0, 1]])
    return m


def apply_matrix(pts: np.ndarray, m: np.ndarray) -> np.ndarray:
    return (pts.astype(np.float64) @ m[:, :2].T + m[:, 2]).astype(np.float32)


class LandmarkDataset(Dataset):
    """items: danh sách (đường dẫn ảnh, mảng điểm mốc (K, 2) theo toạ độ ảnh gốc)."""

    def __init__(self, items: list, input_hw: tuple[int, int] = (136, 96), train: bool = True, seed: int = 0,
                 margin_range: tuple[float, float] = (0.02, 0.35), max_angle: float = 20.0):
        if not items:
            raise ValueError("không có ảnh nào")
        self.items = [(Path(p), np.asarray(q, np.float32)) for p, q in items]
        self.input_hw, self.train, self.seed = tuple(input_hw), train, seed
        self.margin_range, self.max_angle = margin_range, max_angle
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        path, pts = self.items[i]
        img = imread_rgb(path)
        if self.train:
            rng = np.random.default_rng([self.seed, self.epoch, i])
            margin = tuple(rng.uniform(*self.margin_range, 4))
            angle = float(rng.uniform(-self.max_angle, self.max_angle))
            flip = bool(rng.random() < 0.5)
        else:
            mid = float(np.mean(self.margin_range))
            margin, angle, flip, rng = (mid,) * 4, 0.0, False, None
        m = crop_matrix(pts, self.input_hw, margin, angle, flip)
        out = cv2.warpAffine(img, m, (self.input_hw[1], self.input_hw[0]), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REPLICATE)
        if self.train:
            x = out.astype(np.float32)
            x = (x - 127.5) * rng.uniform(0.7, 1.3) + 127.5 + rng.uniform(-25, 25)
            if rng.random() < 0.5:
                x = cv2.GaussianBlur(x, (0, 0), float(rng.uniform(0.3, 1.8)))
            if rng.random() < 0.3:
                x = x + rng.normal(0, rng.uniform(1, 6), x.shape)
            out = np.clip(x, 0, 255).astype(np.uint8)
        t = torch.from_numpy(np.ascontiguousarray(out)).permute(2, 0, 1).float().div_(255.0)
        return t, torch.from_numpy(apply_matrix(pts, m))


def split_items(items: list, part: str = "all", val_frac: float = 0.1) -> tuple[list, list]:
    """Chia theo tên file, tất định. ``part``: 'a' và 'b' là hai nửa không giao
    nhau (cho hai bộ dò khác cách chia dữ liệu); 'all' dùng cả bộ. Trong phần
    được chọn, ``val_frac`` cuối theo thứ tự băm làm validation."""
    import hashlib

    key = lambda it: hashlib.sha1(Path(it[0]).name.encode()).hexdigest()
    items = sorted(items, key=key)
    if part == "a":
        items = items[0::2]
    elif part == "b":
        items = items[1::2]
    elif part != "all":
        raise ValueError("part phải là a, b hoặc all")
    n_val = max(1, int(round(len(items) * val_frac)))
    return items[:-n_val], items[-n_val:]
