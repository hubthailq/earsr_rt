"""Hàm thu phóng duy nhất của project: bicubic có khử răng cưa, tương thích
MATLAB ``imresize`` (cách NTIRE tạo ảnh LR).

Mọi chỗ trong project cần thu phóng ảnh đều phải gọi ``imresize`` ở đây.
Các thư viện khác (PIL, OpenCV, torch) cho kết quả khác nhau ở mức 0,1 đến 1 dB.

Đã kiểm với ảnh LR của Set5 do MATLAB tạo (tests/test_resize.py).
"""
from __future__ import annotations

import numpy as np

__all__ = ["imresize", "modcrop", "center_crop_to_multiple"]


def _cubic(x: np.ndarray) -> np.ndarray:
    """Nhân bicubic của Keys với a = -0.5 (giống MATLAB)."""
    ax = np.abs(x)
    ax2 = ax * ax
    ax3 = ax2 * ax
    return ((1.5 * ax3 - 2.5 * ax2 + 1) * (ax <= 1)
            + (-0.5 * ax3 + 2.5 * ax2 - 4 * ax + 2) * ((ax > 1) & (ax <= 2)))


def _weights_indices(in_len: int, out_len: int, scale: float, antialias: bool):
    """Trọng số và chỉ số nội suy theo một chiều (thuật toán của MATLAB)."""
    kernel_width = 4.0
    if scale < 1 and antialias:
        kernel_width = kernel_width / scale
    x = np.arange(1, out_len + 1, dtype=np.float64)
    u = x / scale + 0.5 * (1 - 1 / scale)
    left = np.floor(u - kernel_width / 2)
    p = int(np.ceil(kernel_width)) + 2
    ind = left[:, None] + np.arange(p, dtype=np.float64)[None, :]
    dist = u[:, None] - ind
    if scale < 1 and antialias:
        w = scale * _cubic(dist * scale)
    else:
        w = _cubic(dist)
    w = w / w.sum(axis=1, keepdims=True)
    # phản chiếu ở biên, giống MATLAB
    aux = np.concatenate([np.arange(in_len), np.arange(in_len - 1, -1, -1)])
    ind = aux[np.mod(ind.astype(np.int64) - 1, aux.size)]
    keep = np.any(w != 0, axis=0)
    return w[:, keep], ind[:, keep]


def _resize_axis(img: np.ndarray, w: np.ndarray, ind: np.ndarray, axis: int) -> np.ndarray:
    img = np.moveaxis(img, axis, 0)
    out = np.einsum("op,op...->o...", w, img[ind])
    return np.moveaxis(out, 0, axis)


def imresize(img: np.ndarray, scale: float | None = None,
             out_size: tuple[int, int] | None = None, antialias: bool = True) -> np.ndarray:
    """Thu phóng ảnh bằng bicubic kiểu MATLAB.

    img: mảng H×W hoặc H×W×C; uint8 (0..255) hoặc float (0..1).
    scale: hệ số (ví dụ 0.25); hoặc out_size=(H, W).
    Ảnh uint8 trả về uint8 (làm tròn và cắt ngưỡng như MATLAB); ảnh float trả
    về float64 không cắt ngưỡng.
    """
    if (scale is None) == (out_size is None):
        raise ValueError("cần đúng một trong scale hoặc out_size")
    is_uint8 = img.dtype == np.uint8
    x = img.astype(np.float64)
    h, w = x.shape[:2]
    if out_size is None:
        oh, ow = int(np.ceil(h * scale)), int(np.ceil(w * scale))
        sh = sw = float(scale)
    else:
        oh, ow = int(out_size[0]), int(out_size[1])
        sh, sw = oh / h, ow / w
    wh, ih = _weights_indices(h, oh, sh, antialias)
    ww, iw = _weights_indices(w, ow, sw, antialias)
    # MATLAB xử lý chiều có hệ số nhỏ hơn trước
    if sh <= sw:
        x = _resize_axis(x, wh, ih, 0)
        x = _resize_axis(x, ww, iw, 1)
    else:
        x = _resize_axis(x, ww, iw, 1)
        x = _resize_axis(x, wh, ih, 0)
    if is_uint8:
        return np.clip(np.round(x), 0, 255).astype(np.uint8)
    return x


def modcrop(img: np.ndarray, m: int) -> np.ndarray:
    """Cắt góc trên trái để hai cạnh chia hết cho m (quy ước của NTIRE)."""
    h, w = img.shape[:2]
    return img[: h - h % m, : w - w % m]


def center_crop_to_multiple(img: np.ndarray, m: int) -> np.ndarray:
    """Cắt giữa để hai cạnh chia hết cho m."""
    h, w = img.shape[:2]
    nh, nw = h - h % m, w - w % m
    t, l = (h - nh) // 2, (w - nw) // 2
    return img[t: t + nh, l: l + nw]
