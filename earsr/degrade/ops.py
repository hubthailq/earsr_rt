"""Các phép suy giảm cơ bản trên ảnh uint8 RGB. Mọi phép ngẫu nhiên nhận một
``numpy.random.Generator`` để tất định theo hạt giống."""
from __future__ import annotations

import cv2
import numpy as np

from ..data.resize import imresize


def bicubic_down(hr: np.ndarray, scale: int) -> np.ndarray:
    """Thu nhỏ bicubic kiểu MATLAB, như NTIRE. ``hr`` phải chia hết cho ``scale``."""
    h, w = hr.shape[:2]
    if h % scale or w % scale:
        raise ValueError(f"ảnh {h}×{w} không chia hết cho {scale}")
    return imresize(hr, out_size=(h // scale, w // scale))


def jpeg(img: np.ndarray, quality: int) -> np.ndarray:
    """Nén rồi giải nén JPEG ở mức ``quality`` (thang 1..100 của libjpeg)."""
    ok, buf = cv2.imencode(".jpg", img[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        raise RuntimeError("nén JPEG thất bại")
    return np.ascontiguousarray(cv2.imdecode(buf, cv2.IMREAD_COLOR)[:, :, ::-1])


def gaussian_blur(img: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return img
    k = int(2 * np.ceil(3 * sigma) + 1)
    return cv2.GaussianBlur(img, (k, k), sigmaX=float(sigma), sigmaY=float(sigma),
                            borderType=cv2.BORDER_REFLECT_101)


def gaussian_noise(img: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Nhiễu Gauss cộng, ``sigma`` theo thang 0..255."""
    if sigma <= 0:
        return img
    x = img.astype(np.float64) + rng.normal(0.0, sigma, img.shape)
    return np.clip(np.round(x), 0, 255).astype(np.uint8)
