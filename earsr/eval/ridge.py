"""Bảng kiểm tra gờ giả và gờ mất: đối chứng cho câu hỏi "ảnh rõ hơn thật hay do
bịa cấu trúc", không dùng điểm mốc và không dùng mạng học sâu.

Cách tính. Lấy bản đồ cạnh của ảnh đáp án và ảnh SR bằng Canny trên kênh Y,
với hai ngưỡng đặt theo trung vị độ lớn gradient của CHÍNH ảnh đáp án (cùng
ngưỡng cho cả hai ảnh, để ảnh SR sắc hơn không được lợi). Với dung sai ``tol``
điểm ảnh:
    gờ giả  (false ridge)  = tỉ lệ điểm cạnh của SR không có điểm cạnh nào của đáp án trong vòng ``tol``
    gờ mất  (missed ridge) = tỉ lệ điểm cạnh của đáp án không có điểm cạnh nào của SR trong vòng ``tol``
Cả hai càng thấp càng tốt. ``ridge_f1`` gộp hai tỉ lệ.
"""
from __future__ import annotations

import cv2
import numpy as np

from .metrics import rgb_to_y


def _edges(y: np.ndarray, lo: float, hi: float) -> np.ndarray:
    g = np.clip(np.round(y), 0, 255).astype(np.uint8)
    g = cv2.GaussianBlur(g, (0, 0), 1.0)
    return cv2.Canny(g, lo, hi, L2gradient=True) > 0


def canny_thresholds(hr_y: np.ndarray) -> tuple[float, float]:
    g = cv2.GaussianBlur(np.clip(np.round(hr_y), 0, 255).astype(np.uint8), (0, 0), 1.0).astype(np.float64)
    gx = cv2.Sobel(g, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy)
    hi = float(max(20.0, np.quantile(mag, 0.85)))
    return 0.5 * hi, hi


def ridge_check(sr: np.ndarray, hr: np.ndarray, tol: int = 1, border: int = 4) -> dict:
    """``sr``, ``hr``: uint8 RGB cùng cỡ. ``border``: bỏ viền trước khi đếm."""
    if sr.shape != hr.shape:
        raise ValueError("sr và hr phải cùng kích thước")
    hy, sy = rgb_to_y(hr), rgb_to_y(sr)
    lo, hi = canny_thresholds(hy)
    eh, es = _edges(hy, lo, hi), _edges(sy, lo, hi)
    if border > 0:
        eh, es = eh[border:-border, border:-border], es[border:-border, border:-border]
    # khoảng cách từ mỗi điểm tới điểm cạnh gần nhất của ảnh kia
    dist_to_h = cv2.distanceTransform((~eh).astype(np.uint8), cv2.DIST_L2, 3)
    dist_to_s = cv2.distanceTransform((~es).astype(np.uint8), cv2.DIST_L2, 3)
    n_s, n_h = int(es.sum()), int(eh.sum())
    false_r = float((dist_to_h[es] > tol).mean()) if n_s else 0.0
    missed = float((dist_to_s[eh] > tol).mean()) if n_h else 0.0
    if n_h == 0 and n_s > 0:
        false_r = 1.0
    prec, rec = 1 - false_r, 1 - missed
    f1 = 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)
    return {"ridge_false": false_r, "ridge_missed": missed, "ridge_f1": float(f1),
            "ridge_n_hr": n_h, "ridge_n_sr": n_s}
