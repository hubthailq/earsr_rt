"""T4: đường trộn và cổng oracle, cho quyết định giữ hay bỏ N3. Không huấn luyện.

Cho hai ảnh ra của cùng một ảnh vào: f_S (mô hình tối ưu PSNR) và f_T (mô hình
cảm nhận).
- Đường trộn: α·f_S + (1 − α)·f_T, cùng một α cho mọi điểm ảnh. Đây là mốc mà
  cổng điểm ảnh phải vượt.
- Cổng oracle: biết đáp án, chọn theo từng điểm ảnh. Lấy f_S ở nơi f_T sai hơn
  f_S quá τ, ngược lại lấy f_T. Hai định nghĩa sai số:
    'pixel' : sai số L1 tại đúng điểm ảnh đó (trần trên tuyệt đối);
    'window': sai số L1 trung bình trong cửa sổ k×k quanh điểm ảnh (gần với thứ
              một cổng học được có thể làm).
Nếu cả hai oracle không hơn đường trộn ít nhất 10% LPIPS ở cùng PSNR thì bỏ N3.
"""
from __future__ import annotations

import cv2
import numpy as np

ALPHAS = tuple(round(x, 2) for x in np.linspace(0, 1, 11))     # lưới ghi trước
TAUS = (0.0, 0.01, 0.02, 0.04, 0.08)                           # theo dải [0, 1]


def _f(a: np.ndarray) -> np.ndarray:
    return a.astype(np.float64) / 255.0


def _u8(a: np.ndarray) -> np.ndarray:
    return np.clip(np.round(a * 255.0), 0, 255).astype(np.uint8)


def blend(f_s: np.ndarray, f_t: np.ndarray, alpha: float) -> np.ndarray:
    return _u8(alpha * _f(f_s) + (1 - alpha) * _f(f_t))


def oracle_mask(f_s: np.ndarray, f_t: np.ndarray, hr: np.ndarray, tau: float, mode: str = "pixel",
                k: int = 7) -> np.ndarray:
    """Mặt nạ bool H×W: True = giữ f_S."""
    e_s = np.abs(_f(f_s) - _f(hr)).mean(2)
    e_t = np.abs(_f(f_t) - _f(hr)).mean(2)
    if mode == "window":
        e_s = cv2.blur(e_s, (k, k), borderType=cv2.BORDER_REFLECT)
        e_t = cv2.blur(e_t, (k, k), borderType=cv2.BORDER_REFLECT)
    elif mode != "pixel":
        raise ValueError("mode phải là 'pixel' hoặc 'window'")
    return e_t > e_s + tau


def oracle_gate(f_s: np.ndarray, f_t: np.ndarray, hr: np.ndarray, tau: float, mode: str = "pixel",
                k: int = 7) -> tuple[np.ndarray, float]:
    """Trả về (ảnh ra uint8, tỉ lệ điểm ảnh lấy từ f_S)."""
    m = oracle_mask(f_s, f_t, hr, tau, mode, k)
    return np.where(m[..., None], f_s, f_t), float(m.mean())


def candidates(f_s: np.ndarray, f_t: np.ndarray, hr: np.ndarray, alphas=ALPHAS, taus=TAUS, k: int = 7):
    """Sinh (method, param, ảnh ra, tỉ lệ f_S) cho mọi điểm của đường trộn và hai oracle."""
    for a in alphas:
        yield "blend", float(a), blend(f_s, f_t, a), float(a)
    for mode in ("pixel", "window"):
        for t in taus:
            out, frac = oracle_gate(f_s, f_t, hr, t, mode, k)
            yield f"oracle_{mode}", float(t), out, frac
