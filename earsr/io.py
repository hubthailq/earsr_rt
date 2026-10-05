"""Đọc và ghi ảnh; chuyển giữa uint8 H×W×3 (RGB) và tensor 1×3×H×W trong [0, 1]."""
from __future__ import annotations

import hashlib
from pathlib import Path

import cv2
import numpy as np
import torch


def imread_rgb(path: str | Path) -> np.ndarray:
    """Đọc ảnh thành uint8 H×W×3, thứ tự RGB. Ảnh xám được lặp thành 3 kênh."""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"không đọc được ảnh: {path}")
    if img.dtype != np.uint8:
        raise ValueError(f"chỉ hỗ trợ ảnh 8 bit: {path} ({img.dtype})")
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    elif img.shape[2] == 4:
        img = img[:, :, :3][:, :, ::-1]
    else:
        img = img[:, :, ::-1]
    return np.ascontiguousarray(img)


def imwrite_rgb(path: str | Path, img: np.ndarray) -> None:
    """Ghi ảnh uint8 RGB thành PNG (không mất mát)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if img.dtype != np.uint8:
        raise ValueError("imwrite_rgb chỉ nhận uint8")
    ok = cv2.imwrite(str(path), img[:, :, ::-1], [cv2.IMWRITE_PNG_COMPRESSION, 3])
    if not ok:
        raise IOError(f"không ghi được ảnh: {path}")


def to_tensor(img: np.ndarray) -> torch.Tensor:
    """uint8 H×W×3 -> float32 1×3×H×W trong [0, 1]."""
    return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float().div_(255.0).unsqueeze(0)


def to_uint8(t: torch.Tensor) -> np.ndarray:
    """float 1×3×H×W (hoặc 3×H×W) trong [0, 1] -> uint8 H×W×3, làm tròn và cắt ngưỡng."""
    if t.dim() == 4:
        t = t[0]
    x = t.detach().float().clamp_(0, 1).mul(255.0).round().permute(1, 2, 0).cpu().numpy()
    return x.astype(np.uint8)


def sha1_file(path: str | Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha1_array(a: np.ndarray) -> str:
    return hashlib.sha1(np.ascontiguousarray(a).tobytes()).hexdigest()
