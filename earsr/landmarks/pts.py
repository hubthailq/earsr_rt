"""Đọc và ghi file điểm mốc dạng ``.pts`` (định dạng của ibug):

    version: 1
    n_points: 55
    {
    x1 y1
    ...
    }

Toạ độ trong file ``.pts`` của ibug đếm từ 1 (kiểu MATLAB). ``read_pts`` mặc
định trừ 1 để về toạ độ đếm từ 0 theo tâm điểm ảnh. KIỂM TRA LẠI quy ước này
trên bộ dữ liệu thật trước khi dùng (vẽ thử vài ảnh bằng scripts/train_landmarks.py --preview).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp")


def read_pts(path: str | Path, one_based: bool = True) -> np.ndarray:
    lines = [l.strip() for l in Path(path).read_text().splitlines() if l.strip()]
    n = None
    for l in lines:
        if l.lower().startswith("n_points"):
            n = int(l.split(":")[1])
    try:
        a, b = lines.index("{"), lines.index("}")
    except ValueError as e:
        raise ValueError(f"{path}: thiếu dấu ngoặc nhọn") from e
    pts = np.array([[float(v) for v in l.split()[:2]] for l in lines[a + 1:b]], dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 2 or (n is not None and len(pts) != n):
        raise ValueError(f"{path}: đọc được {pts.shape}, n_points = {n}")
    return pts - 1.0 if one_based else pts


def write_pts(path: str | Path, pts: np.ndarray, one_based: bool = True) -> None:
    pts = np.asarray(pts, np.float64) + (1.0 if one_based else 0.0)
    body = "\n".join(f"{x:.3f} {y:.3f}" for x, y in pts)
    Path(path).write_text(f"version: 1\nn_points: {len(pts)}\n{{\n{body}\n}}\n")


def find_pairs(root: str | Path) -> list[tuple[Path, Path]]:
    """Các cặp (ảnh, file .pts cùng tên) dưới ``root``, theo thứ tự tên."""
    out = []
    for p in sorted(Path(root).rglob("*.pts")):
        for ext in IMG_EXTS:
            q = p.with_suffix(ext)
            if q.exists():
                out.append((q, p))
                break
    return out
