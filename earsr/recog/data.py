"""Liệt kê ảnh theo vai và theo cỡ; chia ảnh đăng ký (lớn) và ảnh dò (nhỏ thật)."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

EXTS = (".jpg", ".jpeg", ".png", ".bmp")
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
INPUT_HW = (160, 112)   # ảnh tai cao hơn rộng; mọi phương pháp đều qua cùng một phép thu phóng về cỡ này


def list_images(root: str | Path, roles_path: str | Path | None, roles: tuple[str, ...] | None,
                cache: str | Path | None = None) -> list[dict]:
    """Mỗi dòng: path (tương đối), subject, role, h, w, short. Chỉ đọc phần đầu file để lấy kích thước."""
    root = Path(root)
    role_of = json.loads(Path(roles_path).read_text()) if roles_path else {}
    rows = None
    if cache and Path(cache).exists():
        with open(cache, newline="") as f:
            rows = [dict(r, h=int(r["h"]), w=int(r["w"]), short=int(r["short"])) for r in csv.DictReader(f)]
    if rows is None:
        rows = []
        for p in sorted(root.rglob("*")):
            if p.suffix.lower() not in EXTS:
                continue
            try:
                with Image.open(p) as im:
                    w, h = im.size
            except Exception:
                continue
            rows.append({"path": str(p.relative_to(root)), "subject": p.parent.name, "h": h, "w": w, "short": min(h, w)})
        if cache:
            Path(cache).parent.mkdir(parents=True, exist_ok=True)
            with open(cache, "w", newline="") as f:
                wr = csv.DictWriter(f, fieldnames=["path", "subject", "h", "w", "short"])
                wr.writeheader()
                wr.writerows(rows)
    for r in rows:
        r["role"] = role_of.get(r["subject"], "")
    if roles is not None:
        rows = [r for r in rows if r["role"] in roles]
    return rows


def split_gallery_probe(rows: list[dict], probe_short: tuple[int, int] = (24, 48), gallery_min_short: int = 96,
                        min_gallery: int = 4, seed: int = 0) -> dict:
    """Chia ảnh của nhóm chấm.

    gallery: nửa số ảnh lớn của mỗi người (ảnh đăng ký); ref: nửa còn lại (ảnh dò lớn, làm mức tham chiếu của
    mạng nhận dạng); probe: ảnh nhỏ thật. Người có ít hơn ``min_gallery`` ảnh lớn thì bị bỏ. Người có đủ ảnh lớn
    nhưng không có ảnh nhỏ vẫn được đăng ký: họ là danh tính gây nhiễu, làm phép dò khó hơn và sát thực tế hơn.
    (Trong EarVN1.0 ảnh của một người thường cùng một cỡ, nên chỉ một phần số người có cả hai loại ảnh.)
    """
    by = {}
    for r in rows:
        by.setdefault(r["subject"], []).append(r)
    rng = np.random.default_rng(seed)
    out = {"gallery": [], "ref": [], "probe": [], "dropped": [], "gallery_only": []}
    for s in sorted(by):
        large = sorted((r for r in by[s] if r["short"] >= gallery_min_short), key=lambda r: r["path"])
        small = sorted((r for r in by[s] if probe_short[0] <= r["short"] <= probe_short[1]), key=lambda r: r["path"])
        if len(large) < min_gallery:
            out["dropped"].append(s)
            continue
        order = rng.permutation(len(large))
        half = len(large) // 2
        out["gallery"] += [large[i] for i in sorted(order[:half])]
        out["ref"] += [large[i] for i in sorted(order[half:])]
        out["probe"] += small
        if not small:
            out["gallery_only"].append(s)
    return out


def to_input(img: np.ndarray, hw: tuple[int, int] = INPUT_HW) -> torch.Tensor:
    """Ảnh uint8 RGB cỡ bất kỳ -> tensor 3×H×W đã chuẩn hóa. Một phép thu phóng duy nhất (bicubic có chống răng cưa)
    cho mọi phương pháp, cả khi phóng to lẫn khi thu nhỏ."""
    x = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float().div(255.0)[None]
    x = F.interpolate(x, size=hw, mode="bicubic", antialias=True, align_corners=False).clamp(0, 1)[0]
    return (x - torch.tensor(MEAN)[:, None, None]) / torch.tensor(STD)[:, None, None]
