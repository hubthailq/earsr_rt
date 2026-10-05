"""Benchmark AMI thu nhỏ: ảnh đáp án (HR) ở nhiều cỡ, từ ảnh gốc 492×702.

Không cắt vùng tai, không căn chỉnh, không khử nhiễu hay làm nét. Bước duy
nhất làm đổi nội dung là thu nhỏ bằng ``earsr.data.resize.imresize`` rồi cắt
giữa cho hai cạnh chia hết cho 4.
"""
from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..io import imread_rgb, imwrite_rgb, sha1_array
from .resize import center_crop_to_multiple, imresize

# cạnh ngắn của ảnh đáp án -> tên tầng. 244 chỉ dùng cho bộ chỉ số cảm nhận của NTIRE.
TIERS = (96, 144, 192, 244)
MAIN_TIERS = (96, 144, 192)
VIEWS = ("back", "down", "front", "left", "right", "up", "zoom")
_NAME = re.compile(r"^(\d{3})_(back|down|front|left|right|up|zoom)_ear\.jpg$")
EXPECTED_ORIG_SIZE = (702, 492)  # (H, W)


@dataclass(frozen=True)
class AmiItem:
    subject: str
    view: str
    path: Path

    @property
    def key(self) -> str:
        return f"{self.subject}_{self.view}"


def scan_ami(root: str | Path) -> list[AmiItem]:
    """Liệt kê ảnh AMI. Chỉ nhận file đúng mẫu tên ``NNN_view_ear.jpg``."""
    root = Path(root)
    items = []
    for p in sorted(root.rglob("*.jpg")):
        m = _NAME.match(p.name)
        if m:
            items.append(AmiItem(m.group(1), m.group(2), p))
    return items


def check_ami(items: list[AmiItem]) -> dict:
    """Kiểm tra tính đầy đủ: số người, số ảnh mỗi người, không trùng khóa."""
    subj: dict[str, set] = {}
    for it in items:
        subj.setdefault(it.subject, set()).add(it.view)
    keys = [it.key for it in items]
    return {
        "n_images": len(items),
        "n_subjects": len(subj),
        "incomplete_subjects": sorted(s for s, v in subj.items() if v != set(VIEWS)),
        "duplicate_keys": len(keys) - len(set(keys)),
    }


def hr_size(tier: int, orig_hw: tuple[int, int] = EXPECTED_ORIG_SIZE) -> tuple[int, int]:
    """Cỡ ảnh đáp án (H, W) của một tầng, sau khi cắt cho chia hết cho 4."""
    h, w = orig_hw
    short, long_ = (w, h) if w <= h else (h, w)
    new_long = int(round(long_ * tier / short))
    nh, nw = (new_long, tier) if w <= h else (tier, new_long)
    return nh - nh % 4, nw - nw % 4


def make_hr(orig: np.ndarray, tier: int) -> np.ndarray:
    """Ảnh gốc uint8 -> ảnh đáp án uint8 của tầng ``tier`` (cạnh ngắn)."""
    h, w = orig.shape[:2]
    short, long_ = (w, h) if w <= h else (h, w)
    new_long = int(round(long_ * tier / short))
    out_hw = (new_long, tier) if w <= h else (tier, new_long)
    return center_crop_to_multiple(imresize(orig, out_size=out_hw), 4)


def build_ami_benchmark(raw_root: str | Path, out_root: str | Path,
                        tiers: tuple[int, ...] = TIERS, strict: bool = True) -> Path:
    """Tạo ``out_root/hr{tier}/{subject}_{view}.png`` và file ``manifest.csv``.

    ``strict``: dừng nếu bộ ảnh không đủ 100 người × 7 ảnh hoặc sai kích thước.
    Trả về đường dẫn manifest.
    """
    raw_root, out_root = Path(raw_root), Path(out_root)
    items = scan_ami(raw_root)
    rep = check_ami(items)
    if strict and (rep["n_images"] != 700 or rep["n_subjects"] != 100
                   or rep["incomplete_subjects"] or rep["duplicate_keys"]):
        raise RuntimeError(f"bộ AMI không đúng như mong đợi: {rep}")
    rows = []
    for it in items:
        orig = imread_rgb(it.path)
        if strict and orig.shape[:2] != EXPECTED_ORIG_SIZE:
            raise RuntimeError(f"{it.path}: kích thước {orig.shape[:2]}, mong đợi {EXPECTED_ORIG_SIZE}")
        for t in tiers:
            hr = make_hr(orig, t)
            rel = f"hr{t}/{it.key}.png"
            imwrite_rgb(out_root / rel, hr)
            rows.append({"dataset": "ami", "subject": it.subject, "view": it.view, "tier": t,
                         "file": rel, "h": hr.shape[0], "w": hr.shape[1], "sha1": sha1_array(hr)})
    man = out_root / "manifest.csv"
    with open(man, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    return man


def read_manifest(path: str | Path) -> list[dict]:
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in ("tier", "h", "w"):
            if k in r and r[k] != "":
                r[k] = int(r[k])
    return rows
