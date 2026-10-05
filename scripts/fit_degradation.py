#!/usr/bin/env python3
"""N2: ước lượng tham số suy giảm từ ảnh tai thật, một lần, trước mọi lần huấn luyện.

  python scripts/fit_degradation.py --root /path/EarVN1.0_raw --roles splits/earvn_roles.json \
      --out configs/degrade_estimated.json

Chỉ dùng ảnh của nhóm người có vai 'fit'. Nếu chưa có file vai, script tạo file
đó (chia người thành train, test, fit, clf, viewer) và ghi ra để commit.
Script không cần torch; chạy được ở máy chỉ có numpy, OpenCV, Pillow.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402

from earsr.data import wild  # noqa: E402
from earsr.degrade.fit_estimated import fit, save_fit  # noqa: E402

ROLE_FRACTIONS = wild.ROLE_FRACTIONS


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="thư mục gốc, dạng root/<người>/<ảnh>")
    ap.add_argument("--roles", default="splits/earvn_roles.json")
    ap.add_argument("--out", default="configs/degrade_estimated.json")
    ap.add_argument("--small", type=int, nargs=2, default=[24, 48], help="dải cạnh ngắn của ảnh nhỏ thật")
    ap.add_argument("--scale", type=int, default=4)
    a = ap.parse_args()
    root = Path(a.root)
    files = sorted(p for p in root.rglob("*") if p.suffix.lower() in wild.EXTS)
    subjects = sorted({p.parent.name for p in files})
    roles, made = wild.load_or_make_roles(subjects, a.roles)
    if made:
        print(f"đã tạo {a.roles}: hãy commit file này")
    fit_files = [p for p in files if roles.get(p.parent.name) == "fit"]
    small, large = [], []
    for p in fit_files:
        img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if img is None:
            continue
        s = min(img.shape[:2])
        if a.small[0] <= s <= a.small[1]:
            small.append(p)
        elif s >= a.small[1] * a.scale:
            large.append(p)
    print(f"nhóm 'fit': {len(fit_files)} ảnh; ảnh nhỏ thật {len(small)}; ảnh lớn {len(large)}")
    res = fit(small, large, scale=a.scale, target_short=tuple(a.small))
    save_fit(res, a.out, Path(a.out).with_suffix(".diagnostics.json"))
    print(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
