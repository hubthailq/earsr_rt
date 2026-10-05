#!/usr/bin/env python3
"""Huấn luyện bộ dò điểm mốc tai trên một bộ ảnh có file ``.pts`` (ví dụ bộ 55
điểm mốc của Imperial College; kiểm tra điều khoản sử dụng trước, phép thử T1).

Hai bộ dò khác kiến trúc và khác phần dữ liệu (bộ A sinh nhãn cho N1, bộ B để đo):
  python scripts/train_landmarks.py --data /path/landmarks --arch heatmap --part a --out weights/lm_A_heatmap.pt
  python scripts/train_landmarks.py --data /path/landmarks --arch regress --part b --out weights/lm_B_regress.pt
Nếu không làm N1, một bộ dò là đủ:  --part all

``--preview N`` vẽ điểm mốc lên N ảnh đầu để kiểm quy ước toạ độ của file .pts.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import torch  # noqa: E402

from earsr.io import imread_rgb  # noqa: E402
from earsr.landmarks.data import split_items  # noqa: E402
from earsr.landmarks.pts import find_pairs, read_pts  # noqa: E402
from earsr.landmarks.train import train_detector  # noqa: E402


def main(argv=None) -> dict | None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True, help="thư mục chứa ảnh và file .pts cùng tên")
    ap.add_argument("--arch", default="heatmap", choices=["heatmap", "regress"])
    ap.add_argument("--part", default="all", choices=["a", "b", "all"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--zero-based", action="store_true", help="file .pts đếm toạ độ từ 0 (mặc định: từ 1)")
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--iters", type=int, default=20000)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-every", type=int, default=500)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--preview", type=int, default=0)
    a = ap.parse_args(argv)
    pairs = [pq for d in a.data for pq in find_pairs(d)]
    if not pairs:
        raise SystemExit("không tìm thấy cặp ảnh và .pts nào")
    items = [(img, read_pts(pts, one_based=not a.zero_based)) for img, pts in pairs]
    n_pts = {len(q) for _, q in items}
    if len(n_pts) != 1:
        raise SystemExit(f"số điểm mốc không đồng nhất: {sorted(n_pts)}")
    print(f"{len(items)} ảnh, {n_pts.pop()} điểm mốc mỗi ảnh")
    if a.preview:
        out = Path(a.out).with_suffix("") .parent / "lm_preview"
        out.mkdir(parents=True, exist_ok=True)
        for img, pts in items[: a.preview]:
            im = imread_rgb(img)[:, :, ::-1].copy()
            for i, (x, y) in enumerate(pts):
                cv2.circle(im, (int(round(x)), int(round(y))), 2, (0, 255, 0), -1)
            cv2.imwrite(str(out / (Path(img).stem + ".jpg")), im)
        print(f"đã vẽ {a.preview} ảnh vào {out}; kiểm bằng mắt rồi chạy lại không có --preview")
        return None
    tr, va = split_items(items, a.part)
    print(f"phần '{a.part}': {len(tr)} ảnh train, {len(va)} ảnh validation")
    res = train_detector(tr, va, a.out, a.arch, a.width, iters=a.iters, batch_size=a.batch_size, lr=a.lr,
                         val_every=a.val_every, workers=a.workers, device=a.device, seed=a.seed,
                         extra={"part": a.part, "data": a.data})
    print({k: v for k, v in res.items() if k != "history"})
    return res


if __name__ == "__main__":
    main()
