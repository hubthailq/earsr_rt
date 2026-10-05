#!/usr/bin/env python3
"""Ba việc dùng bộ dò điểm mốc đã huấn luyện.

1. Nhãn giả cho N1 (bộ dò A chạy trên ảnh AMI gốc; toạ độ theo ảnh gốc):
     python scripts/landmark_tools.py pseudo --ckpt weights/lm_A_heatmap.pt --ami-raw /path/AMI --out data/ami_lm.npz
   Có thể nối thêm nhãn thật của bộ điểm mốc:  --pts-dir /path/landmarks
2. Hộp bao vùng tai cho một benchmark (để báo số đo "trong vùng tai"):
     python scripts/landmark_tools.py boxes --ckpt weights/lm_B_regress.pt --bench data/bench/ami --tier 192
3. Sàn nhiễu của bộ dò ở từng cỡ ảnh (điều kiện dải động của N1):
     python scripts/landmark_tools.py floor --ckpt weights/lm_B_regress.pt --bench data/bench/ami --tiers 96 144 192
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from earsr.data.ami import read_manifest, scan_ami  # noqa: E402
from earsr.io import imread_rgb  # noqa: E402
from earsr.landmarks.metric import LandmarkScorer  # noqa: E402
from earsr.landmarks.pts import find_pairs, read_pts  # noqa: E402


def cmd_pseudo(a) -> None:
    sc = LandmarkScorer(a.ckpt, a.device)
    paths, pts = [], []
    for it in scan_ami(a.ami_raw):
        paths.append(str(it.path))
        pts.append(sc.points(imread_rgb(it.path)))
    n_pseudo = len(paths)
    for d in a.pts_dir or []:
        for img, p in find_pairs(d):
            q = read_pts(p, one_based=not a.zero_based)
            if len(q) != pts[0].shape[0]:
                raise SystemExit(f"{p}: {len(q)} điểm, bộ dò cho {pts[0].shape[0]} điểm")
            paths.append(str(img))
            pts.append(q)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(a.out, paths=np.array(paths), points=np.stack(pts).astype(np.float32),
             n_pseudo=n_pseudo, detector=str(a.ckpt))
    print(f"đã ghi {a.out}: {n_pseudo} ảnh nhãn giả, {len(paths) - n_pseudo} ảnh nhãn thật")


def cmd_boxes(a) -> None:
    sc = LandmarkScorer(a.ckpt, a.device)
    bench = Path(a.bench)
    rows = []
    for r in read_manifest(bench / "manifest.csv"):
        if r["tier"] != a.tier:
            continue
        t, l, b, rr = sc.box(imread_rgb(bench / r["file"]), a.margin)
        rows.append({"key": f"{r['subject']}_{r['view']}", "top": round(t, 5), "left": round(l, 5),
                     "bottom": round(b, 5), "right": round(rr, 5)})
    if not rows:
        raise SystemExit(f"benchmark không có tầng {a.tier}")
    out = Path(a.out or bench / "boxes.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    area = np.mean([(r["bottom"] - r["top"]) * (r["right"] - r["left"]) for r in rows])
    print(f"đã ghi {out}: {len(rows)} hộp (dò ở tầng {a.tier}); hộp chiếm trung bình {area:.0%} diện tích ảnh")


def cmd_floor(a) -> None:
    sc = LandmarkScorer(a.ckpt, a.device)
    bench = Path(a.bench)
    man = read_manifest(bench / "manifest.csv")
    for tier in a.tiers:
        v = [sc.noise_floor(imread_rgb(bench / r["file"]), seed=i) for i, r in enumerate(man) if r["tier"] == tier]
        if v:
            print(f"tầng {tier}: sàn nhiễu Gauss {np.mean([x['lm_floor_noise'] for x in v]):.5f}, "
                  f"JPEG 90 {np.mean([x['lm_floor_jpeg'] for x in v]):.5f}  ({len(v)} ảnh). "
                  f"Thước đo dùng được nếu độ lệch của thân tinh chỉnh ≥ 2 lần giá trị lớn hơn.")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    p = sub.add_parser("pseudo")
    p.add_argument("--ckpt", required=True); p.add_argument("--ami-raw", required=True)
    p.add_argument("--out", required=True); p.add_argument("--pts-dir", nargs="*")
    p.add_argument("--zero-based", action="store_true"); p.add_argument("--device", default=dev)
    p.set_defaults(fn=cmd_pseudo)
    p = sub.add_parser("boxes")
    p.add_argument("--ckpt", required=True); p.add_argument("--bench", required=True)
    p.add_argument("--tier", type=int, default=192); p.add_argument("--margin", type=float, default=0.10)
    p.add_argument("--out", default=None); p.add_argument("--device", default=dev)
    p.set_defaults(fn=cmd_boxes)
    p = sub.add_parser("floor")
    p.add_argument("--ckpt", required=True); p.add_argument("--bench", required=True)
    p.add_argument("--tiers", type=int, nargs="+", default=[96, 144, 192]); p.add_argument("--device", default=dev)
    p.set_defaults(fn=cmd_floor)
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
