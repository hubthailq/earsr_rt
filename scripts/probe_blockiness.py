#!/usr/bin/env python3
"""Ảnh dò thật có mang lưới khối 8×8 của JPEG không? (chẩn đoán cho phép đo nhận dạng, mục 6.5 của bản thảo)

  python scripts/probe_blockiness.py            # ghi results/recog/blockiness.json

Với mỗi ảnh dò: tỉ số giữa chênh lệch điểm ảnh kề nhau NGANG QUA biên khối 8×8 và chênh lệch bên trong khối. Ảnh JPEG
giải nén có lưới khối thẳng hàng với biên ảnh nên tỉ số lớn hơn 1; ảnh không nén, hoặc đã bị thu phóng sau khi nén, cho
tỉ số quanh 1. Cần dữ liệu gốc (data/raw) và danh sách ảnh dò do recog_eval.py ghi (results/recog/*/probes.csv).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def blockiness(gray: np.ndarray) -> float:
    g = gray.astype(np.float64)
    dh, dv = np.abs(np.diff(g, axis=1)), np.abs(np.diff(g, axis=0))
    cb, rb = np.arange(7, dh.shape[1], 8), np.arange(7, dv.shape[0], 8)
    if len(cb) == 0 or len(rb) == 0:
        return float("nan")
    mh, mv = np.ones(dh.shape[1], bool), np.ones(dv.shape[0], bool)
    mh[cb], mv[rb] = False, False
    on = np.concatenate([dh[:, cb].ravel(), dv[rb, :].ravel()]).mean()
    off = np.concatenate([dh[:, mh].ravel(), dv[mv, :].ravel()]).mean()
    return float(on / max(off, 1e-6))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recog", default="results/recog")
    ap.add_argument("--out", default="results/recog/blockiness.json")
    a = ap.parse_args(argv)
    out = {}
    for name, tag, root in (("earvn", "resnet18", "data/raw/EarVN1.0"), ("awex", "awex_resnet18", "data/raw/awex")):
        pr = pd.read_csv(Path(a.recog) / tag / "probes.csv")
        b = []
        for p in pr.path:
            g = cv2.imread(str(Path(root) / p), cv2.IMREAD_GRAYSCALE)
            b.append(blockiness(g) if g is not None else float("nan"))
        pr["b"] = b
        gal = pd.read_csv(Path(a.recog) / tag / "gallery.csv")
        groups = {"all": pr}
        if pr.jpeg_q.notna().any():
            groups.update(low=pr[pr.jpeg_q.between(70, 80)], high=pr[pr.jpeg_q >= 90])
        out[name] = {"gallery_per_subject_median": float(gal.groupby("subject").size().median()),
                     "probes_per_subject_median": float(pr.groupby("subject").size().median()),
                     "groups": {k: {"n": int(len(v)), "median": float(v.b.median()), "q25": float(v.b.quantile(0.25)),
                                    "q75": float(v.b.quantile(0.75))} for k, v in groups.items()}}
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
