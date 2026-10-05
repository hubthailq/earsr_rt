#!/usr/bin/env python3
"""Phép thử tương tác (N5b c): phần hơn của biến thể ở ảnh vào nhỏ trừ phần hơn ở ảnh vào lớn.

  python scripts/interaction.py --small variant_small.csv ref_small.csv --large variant_large.csv ref_large.csv

Mỗi file là số đo theo từng ảnh (cột key, subject, và cột số đo). Trong mỗi
cặp, hai file phải cùng tập ảnh.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from earsr.stats.bootstrap import interaction, paired_diff  # noqa: E402


def _pair(fa, fb, metric):
    a = pd.read_csv(fa, dtype={"subject": str}).sort_values("key")
    b = pd.read_csv(fb, dtype={"subject": str}).sort_values("key")
    if list(a.key) != list(b.key):
        raise SystemExit(f"{fa} và {fb} không cùng tập ảnh")
    return a[metric].values, b[metric].values, a.subject.values


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--small", nargs=2, required=True, metavar=("VARIANT", "REF"))
    ap.add_argument("--large", nargs=2, required=True, metavar=("VARIANT", "REF"))
    ap.add_argument("--metric", default="psnr_y")
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args()
    a1, b1, c1 = _pair(*a.small, a.metric)
    a2, b2, c2 = _pair(*a.large, a.metric)
    same = sorted(set(c1)) == sorted(set(c2))
    g1, g2 = paired_diff(a1, b1, c1, n_boot=a.n_boot), paired_diff(a2, b2, c2, n_boot=a.n_boot)
    e = interaction(a1, b1, c1, a2, b2, c2, n_boot=a.n_boot, same_clusters=same)
    print(f"phần hơn ở ảnh nhỏ : {g1.point:+.4f} [{g1.lo:+.4f}, {g1.hi:+.4f}]")
    print(f"phần hơn ở ảnh lớn : {g2.point:+.4f} [{g2.lo:+.4f}, {g2.hi:+.4f}]")
    print(f"tương tác (nhỏ - lớn): {e.point:+.4f} [{e.lo:+.4f}, {e.hi:+.4f}]  "
          f"{'ĐẠT' if e.lo > 0 else 'KHÔNG ĐẠT'} (cùng tập người: {same})")


if __name__ == "__main__":
    main()
