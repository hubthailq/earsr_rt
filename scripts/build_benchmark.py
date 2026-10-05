#!/usr/bin/env python3
"""P1 và P2: dựng benchmark AMI và ảnh LR test.

  python scripts/build_benchmark.py --ami-raw /path/AMI --out data/bench/ami

Tạo ảnh đáp án ở các cỡ 96, 144, 192, 244 px (cạnh ngắn), kiểm file fold có sẵn
(splits/ami_5fold.json) khớp với bộ ảnh, rồi sinh ảnh LR test cho ×4 và ×2.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from earsr.data.ami import MAIN_TIERS, TIERS, build_ami_benchmark, check_ami, scan_ami  # noqa: E402
from earsr.data.build_lr import build_lr_set  # noqa: E402
from earsr.data.splits import load_folds, make_folds, save_folds  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ami-raw", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--folds", default="splits/ami_5fold.json")
    ap.add_argument("--kinds", nargs="+", default=["bic", "bicjpeg75"])
    ap.add_argument("--scales", type=int, nargs="+", default=[4, 2])
    a = ap.parse_args()
    items = scan_ami(a.ami_raw)
    print("kiểm bộ ảnh:", check_ami(items))
    man = build_ami_benchmark(a.ami_raw, a.out, TIERS)
    print("đã ghi", man)
    subjects = sorted({i.subject for i in items})
    fp = Path(a.folds)
    if fp.exists():
        d = load_folds(fp)
        fold_subj = sorted({s for f in d["folds"].values() for part in f.values() for s in part})
        if fold_subj != subjects:
            raise SystemExit("file fold không khớp danh sách người của bộ ảnh; không tự chia lại. "
                             "Kiểm lại dữ liệu hoặc xóa file fold nếu thật sự muốn chia mới.")
        print(f"file fold khớp bộ ảnh (sha1 {d['sha1'][:12]})")
    else:
        d = make_folds(subjects)
        save_folds(d, fp)
        print(f"đã tạo file fold mới: {fp} (sha1 {d['sha1'][:12]}). Hãy commit file này.")
    for t in MAIN_TIERS:
        for s in a.scales:
            for k in a.kinds:
                print("LR:", build_lr_set(a.out, t, s, k))


if __name__ == "__main__":
    main()
