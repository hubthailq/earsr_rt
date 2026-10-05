#!/usr/bin/env python3
"""Khảo sát người xem: tạo trang HTML và phân tích phiếu trả lời.

1. Lưu ảnh SR của các mô hình cần so (một thư mục mỗi mô hình):
     python scripts/evaluate.py ... --save-sr results/sr
   -> results/sr/hr144_x4_bic/<mô hình>/<khóa>.png

2. Tạo trang (phần có đáp án, 40 ảnh, chọn ngẫu nhiên cân theo người):
     python scripts/viewer_study.py make --part ref --hr-dir data/bench/ami/hr144 \
         --sr-dir results/sr/hr144_x4_bic --pairs ours:span_ft ours:bicubic --n 40 --out results/viewer/ref
   Phần không có đáp án (ảnh nhỏ thật): --part noref (không cần --hr-dir).
   -> study.html (gửi cho người xem), key.csv (GIỮ LẠI, không gửi).

3. Người xem mở study.html bằng trình duyệt, làm xong bấm tải answers_*.json và gửi lại.

4. Phân tích:
     python scripts/viewer_study.py analyze --key results/viewer/ref/key.csv --answers results/viewer/ref/answers_*.json
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from earsr.io import imread_rgb  # noqa: E402
from earsr.report import viewer  # noqa: E402


def cmd_make(a) -> None:
    pairs = [tuple(p.split(":")) for p in a.pairs]
    if any(len(p) != 2 for p in pairs):
        raise SystemExit("--pairs có dạng A:B")
    models = sorted({m for p in pairs for m in p})
    sr = Path(a.sr_dir)
    for m in models:
        if not (sr / m).is_dir():
            raise SystemExit(f"thiếu thư mục ảnh của mô hình '{m}': {sr / m}")
    keys = sorted(set.intersection(*[{p.stem for p in (sr / m).glob("*.png")} for m in models]))
    if a.part == "ref":
        if not a.hr_dir:
            raise SystemExit("--part ref cần --hr-dir")
        keys = [k for k in keys if (Path(a.hr_dir) / f"{k}.png").exists()]
    if a.subjects:
        allow = set(a.subjects)
        keys = [k for k in keys if k.split("_")[0] in allow]
    if not keys:
        raise SystemExit("không có ảnh chung giữa các mô hình")
    # chọn ngẫu nhiên, mỗi người một ảnh trước, rồi mới lặp lại người
    rng = np.random.default_rng(a.seed)
    by_subj: dict = {}
    for k in [keys[i] for i in rng.permutation(len(keys))]:
        by_subj.setdefault(k.split("_")[0], []).append(k)
    chosen, rnd = [], 0
    while len(chosen) < min(a.n, len(keys)):
        for s in sorted(by_subj):
            if rnd < len(by_subj[s]) and len(chosen) < a.n:
                chosen.append(by_subj[s][rnd])
        rnd += 1
    items = []
    for k in chosen:
        items.append({"key": k, "subject": k.split("_")[0], "part": a.part,
                      "ref": imread_rgb(Path(a.hr_dir) / f"{k}.png") if a.part == "ref" else None,
                      "outputs": {m: imread_rgb(sr / m / f"{k}.png") for m in models}})
    out = Path(a.out)
    key = viewer.build_study(items, pairs, out / "study.html", out / "key.csv", study=a.study, zoom=a.zoom, seed=a.seed)
    print(f"{len(key)} câu hỏi ({len(chosen)} ảnh của {key.subject.nunique()} người × {len(pairs)} cặp)")
    print(f"gửi cho người xem: {out / 'study.html'}\nGIỮ LẠI, không gửi: {out / 'key.csv'}")


def cmd_analyze(a) -> None:
    key = pd.read_csv(a.key, dtype=str)
    m = viewer.load_answers(a.answers, key)
    res = viewer.analyze(m)
    out = Path(a.out or Path(a.key).parent / "analysis.csv")
    res.to_csv(out, index=False)
    m.to_csv(out.with_name("answers_long.csv"), index=False)
    for _, r in res.iterrows():
        print(f"[{r.part}] {r.model_a} so với {r.model_b}: chọn {r.model_a} {r.pref_a:.1%} [{r.lo:.1%}, {r.hi:.1%}], "
              f"{r.n_viewers} người xem, {r.n_answers} câu; kiểm định dấu p = {r.p_sign_viewers:.3g}; "
              f"tỉ lệ chọn bên trái {r.left_rate:.1%}")
    if a.metric_csv:
        d = pd.concat([pd.read_csv(f, dtype={"key": str}) for f in a.metric_csv])
        mv = {(r.key, r.model): getattr(r, a.metric) for r in d.itertuples()}
        print(f"{a.metric} và đa số người xem cùng chọn:", viewer.metric_agreement(m, mv, lower_is_better=not a.higher_is_better))
    print("đã ghi", out)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("make")
    p.add_argument("--part", required=True, choices=["ref", "noref"])
    p.add_argument("--sr-dir", required=True, help="thư mục chứa một thư mục con cho mỗi mô hình")
    p.add_argument("--hr-dir", default=None)
    p.add_argument("--pairs", nargs="+", required=True, help="các cặp A:B (tên thư mục mô hình)")
    p.add_argument("--subjects", nargs="*", default=None, help="chỉ lấy ảnh của những người này (ví dụ người test)")
    p.add_argument("--n", type=int, default=40)
    p.add_argument("--zoom", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--study", default="earsr")
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_make)
    p = sub.add_parser("analyze")
    p.add_argument("--key", required=True)
    p.add_argument("--answers", nargs="+", required=True)
    p.add_argument("--out", default=None)
    p.add_argument("--metric-csv", nargs="*", default=None, help="file số đo theo ảnh (cột key, model, và số đo)")
    p.add_argument("--metric", default="lpips")
    p.add_argument("--higher-is-better", action="store_true")
    p.set_defaults(fn=cmd_analyze)
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
