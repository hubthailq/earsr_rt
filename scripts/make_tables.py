#!/usr/bin/env python3
"""Dựng lại các bảng LaTeX từ ``results/``. Chạy sau ``summarize_t2.py``.

  python scripts/make_tables.py --results results --out paper/tables

Bảng nào thiếu dữ liệu thì bỏ qua và báo.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from earsr.report.latex import booktabs, efficiency_table, t2_table  # noqa: E402


def main(argv=None) -> list[str]:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="paper/tables")
    a = ap.parse_args(argv)
    res, out = Path(a.results), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    made = []

    def write(name: str, tex: str) -> None:
        (out / name).write_text(tex)
        made.append(name)
        print("đã ghi", out / name)

    q = res / "t2_summary" / "quality.csv"
    if q.exists():
        qd = pd.read_csv(q)
        for (tier, scale), g in qd.groupby(["tier", "scale"]):
            kinds = tuple(k for k in ("bic", "bicjpeg75", "est") if k in set(g.degrade))
            extra = tuple(c for c in ("lpips", "dists", "gmsd") if c in g.columns and g[c].notna().any())
            write(f"t2_hr{tier}_x{scale}.tex", t2_table(qd, int(tier), int(scale), kinds, extra))
    else:
        print("bỏ qua bảng T2: chưa có", q)
    for name in ("latency_local.csv", "latency_jetson.csv"):
        f = res / name
        if f.exists():
            write(name.replace(".csv", ".tex"), efficiency_table(pd.read_csv(f)))
    f = res / "n2_realism.json"
    if f.exists():      # Bảng 5: bộ phân loại "mô phỏng hay thật"
        import json

        k = json.loads(f.read_text())["kinds"]
        d = pd.DataFrame([{"degradation": n, "acc": 100 * v["balanced_acc"], "lo": 100 * v["lo"], "hi": 100 * v["hi"],
                           "dist": 100 * v["dist_from_chance"]} for n, v in k.items()])
        write("n2_realism.tex", booktabs(d, "Real-versus-simulated classifier: balanced accuracy (\\%) on held-out subjects; "
                                            "50\\% means the classifier cannot tell simulated from real small ear images.",
                                         "tab:realism", {"acc": 1, "lo": 1, "hi": 1, "dist": 1}, {"dist": "min"},
                                         header={"degradation": "Degradation", "acc": "Accuracy", "lo": "CI low", "hi": "CI high",
                                                 "dist": "|Acc. - 50|"}))
    for part in ("ref", "noref"):      # Bảng 9: khảo sát người xem
        f = res / "viewer" / part / "analysis.csv"
        if f.exists():
            d = pd.read_csv(f)
            d = d.assign(pref=100 * d.pref_a, lo=100 * d.lo, hi=100 * d.hi)[["model_a", "model_b", "pref", "lo", "hi", "n_viewers", "n_images"]]
            write(f"viewer_{part}.tex", booktabs(d, "Viewer study (" + ("reference shown" if part == "ref" else "no reference")
                                                 + "): share of pairs (\\%) in which the first model was chosen; 95\\% interval by viewer.",
                                                 f"tab:viewer_{part}", {"pref": 1, "lo": 1, "hi": 1},
                                                 header={"model_a": "Model A", "model_b": "Model B", "pref": "A chosen", "lo": "CI low",
                                                         "hi": "CI high", "n_viewers": "Viewers", "n_images": "Images"}))
    c = res / "t6_summary" / "context.csv"
    if c.exists():
        cd = pd.read_csv(c)
        write("t6_context.tex", booktabs(cd, "Context test: error with real context minus error with padding only.",
                                         "tab:context", {k: 3 for k in cd.columns}))
    return made


if __name__ == "__main__":
    main()
