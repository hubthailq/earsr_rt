#!/usr/bin/env python3
"""Tóm tắt kết quả của scripts/run_review.sh (các mốc thêm cho vòng phản biện): PSNR-Y trung bình của từng phương pháp
và chênh lệch ghép cặp so với bicubic và so với SPAN nhánh est, theo bộ dữ liệu, cỡ ảnh và kiểu ảnh vào.

  python scripts/summarize_review.py --results results        # ghi results/rev_summary/summary.md và fidelity.csv

Chênh lệch ghép cặp tính trên các ảnh mà cả hai phương pháp đều có (mô hình tự huấn luyện trên một fold của AMI chỉ có
ảnh của người test fold đó); khoảng tin cậy 95% là bootstrap theo người. Trên bộ không chia fold, các mô hình cùng nhánh
(khác fold) được lấy trung bình theo từng ảnh.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from earsr.report.arms import arm_of  # noqa: E402
from earsr.stats.bootstrap import paired_diff  # noqa: E402

SETS = (("AMI", "rev"), ("EarVN1.0", "rev_earvn"), ("AWEx", "rev_awex"))
REFS = ("bicubic", "span/est")
ORDER = ["bicubic", "span_ch48 (published)", "rrdb_psnr (published)", "bsrnet (published)", "realesrnet (published)",
         "fbcnn_bicubic (published)", "fbcnn_span_ch48 (published)", "span/est", "rrdb/est"]
KINDS = ["bic", "bicjpeg93", "bicjpeg85", "bicjpeg75", "est"]


def load(folder: Path) -> pd.DataFrame | None:
    fs = sorted(folder.glob("*.csv"))
    if not fs:
        return None
    d = pd.concat([pd.read_csv(f, dtype={"subject": str, "view": str}) for f in fs], ignore_index=True)
    a = d.model.map(arm_of)
    d["arm"], d["runfold"] = [x[0] for x in a], [x[1] for x in a]
    return d


def rows_of(d: pd.DataFrame, name: str) -> list[dict]:
    folds = d.groupby("arm").runfold.agg(lambda s: " ".join(str(f) for f in sorted(set(s)) if f))
    p = d.groupby(["arm", "tier", "degrade", "key", "subject"], as_index=False).psnr_y.mean()
    out = []
    for (tier, kind), g in p.groupby(["tier", "degrade"]):
        by = {a: x.set_index("key") for a, x in g.groupby("arm")}
        for arm, A in by.items():
            r = {"dataset": name, "tier": int(tier), "input_px": int(tier) // 4, "degrade": kind, "arm": arm,
                 "folds": folds.get(arm, ""), "n": len(A), "psnr_y": float(A.psnr_y.mean())}
            for ref in REFS:
                tag = "bic" if ref == "bicubic" else "ours"
                B = by.get(ref)
                keys = A.index.intersection(B.index) if B is not None and ref != arm else []
                if len(keys):
                    e = paired_diff(A.loc[keys, "psnr_y"].to_numpy(), B.loc[keys, "psnr_y"].to_numpy(),
                                    A.loc[keys, "subject"].to_numpy(), n_boot=2000)
                    r.update({f"vs_{tag}": e.point, f"vs_{tag}_lo": e.lo, f"vs_{tag}_hi": e.hi, f"vs_{tag}_n": len(keys)})
            out.append(r)
    return out


def fmt(r: dict, tag: str) -> str:
    if f"vs_{tag}" not in r or pd.isna(r[f"vs_{tag}"]):
        return "--"
    star = "" if r[f"vs_{tag}_lo"] <= 0 <= r[f"vs_{tag}_hi"] else " *"
    return f"{r[f'vs_{tag}']:+.2f} [{r[f'vs_{tag}_lo']:+.2f}, {r[f'vs_{tag}_hi']:+.2f}]{star} (n={int(r[f'vs_{tag}_n'])})"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default=None, help="mặc định: <results>/rev_summary")
    a = ap.parse_args(argv)
    res = Path(a.results)
    out = Path(a.out) if a.out else res / "rev_summary"
    rows = []
    for name, sub in SETS:
        d = load(res / sub)
        if d is None:
            print(f"chưa có kết quả trong {res / sub}")
            continue
        rows += rows_of(d, name)
    if not rows:
        raise SystemExit("không có kết quả nào để tóm tắt (chạy scripts/run_review.sh trước)")
    out.mkdir(parents=True, exist_ok=True)
    T = pd.DataFrame(rows)
    T.to_csv(out / "fidelity.csv", index=False)
    rank = {a_: i for i, a_ in enumerate(ORDER)}
    md = ["# Các mốc thêm cho vòng phản biện: độ trung thực (PSNR-Y, dB)", "",
          "Chênh lệch ghép cặp kèm khoảng tin cậy 95% (bootstrap theo người); dấu * là khoảng không chứa 0; "
          "n là số ảnh chung của hai phương pháp. Cột fold: các fold đã có của mô hình tự huấn luyện.", ""]
    for (ds, tier), g in T.groupby(["dataset", "tier"], sort=False):
        for kind in [k for k in KINDS if k in set(g.degrade)]:
            x = g[g.degrade == kind].copy()
            x["o"] = x.arm.map(lambda a_: rank.get(a_, len(rank)))
            md += [f"## {ds}, ảnh vào {tier // 4} px, kiểu ảnh vào `{kind}`", "",
                   "| Phương pháp | fold | n | PSNR-Y | so với bicubic | so với SPAN nhánh est |", "|---|---|---|---|---|---|"]
            for r in x.sort_values(["o", "arm"]).to_dict("records"):
                md.append(f"| {r['arm']} | {r['folds'] or '-'} | {r['n']} | {r['psnr_y']:.2f} | {fmt(r, 'bic')} | {fmt(r, 'ours')} |")
            md.append("")
    lat = res / "latency_review.csv"
    if lat.is_file():
        L = pd.read_csv(lat)
        md += ["## Độ trễ trên máy chấm (ảnh vào 48×68, FP32)", "", "| Mô hình | tham số (triệu) | trung vị (ms) | p95 (ms) |", "|---|---|---|---|"]
        for r in L.to_dict("records"):
            if pd.notna(r.get("median_ms")):
                par = f"{r['params'] / 1e6:.2f}" if pd.notna(r.get("params")) else "-"
                md.append(f"| {r['name']} | {par} | {r['median_ms']:.2f} | {r['p95_ms']:.2f} |")
        md.append("")
    recs = sorted((res / "recog").glob("rev_*_summary/summary.md")) if (res / "recog").is_dir() else []
    if recs:
        md += ["## Nhận dạng trên ảnh nhỏ thật", ""] + [f"- `{f}`" for f in recs] + [""]
    (out / "summary.md").write_text("\n".join(md))
    print(f"đã ghi {out / 'summary.md'} ({len(T)} dòng số liệu)")


if __name__ == "__main__":
    main()
