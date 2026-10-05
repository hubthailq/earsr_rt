#!/usr/bin/env python3
"""Dựng một bảng so sánh từ các file số đo theo từng ảnh: mỗi dòng là một phương pháp
(thường nhiều file, một file mỗi fold), mỗi cột là một số đo, kèm chênh lệch ghép cặp so
với một dòng mốc. Dùng cho các bảng 4, 6, 7, 8, 11 của bài (mục 2.5 của kế hoạch).

  python scripts/compare_table.py --name main \
      --row "Bicubic=results/s2/bicubic__hr144_x4_bic.csv" \
      --row "SPAN (fine-tuned)=runs/S2_span-zero_pub_*/test_hr144_x4_bic.csv" \
      --row "Ours=runs/S2_span-replicate+xearvn_pf_*/test_hr144_x4_bic.csv" \
      --ref "SPAN (fine-tuned)" --metrics psnr_y ssim_y ms_ssim_y lpips dists gmsd lr_psnr_y ridge_f1

Mọi dòng phải cùng tập ảnh với dòng mốc (cùng khóa ảnh); dòng nào khác tập ảnh thì vẫn có
trung bình nhưng cột chênh lệch để trống và được báo. Ghi:
  results/tables/<name>.csv   trung bình, khoảng tin cậy 95% (bootstrap theo người), chênh lệch ghép cặp
  paper/tables/<name>.tex     bảng LaTeX: trung bình; chênh lệch có khoảng tin cậy không chứa 0 được đánh dấu †
Thêm cột chi phí (độ trễ, tham số) bằng --cost file.csv --cost-key "tên dòng=tên trong file".
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from earsr.report.criteria import load  # noqa: E402
from earsr.report.latex import booktabs  # noqa: E402
from earsr.stats.bootstrap import mean_ci, paired_diff  # noqa: E402

LOWER_BETTER = {"lpips", "dists", "gmsd", "stlpips", "pieapp", "niqe", "ridge_false", "ridge_missed", "lm_dev"}
DIGITS = {"psnr_y": 2, "psnr_rgb": 2, "grad_psnr": 2, "lr_psnr_y": 2, "ssim_y": 4, "ms_ssim_y": 4, "lpips": 4,
          "dists": 4, "gmsd": 4, "ridge_f1": 3, "ridge_false": 3, "ridge_missed": 3, "lm_dev": 4}
HEAD = {"psnr_y": "PSNR", "ssim_y": "SSIM", "ms_ssim_y": "MS-SSIM", "lpips": "LPIPS", "dists": "DISTS", "gmsd": "GMSD",
        "grad_psnr": "Grad-PSNR", "lr_psnr_y": "LR-PSNR", "ridge_f1": "Ridge F1", "ridge_false": "False ridges",
        "ridge_missed": "Missed ridges", "lm_dev": "Landmark dev.", "median_ms": "Latency (ms)", "params_k": "Params (K)"}


def build(rows: list[tuple[str, list[str]]], metrics: list[str], ref: str | None, n_boot: int = 2000) -> pd.DataFrame:
    data = {name: load(files) for name, files in rows}
    if ref is not None and ref not in data:
        raise SystemExit(f"--ref '{ref}' không trùng tên dòng nào: {list(data)}")
    out = []
    for name, d in data.items():
        r = {"method": name, "n_images": len(d), "n_subjects": d.subject.nunique(),
             "n_folds": d.fold.nunique() if "fold" in d else np.nan}
        same = ref is not None and name != ref and list(d.key) == list(data[ref].key)
        if ref is not None and name != ref and not same:
            print(f"CHÚ Ý: '{name}' không cùng tập ảnh với '{ref}' ({len(d)} so với {len(data[ref])}); bỏ cột chênh lệch")
        for m in metrics:
            if m not in d.columns or not d[m].notna().any():
                continue
            v = d[m].replace([np.inf, -np.inf], np.nan)
            e = mean_ci(v.values, d.subject.values, n_boot=n_boot)
            r.update({m: e.point, f"{m}_lo": e.lo, f"{m}_hi": e.hi})
            if same and m in data[ref].columns:
                b = data[ref][m].replace([np.inf, -np.inf], np.nan)
                p = paired_diff(v.values, b.values, d.subject.values, n_boot=n_boot)
                r.update({f"{m}_diff": p.point, f"{m}_diff_lo": p.lo, f"{m}_diff_hi": p.hi,
                          f"{m}_sig": bool(p.excludes_zero())})
        out.append(r)
    return pd.DataFrame(out)


def to_tex(t: pd.DataFrame, metrics: list[str], name: str, caption: str, extra_cols: list[str]) -> str:
    cols = [m for m in metrics if m in t.columns] + [c for c in extra_cols if c in t.columns]
    view = t[["method"] + cols].copy()
    best = {m: ("min" if m in LOWER_BETTER or m == "median_ms" else "max") for m in cols if m != "params_k"}
    tex = booktabs(view, caption, f"tab:{name}", {m: DIGITS.get(m, 2) for m in cols}, best,
                   header={"method": "Method", **{m: HEAD.get(m, m) for m in cols}})
    # dấu † cho chênh lệch có ý nghĩa so với dòng mốc
    lines = tex.split("\n")
    for _, r in t.iterrows():
        marks = [m for m in cols if r.get(f"{m}_sig") is True]
        if not marks:
            continue
        for i, line in enumerate(lines):
            if line.startswith(str(r.method).replace("_", r"\_") + " & "):
                cells = line[:-3].split(" & ")
                for m in marks:
                    j = 1 + cols.index(m)
                    cells[j] = cells[j] + r"$^{\dagger}$"
                lines[i] = " & ".join(cells) + r" \\"
    return "\n".join(lines)


def main(argv=None) -> pd.DataFrame:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True)
    ap.add_argument("--row", action="append", required=True, help='"Tên=mẫu_file [mẫu_file ...]" (mẫu kiểu glob, cách nhau bằng dấu phẩy)')
    ap.add_argument("--ref", default=None, help="tên dòng mốc để tính chênh lệch ghép cặp")
    ap.add_argument("--metrics", nargs="+", default=["psnr_y", "ssim_y", "ms_ssim_y", "lpips", "dists", "gmsd"])
    ap.add_argument("--cost", default=None, help="file độ trễ (cột name, median_ms, params)")
    ap.add_argument("--cost-key", action="append", default=[], help='"tên dòng=tên trong file độ trễ"')
    ap.add_argument("--caption", default=None)
    ap.add_argument("--out-csv", default="results/tables")
    ap.add_argument("--out-tex", default="paper/tables")
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args(argv)
    rows = []
    for spec in a.row:
        if "=" not in spec:
            raise SystemExit(f'--row phải có dạng "Tên=mẫu_file": {spec}')
        name, pats = spec.split("=", 1)
        files = sorted(f for p in pats.split(",") for f in glob.glob(p.strip()))
        if not files:
            raise SystemExit(f"dòng '{name}': không file nào khớp {pats}")
        rows.append((name.strip(), files))
    t = build(rows, a.metrics, a.ref, a.n_boot)
    extra = []
    if a.cost:
        c = pd.read_csv(a.cost).rename(columns={"name": "model"})
        key = dict(k.split("=", 1) for k in a.cost_key)
        c = c.set_index("model")
        for col, src, f in (("median_ms", "median_ms", 1.0), ("params_k", "params", 1e-3)):
            if src in c.columns:
                t[col] = [float(c.loc[key[m], src]) * f if key.get(m) in c.index else np.nan for m in t.method]
                extra.append(col)
    Path(a.out_csv).mkdir(parents=True, exist_ok=True)
    Path(a.out_tex).mkdir(parents=True, exist_ok=True)
    t.to_csv(Path(a.out_csv) / f"{a.name}.csv", index=False)
    cap = a.caption or (f"Comparison ({a.name}). Mean over images; "
                        + (rf"$\dagger$: paired difference to {a.ref} has a 95\% subject-level bootstrap interval excluding zero." if a.ref else ""))
    (Path(a.out_tex) / f"{a.name}.tex").write_text(to_tex(t, a.metrics, a.name, cap, extra))
    show = ["method", "n_images", "n_folds"] + [m for m in a.metrics if m in t.columns]
    print(t[show].round(4).to_string(index=False))
    for m in a.metrics:
        if f"{m}_diff" in t.columns:
            for _, r in t.dropna(subset=[f"{m}_diff"]).iterrows():
                print(f"  {m}: {r.method} − {a.ref} = {r[f'{m}_diff']:+.4f} [{r[f'{m}_diff_lo']:+.4f}, {r[f'{m}_diff_hi']:+.4f}]")
    print("đã ghi", Path(a.out_csv) / f"{a.name}.csv", "và", Path(a.out_tex) / f"{a.name}.tex")
    return t


if __name__ == "__main__":
    main()
