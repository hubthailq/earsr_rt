#!/usr/bin/env python3
"""Vẽ các hình của bài từ ``results/`` (mục 2.5 của kế hoạch). Hình nào thiếu dữ liệu thì bỏ qua và báo.

  python scripts/make_figures.py --results results --out paper/figures

Tự vẽ được khi có dữ liệu:
  fig2_size_sweep.pdf   từ results/size_sweep/summary.csv          (run_size_sweep.py)
  fig3_quality_cost.pdf từ results/t2_summary/quality.csv + bảng độ trễ (ưu tiên latency_jetson.csv, không có thì latency_local.csv)
  fig4_context.pdf      từ results/t6_context/*.csv                 (run_t6_context.py)
Hình 5 (đường chất lượng theo độ trễ của mô hình đã huấn luyện) dùng cùng hàm với hình 3:
  python scripts/make_figures.py cost --quality results/tables/main.csv --cost results/latency_jetson.csv --out paper/figures/fig5.pdf
Hình 1 và 6 (so sánh định tính) cần chỉ rõ ảnh và cột:
  python scripts/make_figures.py grid --col LR=data/bench/ami/lr/hr144_x4_bic --col Bicubic=results/sr/hr144_x4_bic/bicubic \
      --col Ours=results/sr/hr144_x4_bic/<mã> --col HR=data/bench/ami/hr144 --keys 012_front 047_left --zoom-lr 4 \
      --out paper/figures/fig1.pdf
Lưu ý: ảnh AMI có giấy phép CC BY-NC-ND; hình dùng ảnh AMI cần thư đồng ý của tác giả (docs/AMI_PERMISSION.md).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from earsr.report import figures as F  # noqa: E402
from earsr.report.context import load_dir as load_context  # noqa: E402


def auto(a) -> list[str]:
    res, out = Path(a.results), Path(a.out)
    made = []

    def done(p):
        made.append(Path(p).name)
        print("đã vẽ", p)

    f = res / "size_sweep" / "summary.csv"
    if f.exists():
        done(F.fig_size_sweep(pd.read_csv(f), out / "fig2_size_sweep.pdf", a.models))
    else:
        print("bỏ qua hình 2: chưa có", f)
    q = res / "t2_summary" / "quality.csv"
    lat = next((p for p in (res / "latency_jetson.csv", res / "latency_local.csv") if p.exists()), None)
    if q.exists() and lat is not None:
        c = pd.read_csv(lat)
        if "status" in c.columns:
            c = c[c.status == "ok"]
        thr = 33.0 if lat.name == "latency_jetson.csv" else None
        done(F.fig_quality_vs_cost(pd.read_csv(q), c, out / "fig3_quality_cost.pdf", a.tier, a.scale, models=a.models,
                                   cost_label=f"Latency (ms, median; {'Jetson Nano' if thr else 'this machine'})",
                                   threshold=thr))
        if "params" in c.columns and c.params.notna().any():
            c2 = c.assign(params_k=pd.to_numeric(c.params, errors="coerce") / 1e3).dropna(subset=["params_k"])
            done(F.fig_quality_vs_cost(pd.read_csv(q), c2, out / "fig3b_quality_params.pdf", a.tier, a.scale,
                                       models=a.models, cost_col="params_k", cost_label="Parameters (K)"))
    else:
        print("bỏ qua hình 3: cần", q, "và results/latency_jetson.csv hoặc latency_local.csv")
    d = res / "t6_context"
    if d.is_dir() and any(d.glob("*.csv")):
        done(F.fig_context(load_context(d), out / "fig4_context.pdf", a.context_config, a.models,
                           labels={"ami": "Ear images (AMI)", "generic": "Natural images", "urban100": "Natural images (Urban100)",
                                   "div2k": "Natural images (DIV2K)"}))
    else:
        print("bỏ qua hình 4: chưa có", d)
    return made


def cmd_cost(a) -> None:
    q = pd.read_csv(a.quality)
    if "model" not in q.columns and "method" in q.columns:      # bảng của compare_table.py
        q = q.rename(columns={"method": "model"})
    for col, val in (("tier", a.tier), ("scale", a.scale), ("degrade", a.kinds[0])):
        if col not in q.columns:
            q[col] = val
    print("đã vẽ", F.fig_quality_vs_cost(q, pd.read_csv(a.cost), a.out, a.tier, a.scale, tuple(a.kinds), a.metric,
                                         a.cost_col, a.cost_label, a.models, threshold=a.threshold))


def cmd_grid(a) -> None:
    cols = dict(c.split("=", 1) for c in a.col)
    print("đã vẽ", F.fig_grid(cols, a.keys, a.out, a.zoom_lr, crop=tuple(a.crop) if a.crop else None))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="paper/figures")
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--tier", type=int, default=144)
    ap.add_argument("--scale", type=int, default=4)
    ap.add_argument("--context-config", default="wide")
    sub = ap.add_subparsers(dest="cmd")
    p = sub.add_parser("cost")
    p.add_argument("--quality", required=True, help="bảng có cột model và số đo (quality.csv hoặc bảng của compare_table.py)")
    p.add_argument("--cost", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--metric", default="psnr_y"); p.add_argument("--cost-col", default="median_ms")
    p.add_argument("--cost-label", default="Latency (ms, median)"); p.add_argument("--kinds", nargs="+", default=["bic"])
    p.add_argument("--tier", type=int, default=144); p.add_argument("--scale", type=int, default=4)
    p.add_argument("--models", nargs="*", default=None); p.add_argument("--threshold", type=float, default=None)
    p = sub.add_parser("grid")
    p.add_argument("--col", action="append", required=True, help="Tên=thư_mục; lặp lại theo thứ tự cột")
    p.add_argument("--keys", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--zoom-lr", type=int, default=None)
    p.add_argument("--crop", type=int, nargs=4, default=None, metavar=("TOP", "LEFT", "BOTTOM", "RIGHT"))
    a = ap.parse_args(argv)
    if a.cmd == "cost":
        return cmd_cost(a)
    if a.cmd == "grid":
        return cmd_grid(a)
    return auto(a)


if __name__ == "__main__":
    main()
