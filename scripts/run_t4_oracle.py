#!/usr/bin/env python3
"""T4: cổng oracle so với đường trộn, trên ô chính, người test của fold 2, 3, 4. Không huấn luyện.

  python scripts/run_t4_oracle.py --bench data/bench/ami --fs rrdb_psnr --ft esrgan --out results/t4

``--fs``, ``--ft``: tên mô hình trong kho, hoặc thư mục lần chạy của train.py
(ví dụ thân tinh chỉnh L1 và thân bản GAN, khi đã có). Cần LPIPS (pip install lpips).
Ghi ``<out>/t4_per_image.csv`` và ``<out>/t4_summary.json``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

from earsr.data.ami import read_manifest  # noqa: E402
from earsr.data.build_lr import build_lr_set  # noqa: E402
from earsr.data.splits import load_folds  # noqa: E402
from earsr.eval.infer import make_predictor  # noqa: E402
from earsr.eval.metrics import PerceptualMetrics, fr_metrics  # noqa: E402
from earsr.eval.oracle import candidates  # noqa: E402
from earsr.io import imread_rgb  # noqa: E402
from earsr.report.t4 import summarize  # noqa: E402


def predictor(spec: str, scale: int, device: str):
    if Path(spec).is_dir():
        from earsr.train.finetune import predictor_from_run

        return predictor_from_run(spec, device)
    return make_predictor(spec, scale, device)


def run(bench, tier, scale, kind, fs, ft, subjects, perceptual, limit=None) -> pd.DataFrame:
    """``fs``, ``ft``: hàm lr -> sr. Trả về bảng theo (ảnh, method, param)."""
    bench = Path(bench)
    rows = [r for r in read_manifest(bench / "manifest.csv") if r["tier"] == tier and r["subject"] in subjects]
    if limit:
        rows = rows[:limit]
    out = []
    for r in rows:
        key = f"{r['subject']}_{r['view']}"
        hr = imread_rgb(bench / r["file"])
        lr = imread_rgb(bench / "lr" / f"hr{tier}_x{scale}_{kind}" / f"{key}.png")
        f_s, f_t = fs(lr), ft(lr)
        for method, param, img, frac in candidates(f_s, f_t, hr):
            m = fr_metrics(img, hr, scale, perceptual=perceptual)
            out.append({"subject": r["subject"], "key": key, "method": method, "param": param, "frac_fs": frac, **m})
    return pd.DataFrame(out)


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bench", required=True)
    ap.add_argument("--folds", default="splits/ami_5fold.json")
    ap.add_argument("--use-folds", type=int, nargs="+", default=[2, 3, 4])
    ap.add_argument("--tier", type=int, default=144)
    ap.add_argument("--scale", type=int, default=4)
    ap.add_argument("--kind", default="bic")
    ap.add_argument("--fs", default="rrdb_psnr")
    ap.add_argument("--ft", default="esrgan")
    ap.add_argument("--metric", default="lpips")
    ap.add_argument("--criteria", default="configs/criteria.yaml")
    ap.add_argument("--out", default="results/t4")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args(argv)
    if set(a.use_folds) & {1, 5}:
        raise SystemExit("fold 1 và 5 là hai fold giữ kín; T4 chỉ chạy trên fold 2, 3, 4")
    crit = yaml.safe_load(Path(a.criteria).read_text())["N3"]
    folds = load_folds(a.folds)["folds"]
    subjects = {s for f in a.use_folds for s in folds[str(f)]["test"]}
    perc = PerceptualMetrics(a.device)
    if a.metric not in perc.available:
        raise SystemExit(f"T4 cần số đo '{a.metric}': {perc.reason}")
    build_lr_set(a.bench, a.tier, a.scale, a.kind)
    df = run(a.bench, a.tier, a.scale, a.kind, predictor(a.fs, a.scale, a.device), predictor(a.ft, a.scale, a.device),
             subjects, perc, a.limit)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "t4_per_image.csv", index=False)
    s = summarize(df, a.metric, crit["psnr_budget_db"], crit["oracle_gate_T4"]["min_lpips_gain_over_blend"])
    s.update(fs=a.fs, ft=a.ft, tier=a.tier, scale=a.scale, kind=a.kind, folds=a.use_folds, perceptual=perc.info)
    (out / "t4_summary.json").write_text(json.dumps(s, indent=1, ensure_ascii=False))
    for r in s.get("oracles", []):
        print(f"{r['oracle']:22s} PSNR {r['psnr_y']:.2f}  {a.metric} {r[a.metric]:.4f}  đường trộn {r[f'blend_{a.metric}_same_psnr']:.4f}  "
              f"hơn {r['gain']:+.1%} [{r['gain_lo']:+.1%}, {r['gain_hi']:+.1%}]  {'ĐẠT' if r['pass'] else ''}")
    print("Quyết định:", s.get("decision", s.get("error")))
    return s


if __name__ == "__main__":
    main()
