#!/usr/bin/env python3
"""Đường chất lượng theo cỡ ảnh vào, từ 16 đến 64 px (S4; bằng chứng trực tiếp cho N5). Chỉ suy luận.

  python scripts/run_size_sweep.py --ami-raw /path/AMI --models bicubic span_ch48 rlfn \
      --runs runs/S2_span-..._f2 runs/S2_span-..._f3 --out results/size_sweep

Với mỗi cạnh ngắn s của ảnh vào: ảnh gốc AMI được thu về cạnh ngắn s × hệ số phóng
(đáp án), rồi thu nhỏ bicubic thành ảnh vào. Mô hình trong kho được chấm trên người
test của các fold trong --use-folds; mô hình tự huấn luyện chỉ trên người test của fold của nó.
Dải cỡ đã thấy khi huấn luyện (24 đến 48 px với giao thức 'rand') phải được đánh dấu trên hình.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
import torch  # noqa: E402

from earsr.data.ami import scan_ami  # noqa: E402
from earsr.data.datasets import resize_short_side  # noqa: E402
from earsr.data.splits import fold_of_subject, load_folds  # noqa: E402
from earsr.degrade.pipelines import degrade  # noqa: E402
from earsr.eval.infer import make_predictor  # noqa: E402
from earsr.eval.metrics import fr_metrics  # noqa: E402
from earsr.io import imread_rgb  # noqa: E402
from earsr.stats.bootstrap import mean_ci  # noqa: E402


def sweep(ami_raw, folds_path, use_folds, sizes, scale, predictors: dict, kind: str = "bic", limit=None) -> pd.DataFrame:
    """``predictors``: tên -> (hàm lr->sr, tập người được chấm hoặc None)."""
    folds = load_folds(folds_path)
    fold_of = fold_of_subject(folds)
    items = [it for it in scan_ami(ami_raw) if fold_of.get(it.subject) in use_folds]
    if limit:
        items = items[:limit]
    rows = []
    for it in items:
        orig = imread_rgb(it.path)
        for s in sizes:
            hr = resize_short_side(orig, s * scale, multiple=max(4, scale))
            lr = degrade(hr, scale, kind)
            for name, (pred, subjects) in predictors.items():
                if subjects is not None and it.subject not in subjects:
                    continue
                m = fr_metrics(pred(lr), hr, scale)
                rows.append({"model": name, "subject": it.subject, "key": it.key, "fold": fold_of[it.subject],
                             "lr_short": s, "psnr_y": m["psnr_y"], "ssim_y": m["ssim_y"]})
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame, n_boot: int = 2000) -> pd.DataFrame:
    out = []
    for (model, s), g in df.groupby(["model", "lr_short"]):
        e = mean_ci(g.psnr_y.values, g.subject.values, n_boot=n_boot)
        out.append({"model": model, "lr_short": s, "n": len(g), "psnr_y": e.point, "lo": e.lo, "hi": e.hi,
                    "ssim_y": g.ssim_y.mean()})
    return pd.DataFrame(out)


def main(argv=None) -> pd.DataFrame:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ami-raw", required=True)
    ap.add_argument("--folds", default="splits/ami_5fold.json")
    ap.add_argument("--use-folds", type=int, nargs="+", default=[2, 3, 4])
    ap.add_argument("--sizes", type=int, nargs="+", default=list(range(16, 65, 4)))
    ap.add_argument("--scale", type=int, default=4)
    ap.add_argument("--models", nargs="*", default=["bicubic"])
    ap.add_argument("--runs", nargs="*", default=[])
    ap.add_argument("--out", default="results/size_sweep")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args(argv)
    preds = {m: (make_predictor(m, a.scale, a.device), None) for m in a.models}
    folds = load_folds(a.folds)["folds"]
    for rd in a.runs:
        from earsr.train.finetune import predictor_from_run

        p = predictor_from_run(rd, a.device)
        fold = json.loads((Path(rd) / "config.json").read_text())["extra"].get("fold")
        preds[p.run_id] = (p, set(folds[str(fold)]["test"]) if fold else None)
    df = sweep(a.ami_raw, a.folds, a.use_folds, a.sizes, a.scale, preds, limit=a.limit)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "per_image.csv", index=False)
    s = summarize(df)
    s.to_csv(out / "summary.csv", index=False)
    print(s.pivot(index="lr_short", columns="model", values="psnr_y").round(3).to_string())
    return s


if __name__ == "__main__":
    main()
