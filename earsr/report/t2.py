"""Tổng hợp T2 từ các file số đo theo từng ảnh: bảng chất lượng, phần hơn so với
bicubic, so thứ hạng giữa hai kiểu suy giảm, mức chênh nhỏ nhất phát hiện được."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from ..eval.metrics import LR_PSNR_CAP
from ..stats.bootstrap import mean_ci, paired_diff
from ..stats.mde import mde_paired
from ..stats.ranking import kendall_tau_b, pairwise_signs, tau_bootstrap_ci

_NAME = re.compile(r"^(?P<model>.+)__hr(?P<tier>\d+)_x(?P<scale>\d+)_(?P<kind>[A-Za-z0-9]+)\.csv$")


def load_dir(d: str | Path) -> pd.DataFrame:
    frames = []
    for f in sorted(Path(d).glob("*.csv")):
        if not _NAME.match(f.name):
            continue
        frames.append(pd.read_csv(f, dtype={"subject": str}))
    if not frames:
        raise FileNotFoundError(f"không có file kết quả trong {d}")
    return pd.concat(frames, ignore_index=True)


def quality_table(df: pd.DataFrame, meta: dict | None = None, n_boot: int = 2000) -> pd.DataFrame:
    """Một dòng cho mỗi (tier, scale, degrade, model)."""
    rows = []
    for (tier, scale, kind), g in df.groupby(["tier", "scale", "degrade"]):
        base = g[g.model == "bicubic"].sort_values("key")
        for model, gm in g.groupby("model"):
            gm = gm.sort_values("key")
            e = mean_ci(gm.psnr_y.values, gm.subject.values, n_boot=n_boot)
            row = {"tier": tier, "scale": scale, "degrade": kind, "model": model, "n": len(gm),
                   "psnr_y": e.point, "psnr_y_lo": e.lo, "psnr_y_hi": e.hi,
                   "ssim_y": gm.ssim_y.mean(), "psnr_rgb": gm.psnr_rgb.mean(), "psnr_y_center": gm.psnr_y_c.mean()}
            for col in ("lpips", "dists", "ms_ssim_y", "gmsd", "grad_psnr", "lr_psnr_y", "stlpips", "topiq_fr",
                        "fsim", "vif", "pieapp", "ridge_f1", "lm_dev"):
                if col in gm and gm[col].notna().any():
                    v = gm[col]
                    if col == "lr_psnr_y":  # file chấm trước 06/10/2026 chưa chặn: vô cực nghĩa là trùng khít
                        v = v.replace(float("inf"), LR_PSNR_CAP)
                    row[col] = v.replace([float("inf"), float("-inf")], float("nan")).mean()
            if len(base) == len(gm) and (base.key.values == gm.key.values).all() and model != "bicubic":
                d = paired_diff(gm.psnr_y.values, base.psnr_y.values, gm.subject.values, n_boot=n_boot)
                row.update(gain=d.point, gain_lo=d.lo, gain_hi=d.hi)
            if meta and model in meta:
                row.update(meta[model])
            rows.append(row)
    out = pd.DataFrame(rows)
    return out.sort_values(["scale", "tier", "degrade", "psnr_y"], ascending=[True, True, True, False]).reset_index(drop=True)


def pivot(df: pd.DataFrame, tier: int, scale: int, kind: str, metric: str = "psnr_y", models=None):
    g = df[(df.tier == tier) & (df.scale == scale) & (df.degrade == kind)]
    if models is not None:
        g = g[g.model.isin(models)]
    tab = g.pivot(index="key", columns="model", values=metric).dropna(axis=0, how="any")
    subj = g.drop_duplicates("key").set_index("key").subject.reindex(tab.index).values
    return tab, subj


def rank_agreement(df: pd.DataFrame, tier: int, scale: int, kind_a: str, kind_b: str, models: list[str],
                   metric: str = "psnr_y", n_boot: int = 1000) -> dict:
    """τ-b có hòa giữa hai kiểu suy giảm, các cặp đổi chiều có ý nghĩa, và khoảng tin cậy của τ."""
    ta, sa = pivot(df, tier, scale, kind_a, metric, models)
    tb, sb = pivot(df, tier, scale, kind_b, metric, models)
    cols = [m for m in models if m in ta.columns and m in tb.columns]
    ta, tb = ta[cols], tb[cols]
    r = kendall_tau_b(pairwise_signs(ta, sa, n_boot=n_boot), pairwise_signs(tb, sb, n_boot=n_boot))
    common = ta.index.intersection(tb.index)
    ci = tau_bootstrap_ci(ta.loc[common], tb.loc[common], pd.Series(sa, index=ta.index).loc[common].values,
                          n_boot=min(500, n_boot))
    r.update(tau_mean_rank=ci["tau"], tau_lo=ci["lo"], tau_hi=ci["hi"], models=cols,
             order_a=list(ta.mean().sort_values(ascending=False).index),
             order_b=list(tb.mean().sort_values(ascending=False).index))
    # quy tắc quyết định của kế hoạch (mục 1.10, T2)
    r["flipped"] = bool(r["discordant"] >= 1 or (r["tau_b"] < 0.7 and r["tau_hi"] < 0.9))
    return r


def mde_table(df: pd.DataFrame, tier: int, scale: int, kind: str, pair: tuple[str, str],
              metrics=("psnr_y", "ssim_y"), n_list=(60, 100)) -> pd.DataFrame:
    """MDE cho một cặp mô hình đại diện, với 60 người (fold 2, 3, 4) và 100 người."""
    rows = []
    for metric in metrics:
        tab, subj = pivot(df, tier, scale, kind, metric, list(pair))
        if not all(p in tab.columns for p in pair):
            continue
        diff = (tab[pair[0]] - tab[pair[1]]).values
        for n in n_list:
            r = mde_paired(diff, subj, n)
            rows.append({"metric": metric, "pair": f"{pair[0]} - {pair[1]}", "n_subjects": n,
                         "mean_diff": float(np.mean(diff)), "sd_between_subjects": r["sd_between_subjects"],
                         "mde": r["mde"]})
    return pd.DataFrame(rows)
