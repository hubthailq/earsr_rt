"""So thứ hạng mô hình giữa hai điều kiện (T2): Kendall τ-b có hòa.

Cặp mô hình nào có khoảng tin cậy của chênh lệch chứa 0 trong một điều kiện thì
tính là hòa ở điều kiện đó, để thứ hạng không "đảo" chỉ vì nhiễu.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from .bootstrap import paired_diff


def pairwise_signs(table: pd.DataFrame, clusters, higher_is_better: bool = True, n_boot: int = 1000,
                   seed: int = 0) -> dict:
    """``table``: mỗi dòng một ảnh, mỗi cột một mô hình. Trả về {(m1, m2): -1 | 0 | +1}.

    +1: m1 tốt hơn m2 và khoảng tin cậy không chứa 0. 0: không tách được.
    """
    out = {}
    for m1, m2 in itertools.combinations(table.columns, 2):
        e = paired_diff(table[m1].values, table[m2].values, clusters, n_boot=n_boot, seed=seed)
        s = 0 if not e.excludes_zero() else int(np.sign(e.point))
        out[(m1, m2)] = s if higher_is_better else -s
    return out


def kendall_tau_b(signs_a: dict, signs_b: dict) -> dict:
    """τ-b giữa hai bộ dấu theo cặp, và danh sách cặp đổi chiều có ý nghĩa."""
    keys = sorted(set(signs_a) & set(signs_b))
    conc = disc = ta = tb = 0
    reversals = []
    for k in keys:
        x, y = signs_a[k], signs_b[k]
        if x == 0 and y == 0:
            continue
        if x == 0:
            ta += 1
        elif y == 0:
            tb += 1
        elif x == y:
            conc += 1
        else:
            disc += 1
            reversals.append(k)
    denom = np.sqrt((conc + disc + ta) * (conc + disc + tb))
    tau = float((conc - disc) / denom) if denom > 0 else float("nan")
    return {"tau_b": tau, "concordant": conc, "discordant": disc, "tied_a_only": ta, "tied_b_only": tb,
            "n_pairs": len(keys), "reversals": reversals}


def tau_bootstrap_ci(table_a: pd.DataFrame, table_b: pd.DataFrame, clusters, n_boot: int = 500,
                     alpha: float = 0.05, seed: int = 0) -> dict:
    """Khoảng tin cậy của τ (trên thứ hạng trung bình, không xét hòa) khi lấy mẫu lại người."""
    from scipy.stats import kendalltau

    clusters = np.asarray(clusters)
    uniq = np.unique(clusters)
    rows_of = [np.flatnonzero(clusters == u) for u in uniq]
    rng = np.random.default_rng(seed)
    cols = [c for c in table_a.columns if c in table_b.columns]
    A, B = table_a[cols].values, table_b[cols].values
    taus = []
    for _ in range(n_boot):
        idx = np.concatenate([rows_of[i] for i in rng.integers(0, len(uniq), len(uniq))])
        t = kendalltau(np.nanmean(A[idx], 0), np.nanmean(B[idx], 0)).statistic
        taus.append(t)
    taus = np.array(taus)
    point = kendalltau(np.nanmean(A, 0), np.nanmean(B, 0)).statistic
    lo, hi = np.nanquantile(taus, [alpha / 2, 1 - alpha / 2])
    return {"tau": float(point), "lo": float(lo), "hi": float(hi)}
