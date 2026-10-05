"""Mức chênh nhỏ nhất phát hiện được (MDE) cho một so sánh ghép cặp.

Dùng sau T2 để chỉnh ngưỡng một lần trước khi huấn luyện: ngưỡng nào nhỏ hơn
MDE thì phải nâng lên (mục 1.10 của kế hoạch).
"""
from __future__ import annotations

from statistics import NormalDist

import numpy as np


def mde_paired(diff, clusters, n_subjects: int | None = None, alpha: float = 0.05, power: float = 0.8) -> dict:
    """``diff``: chênh lệch ghép cặp theo ảnh của một cặp phương pháp đại diện.

    Tính độ lệch chuẩn của chênh lệch trung bình theo người, rồi ngoại suy sai
    số chuẩn cho ``n_subjects`` người. MDE = (z_{1-α/2} + z_{power}) × SE.
    """
    diff, clusters = np.asarray(diff, float), np.asarray(clusters)
    ok = np.isfinite(diff)
    diff, clusters = diff[ok], clusters[ok]
    uniq = np.unique(clusters)
    per = np.array([diff[clusters == u].mean() for u in uniq])
    sd = float(per.std(ddof=1))
    n = int(n_subjects or len(uniq))
    se = sd / np.sqrt(n)
    z = NormalDist().inv_cdf(1 - alpha / 2) + NormalDist().inv_cdf(power)
    return {"sd_between_subjects": sd, "n_subjects": n, "se": float(se), "mde": float(z * se),
            "alpha": alpha, "power": power}
