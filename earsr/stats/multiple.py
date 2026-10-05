"""Hiệu chỉnh Holm cho một tập so sánh nêu trước."""
from __future__ import annotations

import numpy as np


def holm(pvals) -> np.ndarray:
    """Trả về p đã hiệu chỉnh Holm (step-down), cùng thứ tự với đầu vào."""
    p = np.asarray(pvals, dtype=np.float64)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj
