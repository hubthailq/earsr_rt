"""Bootstrap theo cụm (người) cho số đo theo từng ảnh.

Đơn vị lấy mẫu lại là người, không phải ảnh, vì các ảnh của cùng một người
không độc lập. Mọi tiêu chí đạt của kế hoạch được phát biểu bằng khoảng tin cậy
của chênh lệch ghép cặp tính ở đây.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Estimate:
    point: float
    lo: float
    hi: float
    p: float          # p hai phía theo bootstrap, cho giả thuyết "bằng 0"
    n_clusters: int
    n_obs: int

    def excludes_zero(self) -> bool:
        return self.lo > 0 or self.hi < 0

    def as_dict(self, prefix: str = "") -> dict:
        return {f"{prefix}point": self.point, f"{prefix}lo": self.lo, f"{prefix}hi": self.hi,
                f"{prefix}p": self.p, f"{prefix}n_clusters": self.n_clusters, f"{prefix}n_obs": self.n_obs}


def _cluster_sums(values: np.ndarray, clusters: np.ndarray):
    """Tổng và số quan sát theo cụm; trả về (uniq, sums[k, ...], counts[k])."""
    uniq, inv = np.unique(clusters, return_inverse=True)
    k = len(uniq)
    v = np.asarray(values, dtype=np.float64)
    if v.ndim == 1:
        v = v[:, None]
    sums = np.zeros((k, v.shape[1]))
    np.add.at(sums, inv, v)
    counts = np.bincount(inv, minlength=k).astype(np.float64)
    return uniq, sums, counts


def _boot_indices(k: int, n_boot: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, k, size=(n_boot, k))


def _summ(point: float, boots: np.ndarray, alpha: float, k: int, n: int) -> Estimate:
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    p = 2 * min((boots <= 0).mean(), (boots >= 0).mean())
    return Estimate(float(point), float(lo), float(hi), float(min(1.0, max(p, 1.0 / len(boots)))), k, n)


def mean_ci(values, clusters, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0) -> Estimate:
    """Trung bình theo ảnh, khoảng tin cậy bootstrap theo cụm."""
    values, clusters = np.asarray(values, float), np.asarray(clusters)
    ok = np.isfinite(values)
    values, clusters = values[ok], clusters[ok]
    _, sums, counts = _cluster_sums(values, clusters)
    idx = _boot_indices(len(counts), n_boot, seed)
    boots = sums[idx, 0].sum(1) / counts[idx].sum(1)
    return _summ(values.mean(), boots, alpha, len(counts), len(values))


def paired_diff(a, b, clusters, relative: bool = False, n_boot: int = 2000, alpha: float = 0.05,
                seed: int = 0) -> Estimate:
    """Chênh lệch ghép cặp theo từng ảnh giữa hai phương pháp, a trừ b.

    relative=False: trung bình của (a - b).
    relative=True : (mean a - mean b) / mean b, dùng cho các tiêu chí "giảm ít
    nhất x%". Tỉ số được tính lại trong từng lần lấy mẫu.
    """
    a, b, clusters = np.asarray(a, float), np.asarray(b, float), np.asarray(clusters)
    if a.shape != b.shape:
        raise ValueError("a và b phải cùng số ảnh, cùng thứ tự")
    ok = np.isfinite(a) & np.isfinite(b)
    a, b, clusters = a[ok], b[ok], clusters[ok]
    _, sums, counts = _cluster_sums(np.stack([a, b], 1), clusters)
    idx = _boot_indices(len(counts), n_boot, seed)
    sa, sb, n = sums[idx, 0].sum(1), sums[idx, 1].sum(1), counts[idx].sum(1)
    if relative:
        boots = (sa - sb) / sb
        point = (a.mean() - b.mean()) / b.mean()
    else:
        boots = (sa - sb) / n
        point = (a - b).mean()
    return _summ(point, boots, alpha, len(counts), len(a))


def interaction(a1, b1, c1, a2, b2, c2, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0,
                same_clusters: bool = True) -> Estimate:
    """Phép thử tương tác: (a1 - b1) trừ (a2 - b2).

    Ví dụ N5b (c): phần hơn của biến thể ở ảnh vào 36 px trừ phần hơn của nó ở
    ảnh vào lớn. ``same_clusters``: hai điều kiện dùng cùng tập người (lấy mẫu
    lại chung); nếu không, hai tập được lấy mẫu lại độc lập.
    """
    a1, b1, c1 = np.asarray(a1, float), np.asarray(b1, float), np.asarray(c1)
    a2, b2, c2 = np.asarray(a2, float), np.asarray(b2, float), np.asarray(c2)
    u1, s1, n1 = _cluster_sums(a1 - b1, c1)
    u2, s2, n2 = _cluster_sums(a2 - b2, c2)
    point = (a1 - b1).mean() - (a2 - b2).mean()
    if same_clusters:
        if list(u1) != list(u2):
            raise ValueError("same_clusters=True nhưng hai điều kiện có tập cụm khác nhau")
        idx1 = idx2 = _boot_indices(len(u1), n_boot, seed)
    else:
        idx1 = _boot_indices(len(u1), n_boot, seed)
        idx2 = _boot_indices(len(u2), n_boot, seed + 1)
    boots = s1[idx1, 0].sum(1) / n1[idx1].sum(1) - s2[idx2, 0].sum(1) / n2[idx2].sum(1)
    return _summ(point, boots, alpha, min(len(u1), len(u2)), len(a1) + len(a2))
