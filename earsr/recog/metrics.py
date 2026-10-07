"""Số đo nhận dạng: thứ hạng, EER, TAR ở FAR cố định; khoảng tin cậy bootstrap theo người."""
from __future__ import annotations

import numpy as np

BINS = 4000   # lưới điểm cosine trong [-1, 1] dùng cho EER; bước 0,0005


def templates(emb: np.ndarray, subjects: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mẫu đăng ký của mỗi người: trung bình các đặc trưng đã chuẩn hóa, rồi chuẩn hóa lại."""
    uniq = np.unique(subjects)
    t = np.stack([emb[subjects == s].mean(0) for s in uniq])
    return uniq, t / np.linalg.norm(t, axis=1, keepdims=True)


def score_probes(probe_emb: np.ndarray, probe_subj: np.ndarray, uniq: np.ndarray, temp: np.ndarray) -> dict:
    """Điểm cosine của từng ảnh dò với mẫu của mọi người. ``rank`` = 1 nghĩa là đúng người đứng đầu."""
    scores = probe_emb @ temp.T
    col = np.minimum(np.searchsorted(uniq, probe_subj), len(uniq) - 1)
    if not np.array_equal(uniq[col], probe_subj):
        raise ValueError("có ảnh dò của người không có mẫu đăng ký")
    genuine = scores[np.arange(len(scores)), col]
    rank = 1 + (scores > genuine[:, None]).sum(1)
    imp = scores.copy()
    imp[np.arange(len(scores)), col] = -np.inf
    return {"scores": scores, "col": col, "genuine": genuine, "rank": rank, "max_impostor": imp.max(1)}


def _hist(x: np.ndarray) -> np.ndarray:
    return np.histogram(np.clip(x, -1, 1), bins=BINS, range=(-1.0, 1.0))[0].astype(np.float64)


def _rates(hg: np.ndarray, hi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """FRR và FAR tại mỗi ngưỡng (biên trái của từng ô): nhận khi điểm >= ngưỡng."""
    frr = np.concatenate([[0.0], np.cumsum(hg)]) / max(hg.sum(), 1.0)
    far = 1.0 - np.concatenate([[0.0], np.cumsum(hi)]) / max(hi.sum(), 1.0)
    return frr, far


def eer_from_hist(hg: np.ndarray, hi: np.ndarray) -> float:
    frr, far = _rates(hg, hi)
    k = int(np.argmin(np.abs(frr - far)))
    return float((frr[k] + far[k]) / 2)


def tar_from_hist(hg: np.ndarray, hi: np.ndarray, far_target: float) -> float:
    frr, far = _rates(hg, hi)
    ok = np.where(far <= far_target)[0]
    return float(1.0 - frr[ok[0]]) if len(ok) else 0.0


def eer(genuine: np.ndarray, impostor: np.ndarray) -> float:
    return eer_from_hist(_hist(genuine), _hist(impostor))


def subject_hists(scores: np.ndarray, col: np.ndarray, clusters: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Biểu đồ điểm đúng người và sai người, gom theo người của ảnh dò: (uniq, hg[k, BINS], hi[k, BINS])."""
    uniq = np.unique(clusters)
    mask = np.zeros(scores.shape, bool)
    mask[np.arange(len(scores)), col] = True
    hg = np.stack([_hist(scores[clusters == s][mask[clusters == s]]) for s in uniq])
    hi = np.stack([_hist(scores[clusters == s][~mask[clusters == s]]) for s in uniq])
    return uniq, hg, hi


def boot_stat(parts: list, fn, n_boot: int = 1000, alpha: float = 0.05, seed: int = 0) -> dict:
    """Bootstrap theo người cho một số đo tính từ tổng theo người.

    ``parts``: danh sách mảng có trục đầu là người (cùng thứ tự). ``fn`` nhận các tổng đã lấy mẫu lại và trả về một số.
    """
    k = len(parts[0])
    idx = np.random.default_rng(seed).integers(0, k, size=(n_boot, k))
    point = fn(*[p.sum(0) for p in parts])
    boots = np.array([fn(*[p[i].sum(0) for p in parts]) for i in idx])
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return {"point": float(point), "lo": float(lo), "hi": float(hi), "n_clusters": k}
