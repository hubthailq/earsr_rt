"""Tổng hợp T4: so cổng oracle với đường trộn ở cùng PSNR, bootstrap theo người."""
from __future__ import annotations

import numpy as np
import pandas as pd


def _interp_blend(psnr_b: np.ndarray, met_b: np.ndarray, at: float) -> float:
    o = np.argsort(psnr_b)
    return float(np.interp(at, psnr_b[o], met_b[o]))


def summarize(df: pd.DataFrame, metric: str = "lpips", psnr_budget_db: float = 0.5, min_gain: float = 0.10,
              n_boot: int = 2000, seed: int = 0, lower_is_better: bool = True) -> dict:
    """``df``: mỗi dòng một (ảnh, method, param) với cột subject, key, psnr_y và ``metric``.

    Với mỗi điểm oracle nằm trong ngân sách PSNR (không thấp hơn f_S quá
    ``psnr_budget_db``; f_S là điểm trộn α = 1):
        gain = 1 − metric(oracle) / metric(đường trộn tại cùng PSNR)
    (đường trộn được nội suy tuyến tính theo PSNR; nếu PSNR của oracle cao hơn
    mọi điểm trộn thì so với điểm trộn có PSNR cao nhất và gắn cờ).
    Đạt khi có ít nhất một điểm oracle có gain ≥ ``min_gain`` và cận dưới khoảng
    tin cậy 95% lớn hơn 0.
    """
    if metric not in df.columns or df[metric].isna().all():
        return {"error": f"không có cột '{metric}' (thiếu số đo cảm nhận?)", "pass": None}
    if not lower_is_better:
        raise NotImplementedError("chỉ hỗ trợ số đo càng thấp càng tốt (LPIPS, DISTS)")
    d = df.copy()
    d["cfg"] = d.method + "@" + d.param.map(lambda v: f"{v:g}")
    cfgs = sorted(d.cfg.unique())
    subs = sorted(d.subject.astype(str).unique())
    si = {s: i for i, s in enumerate(subs)}
    ci = {c: i for i, c in enumerate(cfgs)}
    sp = np.zeros((len(subs), len(cfgs)))
    sm = np.zeros_like(sp)
    cnt = np.zeros_like(sp)
    np.add.at(sp, (d.subject.astype(str).map(si).values, d.cfg.map(ci).values), d.psnr_y.values)
    np.add.at(sm, (d.subject.astype(str).map(si).values, d.cfg.map(ci).values), d[metric].values)
    np.add.at(cnt, (d.subject.astype(str).map(si).values, d.cfg.map(ci).values), 1.0)
    if (cnt != cnt[:, :1]).any():
        return {"error": "các cấu hình không có cùng tập ảnh", "pass": None}
    blends = [c for c in cfgs if c.startswith("blend@")]
    oracles = [c for c in cfgs if c.startswith("oracle_")]
    if "blend@1" not in cfgs or len(blends) < 3:
        return {"error": "thiếu đường trộn (cần α = 1 và ít nhất 3 điểm)", "pass": None}
    bi = [ci[c] for c in blends]

    def gains(idx):
        p = sp[idx].sum(0) / cnt[idx].sum(0)
        m = sm[idx].sum(0) / cnt[idx].sum(0)
        ref = p[ci["blend@1"]]
        out = {}
        for c in oracles:
            j = ci[c]
            b = _interp_blend(p[bi], m[bi], p[j])
            out[c] = (1 - m[j] / b if b > 0 else 0.0, p[j], m[j], b, p[j] >= ref - psnr_budget_db,
                      p[j] > p[bi].max())
        return out, ref

    point, ref = gains(np.arange(len(subs)))
    rng = np.random.default_rng(seed)
    boots = {c: [] for c in oracles}
    for _ in range(n_boot):
        g, _ = gains(rng.integers(0, len(subs), len(subs)))
        for c in oracles:
            boots[c].append(g[c][0])
    rows = []
    for c in oracles:
        g, p, m, b, ok, above = point[c]
        lo, hi = np.quantile(boots[c], [0.025, 0.975])
        rows.append({"oracle": c, "psnr_y": p, metric: m, f"blend_{metric}_same_psnr": b, "gain": g, "gain_lo": lo,
                     "gain_hi": hi, "within_psnr_budget": bool(ok), "psnr_above_blend_range": bool(above),
                     "pass": bool(ok and g >= min_gain and lo > 0)})
    curve = [{"alpha": float(c.split("@")[1]), "psnr_y": float(sp[:, ci[c]].sum() / cnt[:, ci[c]].sum()),
              metric: float(sm[:, ci[c]].sum() / cnt[:, ci[c]].sum())} for c in blends]
    by_mode = {mode: any(r["pass"] for r in rows if r["oracle"].startswith(mode)) for mode in ("oracle_pixel", "oracle_window")}
    return {"metric": metric, "n_subjects": len(subs), "n_images": int(cnt[:, 0].sum()), "psnr_fidelity": float(ref),
            "psnr_budget_db": psnr_budget_db, "min_gain": min_gain, "blend_curve": sorted(curve, key=lambda r: r["alpha"]),
            "oracles": rows, "pass_by_oracle": by_mode, "pass": any(by_mode.values()),
            "decision": "giữ N3 làm ứng viên" if any(by_mode.values()) else "bỏ N3 (cả hai oracle không vượt đường trộn đủ ngưỡng)"}
