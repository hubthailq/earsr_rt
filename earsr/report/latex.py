"""Sinh bảng LaTeX (booktabs) từ các bảng tổng hợp trong ``results/``.

Một lệnh (``scripts/make_tables.py``) dựng lại mọi bảng; không bảng nào được
gõ tay vào bản thảo.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

_ESC = {"_": r"\_", "%": r"\%", "&": r"\&", "#": r"\#"}


def esc(s) -> str:
    s = str(s)
    for k, v in _ESC.items():
        s = s.replace(k, v)
    return s


def fmt(v, digits: int = 2) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "--"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        return f"{v:.{digits}f}"
    return esc(v)


def booktabs(df: pd.DataFrame, caption: str, label: str, digits: dict | None = None, best: dict | None = None,
             align: str | None = None, header: dict | None = None, note: str | None = None,
             group_col: str | None = None) -> str:
    """``digits``: cột -> số chữ số thập phân. ``best``: cột -> 'max' | 'min', in đậm
    giá trị tốt nhất của cột (hòa thì đậm cả). ``header``: cột -> tên hiển thị.
    ``group_col``: chèn \\midrule mỗi khi giá trị cột này đổi (cột này không in)."""
    digits, best, header = digits or {}, best or {}, header or {}
    cols = [c for c in df.columns if c != group_col]
    align = align or ("l" + "r" * (len(cols) - 1))
    marks = {}
    for c, how in best.items():
        if c in df.columns:
            v = pd.to_numeric(df[c], errors="coerce").round(digits.get(c, 2))
            if v.notna().any():
                marks[c] = v == (v.max() if how == "max" else v.min())
    lines = [r"\begin{table}[t]", r"\centering", rf"\caption{{{caption}}}", rf"\label{{{label}}}",
             rf"\begin{{tabular}}{{{align}}}", r"\toprule",
             " & ".join(esc(header.get(c, c)) for c in cols) + r" \\", r"\midrule"]
    prev = None
    for i, (_, r) in enumerate(df.iterrows()):
        if group_col is not None and prev is not None and r[group_col] != prev:
            lines.append(r"\midrule")
        prev = r[group_col] if group_col is not None else None
        cells = []
        for c in cols:
            t = fmt(r[c], digits.get(c, 2))
            if c in marks and bool(marks[c].iloc[i]):
                t = rf"\textbf{{{t}}}"
            cells.append(t)
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    if note:
        lines.append(rf"\par\smallskip\footnotesize {note}")
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


def ci_cell(point: float, lo: float, hi: float, digits: int = 2) -> str:
    return f"{point:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"


def t2_table(quality: pd.DataFrame, tier: int, scale: int, kinds: tuple[str, ...] = ("bic", "bicjpeg75"),
             extra_cols: tuple[str, ...] = ()) -> str:
    """Bảng T2 cho một cỡ: mỗi dòng một mô hình, PSNR-Y và SSIM theo từng kiểu suy giảm."""
    q = quality[(quality.tier == tier) & (quality.scale == scale)]
    if q.empty:
        raise ValueError(f"không có dòng nào cho cỡ {tier}, ×{scale}")
    rows = None
    for k in kinds:
        cols = ["model", "group", "psnr_y", "ssim_y"] + [c for c in extra_cols if c in q.columns]
        part = q[q.degrade == k][cols].rename(columns={c: f"{c}_{k}" for c in cols if c not in ("model", "group")})
        rows = part if rows is None else rows.merge(part.drop(columns="group"), on="model", how="outer")
    order = {"baseline": 0, "light": 1, "mid": 2, "upper": 3, "perceptual": 4}
    rows["group"] = rows.group.fillna("baseline")
    rows = rows.sort_values(["group", f"psnr_y_{kinds[0]}"], key=lambda s: s.map(order) if s.name == "group" else -s)
    head = {"model": "Model"}
    dig, best = {}, {}
    for k in kinds:
        head[f"psnr_y_{k}"], head[f"ssim_y_{k}"] = f"PSNR ({k})", f"SSIM ({k})"
        dig[f"ssim_y_{k}"] = 4
        best[f"psnr_y_{k}"], best[f"ssim_y_{k}"] = "max", "max"
        for c in extra_cols:
            head[f"{c}_{k}"], dig[f"{c}_{k}"], best[f"{c}_{k}"] = f"{c.upper()} ({k})", 4, "min"
    return booktabs(rows.reset_index(drop=True), f"Published weights on the AMI benchmark, HR short side {tier} px, "
                    rf"$\times${scale}. PSNR and SSIM on the Y channel.", f"tab:t2_hr{tier}_x{scale}", dig, best,
                    header=head, group_col="group")


def efficiency_table(lat: pd.DataFrame) -> str:
    """Bảng hiệu quả từ file độ trễ (bench_local.py hoặc bench_trtexec.py)."""
    cols = [c for c in ["name", "params", "flops_g_256", "median_ms", "p95_ms"] if c in lat.columns]
    d = lat[cols].copy()
    if "params" in d:
        d["params"] = pd.to_numeric(d.params, errors="coerce") / 1e3
    return booktabs(d, "Efficiency: parameters (K), FLOPs (G, at $256\\times256$) and latency (ms).", "tab:efficiency",
                    {"params": 0, "flops_g_256": 2, "median_ms": 2, "p95_ms": 2},
                    {"median_ms": "min"}, header={"name": "Model", "params": "Params (K)", "flops_g_256": "FLOPs (G)",
                                                  "median_ms": "Median (ms)", "p95_ms": "p95 (ms)"})
