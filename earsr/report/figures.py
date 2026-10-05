"""Hình của bài (mục 2.5 của kế hoạch), vẽ từ các file trong ``results/``.

Quy ước: PDF vectơ, cỡ một cột (3,4 inch) hoặc hai cột (7 inch) của bài; bảng màu
Okabe-Ito (phân biệt được với người mù màu), mỗi đường còn khác nhau ở kiểu điểm
đánh dấu nên in đen trắng vẫn đọc được; không tiêu đề trong hình (chú thích nằm
ở bài); chữ tiếng Anh.
"""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

GROUP_ORDER = ["light", "mid", "upper", "perceptual", "interpolation", "ours"]
OKABE_ITO = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000", "#F0E442"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
COL1, COL2 = 3.4, 7.0


def _style() -> None:
    plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7,
                         "ytick.labelsize": 7, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5, "lines.linewidth": 1.3,
                         "lines.markersize": 4, "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02})


def _save(fig, out: str | Path) -> Path:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    if out.suffix == ".pdf":
        fig.savefig(out.with_suffix(".png"), dpi=200)
    plt.close(fig)
    return out


def _pick(models_all: list[str], models: list[str] | None, limit: int = 8) -> list[str]:
    if models:
        miss = [m for m in models if m not in models_all]
        if miss:
            raise ValueError(f"không có mô hình {miss} trong dữ liệu; có: {sorted(models_all)}")
        return list(models)
    return list(models_all)[:limit]


def fig_size_sweep(summary: pd.DataFrame, out: str | Path, models: list[str] | None = None,
                   seen: tuple[int, int] | None = (24, 48), labels: dict | None = None, relative_to: str | None = "bicubic") -> Path:
    """Hình 2: chất lượng theo cỡ ảnh vào. ``summary``: cột model, lr_short, psnr_y, lo, hi
    (``run_size_sweep.py``). ``relative_to``: vẽ phần hơn so với mô hình này (mặc định
    bicubic), vì PSNR tuyệt đối đổi mạnh theo cỡ và che mất khoảng cách giữa các mô hình;
    None để vẽ PSNR tuyệt đối. ``seen``: dải cỡ đã thấy khi huấn luyện, được tô nền."""
    _style()
    labels = labels or {}
    order = list(summary.groupby("model").psnr_y.mean().sort_values(ascending=False).index)
    use = _pick(order, models)
    fig, ax = plt.subplots(figsize=(COL1, 2.5))
    base = None
    if relative_to is not None:
        if relative_to not in set(summary.model):
            raise ValueError(f"không có '{relative_to}' để làm mốc")
        base = summary[summary.model == relative_to].set_index("lr_short").psnr_y
        use = [m for m in use if m != relative_to]
    for i, m in enumerate(use):
        g = summary[summary.model == m].sort_values("lr_short")
        off = base.reindex(g.lr_short).values if base is not None else 0.0
        c = OKABE_ITO[i % len(OKABE_ITO)]
        ax.plot(g.lr_short, g.psnr_y - off, color=c, marker=MARKERS[i % len(MARKERS)], label=labels.get(m, m))
        if base is None and {"lo", "hi"} <= set(g.columns):
            ax.fill_between(g.lr_short, g.lo, g.hi, color=c, alpha=0.15, linewidth=0)
    if seen:
        ax.axvspan(seen[0], seen[1], color="0.85", zorder=0, linewidth=0)
        ax.text(sum(seen) / 2, ax.get_ylim()[1], "sizes seen in training", ha="center", va="top", fontsize=6, color="0.35")
    ax.set_xlabel("Input short side (px)")
    ax.set_ylabel("PSNR-Y (dB)" if base is None else f"PSNR-Y gain over {labels.get(relative_to, relative_to)} (dB)")
    ax.legend(frameon=False)
    return _save(fig, out)


def fig_quality_vs_cost(quality: pd.DataFrame, cost: pd.DataFrame, out: str | Path, tier: int = 144, scale: int = 4,
                        kinds: tuple[str, ...] = ("bic", "bicjpeg75"), metric: str = "psnr_y", cost_col: str = "median_ms",
                        cost_label: str = "Latency (ms, median)", models: list[str] | None = None,
                        labels: dict | None = None, baseline: str = "bicubic", threshold: float | None = None) -> Path:
    """Hình 3 và 5: chất lượng theo chi phí (độ trễ hoặc số tham số), một ô cho mỗi kiểu suy giảm.
    ``quality``: bảng ``quality.csv``. ``cost``: bảng có cột name (hoặc model) và ``cost_col``.
    Mốc nội suy (``baseline``) được vẽ thành đường ngang. ``threshold``: vạch dọc ngưỡng real-time."""
    _style()
    labels = labels or {}
    cost = cost.rename(columns={"name": "model"}) if "name" in cost.columns else cost
    q = quality[(quality.tier == tier) & (quality.scale == scale)]
    kinds = [k for k in kinds if k in set(q.degrade)]
    if not kinds:
        raise ValueError(f"không có dữ liệu cho cỡ {tier}, ×{scale}")
    fig, axes = plt.subplots(1, len(kinds), figsize=(COL1 * len(kinds), 2.6), squeeze=False)
    fig.subplots_adjust(wspace=0.32)
    seen_groups: dict = {}
    for ax, kind in zip(axes[0], kinds):
        d = q[q.degrade == kind].merge(cost[["model", cost_col]], on="model", how="inner")
        d = d[d.model != baseline]
        if models:
            d = d[d.model.isin(models)]
        if d.empty:
            raise ValueError(f"không mô hình nào có cả số đo lẫn '{cost_col}' ở kiểu {kind}")
        grp = d["group"].fillna("interpolation") if "group" in d.columns else pd.Series("model", index=d.index)
        for name in [x for x in GROUP_ORDER if x in set(grp)] + sorted(set(grp) - set(GROUP_ORDER)):
            i = GROUP_ORDER.index(name) if name in GROUP_ORDER else len(GROUP_ORDER)
            g = d[grp == name]
            ax.scatter(g[cost_col], g[metric], color=OKABE_ITO[i % 8], marker=MARKERS[i % 8], s=18, zorder=3)
            seen_groups[name] = i
        for _, r in d.iterrows():
            ax.annotate(labels.get(r.model, r.model), (r[cost_col], r[metric]), xytext=(3, 3), textcoords="offset points",
                        fontsize=5.5)
        b = q[(q.degrade == kind) & (q.model == baseline)]
        if len(b):
            y = float(b[metric].iloc[0])
            ax.axhline(y, color="0.3", linestyle="--", linewidth=0.9)
            ax.annotate(labels.get(baseline, baseline), (0.01, y), xycoords=("axes fraction", "data"), xytext=(0, 2),
                        textcoords="offset points", fontsize=6, color="0.3")
        if threshold is not None:
            ax.axvline(threshold, color="0.3", linestyle=":", linewidth=0.9)
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel(cost_label)
        ax.set_ylabel({"psnr_y": "PSNR-Y (dB)", "lpips": "LPIPS"}.get(metric, metric))
        ax.set_title({"bic": "Bicubic", "bicjpeg75": "Bicubic + JPEG 75", "est": "Estimated"}.get(kind, kind), fontsize=8)
    if len(seen_groups) > 1:
        from matplotlib.lines import Line2D

        hs = [Line2D([], [], linestyle="", marker=MARKERS[i % 8], color=OKABE_ITO[i % 8], markersize=4, label=n)
              for n, i in sorted(seen_groups.items(), key=lambda kv: kv[1])]
        fig.legend(handles=hs, frameon=False, ncol=len(hs), loc="lower center", bbox_to_anchor=(0.5, 1.0))
    return _save(fig, out)


def context_curve(per_image: pd.DataFrame) -> pd.DataFrame:
    """Từ số đo theo ảnh của phép thử ngữ cảnh: phần PSNR mất đi (so với ngữ cảnh đầy đủ)
    theo bề rộng ngữ cảnh, cho mỗi (dataset, config, model)."""
    rows = []
    for (ds, cfg, model), g in per_image.groupby(["dataset", "config", "model"]):
        p = g.pivot(index="key", columns="cond", values="psnr_y")
        ms = sorted((c for c in p.columns if re.match(r"^m\d+$", c)), key=lambda c: int(c[1:]))
        full = p[ms[-1]]
        for c in ms:
            rows.append({"dataset": ds, "config": cfg, "model": model, "context_px": int(c[1:]),
                         "loss_db": float((full - p[c]).mean()), "n": len(p)})
    return pd.DataFrame(rows)


def fig_context(per_image: pd.DataFrame, out: str | Path, config: str = "wide", models: list[str] | None = None,
                labels: dict | None = None) -> Path:
    """Hình 4: phần PSNR mất theo bề rộng ngữ cảnh thật quanh cửa sổ, một ô cho mỗi bộ ảnh."""
    _style()
    labels = labels or {}
    cur = context_curve(per_image)
    cur = cur[cur.config == config]
    if cur.empty:
        raise ValueError(f"không có cấu hình '{config}'")
    sets = list(dict.fromkeys(cur.dataset))
    fig, axes = plt.subplots(1, len(sets), figsize=(COL1 * len(sets), 2.4), squeeze=False, sharey=True)
    for ax, ds in zip(axes[0], sets):
        d = cur[cur.dataset == ds]
        order = list(d[d.context_px == 0].sort_values("loss_db", ascending=False).model)
        for i, m in enumerate(_pick(order, [x for x in (models or []) if x in order] or None)):
            g = d[d.model == m].sort_values("context_px")
            ax.plot(g.context_px, g.loss_db, color=OKABE_ITO[i % 8], marker=MARKERS[i % 8], label=labels.get(m, m))
        ax.axhline(0, color="0.3", linewidth=0.6)
        ax.set_xlabel("Real context around the window (LR px)")
        ax.set_title(labels.get(ds, ds), fontsize=8)
    axes[0][0].set_ylabel("PSNR-Y lost vs. full context (dB)")
    axes[0][-1].legend(frameon=False)
    return _save(fig, out)


def fig_grid(columns: dict, keys: list[str], out: str | Path, zoom_lr: int | None = None, labels: dict | None = None,
             crop: tuple[int, int, int, int] | None = None) -> Path:
    """Hình 1 và 6: lưới so sánh định tính. ``columns``: tên cột -> thư mục chứa ``<khóa>.png``
    (thứ tự cột theo thứ tự dict; ví dụ LR, bicubic, mốc, đề xuất, HR). ``zoom_lr``: phóng
    cột đầu (ảnh LR) lên ngần này lần bằng lặp điểm ảnh. ``crop`` = (trên, trái, dưới, phải)
    theo toạ độ ảnh HR, áp cho mọi cột. Ảnh hiển thị không nội suy."""
    import cv2

    _style()
    labels = labels or {}
    names = list(columns)
    fig, axes = plt.subplots(len(keys), len(names), figsize=(1.25 * len(names), 1.25 * 1.42 * len(keys)), squeeze=False)
    for r, key in enumerate(keys):
        for c, name in enumerate(names):
            f = Path(columns[name]) / f"{key}.png"
            img = cv2.imread(str(f), cv2.IMREAD_COLOR)
            if img is None:
                raise FileNotFoundError(f"thiếu ảnh {f}")
            img = img[:, :, ::-1]
            if c == 0 and zoom_lr:
                img = np.repeat(np.repeat(img, zoom_lr, 0), zoom_lr, 1)
            if crop is not None:
                t, l, b, rr = crop
                img = img[t:b, l:rr]
            ax = axes[r][c]
            ax.imshow(img, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            for s in ax.spines.values():
                s.set_visible(False)
            if r == 0:
                ax.set_title(labels.get(name, name), fontsize=7)
    fig.subplots_adjust(wspace=0.03, hspace=0.03)
    return _save(fig, out)
