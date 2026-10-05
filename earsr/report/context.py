"""Tổng hợp phép thử ngữ cảnh: mất bao nhiêu dB khi không có ngữ cảnh, mất ở
vành nào, và đệm trước bằng lặp viền hay phản chiếu có bù được không."""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from ..stats.bootstrap import paired_diff

_NAME = re.compile(r"^(?P<model>.+)__(?P<dataset>[^_]+)__(?P<config>[a-z]+)\.csv$")


def load_dir(d: str | Path) -> pd.DataFrame:
    frames = [pd.read_csv(f, dtype={"subject": str}) for f in sorted(Path(d).glob("*.csv")) if _NAME.match(f.name)]
    if not frames:
        raise FileNotFoundError(f"không có file kết quả trong {d}")
    return pd.concat(frames, ignore_index=True)


def summarize(df: pd.DataFrame, n_boot: int = 2000) -> pd.DataFrame:
    """Một dòng cho mỗi (dataset, config, model).

    loss_db      : PSNR khi có ngữ cảnh đầy đủ trừ PSNR khi không có (m lớn nhất trừ m0), ghép cặp.
    recovered_at : bề rộng ngữ cảnh nhỏ nhất mà PSNR cách mức đầy đủ không quá 0,01 dB.
    ring0_ratio  : MSE ở vành sát mép khi không có ngữ cảnh chia cho khi có.
    inner_ratio  : cùng tỉ số ở vành trong cùng (từ 8 px trở vào).
    rep_db/ref_db: PSNR khi đệm trước bằng lặp viền / phản chiếu, trừ PSNR ở m0.
    """
    rows = []
    for (ds, cfg, model), g in df.groupby(["dataset", "config", "model"]):
        p = g.pivot(index="key", columns="cond", values="psnr_y")
        subj = g.drop_duplicates("key").set_index("key").subject.reindex(p.index).values
        ms = sorted((c for c in p.columns if re.match(r"^m\d+$", c)), key=lambda c: int(c[1:]))
        full = ms[-1]
        e = paired_diff(p[full].values, p["m0"].values, subj, n_boot=n_boot)
        rec = next((int(c[1:]) for c in ms if p[full].mean() - p[c].mean() <= 0.01), int(full[1:]))
        r0 = g.pivot(index="key", columns="cond", values="mse_ring0")
        r5 = g.pivot(index="key", columns="cond", values="mse_ring5")
        row = {"dataset": ds, "config": cfg, "model": model, "n": len(p), "psnr_full": p[full].mean(),
               "psnr_m0": p["m0"].mean(), "loss_db": e.point, "loss_lo": e.lo, "loss_hi": e.hi,
               "recovered_at_px": rec, "ring0_ratio": r0["m0"].mean() / r0[full].mean(),
               "inner_ratio": r5["m0"].mean() / r5[full].mean()}
        for c in p.columns:
            if c.startswith("rep"):
                row["rep_db"] = p[c].mean() - p["m0"].mean()
            if c.startswith("ref"):
                row["ref_db"] = p[c].mean() - p["m0"].mean()
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["dataset", "config", "loss_db"], ascending=[True, True, False])


def to_markdown(s: pd.DataFrame) -> str:
    lines = ["# T6 (i), (v): phép thử ngữ cảnh (không huấn luyện)", "",
             "Mất khi thiếu ngữ cảnh = PSNR-Y của cửa sổ khi mạng thấy ngữ cảnh thật đầy đủ, trừ PSNR-Y khi mạng "
             "chỉ thấy cửa sổ. Cùng cửa sổ, cùng điểm ảnh; so ghép cặp, bootstrap theo người (hoặc theo ảnh).", ""]
    for (ds, cfg), g in s.groupby(["dataset", "config"]):
        lines += [f"## {ds}, cấu hình {cfg} ({int(g.n.max())} ảnh)", "",
                  "| Mô hình | PSNR đủ ngữ cảnh | Mất khi thiếu ngữ cảnh (dB) [KTC 95%] | Hồi phục hết ở | "
                  "MSE vành sát mép (lần) | MSE vành trong (lần) | Đệm trước lặp viền (dB) | Đệm trước phản chiếu (dB) |",
                  "|---|---|---|---|---|---|---|---|"]
        for _, r in g.iterrows():
            lines.append(f"| {r.model} | {r.psnr_full:.2f} | {r.loss_db:.3f} [{r.loss_lo:.3f}, {r.loss_hi:.3f}] | "
                         f"{int(r.recovered_at_px)} px | {r.ring0_ratio:.2f} | {r.inner_ratio:.3f} | "
                         f"{r.get('rep_db', float('nan')):+.2f} | {r.get('ref_db', float('nan')):+.2f} |")
        lines.append("")
    return "\n".join(lines)
