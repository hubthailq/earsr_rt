#!/usr/bin/env python3
"""Tổng hợp T2: bảng chất lượng, so thứ hạng bic và bicjpeg75, MDE.

  python scripts/summarize_t2.py --in results/t2 --out results/t2_summary
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from earsr.models.registry import SPECS  # noqa: E402
from earsr.report.t2 import load_dir, mde_table, quality_table, rank_agreement  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    df = load_dir(a.inp)
    meta = {n: {"group": s.group, "objective": s.objective} for n, s in SPECS.items()}
    q = quality_table(df, meta, a.n_boot)
    q.to_csv(out / "quality.csv", index=False)

    lines = ["# T2: mô hình có sẵn trên benchmark AMI (không huấn luyện)", ""]
    for (scale, tier, kind), g in q.groupby(["scale", "tier", "degrade"]):
        lines += [f"## ×{scale}, đáp án {tier} px, suy giảm {kind} ({int(g.n.max())} ảnh)", "",
                  "| Mô hình | Nhóm | PSNR-Y (dB) [KTC 95%] | Hơn bicubic (dB) [KTC 95%] | SSIM-Y | PSNR-RGB | PSNR-Y vùng giữa |",
                  "|---|---|---|---|---|---|---|"]
        for _, r in g.iterrows():
            gain = "" if r.get("gain") != r.get("gain") else f"{r.gain:+.2f} [{r.gain_lo:+.2f}, {r.gain_hi:+.2f}]"
            grp = r.get("group") if isinstance(r.get("group"), str) else "mốc"
            lines.append(f"| {r.model} | {grp} | {r.psnr_y:.2f} [{r.psnr_y_lo:.2f}, {r.psnr_y_hi:.2f}] | {gain} | "
                         f"{r.ssim_y:.4f} | {r.psnr_rgb:.2f} | {r.psnr_y_center:.2f} |")
        lines.append("")

    # so thứ hạng giữa hai kiểu suy giảm, trong nhóm tối ưu PSNR (nhẹ + vừa)
    rank = {}
    psnr_models = [n for n, s in SPECS.items() if s.objective == "psnr" and s.group in ("light", "mid")]
    kinds = sorted(df.degrade.unique())
    if "bic" in kinds and "bicjpeg75" in kinds:
        lines += ["## Thứ hạng: bicubic so với bicubic + JPEG 75 (mô hình nhẹ và vừa, tối ưu PSNR)", ""]
        for (scale, tier), g in df.groupby(["scale", "tier"]):
            ms = [m for m in psnr_models if m in set(g.model) and SPECS[m].scale == scale]
            if len(ms) < 3 or not {"bic", "bicjpeg75"} <= set(g.degrade):
                continue
            r = rank_agreement(df, tier, scale, "bic", "bicjpeg75", ms, n_boot=min(1000, a.n_boot))
            rank[f"x{scale}_hr{tier}"] = {k: (v if not isinstance(v, list) or not v or not isinstance(v[0], tuple)
                                              else [list(t) for t in v]) for k, v in r.items()}
            lines += [f"**×{scale}, {tier} px.** τ-b (có hòa) = {r['tau_b']:.2f}; cặp cùng chiều {r['concordant']}, "
                      f"đổi chiều có ý nghĩa {r['discordant']}; τ trên thứ hạng trung bình = {r['tau_mean_rank']:.2f} "
                      f"[{r['tau_lo']:.2f}, {r['tau_hi']:.2f}]. Kết luận theo quy tắc ghi trước: "
                      f"**{'thứ hạng đảo' if r['flipped'] else 'thứ hạng không đảo'}**.", "",
                      f"- thứ tự ở bic: {', '.join(r['order_a'])}", f"- thứ tự ở bicjpeg75: {', '.join(r['order_b'])}"]
            if r["reversals"]:
                lines.append(f"- cặp đổi chiều: {', '.join(a_ + ' / ' + b_ for a_, b_ in r['reversals'])}")
            lines.append("")
    with open(out / "ranking.json", "w") as f:
        json.dump(rank, f, indent=1, default=str)

    # MDE trên ô chính
    main_cell = df[(df.tier == 144) & (df.scale == 4) & (df.degrade == "bic")]
    if not main_cell.empty and {"span_ch48", "rlfn"} <= set(main_cell.model):
        m = mde_table(df, 144, 4, "bic", ("span_ch48", "rlfn"))
        m.to_csv(out / "mde.csv", index=False)
        lines += ["## Mức chênh nhỏ nhất phát hiện được (ô chính, cặp span_ch48 và rlfn)", "",
                  "| Số đo | Số người | Độ lệch chuẩn giữa người | MDE (α = 0,05; lực 0,8) |", "|---|---|---|---|"]
        for _, r in m.iterrows():
            lines.append(f"| {r.metric} | {int(r.n_subjects)} | {r.sd_between_subjects:.4f} | {r.mde:.4f} |")
        lines += ["", "MDE này tính từ độ biến thiên của chênh lệch giữa hai mô hình có sẵn. Chênh lệch giữa hai "
                  "mô hình cùng thân, cùng fold thường biến thiên ít hơn, nên đây là ước lượng thận trọng.", ""]
    (out / "T2_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"đã ghi {out}/T2_summary.md, quality.csv, ranking.json")


if __name__ == "__main__":
    main()
