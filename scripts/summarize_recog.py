#!/usr/bin/env python3
"""Tổng hợp kết quả của recog_eval.py: rank-1, rank-5, EER, TAR ở FAR 1% theo nhánh, và chênh lệch ghép cặp.

  python scripts/summarize_recog.py --in results/recog/resnet18 --out results/recog/resnet18_summary

Các lần chạy cùng thân, cùng kiểu suy giảm huấn luyện (khác fold) được gộp thành một nhánh: với rank-1 lấy trung bình
độ đúng của các mô hình theo từng ảnh dò; với EER và TAR tính riêng từng mô hình rồi lấy trung bình. Mọi khoảng tin
cậy là bootstrap theo người (đơn vị lấy mẫu lại là người của ảnh dò; mẫu đăng ký giữ nguyên).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from earsr.recog.metrics import boot_stat, eer_from_hist, subject_hists, tar_from_hist  # noqa: E402
from earsr.report.arms import arm_of  # noqa: E402
from earsr.stats.bootstrap import mean_ci, paired_diff  # noqa: E402


LISTS = ("scan.csv", "probes.csv", "refs.csv", "gallery.csv")


def load_arms(folder: Path) -> dict:
    """nhánh -> danh sách mô hình; mỗi mô hình là dict(df, scores)."""
    arms = {}
    for f in sorted(folder.glob("*.csv")):
        if f.name in LISTS:
            continue
        df = pd.read_csv(f, dtype={"subject": str})
        arms.setdefault(arm_of(f.stem)[0], []).append({"name": f.stem, "df": df, "scores": np.load(f.with_suffix(".npy"))})
    return arms


def arm_stats(models: list[dict], subjects: list[str], n_boot: int) -> dict:
    df = models[0]["df"]
    for m in models[1:]:
        if not m["df"]["subject"].equals(df["subject"]):
            raise ValueError(f"{m['name']}: danh sách ảnh dò khác các mô hình cùng nhánh")
    clusters = df["subject"].to_numpy()
    col = np.searchsorted(np.array(subjects), clusters)
    hit1 = np.mean([(m["df"]["rank"] == 1).to_numpy(float) for m in models], axis=0)
    hit5 = np.mean([(m["df"]["rank"] <= 5).to_numpy(float) for m in models], axis=0)
    parts = []
    for m in models:
        _, hg, hi = subject_hists(m["scores"], col, clusters)
        parts += [hg, hi]
    k = len(models)
    return {"n_models": k, "n_probe": len(df), "clusters": clusters, "hit1": hit1, "hit5": hit5, "parts": parts,
            "rank1": mean_ci(hit1, clusters, n_boot), "rank5": mean_ci(hit5, clusters, n_boot),
            "eer": boot_stat(parts, lambda *p: float(np.mean([eer_from_hist(p[2 * i], p[2 * i + 1]) for i in range(k)])),
                             n_boot),
            "tar": boot_stat(parts, lambda *p: float(np.mean([tar_from_hist(p[2 * i], p[2 * i + 1], 0.01)
                                                              for i in range(k)])), n_boot)}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--baselines", nargs="+", default=["bicubic", "direct", "span_ch48 (published)", "span/est"])
    ap.add_argument("--n-boot", type=int, default=1000)
    a = ap.parse_args(argv)
    inp, out = Path(a.inp), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = json.loads((inp / "meta.json").read_text())
    st = {arm: arm_stats(ms, meta["subjects"], a.n_boot) for arm, ms in load_arms(inp).items()}
    rows = []
    for arm, s in st.items():
        rows.append({"arm": arm, "n_models": s["n_models"], "n_probe": s["n_probe"], "n_subjects": s["rank1"].n_clusters,
                     "rank1": 100 * s["rank1"].point, "rank1_lo": 100 * s["rank1"].lo, "rank1_hi": 100 * s["rank1"].hi,
                     "rank5": 100 * s["rank5"].point, "rank5_lo": 100 * s["rank5"].lo, "rank5_hi": 100 * s["rank5"].hi,
                     "eer": 100 * s["eer"]["point"], "eer_lo": 100 * s["eer"]["lo"], "eer_hi": 100 * s["eer"]["hi"],
                     "tar_far1": 100 * s["tar"]["point"], "tar_far1_lo": 100 * s["tar"]["lo"],
                     "tar_far1_hi": 100 * s["tar"]["hi"]})
    summ = pd.DataFrame(rows).sort_values("rank1", ascending=False)
    summ.to_csv(out / "summary.csv", index=False)
    pairs = []
    for base in a.baselines:
        if base not in st:
            continue
        b = st[base]
        for arm, s in st.items():
            if arm in (base, "ref_large") or base == "ref_large" or s["n_probe"] != b["n_probe"]:
                continue
            d1 = paired_diff(s["hit1"], b["hit1"], s["clusters"], n_boot=a.n_boot)
            ka, kb = s["n_models"], b["n_models"]

            def eer_gap(*p, ka=ka, kb=kb):
                ea = np.mean([eer_from_hist(p[2 * i], p[2 * i + 1]) for i in range(ka)])
                eb = np.mean([eer_from_hist(p[2 * (ka + i)], p[2 * (ka + i) + 1]) for i in range(kb)])
                return float(ea - eb)

            de = boot_stat(s["parts"] + b["parts"], eer_gap, a.n_boot)
            pairs.append({"arm": arm, "minus": base, "d_rank1": 100 * d1.point, "d_rank1_lo": 100 * d1.lo,
                          "d_rank1_hi": 100 * d1.hi, "d_rank1_p": d1.p, "d_eer": 100 * de["point"],
                          "d_eer_lo": 100 * de["lo"], "d_eer_hi": 100 * de["hi"]})
    pd.DataFrame(pairs).to_csv(out / "paired.csv", index=False)
    lines = [f"# Nhận dạng tai trên ảnh nhỏ thật ({', '.join(meta['role'])}): {len(meta['subjects'])} người, "
             f"{meta['n_probe']} ảnh dò, {meta['n_gallery']} ảnh đăng ký", "",
             f"Mạng nhận dạng: `{meta['recognizer']}`. Khoảng tin cậy 95%, bootstrap theo người. Đơn vị: %.", "",
             "| Nhánh | Số mô hình | Rank-1 | Rank-5 | EER | TAR ở FAR 1% |", "|---|---|---|---|---|---|"]
    for r in summ.to_dict("records"):
        lines.append(f"| {r['arm']} | {r['n_models']} | {r['rank1']:.2f} [{r['rank1_lo']:.2f}; {r['rank1_hi']:.2f}] | "
                     f"{r['rank5']:.2f} | {r['eer']:.2f} [{r['eer_lo']:.2f}; {r['eer_hi']:.2f}] | {r['tar_far1']:.2f} |")
    for base in a.baselines:
        sub = [p for p in pairs if p["minus"] == base]
        if not sub:
            continue
        lines += ["", f"## Chênh lệch ghép cặp so với `{base}` (điểm phần trăm; dấu * là khoảng tin cậy không chứa 0)", "",
                  "| Nhánh | Rank-1 | EER |", "|---|---|---|"]
        for p in sorted(sub, key=lambda p: -p["d_rank1"]):
            s1 = "*" if p["d_rank1_lo"] * p["d_rank1_hi"] > 0 else ""
            s2 = "*" if p["d_eer_lo"] * p["d_eer_hi"] > 0 else ""
            lines.append(f"| {p['arm']} | {p['d_rank1']:+.2f} [{p['d_rank1_lo']:+.2f}; {p['d_rank1_hi']:+.2f}]{s1} | "
                         f"{p['d_eer']:+.2f} [{p['d_eer_lo']:+.2f}; {p['d_eer_hi']:+.2f}]{s2} |")
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
