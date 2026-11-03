#!/usr/bin/env python3
"""Phép thử khả thi của luật chọn lúc chạy (góp ý ngày 04/11/2026; TODO.md, việc N): với mỗi ảnh vào đã nén, chọn giữa
bicubic và một mô hình công bố học bằng bicubic, chỉ dựa trên thông tin có lúc chạy (mức JPEG đọc từ file và một thống kê
độ sắc nét của chính ảnh nhỏ), không dùng ảnh gốc.

Các luật so sánh (chọn tham số trên tập hiệu chỉnh, thử trên tập giữ riêng):
  model   luôn dùng mô hình công bố
  bicubic luôn dùng bicubic
  jpeg    chỉ dựa trên mức JPEG: với mỗi mức, chọn bên có PSNR trung bình cao hơn trên tập hiệu chỉnh
  ratio   ước lượng tỉ số E_c/E_i: E_c theo mức JPEG (trung bình trên tập hiệu chỉnh), E_i từ sai số tự nội suy của ảnh
          nhỏ (log E_i hồi quy tuyến tính theo log của thống kê đó); dùng mô hình khi tỉ số ước lượng dưới một ngưỡng chung
  qc      theo từng mức JPEG, một ngưỡng riêng trên thống kê độ sắc nét
  oracle  biết trước bên nào tốt hơn cho từng ảnh (giới hạn trên)

Thống kê độ sắc nét ("sai số tự nội suy"): MSE trên kênh Y giữa ảnh nhỏ và ảnh nhỏ thu nhỏ ×0,5 rồi phóng lại ×2 bằng
bicubic. Ảnh gốc chỉ dùng để chấm (PSNR đã có trong results/t2*) và để hiệu chỉnh.

Tập hiệu chỉnh: người của fold 2, 3, 4 (AMI) và ảnh 0801 đến 0850 (DIV2K). Tập thử: người của fold 1, 5 (AMI), ảnh 0851 đến
0900 (DIV2K), và toàn bộ EarVN1.0, AWEx (không dùng để hiệu chỉnh). Ba kịch bản hiệu chỉnh: trên AMI, trên DIV2K, trên cả
hai. Tiêu chí chốt trước: luật dùng độ sắc nét có ích nếu hơn luật chỉ dựa trên mức JPEG một cách có ý nghĩa (khoảng tin
cậy 95%, bootstrap theo người) trên các tập thử.

Cần ảnh nhỏ trong data/bench (dựng lại được bằng build_benchmark.py, build_wild.py và build_lr_set, giống trên labai217).

  python3 scripts/analyze_selector.py                    # mô hình span_ch48; ghi results/selector/
  python3 scripts/analyze_selector.py --model rrdb_psnr
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from earsr.data.resize import imresize  # noqa: E402
from earsr.io import imread_rgb  # noqa: E402
from earsr.stats.bootstrap import paired_diff  # noqa: E402

SETS = {  # tên: (thư mục kết quả, thư mục benchmark, các cỡ, các kiểu ảnh vào)
    "AMI": ("t2", "ami", (96, 144), ("bic", "bicjpeg93", "bicjpeg85", "bicjpeg75", "bicjpeg60")),
    "DIV2K": ("t2_div2k", "div2k", (96, 144, 192), ("bic", "bicjpeg93", "bicjpeg85", "bicjpeg75", "bicjpeg60")),
    "EarVN1.0": ("t2_earvn", "earvn", (96,), ("bic", "bicjpeg93", "bicjpeg75")),
    "AWEx": ("t2_awex", "awex_s20", (96, 144), ("bic", "bicjpeg93", "bicjpeg75")),
}
RULES = ("model", "bicubic", "jpeg", "ratio", "qc", "oracle")


def qual(kind: str) -> int:
    return 100 if kind == "bic" else int(kind.replace("bicjpeg", ""))


def y_of(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)[..., 0].astype(np.float64)


def self_interp(lr: np.ndarray) -> float:
    """MSE kênh Y giữa ảnh nhỏ và bản thu nhỏ ×0,5 rồi phóng ×2 của nó (cắt về cỡ chẵn)."""
    h, w = lr.shape[0] // 2 * 2, lr.shape[1] // 2 * 2
    x = lr[:h, :w]
    back = imresize(imresize(x, 0.5), 2.0)
    return float(np.mean((y_of(x) - y_of(back)) ** 2))


def load(res: Path, bench: Path, model: str) -> pd.DataFrame:
    rows = []
    for name, (rdir, bdir, tiers, kinds) in SETS.items():
        for t in tiers:
            clean = None
            for k in kinds:
                fb, fm = res / rdir / f"bicubic__hr{t}_x4_{k}.csv", res / rdir / f"{model}__hr{t}_x4_{k}.csv"
                if not (fb.is_file() and fm.is_file()):
                    print(f"bỏ qua {name} {t} {k}: thiếu {fb.name if not fb.is_file() else fm.name}")
                    continue
                b = pd.read_csv(fb, dtype={"subject": str, "view": str}).set_index("key")
                m = pd.read_csv(fm, dtype={"subject": str, "view": str}).set_index("key")
                lrdir = bench / bdir / "lr" / f"hr{t}_x4_{k}"
                for key in b.index.intersection(m.index):
                    rows.append({"dataset": name, "tier": t, "kind": k, "q": qual(k), "key": key,
                                 "subject": f"{name}:{b.loc[key, 'subject']}", "fold": int(b.loc[key, "fold"]),
                                 "p_bic": float(b.loc[key, "psnr_y"]), "p_model": float(m.loc[key, "psnr_y"]),
                                 "s_self": self_interp(imread_rgb(lrdir / f"{key}.png"))})
    d = pd.DataFrame(rows)
    d["d"] = d.p_model - d.p_bic
    mse = lambda p: 255.0 ** 2 * 10 ** (-p / 10)   # noqa: E731
    d["mse_bic"] = mse(d.p_bic)
    ei = d[d.kind == "bic"].set_index(["dataset", "tier", "key"]).mse_bic.rename("E_i")
    d = d.join(ei, on=["dataset", "tier", "key"])
    d["E_c"] = (d.mse_bic - d.E_i).clip(lower=0)
    d["ratio_true"] = d.E_c / d.E_i
    return d


def split(d: pd.DataFrame, folds_path: Path) -> pd.Series:
    """'dev' (hiệu chỉnh) hay 'test' cho từng dòng."""
    fj = json.loads(folds_path.read_text())["folds"]
    dev_ami = {s for f in ("2", "3", "4") for s in fj[f]["test"]}
    sub = d.subject.str.split(":", n=1).str[1]
    num = pd.to_numeric(sub.str[:4], errors="coerce")
    is_dev = ((d.dataset == "AMI") & sub.isin(dev_ami)) | ((d.dataset == "DIV2K") & (num <= 850))
    return pd.Series(np.where(is_dev, "dev", "test"), index=d.index)


def best_threshold(x: np.ndarray, d: np.ndarray, below: bool) -> float:
    """Ngưỡng trên x làm tổng chênh lệch d lớn nhất khi chọn mô hình với x < ngưỡng (below) hoặc x > ngưỡng."""
    o = np.argsort(x)
    xs, ds = x[o], d[o]
    cs = np.concatenate([[0.0], np.cumsum(ds)])          # cs[i]: tổng d của i phần tử có x nhỏ nhất
    gains = cs if below else cs[-1] - cs
    cuts = np.concatenate([[-np.inf], (xs[:-1] + xs[1:]) / 2, [np.inf]])
    return float(cuts[int(np.argmax(gains))])


class Rules:
    """Các luật chọn, hiệu chỉnh trên ``cal``."""

    def __init__(self, cal: pd.DataFrame):
        self.q_pick = cal.groupby("q").d.mean().gt(0).to_dict()            # True: dùng mô hình ở mức này
        # E_i từ sai số tự nội suy (ảnh sạch của tập hiệu chỉnh), E_c theo mức JPEG
        c = cal[cal.kind == "bic"]
        self.b, self.a = np.polyfit(np.log(c.s_self), np.log(c.E_i), 1)
        self.ec = cal.groupby("q").E_c.mean().to_dict()
        r = self.r_hat(cal)
        self.tau = best_threshold(r, cal.d.to_numpy(), below=True)
        self.qc = {}
        for q, g in cal.groupby("q"):
            x, dd = np.log(g.s_self.to_numpy()), g.d.to_numpy()
            t_hi, t_lo = best_threshold(x, dd, below=False), best_threshold(x, dd, below=True)
            gain = lambda thr, below: dd[(x < thr) if below else (x > thr)].sum()   # noqa: E731
            self.qc[q] = (t_hi, False) if gain(t_hi, False) >= gain(t_lo, True) else (t_lo, True)

    def ei_hat(self, d: pd.DataFrame) -> np.ndarray:
        return np.exp(self.a + self.b * np.log(d.s_self.to_numpy()))

    def r_hat(self, d: pd.DataFrame) -> np.ndarray:
        ec = d.q.map(lambda q: self.ec.get(q, np.nan)).to_numpy()
        return ec / self.ei_hat(d)

    def choose(self, d: pd.DataFrame, rule: str) -> np.ndarray:
        """True: dùng mô hình."""
        if rule == "model":
            return np.ones(len(d), bool)
        if rule == "bicubic":
            return np.zeros(len(d), bool)
        if rule == "oracle":
            return d.d.to_numpy() > 0
        if rule == "jpeg":
            return d.q.map(lambda q: self.q_pick.get(q, True)).to_numpy(bool)
        if rule == "ratio":
            r = self.r_hat(d)
            return np.where(np.isnan(r), True, r < self.tau)
        if rule == "qc":
            x = np.log(d.s_self.to_numpy())
            out = np.ones(len(d), bool)
            for q, (thr, below) in self.qc.items():
                m = (d.q == q).to_numpy()
                out[m] = (x[m] < thr) if below else (x[m] > thr)
            return out
        raise KeyError(rule)


def evaluate(test: pd.DataFrame, rules: Rules) -> tuple[pd.DataFrame, dict]:
    out = {r: np.where(rules.choose(test, r), test.p_model, test.p_bic) for r in RULES}
    rows = []
    for (ds, q), idx in test.groupby(["dataset", "q"]).groups.items():
        i = test.index.get_indexer(idx)
        rows.append({"dataset": ds, "q": q, "n": len(i), **{r: float(out[r][i].mean()) for r in RULES},
                     **{f"use_{r}": float(rules.choose(test.iloc[i], r).mean()) for r in ("jpeg", "ratio", "qc")}})
    comp = {}
    for ds, g in test.groupby("dataset"):
        i = test.index.get_indexer(g.index)
        subj = g.subject.to_numpy()
        comp[ds] = {f"{a}-{b}": paired_diff(out[a][i], out[b][i], subj, n_boot=2000)
                    for a, b in (("jpeg", "model"), ("ratio", "model"), ("qc", "model"), ("ratio", "jpeg"),
                                 ("qc", "jpeg"), ("oracle", "jpeg"))}
    return pd.DataFrame(rows), comp


def fmt(e) -> str:
    star = " *" if e.lo > 0 or e.hi < 0 else ""
    return f"{e.point:+.3f} [{e.lo:+.3f}, {e.hi:+.3f}]{star}"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="results")
    ap.add_argument("--bench", default="data/bench")
    ap.add_argument("--folds", default="splits/ami_5fold.json")
    ap.add_argument("--model", default="span_ch48")
    ap.add_argument("--out", default=None, help="mặc định: <results>/selector")
    a = ap.parse_args(argv)
    out = Path(a.out) if a.out else Path(a.results) / "selector"
    out.mkdir(parents=True, exist_ok=True)
    d = load(Path(a.results), Path(a.bench), a.model)
    d["split"] = split(d, Path(a.folds))
    d.to_csv(out / f"per_image_{a.model}.csv", index=False)
    md = [f"# Phép thử luật chọn lúc chạy: bicubic hay `{a.model}` (PSNR-Y, dB)", "",
          f"{len(d)} quyết định (ảnh × cỡ × mức nén). Dấu * là khoảng tin cậy 95% không chứa 0 (bootstrap theo người).", ""]

    # 1. thống kê độ sắc nét ước lượng E_i tốt tới đâu (trong từng bộ, ảnh sạch)
    md += ["## 1. Sai số tự nội suy của ảnh nhỏ so với E_i thật", "", "| Bộ | Spearman (s_self, E_i) | Spearman (r̂, tỉ số thật), ảnh có nén |", "|---|---|---|"]
    allr = Rules(d[d.split == "dev"])
    keep = {"model": a.model, "corr": {}, "scenarios": {}}    # cho make_paper.py
    for ds, g in d.groupby("dataset"):
        c = g[g.kind == "bic"]
        z = g[g.kind != "bic"]
        rho_i, rho_r = spearmanr(c.s_self, c.E_i).statistic, spearmanr(allr.r_hat(z), z.ratio_true).statistic
        keep["corr"][ds] = {"s_self_vs_Ei": float(rho_i), "rhat_vs_ratio": float(rho_r)}
        md.append(f"| {ds} | {rho_i:.2f} | {rho_r:.2f} |")
    md.append("")

    # 2. ba kịch bản hiệu chỉnh
    summ = []
    for scen, cal_sets in (("hiệu chỉnh trên AMI", ("AMI",)), ("hiệu chỉnh trên DIV2K", ("DIV2K",)),
                           ("hiệu chỉnh trên AMI và DIV2K", ("AMI", "DIV2K"))):
        cal = d[(d.split == "dev") & d.dataset.isin(cal_sets)]
        test = d[d.split == "test"].reset_index(drop=True)
        R = Rules(cal)
        tab, comp = evaluate(test, R)
        keep["scenarios"]["+".join(cal_sets)] = {
            "jpeg_levels_model": sorted(int(q) for q, v in R.q_pick.items() if v), "tau": R.tau,
            "comparisons": {ds: {k: {"point": e.point, "lo": e.lo, "hi": e.hi} for k, e in c.items()} for ds, c in comp.items()}}
        tab.insert(0, "scenario", scen)
        summ.append(tab)
        md += [f"## 2. {scen}", "",
               f"Luật jpeg chọn mô hình ở các mức: {sorted(q for q, v in R.q_pick.items() if v)}; luật ratio: ngưỡng "
               f"r̂ < {R.tau:.3f}.", "",
               "| Bộ thử | jpeg so với luôn mô hình | ratio so với luôn mô hình | qc so với luôn mô hình | "
               "**ratio so với jpeg** | **qc so với jpeg** | oracle so với jpeg |", "|---|---|---|---|---|---|---|"]
        for ds, c in comp.items():
            md.append(f"| {ds} | " + " | ".join(fmt(c[k]) for k in ("jpeg-model", "ratio-model", "qc-model",
                                                                    "ratio-jpeg", "qc-jpeg", "oracle-jpeg")) + " |")
        md += ["", "PSNR trung bình theo mức nén (tập thử):", "",
               "| Bộ | mức | n | luôn mô hình | luôn bicubic | jpeg | ratio | qc | oracle | tỉ lệ dùng mô hình (jpeg / ratio / qc) |",
               "|---|---|---|---|---|---|---|---|---|---|"]
        for r in tab.to_dict("records"):
            md.append(f"| {r['dataset']} | {r['q']} | {r['n']} | {r['model']:.2f} | {r['bicubic']:.2f} | {r['jpeg']:.2f} | "
                      f"{r['ratio']:.2f} | {r['qc']:.2f} | {r['oracle']:.2f} | "
                      f"{r['use_jpeg']:.2f} / {r['use_ratio']:.2f} / {r['use_qc']:.2f} |")
        md.append("")
    pd.concat(summ).to_csv(out / f"summary_{a.model}.csv", index=False)
    (out / f"comparisons_{a.model}.json").write_text(json.dumps(keep, indent=1, ensure_ascii=False))
    (out / f"summary_{a.model}.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
