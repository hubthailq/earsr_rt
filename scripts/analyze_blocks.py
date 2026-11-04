"""Phần dư nén nằm ở đâu: theo vị trí trong khối 8×8 của ảnh vào và theo tần số (phản biện ngày 04/11).

Với mỗi mô hình m và mỗi mức nén Q, phần dư theo phân rã của mục 4.2 là
    (b-1) E_c = [E'_m - E_m] - [E'_bic - E_bic],
tức sai số mà nén thêm vào ảnh của mô hình, trừ phần nén thêm vào ảnh của bicubic. Script tách đại lượng này
  (a) theo khoảng cách (điểm ảnh LR) từ điểm ảnh tới biên khối 8×8 gần nhất: 0 (sát biên), 1, 2, 3 (giữa khối);
  (b) theo tần số không gian của ảnh sai số (kênh Y, chu kỳ trên điểm ảnh HR, chia thành các vành tròn).
Cả hai cách tách đều cộng lại đúng bằng phần dư toàn ảnh (cùng vùng bỏ viền), nên đọc được là phần nào chiếm bao nhiêu.

Chỉ chấm trên ảnh LR đã có trong benchmark (bic, bicjpeg93, bicjpeg75), không huấn luyện gì.

Dùng:  python scripts/analyze_blocks.py --sets ami div2k --tier 144 --out results/mechanism
       (thử: --limit 5 --models span_ch48)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from earsr.data.ami import read_manifest  # noqa: E402
from earsr.eval.infer import make_predictor  # noqa: E402
from earsr.eval.metrics import rgb_to_y  # noqa: E402
from earsr.io import imread_rgb  # noqa: E402

PSNR16 = ["span_ch48", "span_ch28", "span_ch26", "rlfn", "efdn", "safmnpp", "smfan", "msrresnet", "edsr_baseline",
          "swinir_light", "rrdb_psnr", "pds26", "pkdsr26", "dscf26", "disp26", "errn26"]
KINDS = ["bic", "bicjpeg93", "bicjpeg75"]
SCALE = 4
FREQ_EDGES = np.linspace(0.0, 0.5, 21)   # 20 vành rộng 0,025, chu kỳ trên điểm ảnh HR; Nyquist của ảnh LR ở 0,125


def block_class(h: int, w: int, scale: int = SCALE) -> np.ndarray:
    """Khoảng cách (điểm ảnh LR) tới biên khối 8×8 gần nhất, cho từng điểm ảnh HR: 0..3."""
    def d1(n):
        i = (np.arange(n) // scale) % 8
        return np.minimum(i, 7 - i)
    return np.minimum(d1(h)[:, None], d1(w)[None, :])


def radial_power(e: np.ndarray) -> np.ndarray:
    """Phần MSE của ảnh sai số ``e`` nằm trong từng vành tần số (tổng các vành = MSE, theo Parseval)."""
    n = e.size
    p = np.abs(np.fft.fft2(e)) ** 2 / (n * n)
    fy = np.fft.fftfreq(e.shape[0])[:, None]
    fx = np.fft.fftfreq(e.shape[1])[None, :]
    r = np.minimum(np.sqrt(fy ** 2 + fx ** 2), 0.5)   # góc vượt 0,5 dồn vào vành cuối
    idx = np.clip(np.digitize(r, FREQ_EDGES[1:-1]), 0, len(FREQ_EDGES) - 2)
    return np.bincount(idx.ravel(), weights=p.ravel(), minlength=len(FREQ_EDGES) - 1)


def stats_one(sr: np.ndarray, hr: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """(MSE theo lớp vị trí 0..3 [tổng có trọng số = MSE], MSE theo vành tần số, MSE) trên kênh Y, bỏ viền 4 px."""
    e = rgb_to_y(sr) - rgb_to_y(hr)
    cls = block_class(*e.shape)
    b = SCALE
    e, cls = e[b:-b, b:-b], cls[b:-b, b:-b]
    sq = e ** 2
    by_pos = np.bincount(cls.ravel(), weights=sq.ravel(), minlength=4) / sq.size   # phần đóng góp vào MSE
    return by_pos, radial_power(e), float(sq.mean())


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bench-root", default="data/bench")
    ap.add_argument("--sets", nargs="+", default=["ami", "div2k"])
    ap.add_argument("--tier", type=int, default=144)
    ap.add_argument("--models", nargs="*", default=PSNR16)
    ap.add_argument("--out", default="results/mechanism")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for st in a.sets:
        root = Path(a.bench_root) / st
        rows = [r for r in read_manifest(root / "manifest.csv") if r["tier"] == a.tier][: a.limit]
        hrs = [imread_rgb(root / r["file"]) for r in rows]
        keys = [f"{r['subject']}_{r['view']}" for r in rows]
        lrs = {k: [imread_rgb(root / "lr" / f"hr{a.tier}_x{SCALE}_{k}" / f"{key}.png") for key in keys] for k in KINDS}
        # tỉ lệ điểm ảnh của từng lớp vị trí (để quy về MSE trong lớp)
        frac = np.zeros(4)
        for hr in hrs:
            c = block_class(*hr.shape[:2])[SCALE:-SCALE, SCALE:-SCALE]
            frac += np.bincount(c.ravel(), minlength=4) / c.size
        frac /= len(hrs)
        res = {"set": st, "tier": a.tier, "n": len(rows), "kinds": KINDS, "pos_frac": frac.tolist(),
               "freq_edges": FREQ_EDGES.tolist(), "models": {}}
        for name in ["bicubic"] + list(a.models):
            t0 = time.time()
            pred = make_predictor(name, SCALE, a.device)
            d = {}
            for k in KINDS:
                pos, frq, mse = np.zeros(4), np.zeros(len(FREQ_EDGES) - 1), 0.0
                for lr, hr in zip(lrs[k], hrs):
                    p_, f_, m_ = stats_one(pred(lr), hr)
                    pos += p_
                    frq += f_
                    mse += m_
                n = len(hrs)
                d[k] = {"pos": (pos / n).tolist(), "freq": (frq / n).tolist(), "mse": mse / n}
            res["models"][name] = d
            print(f"{st} {name}: {time.time() - t0:.1f} s", flush=True)
        path = out / f"blocks_{st}_hr{a.tier}.json"
        path.write_text(json.dumps(res, indent=1))
        print(f"ghi {path}")


if __name__ == "__main__":
    main()
