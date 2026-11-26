"""Đường CMC và TAR@FAR từ ma trận điểm đã lưu của recog_eval.py.

Mỗi thư mục kết quả của recog_eval.py có ``meta.json`` (danh sách người đăng ký, theo thứ tự cột), ``probes.csv`` (người
của từng ảnh dò, theo thứ tự hàng) và một file ``<tên>.npy`` cho mỗi cách phóng: ma trận điểm cosine ảnh dò × người
đăng ký. Không chạy lại mạng nào.

- CMC: tỉ lệ ảnh dò có người đúng nằm trong k hạng đầu, k = 1..K.
- TAR@FAR: coi mỗi cặp (ảnh dò, người đăng ký) là một phép xác minh; cặp đúng người là genuine, các cặp còn lại là
  impostor. Ngưỡng đặt sao cho tỉ lệ impostor vượt ngưỡng bằng FAR; TAR là tỉ lệ genuine vượt ngưỡng đó.
Với cách phóng có nhiều mô hình fold (nhiều file .npy), mỗi số được tính cho từng file rồi lấy trung bình, giống cách bài
gộp các mô hình fold. Khoảng tin cậy bootstrap theo người cho hiệu số với một cách phóng gốc (mặc định bicubic).

Dùng:
  python scripts/recog_curves.py --dir results/recog_up/earvn_{m} ... --out results/recog_curves/earvn
  (mỗi --spec có dạng tên=thư_mục[:mẫu_file], mẫu mặc định là tất cả *.npy trừ ref_large và direct)
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

FARS = (0.01, 0.001)


def load(dir_: Path, pattern: str | None):
    meta = json.loads((dir_ / "meta.json").read_text())
    subj = np.array(meta["subjects"]).astype(str)
    with open(dir_ / "probes.csv") as fh:
        probe_subj = np.array([r["subject"] for r in csv.DictReader(fh)]).astype(str)
    files = sorted(dir_.glob(pattern or "*.npy"))
    files = [f for f in files if f.stem not in ("ref_large", "direct")]
    if not files:
        raise SystemExit(f"không có file .npy trong {dir_} khớp {pattern}")
    col = {s: i for i, s in enumerate(subj)}
    true_idx = np.array([col[s] for s in probe_subj])
    mats = [np.load(f).astype(np.float64) for f in files]
    for m in mats:
        if m.shape != (len(probe_subj), len(subj)):
            raise SystemExit(f"kích thước ma trận điểm sai trong {dir_}: {m.shape}")
    return mats, true_idx, probe_subj


def per_probe(mats, true_idx, kmax):
    """Trả về (hits[k, probe]: trung bình qua fold của 1{hạng <= k}, genuine, impostor theo từng fold)."""
    hits = np.zeros((kmax, len(true_idx)))
    gen, imp = [], []
    rows = np.arange(len(true_idx))
    for m in mats:
        g = m[rows, true_idx]
        rank = 1 + (m > g[:, None]).sum(1)
        for k in range(1, kmax + 1):
            hits[k - 1] += (rank <= k)
        mask = np.ones_like(m, bool)
        mask[rows, true_idx] = False
        gen.append(g)
        imp.append(m[mask].reshape(len(rows), -1))
    return hits / len(mats), gen, imp


def tar_at_far(gen, imp, far):
    out = []
    for g, i in zip(gen, imp):
        thr = np.quantile(i.ravel(), 1 - far)
        out.append(float((g > thr).mean()))
    return float(np.mean(out))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spec", nargs="+", required=True, help="tên=thư_mục[:mẫu_file]")
    ap.add_argument("--base", default="bicubic", help="cách phóng gốc để tính hiệu số")
    ap.add_argument("--kmax", type=int, default=10)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    data = {}
    for sp in a.spec:
        name, rest = sp.split("=", 1)
        d, _, pat = rest.partition(":")
        data[name] = load(Path(d), pat or None)
    subj0 = next(iter(data.values()))[2]
    for name, (_, _, ps) in data.items():
        if not np.array_equal(ps, subj0):
            raise SystemExit(f"ảnh dò của {name} không khớp thứ tự với các cách phóng khác")
    res = {n: per_probe(m, t, a.kmax) for n, (m, t, _) in data.items()}
    rng = np.random.default_rng(0)
    us = np.unique(subj0)
    idx = {u: np.where(subj0 == u)[0] for u in us}
    rows = []
    for n, (hits, gen, imp) in res.items():
        r = {"method": n, **{f"cmc{k}": round(100 * float(hits[k - 1].mean()), 2) for k in range(1, a.kmax + 1)}}
        for far in FARS:
            r[f"tar@far{far:g}"] = round(100 * tar_at_far(gen, imp, far), 2)
        if a.base in res and n != a.base:   # hiệu số CMC hạng 1 và 5 với cách phóng gốc, bootstrap theo người
            hb = res[a.base][0]
            for k in (1, 5):
                d = 100 * (hits[k - 1] - hb[k - 1])
                bs = [d[np.concatenate([idx[u] for u in rng.choice(us, len(us))])].mean() for _ in range(a.n_boot)]
                r[f"d_cmc{k}"] = round(float(d.mean()), 2)
                r[f"d_cmc{k}_lo"], r[f"d_cmc{k}_hi"] = (round(float(np.percentile(bs, q)), 2) for q in (2.5, 97.5))
        rows.append(r)
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k != "method", k))
    with open(out / "curves.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=keys)
        wr.writeheader()
        wr.writerows(rows)
    for r in rows:
        print(r["method"], "CMC1", r["cmc1"], "CMC5", r["cmc5"], "TAR@FAR1%", r["tar@far0.01"],
              "TAR@FAR0.1%", r["tar@far0.001"])
    print("đã ghi", out / "curves.csv")


if __name__ == "__main__":
    main()
