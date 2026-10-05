#!/usr/bin/env python3
"""N2 (a): bộ phân loại "mô phỏng hay thật" trên EarVN1.0.

  python scripts/realism_classifier.py --root /path/EarVN1.0 --roles splits/earvn_roles.json \
      --kinds bic generic est --degrade-params configs/degrade_estimated.json --out results/n2_realism.json

Chỉ dùng người có vai 'clf' (tách khỏi nhóm dùng để khớp tham số suy giảm).
Ảnh nhỏ thật: cạnh ngắn trong --small. Ảnh mô phỏng: tạo từ ảnh lớn của cùng nhóm
người, đưa về cùng phân bố cạnh ngắn với ảnh nhỏ thật.
Tiêu chí N2 (a): kiểu 'est' gần 50% hơn kiểu 'generic' (hiệu âm, khoảng tin cậy không chứa 0).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from earsr.data import wild  # noqa: E402
from earsr.degrade.pipelines import DegradeParams  # noqa: E402
from earsr.eval.realism import realism_test, simulate_small  # noqa: E402
from earsr.io import imread_rgb  # noqa: E402


def collect(root: Path, roles: dict, role: str, small: tuple[int, int], scale: int):
    """Trả về (ảnh nhỏ thật theo người, đường dẫn ảnh lớn theo người)."""
    real, large = {}, {}
    for p in sorted(q for q in root.rglob("*") if q.suffix.lower() in wild.EXTS):
        s = p.parent.name
        if roles.get(s) != role:
            continue
        im = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if im is None or im.dtype != np.uint8 or im.ndim != 3 or im.shape[2] != 3:
            continue
        short = min(im.shape[:2])
        if small[0] <= short <= small[1]:
            real.setdefault(s, []).append(np.ascontiguousarray(im[:, :, ::-1]))
        elif short >= small[0] * scale:
            large.setdefault(s, []).append(p)
    return real, large


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--roles", default="splits/earvn_roles.json")
    ap.add_argument("--role", default="clf")
    ap.add_argument("--kinds", nargs="+", default=["bic", "generic", "est"])
    ap.add_argument("--degrade-params", default=None)
    ap.add_argument("--small", type=int, nargs=2, default=[24, 48])
    ap.add_argument("--scale", type=int, default=4)
    ap.add_argument("--sim-per-image", type=int, default=4)
    ap.add_argument("--patch", type=int, default=16)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/n2_realism.json")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args(argv)
    if "est" in a.kinds and not a.degrade_params:
        raise SystemExit("kiểu 'est' cần --degrade-params")
    params = DegradeParams.load(a.degrade_params) if a.degrade_params else None
    roles = json.loads(Path(a.roles).read_text())
    real, large = collect(Path(a.root), roles, a.role, tuple(a.small), a.scale)
    shorts = np.array([min(im.shape[:2]) for v in real.values() for im in v])
    if len(shorts) == 0 or not large:
        raise SystemExit(f"vai '{a.role}': {len(shorts)} ảnh nhỏ thật, {sum(map(len, large.values()))} ảnh lớn; không đủ")
    print(f"vai '{a.role}': {len(shorts)} ảnh nhỏ thật của {len(real)} người; "
          f"{sum(map(len, large.values()))} ảnh lớn của {len(large)} người")
    rng = np.random.default_rng(a.seed)
    sims = {k: {} for k in a.kinds}
    for s, paths in large.items():
        for p in paths:
            img = imread_rgb(p)
            ok = shorts[shorts * a.scale <= min(img.shape[:2])]
            if len(ok) == 0:
                continue
            for j in range(a.sim_per_image):
                short = int(rng.choice(ok))
                sd = int(rng.integers(1 << 31))
                for k in a.kinds:   # cùng cỡ và cùng hạt giống cho mọi kiểu
                    lr = simulate_small(img, short, k, a.scale, params if k == "est" else None, sd, f"{p.name}|{j}")
                    if lr is not None and min(lr.shape[:2]) >= a.patch:
                        sims[k].setdefault(s, []).append(lr)
    real = {s: [im for im in v if min(im.shape[:2]) >= a.patch] for s, v in real.items()}
    real = {s: v for s, v in real.items() if v}
    res = realism_test(real, sims, patch=a.patch, steps=a.steps, n_seeds=a.seeds, device=a.device, seed=a.seed)
    res["args"] = vars(a)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1, ensure_ascii=False))
    for k, v in res["kinds"].items():
        print(f"{k:10s} độ chính xác cân bằng {v['balanced_acc']:.3f} [{v['lo']:.3f}, {v['hi']:.3f}]")
    for k, v in res["pairs"].items():
        print(f"|acc−0,5| của {k}: {v['diff_dist_from_chance']:+.3f} [{v['lo']:+.3f}, {v['hi']:+.3f}]")
    return res


if __name__ == "__main__":
    main()
