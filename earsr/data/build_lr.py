"""Sinh ảnh LR test một lần và lưu đĩa, để mọi mô hình test trên cùng bộ ảnh."""
from __future__ import annotations

import csv
from pathlib import Path

from ..degrade.pipelines import DegradeParams, degrade, needs_params
from ..io import imread_rgb, imwrite_rgb, sha1_array
from .ami import read_manifest


def build_lr_set(bench_root: str | Path, tier: int, scale: int, kind: str, seed: int = 0,
                 params: DegradeParams | None = None, overwrite: bool = False) -> Path:
    """Tạo ``bench_root/lr/hr{tier}_x{scale}_{kind}/{key}.png`` và manifest.

    Trả về thư mục chứa ảnh LR. Nếu đã có và ``overwrite`` là False thì kiểm lại
    mã băm của manifest thay vì sinh lại.
    """
    bench_root = Path(bench_root)
    rows = [r for r in read_manifest(bench_root / "manifest.csv") if r["tier"] == tier]
    if not rows:
        raise RuntimeError(f"manifest không có tầng {tier}")
    out_dir = bench_root / "lr" / f"hr{tier}_x{scale}_{kind}"
    man = out_dir / "manifest.csv"
    if needs_params(kind) and params is None:
        raise ValueError(f"kiểu '{kind}' cần params (configs/degrade_estimated.json)")
    if man.exists() and not overwrite:
        return out_dir
    out = []
    for r in rows:
        hr = imread_rgb(bench_root / r["file"])
        key = f"{r['subject']}_{r['view']}"
        lr = degrade(hr, scale, kind, seed=seed, key=f"{key}|hr{tier}|x{scale}", params=params)
        imwrite_rgb(out_dir / f"{key}.png", lr)
        out.append({"key": key, "subject": r["subject"], "view": r["view"], "tier": tier, "scale": scale,
                    "kind": kind, "seed": seed, "h": lr.shape[0], "w": lr.shape[1], "sha1": sha1_array(lr),
                    "params": params.source if params is not None else ""})
    with open(man, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        wr.writeheader()
        wr.writerows(out)
    return out_dir
