#!/usr/bin/env python3
"""T6 (i) và (v): phép thử ngữ cảnh, không huấn luyện.

  # ảnh tai (AMI gốc)
  python scripts/run_t6_context.py --ami-raw /path/AMI --out results/t6_context
  # ảnh tự nhiên (thư mục ảnh HR bất kỳ, ví dụ DIV2K valid hoặc Urban100)
  python scripts/run_t6_context.py --generic /path/images --generic-name urban100 --out results/t6_context

Hai cấu hình khung:
  wide: ngữ cảnh tới 24 px mỗi cạnh (lớn hơn vùng nhìn 21 px của SPAN); cửa sổ
        chiếm khoảng 43% bề ngang khung.
  near: ngữ cảnh tới 8 px; cửa sổ chiếm khoảng 69% bề ngang khung, gần với độ
        phóng của benchmark hơn.
  small: cửa sổ 24×34 (ảnh vào nhỏ nhất của benchmark), ngữ cảnh tới 8 px.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from earsr.data.ami import scan_ami  # noqa: E402
from earsr.device import describe_device  # noqa: E402
from earsr.eval.context_test import ContextConfig, make_context_frame, run_context_test  # noqa: E402
from earsr.eval.infer import make_predictor  # noqa: E402
from earsr.io import imread_rgb  # noqa: E402
from earsr.models.registry import SPECS, list_models  # noqa: E402

CONFIGS = {
    "wide": ContextConfig(win_hw=(51, 36), margins=(0, 4, 8, 16, 24)),
    "near": ContextConfig(win_hw=(51, 36), margins=(0, 2, 4, 8), prepad=(("replicate", 8), ("reflect", 8))),
    # cửa sổ bằng ảnh vào nhỏ nhất của benchmark (24×34): tỉ lệ điểm ảnh sát mép lớn nhất
    "small": ContextConfig(win_hw=(34, 24), margins=(0, 2, 4, 8), prepad=(("replicate", 8), ("reflect", 8))),
}


def iter_images(a):
    if a.ami_raw:
        for it in scan_ami(a.ami_raw):
            yield "ami", it.subject, it.key, it.path
    if a.generic:
        exts = {".png", ".jpg", ".jpeg", ".bmp"}
        for p in sorted(Path(a.generic).rglob("*")):
            if p.suffix.lower() in exts and (a.generic_filter is None or a.generic_filter in p.name):
                yield a.generic_name, p.stem, p.stem, p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ami-raw", default=None)
    ap.add_argument("--generic", default=None, help="thư mục ảnh HR tự nhiên")
    ap.add_argument("--generic-name", default="generic")
    ap.add_argument("--generic-filter", default=None, help="chỉ lấy file có chuỗi này trong tên (ví dụ _HR)")
    ap.add_argument("--out", default="results/t6_context")
    ap.add_argument("--configs", nargs="+", default=["wide", "near"], choices=list(CONFIGS))
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--groups", nargs="*", default=["light", "mid"])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    if not (a.ami_raw or a.generic):
        ap.error("cần --ami-raw hoặc --generic")
    print(describe_device(a.device) + ". Mọi phép chấm chạy ở FP32 đầy đủ (TF32 tắt)", flush=True)
    models = a.models if a.models else ["bicubic"] + [m for m in list_models(4, available_only=True)
                                                       if SPECS[m].group in a.groups]
    images = list(iter_images(a))
    if a.limit:
        images = images[:a.limit]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    datasets = sorted({d for d, *_ in images})
    for cname in a.configs:
        cfg0 = CONFIGS[cname]
        for m in models:
            for ds in datasets:
                f = out / f"{m}__{ds}__{cname}.csv"
                if f.exists() and not a.overwrite:
                    continue
                predict = make_predictor(m, cfg0.scale, a.device)
                rows, skipped = [], 0
                for d, subj, key, path in images:
                    if d != ds:
                        continue
                    try:
                        frame, cfg = make_context_frame(imread_rgb(path), cfg0)
                    except ValueError:
                        skipped += 1
                        continue
                    for r in run_context_test(predict, frame, cfg):
                        rows.append({"model": m, "dataset": ds, "config": cname, "subject": subj, "key": key, **r})
                if not rows:
                    continue
                with open(f, "w", newline="") as fh:
                    wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
                    wr.writeheader()
                    wr.writerows(rows)
                print(f"xong {f.name} ({len(rows)} dòng, bỏ {skipped} ảnh quá nhỏ)", flush=True)


if __name__ == "__main__":
    main()
