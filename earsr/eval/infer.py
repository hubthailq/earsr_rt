"""Chạy một mô hình trên một bộ ảnh LR đã lưu và ghi số đo theo từng ảnh.

Bước này không tính trung bình. Mọi thống kê nằm ở ``earsr.stats``.
"""
from __future__ import annotations

import csv
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from ..data.ami import read_manifest
from ..data.resize import imresize
from ..data.splits import fold_of_subject, load_folds
from ..device import GpuFirst, full_precision
from ..io import imread_rgb, imwrite_rgb, to_tensor, to_uint8
from .metrics import PerceptualMetrics, fr_metrics

BASELINES = ("bicubic", "bicubic_sharp")
FIELDS_HEAD = ["model", "dataset", "subject", "view", "key", "fold", "scale", "tier", "degrade"]


def unsharp(img: np.ndarray, sigma: float = 1.0, amount: float = 0.5) -> np.ndarray:
    """Lọc làm nét đơn giản (unsharp mask), dùng cho mốc 'bicubic + làm nét'."""
    blur = cv2.GaussianBlur(img, (0, 0), sigma)
    return np.clip(np.round(img.astype(np.float64) * (1 + amount) - blur.astype(np.float64) * amount),
                   0, 255).astype(np.uint8)


def make_predictor(model_name: str, scale: int, device: str = "cpu"):
    """Trả về hàm ``lr uint8 -> sr uint8`` cho một mô hình hoặc một mốc nội suy."""
    if model_name == "bicubic":
        return lambda lr: imresize(lr, float(scale))
    if model_name == "bicubic_sharp":
        return lambda lr: unsharp(imresize(lr, float(scale)))
    from ..models import registry

    if model_name in registry.CHAINS:
        # Khử nén rồi mới phóng. Ảnh trung gian được làm tròn về 8 bit, như một ảnh đã khử nén được lưu lại.
        pre_name, post_name = registry.CHAINS[model_name]
        post = make_predictor(post_name, scale, device)
        pre = GpuFirst(lambda dev: registry.build_restorer(pre_name).to(dev), device, name=pre_name).warm()

        @torch.no_grad()
        def chained(lr: np.ndarray) -> np.ndarray:
            with full_precision():
                clean = pre.run(lambda model, dev: to_uint8(model(to_tensor(lr).to(dev))))
            return post(clean)

        chained.runner = pre
        return chained
    if model_name in registry.SPECS and registry.SPECS[model_name].scale != scale:
        raise ValueError(f"{model_name} là mô hình ×{registry.SPECS[model_name].scale}, không phải ×{scale}")
    # Ưu tiên ``device``; hết bộ nhớ thì ảnh đó chạy trên CPU và GPU được thử lại sau (earsr/device.py).
    runner = GpuFirst(lambda dev: registry.build_model(model_name).to(dev), device, name=model_name).warm()

    @torch.no_grad()
    def predict(lr: np.ndarray) -> np.ndarray:
        with full_precision():
            return runner.run(lambda model, dev: to_uint8(model(to_tensor(lr).to(dev))))

    predict.runner = runner
    return predict


def evaluate_on_bench(model_name: str, bench_root: str | Path, tier: int, scale: int, kind: str,
                      folds_path: str | Path | None, out_csv: str | Path, device: str = "cpu",
                      perceptual: PerceptualMetrics | None = None, predictor=None,
                      save_sr_dir: str | Path | None = None, limit: int | None = None,
                      extra_metrics=None, subjects: set | None = None,
                      boxes: dict | None = None) -> Path:
    """Ghi ``out_csv``: mỗi dòng là một ảnh test. Trả về đường dẫn file.

    Dùng cho mọi benchmark có ``manifest.csv`` (AMI, EarVN1.0, AWEx).
    ``folds_path``: file fold của AMI; None cho bộ không chia fold (cột fold = 0).
    ``predictor``: hàm lr->sr tự cấp (cho mô hình tự huấn luyện); nếu None thì
    dựng từ kho mô hình theo ``model_name``.
    ``extra_metrics``: danh sách hàm ``f(sr, hr, row) -> dict`` (điểm mốc, gờ, ...).
    ``subjects``: chỉ chấm ảnh của những người này (ví dụ người test của một fold).
    ``boxes``: dict khóa ảnh -> hộp bao vùng tai (trên, trái, dưới, phải) theo tỉ lệ
    của ảnh (``load_boxes``); nếu có, thêm cột ``psnr_y_box``.
    """
    bench_root = Path(bench_root)
    fold_of = fold_of_subject(load_folds(folds_path)) if folds_path else {}
    rows = [r for r in read_manifest(bench_root / "manifest.csv") if r["tier"] == tier]
    if subjects is not None:
        rows = [r for r in rows if r["subject"] in subjects]
    if not rows:
        raise RuntimeError(f"không có ảnh nào ở tầng {tier} trong {bench_root}")
    lr_dir = bench_root / "lr" / f"hr{tier}_x{scale}_{kind}"
    if not lr_dir.is_dir():
        raise FileNotFoundError(f"chưa sinh ảnh LR: {lr_dir} (gọi build_lr_set trước)")
    predict = predictor or make_predictor(model_name, scale, device)
    runner = getattr(predict, "runner", None)
    if limit:
        rows = rows[:limit]
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_rows = []
    for r in rows:
        key = f"{r['subject']}_{r['view']}"
        hr = imread_rgb(bench_root / r["file"])
        lr = imread_rgb(lr_dir / f"{key}.png")
        t0 = time.perf_counter()
        with full_precision():   # cả với hàm dự đoán tự cấp (ví dụ lúc chấm test ở cuối train.py)
            sr = predict(lr)
        dt = (time.perf_counter() - t0) * 1000.0
        if sr.shape != hr.shape:
            raise RuntimeError(f"{model_name} {key}: đầu ra {sr.shape}, đáp án {hr.shape}")
        box = None
        if boxes is not None and key in boxes:
            t, l, b, rr = boxes[key]
            hh, ww = hr.shape[:2]
            box = (max(0, int(np.floor(t * hh))), max(0, int(np.floor(l * ww))),
                   min(hh, int(np.ceil(b * hh))), min(ww, int(np.ceil(rr * ww))))
            if box[2] - box[0] < 8 or box[3] - box[1] < 8:
                box = None
        m = fr_metrics(sr, hr, scale, perceptual=perceptual, box=box)
        with full_precision():
            for f in (extra_metrics or []):
                m.update(f(sr, hr, r))
        if save_sr_dir is not None:
            imwrite_rgb(Path(save_sr_dir) / f"{key}.png", sr)
        row = {"model": model_name, "dataset": r.get("dataset", "ami"), "subject": r["subject"],
               "view": r["view"], "key": key, "fold": fold_of.get(r["subject"], 0), "scale": scale,
               "tier": tier, "degrade": kind}
        row.update({k: (v if isinstance(v, (int, np.integer)) else round(v, 6) if np.isfinite(v) else v)
                    for k, v in m.items()})
        row["host_ms"] = round(dt, 3)
        # thiết bị đã chạy ảnh này (khác ``device`` khi GPU hết bộ nhớ); host_ms chỉ so được trong cùng thiết bị
        if runner is not None:
            row["device"] = runner.last_device or str(device)
        else:  # mốc nội suy chạy trên CPU; hàm dự đoán tự cấp chạy trên thiết bị người gọi truyền vào
            row["device"] = "cpu" if predictor is None else str(device)
        out_rows.append(row)
    if len({r["key"] for r in out_rows}) != len(out_rows):
        raise RuntimeError("trùng khóa ảnh trong kết quả")
    for k in (perceptual.available if perceptual is not None else []):
        n_nan = sum(1 for r in out_rows if not (isinstance(r.get(k), (int, float)) and np.isfinite(r[k])))
        if n_nan:
            print(f"CẢNH BÁO: {model_name} hr{tier} {kind}: cột {k} trống ở {n_nan}/{len(out_rows)} ảnh "
                  f"(số đo báo lỗi khi tính)", flush=True)
    if runner is not None and runner.fallbacks:
        n_cpu = sum(r["device"] == "cpu" for r in out_rows)
        print(f"[thiết bị] {model_name}: {n_cpu}/{len(out_rows)} ảnh chạy trên CPU vì {runner.primary} hết bộ nhớ "
              f"({runner.fallbacks} lần chuyển)", flush=True)
    fields = []
    for r in out_rows:
        fields += [k for k in r if k not in fields]
    with open(out_csv, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fields)
        wr.writeheader()
        wr.writerows(out_rows)
    return out_csv


def load_boxes(path: str | Path) -> dict:
    """Đọc file hộp bao (cột key, top, left, bottom, right theo tỉ lệ của ảnh)."""
    with open(path, newline="") as f:
        return {r["key"]: (float(r["top"]), float(r["left"]), float(r["bottom"]), float(r["right"]))
                for r in csv.DictReader(f)}


# tên cũ, giữ để mã gọi cũ vẫn chạy
evaluate_on_ami = evaluate_on_bench
