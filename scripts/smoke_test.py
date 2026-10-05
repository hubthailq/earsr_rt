#!/usr/bin/env python3
"""Kiểm tra nhanh trên máy sẽ chạy thật (khoảng 5 đến 10 phút trên GPU), TRƯỚC khi chạy việc dài.

  python scripts/smoke_test.py --ami-raw /path/AMI

Project được viết và kiểm thử trên máy chỉ có CPU. Script này chạy những đường mã
chưa từng chạy ở đó: suy luận và huấn luyện trên GPU, AMP, LPIPS và DISTS, loss
cảm nhận (tải trọng số VGG19), và đo tốc độ huấn luyện để ước lượng thời gian.
Mỗi bước in ĐẠT, LỖI hoặc BỎ QUA; kết quả ghi vào results/smoke_test.json.
Gửi lại file đó (hoặc phần in ra) nếu có bước LỖI.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import tempfile
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402

RESULTS: list[dict] = []


def step(name: str):
    def deco(fn):
        def run(*args, **kw):
            t0 = time.time()
            try:
                info = fn(*args, **kw)
                status = "BỎ QUA" if isinstance(info, dict) and info.get("skip") else "ĐẠT"
            except Exception as e:  # noqa: BLE001
                info = {"error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-1500:]}
                status = "LỖI"
            RESULTS.append({"step": name, "status": status, "seconds": round(time.time() - t0, 1), "info": info})
            short = {k: v for k, v in (info or {}).items() if k != "trace"}
            print(f"[{status:6s}] {name}  ({RESULTS[-1]['seconds']} s)  {short}", flush=True)
            return info if status == "ĐẠT" else None
        return run
    return deco


@step("môi trường")
def env(a):
    out = {"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.cuda.is_available(),
           "device": a.device}
    if a.device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda nhưng torch không thấy GPU (cài bản torch có CUDA)")
        out["gpu"] = torch.cuda.get_device_name(0)
        out["gpu_mem_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 2 ** 30, 1)
    for m in ("torchvision", "cv2", "pandas", "scipy", "yaml", "onnx", "onnxruntime", "lpips", "pyiqa"):
        try:
            out[m] = getattr(__import__(m), "__version__", "có")
        except Exception as e:  # noqa: BLE001
            out[m] = f"THIẾU ({type(e).__name__})"
    return out


@step("trọng số công bố")
def weights(a):
    from earsr.models.registry import SPECS, list_models

    have = list_models(available_only=True)
    miss = sorted(set(SPECS) - set(have))
    if "span_ch48" not in have:
        raise RuntimeError("thiếu trọng số SPAN (chạy: bash scripts/get_weights.sh)")
    return {"có": len(have), "thiếu": miss}


@step("bộ AMI và benchmark")
def bench(a):
    from earsr.data import ami
    from earsr.data.build_lr import build_lr_set
    from earsr.data.splits import load_folds

    rep = ami.check_ami(ami.scan_ami(a.ami_raw))
    if rep["n_images"] != 700 or rep["n_subjects"] != 100:
        raise RuntimeError(f"bộ AMI không đủ 700 ảnh của 100 người: {rep}")
    load_folds(a.folds)
    if not (Path(a.bench) / "manifest.csv").exists():
        ami.build_ami_benchmark(a.ami_raw, a.bench)
    build_lr_set(a.bench, 144, 4, "bic")
    return {"n_images": rep["n_images"], "bench": str(a.bench)}


@step("suy luận trên thiết bị, LPIPS và DISTS")
def infer(a):
    import pandas as pd

    from earsr.eval.infer import evaluate_on_bench
    from earsr.eval.metrics import PerceptualMetrics

    perc = PerceptualMetrics(a.device)
    tmp = Path(tempfile.mkdtemp())
    res = {}
    for m in ("bicubic", "span_ch48"):
        evaluate_on_bench(m, a.bench, 144, 4, "bic", a.folds, tmp / f"{m}.csv", device=a.device,
                          perceptual=perc if perc.available else None, limit=a.limit)
        d = pd.read_csv(tmp / f"{m}.csv")
        res[m] = {k: round(float(d[k].mean()), 4) for k in ("psnr_y", "lpips", "dists") if k in d}
    if res["span_ch48"]["psnr_y"] < res["bicubic"]["psnr_y"] + 1.0:
        raise RuntimeError(f"SPAN không hơn bicubic như mong đợi (khoảng +2 dB): {res}")
    res["cảm nhận dùng được"] = perc.available
    res["cảm nhận không dùng được"] = perc.reason
    missing = sorted({"lpips", "dists"} - set(perc.available))
    if missing:
        msg = (f"không dùng được {missing}: {perc.reason}. Cài lpips và pyiqa; máy cần mạng để tải trọng số ở "
               f"lần đầu. Thiếu chúng thì kết quả giai đoạn 1 không có cột LPIPS, DISTS")
        if "perceptual" not in a.skip:
            raise RuntimeError(msg + " (chấp nhận thiếu: --skip perceptual)")
        res["cảnh báo"] = msg
    return res


@step("GPU và CPU cho cùng kết quả")
def consistency(a):
    if not a.device.startswith("cuda"):
        return {"skip": True, "lý do": "đang chạy trên CPU"}
    from earsr.device import full_precision
    from earsr.io import imread_rgb, to_tensor
    from earsr.models.registry import build_model

    # Ảnh LR thật của benchmark. Không dùng nhiễu trắng: với nhiễu trắng đầu ra của SPAN vọt ra ngoài
    # [0, 1] hàng chục lần và mạng nhạy tới mức sai số làm tròn cũng bị khuếch đại thành vài đơn vị.
    f = sorted((Path(a.bench) / "lr" / "hr144_x4_bic").glob("*.png"))[0]
    x = to_tensor(imread_rgb(f))
    m = build_model("span_ch48")
    with torch.no_grad():
        y_cpu = m(x)
        m, xg = m.to(a.device), x.to(a.device)
        d_default = float((m(xg).cpu() - y_cpu).abs().max())          # theo mặc định của máy (TF32 nếu GPU có)
        with full_precision():
            d = float((m(xg).cpu() - y_cpu).abs().max())               # cách mọi phép chấm của project chạy
            dh = float((m.half()(xg.half()).float().cpu() - y_cpu).abs().max())
    res = {"ảnh": f.name, "lệch khi chấm (FP32 đầy đủ)": d, "lệch theo mặc định của máy": d_default,
           "TF32 mặc định bật": bool(torch.backends.cudnn.allow_tf32), "lệch FP16": dh}
    if d > 1e-4:
        raise RuntimeError(f"GPU lệch CPU {d:.2e} ở FP32 đầy đủ (mong đợi cỡ 1e-6): {res}")
    return res


def _train(a, amp: bool, objective: str, iters: int, tag: str):
    from earsr.train.finetune import build_trainable, make_train_set, make_validator
    from earsr.train.objectives import make_objective
    from earsr.train.trainer import TrainConfig, train

    model = build_trainable("span", "zero", "pub", 4)
    ds = make_train_set(a.ami_raw, a.folds, 2, 4, "rand", 144, 24, seed=2)
    val = make_validator(a.bench, a.folds, 2, 4, (144,), "bic", a.device)
    out = Path(tempfile.mkdtemp())
    cfg = TrainConfig(run_id=f"SMOKE_span-zero+{tag}_pub_rand_x4_hrall_bic_f2", out_dir=str(out), iters=iters,
                      batch_size=a.batch_size, lr=1e-4, val_every=iters, save_every=iters, num_workers=a.workers,
                      seed=2, device=a.device, amp=amp)
    model.eval()
    before = val(model.to(a.device))
    s = train(model, ds, val, cfg, make_objective(objective, model))
    rate = iters / max(s["elapsed_s"], 1e-9)
    return {"bước/giây": round(rate, 2), "ước lượng 20.000 bước (giờ)": round(20000 / rate / 3600, 2),
            "PSNR val trước": round(before, 3), "PSNR val sau": round(s["best_val_psnr_y"], 3)}


@step("huấn luyện L1, FP32")
def train_fp32(a):
    r = _train(a, False, "l1", a.iters, "fp32")
    if r["PSNR val sau"] < r["PSNR val trước"] - 0.3:
        raise RuntimeError(f"PSNR validation tụt sau vài bước tinh chỉnh: {r}")
    return r


@step("huấn luyện L1, AMP (FP16)")
def train_amp(a):
    if not a.device.startswith("cuda"):
        return {"skip": True, "lý do": "AMP chỉ có nghĩa trên GPU"}
    return _train(a, True, "l1", a.iters, "amp")


@step("huấn luyện GAN (tải VGG19)")
def train_gan(a):
    return _train(a, False, "gan", max(4, a.iters // 5), "gan")


@step("xuất ONNX và so với PyTorch")
def onnx_export(a):
    from earsr.deploy.export_onnx import export_onnx
    from earsr.models.registry import build_model

    info = export_onnx(build_model("span_ch48"), Path(tempfile.mkdtemp()) / "span.onnx", (68, 48), 13)
    return {k: info.get(k) for k in ("max_abs_diff", "rel_diff", "opset")}


@step("chỉ số không tham chiếu (pyiqa)")
def nr(a):
    from earsr.eval.metrics_nr import NoReferenceMetrics

    m = NoReferenceMetrics(a.device)
    if not m.available:
        return {"skip": True, "lý do": m.reason}
    img = (np.random.default_rng(0).random((348, 244, 3)) * 255).astype(np.uint8)
    return {"dùng được": m.available, "không dùng được": m.reason, "giá trị thử": m(img)}


def main(argv=None) -> list[dict]:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ami-raw", required=True)
    ap.add_argument("--bench", default="data/bench/ami")
    ap.add_argument("--folds", default="splits/ami_5fold.json")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--skip", nargs="*", default=[], help="tên bước cần bỏ: gan nr onnx amp; 'perceptual' = chấp nhận thiếu LPIPS, DISTS")
    ap.add_argument("--out", default="results/smoke_test.json")
    a = ap.parse_args(argv)
    RESULTS.clear()
    env(a)
    weights(a)
    if bench(a) is not None:
        infer(a)
        consistency(a)
        train_fp32(a)
        if "amp" not in a.skip:
            train_amp(a)
        if "gan" not in a.skip:
            train_gan(a)
    if "onnx" not in a.skip:
        onnx_export(a)
    if "nr" not in a.skip:
        nr(a)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(RESULTS, indent=1, ensure_ascii=False, default=str))
    bad = [r["step"] for r in RESULTS if r["status"] == "LỖI"]
    print("\nTÓM TẮT:", "mọi bước đạt hoặc được bỏ qua" if not bad else f"CÓ LỖI ở: {bad}")
    print("đã ghi", a.out)
    return RESULTS


if __name__ == "__main__":
    sys.exit(1 if any(r["status"] == "LỖI" for r in main()) else 0)
