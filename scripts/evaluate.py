#!/usr/bin/env python3
"""Đánh giá mô hình trên một benchmark có ``manifest.csv`` (AMI, EarVN1.0, AWEx):
mô hình trong kho (trọng số công bố), mốc nội suy, và mô hình tự huấn luyện.

T2 (không huấn luyện):
  python scripts/evaluate.py --bench data/bench/ami --out results/t2 --tiers 96 144 192 --kinds bic bicjpeg75

Mô hình tự huấn luyện (thư mục lần chạy của train.py), trên một bộ ngoài thực tế:
  python scripts/evaluate.py --bench data/bench/earvn --folds none --tiers 96 --kinds bic est \
      --degrade-params configs/degrade_estimated.json --models bicubic --runs runs/S2_... runs/S2_...

Thêm số đo: --ridge (gờ giả, gờ mất); --landmark-ckpt (độ lệch điểm mốc và sàn nhiễu);
--boxes (PSNR trong hộp bao vùng tai); --nr (chỉ số không tham chiếu, cỡ 244).

Mỗi (mô hình, cỡ, kiểu suy giảm) cho một file CSV số đo theo từng ảnh. Chạy lại
sẽ bỏ qua file đã có (dùng --overwrite để chạy lại).

Lưu ý: mô hình tự huấn luyện trên một fold của AMI chỉ được chấm trên người test
của fold đó. Với benchmark AMI, script tự giới hạn như vậy (đọc fold từ cấu hình
của lần chạy), trừ khi có --all-subjects.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from earsr.data.build_lr import build_lr_set  # noqa: E402
from earsr.data.splits import load_folds  # noqa: E402
from earsr.degrade.pipelines import DegradeParams, needs_params  # noqa: E402
from earsr.device import describe_device  # noqa: E402
from earsr.eval.infer import BASELINES, evaluate_on_bench, load_boxes  # noqa: E402
from earsr.eval.metrics import PerceptualMetrics  # noqa: E402
from earsr.models.registry import SPECS, list_models  # noqa: E402


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bench", required=True, help="thư mục benchmark (có manifest.csv)")
    ap.add_argument("--folds", default="splits/ami_5fold.json", help="'none' cho bộ không chia fold (EarVN1.0, AWEx)")
    ap.add_argument("--degrade-params", default=None, help="file tham số cho kiểu 'est' (configs/degrade_estimated.json)")
    ap.add_argument("--out", default="results/t2")
    ap.add_argument("--scale", type=int, default=4)
    ap.add_argument("--tiers", type=int, nargs="+", default=[96, 144, 192])
    ap.add_argument("--kinds", nargs="+", default=["bic", "bicjpeg75"])
    ap.add_argument("--models", nargs="*", default=None,
                    help="mặc định: mọi mô hình có trọng số ở hệ số này (danh sách rỗng nếu có --runs)")
    ap.add_argument("--groups", nargs="*", default=None, help="chỉ chạy các nhóm này (light mid upper perceptual)")
    ap.add_argument("--runs", nargs="*", default=[], help="thư mục lần chạy của train.py hoặc pretrain.py")
    ap.add_argument("--ckpt", default="best", help="tên checkpoint trong ckpt/ của lần chạy (best, step123, ...)")
    ap.add_argument("--gate-bias", type=float, default=None, help="điểm vận hành cho mô hình N3")
    ap.add_argument("--all-subjects", action="store_true", help="chấm mô hình tự huấn luyện trên mọi người của AMI")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--no-perceptual", action="store_true", help="không tính LPIPS, DISTS")
    ap.add_argument("--require-perceptual", action="store_true",
                    help="thoát ngay nếu một số đo cảm nhận không nạp được (mặc định: in lý do rồi chạy thiếu cột đó)")
    ap.add_argument("--more-metrics", action="store_true",
                    help="thêm ST-LPIPS, TOPIQ-FR, FSIM, VIF, PieAPP qua pyiqa (chậm hơn; tải trọng số ở lần đầu)")
    ap.add_argument("--nr", action="store_true", help="chỉ số không tham chiếu và điểm cảm nhận NTIRE (cỡ 244)")
    ap.add_argument("--ridge", action="store_true", help="bảng kiểm tra gờ giả và gờ mất")
    ap.add_argument("--landmark-ckpt", default=None, help="bộ dò điểm mốc dùng để ĐO (khác bộ sinh nhãn cho N1)")
    ap.add_argument("--boxes", default=None, help="file hộp bao vùng tai (scripts/landmark_tools.py boxes)")
    ap.add_argument("--save-sr", default=None, help="lưu ảnh SR vào thư mục này (cho khảo sát người xem)")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="chỉ chạy N ảnh đầu (để thử)")
    a = ap.parse_args(argv)
    if a.threads:
        torch.set_num_threads(a.threads)
    if a.no_perceptual and a.require_perceptual:
        ap.error("--no-perceptual và --require-perceptual không đi cùng nhau")

    if a.models is not None:
        models = list(a.models)
    else:
        models = [] if a.runs else list(BASELINES) + list_models(a.scale, available_only=True)
    if a.groups:
        models = [m for m in models if m in BASELINES or SPECS[m].group in a.groups]
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(describe_device(a.device) + ". Mọi phép chấm chạy ở FP32 đầy đủ (TF32 tắt)", flush=True)
    perc = None
    if not a.no_perceptual:
        from earsr.eval.metrics import PYIQA_FR

        perc = PerceptualMetrics(a.device, want=("lpips",) + (PYIQA_FR if a.more_metrics else ("dists",)))
        print(f"số đo cảm nhận dùng được: {perc.available}; không dùng được: {perc.reason}")
        if a.require_perceptual and perc.reason:
            raise SystemExit("DỪNG (--require-perceptual): không nạp được " + "; ".join(
                f"{k} ({v})" for k, v in perc.reason.items()) + ". Cài lpips và pyiqa, và cho máy tải trọng số ở "
                "lần đầu; hoặc bỏ cờ này để chạy thiếu các cột đó.")
        with open(out_dir / "perceptual_info.txt", "w") as f:
            f.write(f"available: {perc.info}\nunavailable: {perc.reason}\n")
        if not perc.available:
            perc = None

    folds = None if a.folds.lower() == "none" else a.folds
    params = DegradeParams.load(a.degrade_params) if a.degrade_params else None
    extra = []
    if a.nr:
        from earsr.eval.metrics_nr import NoReferenceMetrics, add_ntire_score

        nr = NoReferenceMetrics(a.device)
        print(f"chỉ số không tham chiếu dùng được: {nr.available}; không dùng được: {nr.reason}")
        extra.append(lambda sr, hr, row: nr(sr))
    if a.ridge:
        from earsr.eval.ridge import ridge_check

        extra.append(lambda sr, hr, row: ridge_check(sr, hr))
    if a.landmark_ckpt:
        from earsr.landmarks.metric import LandmarkScorer

        extra.append(LandmarkScorer(a.landmark_ckpt, a.device).metric_fn())
    boxes = load_boxes(a.boxes) if a.boxes else None

    # (tên, hàm dự đoán hoặc None, tập người được chấm hoặc None)
    jobs = [(m, None, None) for m in models]
    for rd in a.runs:
        from earsr.train.finetune import predictor_from_run

        pred = predictor_from_run(rd, a.device, a.ckpt, gate_bias=a.gate_bias)
        if pred.scale != a.scale:
            raise SystemExit(f"{rd}: mô hình ×{pred.scale}, đang chấm ×{a.scale}")
        cfg = json.loads((Path(rd) / "config.json").read_text())
        fold = cfg.get("extra", {}).get("fold")
        subjects = None
        if folds and fold and not a.all_subjects:
            subjects = set(load_folds(folds)["folds"][str(fold)]["test"])
        name = pred.run_id + ("" if a.ckpt == "best" else f"@{a.ckpt}") + \
            ("" if a.gate_bias is None else f"@b{a.gate_bias:g}")
        jobs.append((name, pred, subjects))
    if not jobs:
        raise SystemExit("không có mô hình nào để chấm")

    for tier in a.tiers:
        for kind in a.kinds:
            build_lr_set(a.bench, tier, a.scale, kind, params=params if needs_params(kind) else None)
            for name, pred, subjects in jobs:
                out = out_dir / f"{name}__hr{tier}_x{a.scale}_{kind}.csv"
                if out.exists() and not a.overwrite:
                    continue
                save = Path(a.save_sr) / f"hr{tier}_x{a.scale}_{kind}" / name if a.save_sr else None
                evaluate_on_bench(name, a.bench, tier, a.scale, kind, folds, out, device=a.device, perceptual=perc,
                                  predictor=pred, limit=a.limit, extra_metrics=extra, subjects=subjects,
                                  boxes=boxes, save_sr_dir=save)
                if a.nr:
                    import pandas as pd

                    d = pd.read_csv(out, dtype={"subject": str, "view": str})
                    pd.DataFrame([add_ntire_score(r) for r in d.to_dict("records")]).to_csv(out, index=False)
                print(f"xong {out.name}", flush=True)


if __name__ == "__main__":
    main()
