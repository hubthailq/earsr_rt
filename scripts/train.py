#!/usr/bin/env python3
"""Tinh chỉnh một mô hình trên ảnh tai (một fold của AMI) rồi chấm trên người test của fold đó.

Mốc SPAN, giao thức tỉ lệ ngẫu nhiên, fold 2:
  python scripts/train.py --exp T6 --backbone span --variant zero --pretrain pub \
      --protocol rand --fold 2 --ami-raw /path/AMI --bench data/bench/ami

Đối chứng N2 (suy giảm ước lượng), thêm ảnh EarVN1.0 nhóm train:
  python scripts/train.py --exp S3 ... --degrade est --degrade-params configs/degrade_estimated.json \
      --extra-dir /path/EarVN1.0 --extra-roles splits/earvn_roles.json --extra-name earvn

Mốc nới rộng cho bằng độ trễ (số kênh lấy từ scripts/match_latency.py), thân bản GAN, thân + LDL:
  ... --variant zero-c56 --pretrain ps20 --init-ckpt <ckpt tiền huấn luyện>
  ... --objective gan --init-ckpt <best.pt của thân đã tinh chỉnh> --pretrain pf
  ... --objective ldl --init-ckpt <...> --pretrain pf

Seed bằng số fold. Checkpoint chọn trên 10 người validation của fold.
Mỗi lần chạy được ghi vào ``results/runs.csv``.
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from earsr import runlog  # noqa: E402
from earsr.data.ami import read_manifest  # noqa: E402
from earsr.data.build_lr import build_lr_set  # noqa: E402
from earsr.data.datasets import list_images  # noqa: E402
from earsr.data.splits import load_folds  # noqa: E402
from earsr.degrade.pipelines import DegradeParams, needs_params  # noqa: E402
from earsr.eval.infer import evaluate_on_bench  # noqa: E402
from earsr.io import to_tensor, to_uint8  # noqa: E402
from earsr.models.optional.heads import TwoHeadGated  # noqa: E402
from earsr.models.variants import arch_part  # noqa: E402
from earsr.runid import RunId  # noqa: E402
from earsr.train.finetune import (build_n3, build_trainable, make_train_set, make_validator,  # noqa: E402
                                  trainable_fraction, val_pairs)
from earsr.train.objectives import make_objective  # noqa: E402
from earsr.train.trainer import TrainConfig, train  # noqa: E402

HPARAMS = ("iters", "batch_size", "patch_lr", "lr", "ema", "amp", "init_ckpt", "degrade_params", "extra_dir",
           "extra_roles", "extra_role", "extra_min_downscale", "extra_prob", "landmarks", "aux_mode", "aux_sigma",
           "w_aux", "tau", "w_gan", "w_percep", "lr_d", "select_metric", "select", "val_tiers", "photo_aug")
OBJECTIVES = ("l1", "gan", "ldl", "aux", "n3s1A", "n3s1B", "n3s2")


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp", required=True)
    ap.add_argument("--backbone", default="span")
    ap.add_argument("--variant", default="zero", help="zero|replicate|reflect[-wide|-deep][-cNN][-bNN]")
    ap.add_argument("--pretrain", default="pub", help="pub | none | psN | pf")
    ap.add_argument("--init-ckpt", default=None, help="checkpoint khởi tạo (bắt buộc với psN, pf)")
    ap.add_argument("--protocol", default="rand", choices=["native", "fixed", "rand"])
    ap.add_argument("--scale", type=int, default=4)
    ap.add_argument("--tier", type=int, default=144, help="cỡ của ô đang xét (cho 'fixed')")
    ap.add_argument("--degrade", default="bic", help="bic | bicjpegQ | generic | est | jpegmix | jpegu")
    ap.add_argument("--degrade-params", default=None, help="file tham số cho 'est' và 'jpegmix' (configs/degrade_estimated.json)")
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--ami-raw", required=True)
    ap.add_argument("--bench", required=True)
    ap.add_argument("--folds", default="splits/ami_5fold.json")
    ap.add_argument("--out", default="runs")
    ap.add_argument("--runs-csv", default="results/runs.csv")
    ap.add_argument("--tag", default="", help="nhãn thêm vào mã lần chạy, ví dụ lr1e-4")
    ap.add_argument("--force", action="store_true", help="chạy lại dù sổ ghi đã có trạng thái 'done'")
    ap.add_argument("--print-run-id", action="store_true", help="chỉ in mã lần chạy rồi thoát")
    # tối ưu
    ap.add_argument("--objective", default="l1", choices=OBJECTIVES)
    ap.add_argument("--iters", type=int, default=20000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--patch-lr", type=int, default=24)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--val-every", type=int, default=1000)
    ap.add_argument("--log-every", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--select-metric", default="psnr", choices=["psnr", "lpips"],
                    help="số đo validation để chọn checkpoint; mục tiêu có GAN mặc định giữ checkpoint cuối")
    ap.add_argument("--select", default="auto", choices=["auto", "max", "last"])
    ap.add_argument("--val-tiers", type=int, nargs="*", default=None)
    # dữ liệu thêm
    ap.add_argument("--extra-dir", nargs="*", default=[], help="thư mục ảnh thêm (root/<người>/<ảnh>)")
    ap.add_argument("--extra-roles", default=None, help="file vai theo người; chỉ lấy người có vai --extra-role")
    ap.add_argument("--extra-role", default="train")
    ap.add_argument("--extra-name", default="extra", help="tên ngắn đưa vào mã lần chạy")
    ap.add_argument("--extra-min-downscale", type=float, default=2.0)
    ap.add_argument("--extra-prob", type=float, default=None)
    ap.add_argument("--photo-aug", type=float, default=0.75,
                    help="xác suất đổi độ sáng, tông và màu của ảnh HR trước khi suy giảm (0 để tắt). Bật theo mặc định: "
                         "tinh chỉnh chỉ trên AMI mà không có nó cho mô hình hỏng trên ảnh có vùng sáng")
    ap.add_argument("--no-stress", action="store_true", help="bỏ phép thử ảnh sáng sau khi huấn luyện")
    # N1
    ap.add_argument("--landmarks", default=None, help=".npz: paths (đường dẫn ảnh), points (N, K, 2)")
    ap.add_argument("--aux-mode", default="heatmap", choices=["heatmap", "points"])
    ap.add_argument("--aux-sigma", type=float, default=1.0)
    ap.add_argument("--w-aux", type=float, default=0.1)
    # N3 và GAN
    ap.add_argument("--tau", type=float, default=0.02)
    ap.add_argument("--w-gan", type=float, default=0.1)
    ap.add_argument("--w-percep", type=float, default=1.0)
    ap.add_argument("--lr-d", type=float, default=1e-4)
    # chấm test
    ap.add_argument("--eval-tiers", type=int, nargs="*", default=None)
    ap.add_argument("--no-test", action="store_true")
    ap.add_argument("--no-perceptual", action="store_true")
    ap.add_argument("--ridge", action="store_true", help="thêm bảng kiểm tra gờ vào số đo test")
    return ap.parse_args(argv)


def load_landmarks(path: str) -> dict:
    z = np.load(path, allow_pickle=False)
    return {str(p): pts for p, pts in zip(z["paths"], z["points"])}


def extra_paths(a) -> list:
    out = []
    roles = json.loads(Path(a.extra_roles).read_text()) if a.extra_roles else None
    for d in a.extra_dir:
        for p in list_images(d):
            if roles is None or roles.get(p.parent.name) == a.extra_role:
                out.append(p)
    if a.extra_dir and not out:
        raise SystemExit("--extra-dir không cho ảnh nào (kiểm tra --extra-roles và --extra-role)")
    return out


def main(argv=None) -> dict:
    a = parse_args(argv)
    if a.fold in (1, 5) and a.exp.upper().startswith(("T4", "T5", "T6", "LR")):
        raise SystemExit("fold 1 và 5 là hai fold giữ kín: các phép thử quyết định và việc dò tốc độ học "
                         "chỉ chạy trên fold 2, 3, 4")
    tags = [t for t in ([] if a.objective == "l1" else [a.objective]) + ([f"x{a.extra_name}"] if a.extra_dir else [])
            + ([a.tag] if a.tag else []) if t]
    variant_id = "+".join([a.variant] + tags)
    tier_tag = "all" if a.protocol == "rand" else str(a.tier)
    rid = RunId(a.exp, a.backbone, variant_id, a.pretrain, a.protocol, a.scale, tier_tag, a.degrade, a.fold)
    run_dir = Path(a.out) / str(rid)
    if a.print_run_id:
        print(rid)
        return {"run_id": str(rid)}
    # Mã lần chạy không chứa mọi siêu tham số. Nếu thư mục đã có một lần chạy với siêu tham số khác,
    # dừng lại thay vì lặng lẽ tiếp tục từ checkpoint của cấu hình kia.
    hp = {k: getattr(a, k) for k in HPARAMS}
    old_cfg = run_dir / "config.json"
    if old_cfg.exists() and not a.force:
        old = json.loads(old_cfg.read_text()).get("extra", {}).get("hparams")
        diff = {k: (old.get(k), v) for k, v in hp.items() if old is not None and old.get(k) != v}
        if diff:
            raise SystemExit(f"{rid}: thư mục đã có lần chạy với siêu tham số khác {diff}. "
                             "Thêm --tag để tạo mã mới, hoặc --force để ghi tiếp vào thư mục này.")
    if not a.force and runlog.status_of(a.runs_csv, str(rid)) == "done" and (run_dir / "summary.json").exists():
        print(f"bỏ qua {rid}: sổ ghi đã có trạng thái 'done' (dùng --force để chạy lại)")
        return {"run_id": str(rid), "skipped": True}
    folds = load_folds(a.folds)
    params = DegradeParams.load(a.degrade_params) if a.degrade_params else None
    if needs_params(a.degrade) and params is None:
        raise SystemExit(f"--degrade {a.degrade} cần --degrade-params")
    lr_params = params if needs_params(a.degrade) else None
    rand_params = params if needs_params(a.degrade) or a.degrade == "generic" else None
    have = {r["tier"] for r in read_manifest(Path(a.bench) / "manifest.csv")}
    main_tiers = [t for t in (96, 144, 192) if t in have]
    val_tiers = a.val_tiers or (main_tiers if a.protocol == "rand" else [a.tier])
    eval_tiers = a.eval_tiers or (main_tiers if a.protocol == "rand" else [a.tier])
    for t in sorted(set(val_tiers) | set(eval_tiers)):
        build_lr_set(a.bench, t, a.scale, a.degrade, params=lr_params)

    # ---- mô hình và mục tiêu
    is_n3 = a.objective.startswith("n3")
    if is_n3:
        if a.backbone != "span":
            raise SystemExit("N3 chỉ hỗ trợ thân span")
        if a.objective == "n3s2":
            if not a.init_ckpt:
                raise SystemExit("n3s2 cần --init-ckpt là best.pt của giai đoạn 1")
            model = build_n3(a.variant, a.scale)
            model.load_state_dict(torch.load(a.init_ckpt, map_location="cpu", weights_only=False)["model"])
        else:
            if a.objective == "n3s1A" and not a.init_ckpt:
                raise SystemExit("n3s1A cần --init-ckpt là thân đã tinh chỉnh")
            model = build_n3(a.variant, a.scale, a.init_ckpt)
    else:
        model = build_trainable(a.backbone, a.variant, a.pretrain, a.scale, a.init_ckpt)
    lm = load_landmarks(a.landmarks) if a.landmarks else None
    if a.objective == "aux" and lm is None:
        raise SystemExit("--objective aux cần --landmarks")
    adv = dict(w_gan=a.w_gan, w_percep=a.w_percep, lr_d=a.lr_d)
    n_maps = (next(iter(lm.values())).shape[0] if a.aux_mode == "heatmap" else 1) if lm else 0
    obj_kw = {"gan": adv, "ldl": adv, "n3s1A": adv, "n3s1B": adv, "n3s2": {"tau": a.tau},
              "aux": {"n_maps": n_maps, "w_aux": a.w_aux}}.get(a.objective, {})
    objective = make_objective(a.objective, model, **obj_kw)

    # ---- dữ liệu
    ds_kw = {}
    if a.objective == "aux":
        ds_kw = dict(landmarks=lm, n_landmarks=next(iter(lm.values())).shape[0], aux_mode=a.aux_mode,
                     aux_sigma=a.aux_sigma)
    ds_kw["photo_prob"] = a.photo_aug
    extras = extra_paths(a)
    train_set = make_train_set(a.ami_raw, a.folds, a.fold, a.scale, a.protocol, a.tier, a.patch_lr,
                               degrade_kind=a.degrade, params=rand_params,
                               seed=rid.seed, extra_paths=extras, extra_min_downscale=a.extra_min_downscale,
                               extra_prob=a.extra_prob, **ds_kw)
    perc = None
    if a.select_metric == "lpips" or not (a.no_test or a.no_perceptual):
        from earsr.eval.metrics import PerceptualMetrics

        perc = PerceptualMetrics(a.device)
        if not perc.available:
            print(f"không có số đo cảm nhận: {perc.reason}")
            perc = None
    validate = make_validator(a.bench, a.folds, a.fold, a.scale, tuple(val_tiers), a.degrade, a.device,
                              metric=a.select_metric, perceptual=perc)
    cfg = TrainConfig(run_id=str(rid), out_dir=a.out, iters=a.iters, batch_size=a.batch_size, lr=a.lr,
                      val_every=a.val_every, save_every=a.val_every, num_workers=a.workers, seed=rid.seed,
                      device=a.device, ema=a.ema, amp=a.amp, select=a.select, log_every=a.log_every,
                      extra={"backbone": a.backbone, "variant": arch_part(a.variant), "scale": a.scale,
                             "model_kind": "n3" if is_n3 else "plain", "objective": a.objective,
                             "pretrain": a.pretrain, "init_ckpt": a.init_ckpt, "protocol": a.protocol,
                             "tier": a.tier, "degrade": a.degrade, "degrade_params": a.degrade_params,
                             "fold": a.fold, "folds_sha1": folds["sha1"], "patch_lr": a.patch_lr,
                             "val_tiers": val_tiers, "select_metric": a.select_metric,
                             "n_train_images": len(train_set.paths), "n_extra_images": len(extras),
                             "extra_dir": a.extra_dir, "landmarks": a.landmarks, "tau": a.tau,
                             "ami_raw": str(a.ami_raw), "bench": str(a.bench), "argv": sys.argv[1:]})
    cfg.extra["hparams"] = hp
    runlog.record(a.runs_csv, str(rid), "running", exp=a.exp, fold=a.fold, objective=a.objective)
    try:
        summary = train(model, train_set, validate, cfg, objective)
        print(summary, f"| tỉ lệ tham số được học: {trainable_fraction(model):.3f}")
        out = dict(summary)
        if not a.no_stress:
            out.update(stress(a, run_dir, model, val_tiers, rand_params))
        if not a.no_test:
            out.update(test(a, rid, run_dir, model, folds, eval_tiers, val_tiers, perc))
    except BaseException as e:
        runlog.record(a.runs_csv, str(rid), "failed", reason=f"{type(e).__name__}: {str(e)[:200]}")
        traceback.print_exc()
        raise
    runlog.record(a.runs_csv, str(rid), "done", **{k: v for k, v in out.items() if k != "run_id"})
    return out


def stress(a, run_dir: Path, model, val_tiers, params) -> dict:
    """Phép thử ảnh sáng trên checkpoint đã chọn (earsr/eval/stress.py). Chỉ báo động, không chọn checkpoint."""
    from earsr.device import full_precision
    from earsr.eval.stress import brightness_stress, describe

    best = torch.load(run_dir / "ckpt" / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(best["model"])
    model = model.to(a.device).eval()
    hrs = [hr for _, hr in val_pairs(a.bench, a.folds, a.fold, a.scale, tuple(val_tiers), a.degrade)]

    @torch.no_grad()
    def predict(lr):
        with full_precision():
            return to_uint8(model(to_tensor(lr).to(a.device)))

    res = brightness_stress(predict, hrs, a.scale, a.degrade, params)
    with open(run_dir / "stress.json", "w") as f:
        json.dump(res, f, indent=1)
    print(("" if res["stable"] else "CẢNH BÁO: ") + "phép thử ảnh sáng: " + describe(res), flush=True)
    return {"stress_stable": res["stable"], "stress_psnr_y": round(res["model"][-1], 3),
            "stress_bicubic_psnr_y": round(res["bicubic"][-1], 3), "stress_gain": res["gains"][-1],
            "stress_gap_drop": round(res["gap_drop"], 3), "stress_n_broken": sum(res["n_broken"])}


def test(a, rid, run_dir: Path, model, folds, eval_tiers, val_tiers, perc) -> dict:
    """Chấm trên người test của fold, bằng checkpoint đã chọn theo validation."""
    best = torch.load(run_dir / "ckpt" / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(best["model"])
    model = model.to(a.device).eval()
    points = {"": None}
    if isinstance(model, TwoHeadGated):
        if a.objective == "n3s2":
            from earsr.models.optional.n3 import choose_operating_points

            from earsr.device import full_precision

            with full_precision():
                ops = choose_operating_points(model, val_pairs(a.bench, a.folds, a.fold, a.scale, tuple(val_tiers),
                                                               a.degrade), a.scale)
            with open(run_dir / "operating_points.json", "w") as f:
                json.dump(ops, f, indent=1)
            print(f"điểm vận hành (chọn trên validation): F={ops['F']}  P={ops['P']}")
            points = {f"_op{k}": {"gate_bias": ops[k]} for k in ("F", "P") if ops[k] is not None}
        points.update({"_headS": {"gate": 1.0}, "_headT": {"gate": 0.0}})
        points.pop("", None)
    extra = []
    if a.ridge:
        from earsr.eval.ridge import ridge_check

        extra.append(lambda sr, hr, row: ridge_check(sr, hr))
    test_subj = set(folds["folds"][str(a.fold)]["test"])
    res = {}
    for suffix, op in points.items():
        @torch.no_grad()
        def predict(lr, op=op):
            x = to_tensor(lr).to(a.device)
            if op is None:
                return to_uint8(model(x))
            if "gate_bias" in op:
                model.gate_bias = op["gate_bias"]
                return to_uint8(model(x))
            return to_uint8(model(x, gate=op["gate"]))

        for t in eval_tiers:
            out = run_dir / f"test_hr{t}_x{a.scale}_{a.degrade}{suffix}.csv"
            evaluate_on_bench(str(rid) + suffix, a.bench, t, a.scale, a.degrade, a.folds, out, device=a.device,
                              predictor=predict, perceptual=perc, extra_metrics=extra, subjects=test_subj)
            import pandas as pd

            d = pd.read_csv(out, dtype={"subject": str})
            res[f"test_psnr_y_hr{t}{suffix}"] = round(float(d.psnr_y.mean()), 4)
            print(f"{out.name}: {len(d)} ảnh test, PSNR-Y {d.psnr_y.mean():.3f}"
                  + (f", LPIPS {d.lpips.mean():.4f}" if "lpips" in d else ""))
    return res


if __name__ == "__main__":
    main()
