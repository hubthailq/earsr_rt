"""Chạy thử đầu cuối trên dữ liệu giả: dựng benchmark, chia fold, sinh LR, huấn
luyện vài bước, chấm, tổng hợp. Bắt lỗi nối giữa các giai đoạn."""
import cv2
import numpy as np
import pandas as pd
import torch

from earsr.data import ami
from earsr.data.build_lr import build_lr_set
from earsr.data.datasets import SRTrainDataset
from earsr.data.splits import make_folds, save_folds
from earsr.eval.infer import evaluate_on_ami
from earsr.io import to_tensor, to_uint8
from earsr.models.span import SPAN
from earsr.report.t2 import load_dir, quality_table
from earsr.train.finetune import ami_train_paths, make_validator
from earsr.train.trainer import TrainConfig, curve_is_flat, train


def test_end_to_end_on_synthetic_data(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    rng = np.random.default_rng(0)
    subjects = [f"{i:03d}" for i in range(10)]
    for s in subjects:
        for v in ami.VIEWS:
            img = cv2.GaussianBlur(rng.integers(0, 256, (702, 492, 3), dtype=np.uint8), (0, 0), 6.0)
            cv2.imwrite(str(raw / f"{s}_{v}_ear.jpg"), img)
    bench = tmp_path / "bench"
    ami.build_ami_benchmark(raw, bench, tiers=(96,), strict=False)
    folds = tmp_path / "folds.json"
    save_folds(make_folds(subjects, n_folds=5, n_val=2), folds)
    build_lr_set(bench, 96, 4, "bic")

    paths = ami_train_paths(raw, folds, 2, "train")
    assert len(paths) == 6 * 7
    ds = SRTrainDataset(paths, scale=4, protocol="fixed", patch_lr=16, tier=96, seed=2)
    torch.manual_seed(0)
    model = SPAN(feature_channels=8, n_blocks=2)
    validate = make_validator(bench, folds, 2, 4, (96,), "bic")
    cfg = TrainConfig(run_id="TEST_span-zero_none_fixed_x4_hr96_bic_f2", out_dir=str(tmp_path / "runs"), iters=6,
                      batch_size=2, val_every=2, save_every=2, num_workers=0, seed=2, device="cpu", ema=0.0,
                      checkpoints_at=[3])
    s1 = train(model, ds, validate, cfg)
    run = tmp_path / "runs" / cfg.run_id
    assert s1["iters"] == 6 and (run / "ckpt" / "best.pt").exists() and (run / "ckpt" / "step3.pt").exists()
    log = pd.read_csv(run / "log.csv")
    assert list(log.step) == [2, 4, 6]
    assert "flat" in curve_is_flat(run / "log.csv")
    # chạy lại cùng cấu hình: tiếp tục từ checkpoint, không huấn luyện thêm bước nào
    s2 = train(SPAN(feature_channels=8, n_blocks=2), ds, validate, cfg)
    assert s2["iters"] == 6 and len(pd.read_csv(run / "log.csv")) == 3

    model.eval()

    @torch.no_grad()
    def predict(lr):
        return to_uint8(model(to_tensor(lr)))

    out = tmp_path / "res"
    evaluate_on_ami("tiny", bench, 96, 4, "bic", folds, out / "tiny__hr96_x4_bic.csv", predictor=predict)
    evaluate_on_ami("bicubic", bench, 96, 4, "bic", folds, out / "bicubic__hr96_x4_bic.csv")
    df = load_dir(out)
    assert len(df) == 140 and set(df.fold) == {1, 2, 3, 4, 5}
    q = quality_table(df, n_boot=100)
    assert set(q.model) == {"tiny", "bicubic"} and "gain" in q.columns
    assert q[q.model == "bicubic"].psnr_y.iloc[0] > 20
