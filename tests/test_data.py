import json
from pathlib import Path

import numpy as np
import pytest

from earsr.data import ami
from earsr.data.build_lr import build_lr_set
from earsr.data.datasets import SRTrainDataset, resize_short_side
from earsr.data.splits import check_folds, fold_of_subject, load_folds, make_folds
from earsr.degrade.pipelines import DegradeParams, degrade
from earsr.io import imread_rgb, imwrite_rgb

ROOT = Path(__file__).resolve().parents[1]


def _fake_ami(tmp, n_subj=10, hw=(140, 100)):
    rng = np.random.default_rng(0)
    import cv2

    for s in range(n_subj):
        for v in ami.VIEWS:
            img = cv2.GaussianBlur(rng.integers(0, 256, (*hw, 3), dtype=np.uint8), (0, 0), 2.0)
            cv2.imwrite(str(tmp / f"{s:03d}_{v}_ear.jpg"), img)
    (tmp / ".w_test").write_text("")       # file lạ phải bị bỏ qua
    (tmp / "note_ear.jpg").write_text("x")  # tên sai mẫu phải bị bỏ qua
    return tmp


def test_hr_sizes_divisible_by_4():
    assert {t: ami.hr_size(t) for t in ami.TIERS} == {96: (136, 96), 144: (204, 144), 192: (272, 192), 244: (348, 244)}


def test_scan_ignores_foreign_files(tmp_path):
    (tmp_path / "raw").mkdir()
    raw = _fake_ami(tmp_path / "raw")
    items = ami.scan_ami(raw)
    rep = ami.check_ami(items)
    assert rep == {"n_images": 70, "n_subjects": 10, "incomplete_subjects": [], "duplicate_keys": 0}


def test_strict_build_rejects_incomplete_set(tmp_path):
    (tmp_path / "raw").mkdir()
    raw = _fake_ami(tmp_path / "raw")
    with pytest.raises(RuntimeError):
        ami.build_ami_benchmark(raw, tmp_path / "bench")  # không đủ 100 người


def test_folds_have_no_leak_and_each_subject_tested_once():
    d = make_folds([f"{i:03d}" for i in range(100)])
    check_folds(d)
    f = fold_of_subject(d)
    assert len(f) == 100 and sorted(set(f.values())) == [1, 2, 3, 4, 5]
    for k, fd in d["folds"].items():
        assert (len(fd["train"]), len(fd["val"]), len(fd["test"])) == (70, 10, 20)
    assert make_folds([f"{i:03d}" for i in range(100)])["sha1"] == d["sha1"]  # tất định


def test_shipped_fold_file_is_valid():
    d = load_folds(ROOT / "splits" / "ami_5fold.json")  # load_folds tự kiểm rò rỉ và mã băm
    assert d["held_out_folds"] == [1, 5] and d["dev_folds"] == [2, 3, 4]


def test_tampered_fold_file_is_rejected(tmp_path):
    d = json.loads((ROOT / "splits" / "ami_5fold.json").read_text())
    d["folds"]["1"]["train"].append(d["folds"]["1"]["test"][0])
    p = tmp_path / "bad.json"
    p.write_text(json.dumps(d))
    with pytest.raises(AssertionError):
        load_folds(p)


def test_degrade_is_deterministic_and_keyed():
    rng = np.random.default_rng(1)
    hr = rng.integers(0, 256, (64, 48, 3), dtype=np.uint8)
    for kind in ("bic", "bicjpeg75", "generic"):
        a = degrade(hr, 4, kind, seed=7, key="x")
        b = degrade(hr, 4, kind, seed=7, key="x")
        assert np.array_equal(a, b) and a.shape == (16, 12, 3)
    assert not np.array_equal(degrade(hr, 4, "generic", seed=7, key="x"), degrade(hr, 4, "generic", seed=7, key="y"))
    with pytest.raises(ValueError):
        degrade(hr, 4, "est")  # thiếu tham số ước lượng
    assert degrade(hr, 4, "est", params=DegradeParams()).shape == (16, 12, 3)


def test_lr_set_is_reproducible(tmp_path):
    (tmp_path / "raw").mkdir()
    raw = _fake_ami(tmp_path / "raw", n_subj=3, hw=(702, 492))
    ami.build_ami_benchmark(raw, tmp_path / "b", tiers=(96,), strict=False)
    d1 = build_lr_set(tmp_path / "b", 96, 4, "generic", seed=3)
    m1 = (d1 / "manifest.csv").read_text()
    build_lr_set(tmp_path / "b", 96, 4, "generic", seed=3, overwrite=True)
    assert (d1 / "manifest.csv").read_text() == m1
    lr = imread_rgb(d1 / "000_back.png")
    assert lr.shape == (34, 24, 3)


def test_train_dataset_protocols_and_alignment(tmp_path):
    rng = np.random.default_rng(0)
    import cv2

    paths = []
    for i in range(3):
        p = tmp_path / f"{i}.png"
        imwrite_rgb(p, cv2.GaussianBlur(rng.integers(0, 256, (702, 492, 3), dtype=np.uint8), (0, 0), 3.0))
        paths.append(p)
    for proto in ("native", "fixed", "rand"):
        ds = SRTrainDataset(paths, scale=4, protocol=proto, patch_lr=24, tier=144, seed=5)
        lr, hr = ds[11]
        assert lr.shape == (3, 24, 24) and hr.shape == (3, 96, 96)
        lr2, hr2 = ds[11]
        assert (lr == lr2).all() and (hr == hr2).all()      # cùng chỉ số, cùng mẫu
        assert not (ds[12][1] == hr).all()
    # LR của patch phải bằng đúng phần cắt của LR cả ảnh (không thu nhỏ riêng patch)
    ds = SRTrainDataset(paths, scale=4, protocol="fixed", patch_lr=24, tier=96, hflip=False, seed=1)
    lr, hr = ds[0]
    found = False
    for p in paths:
        full_hr = resize_short_side(imread_rgb(p), 96)
        full_lr = degrade(full_hr, 4, "bic")
        hp = (hr.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
        lp = (lr.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
        for y in range(full_lr.shape[0] - 23):
            for x in range(full_lr.shape[1] - 23):
                if np.array_equal(full_lr[y:y + 24, x:x + 24], lp):
                    assert np.array_equal(full_hr[4 * y:4 * y + 96, 4 * x:4 * x + 96], hp)
                    found = True
    assert found
