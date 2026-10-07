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


EST_SHA1 = "cbcb522b176d94e73354fbf09137b1c99789279e"   # tính ngày 07/10/2026, trước khi thêm jpegmix và jpegu


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


def test_jpeg_only_kinds_isolate_the_quality_distribution():
    """jpegmix và jpegu chỉ nén: với cùng mức nén, ảnh ra phải trùng bicjpegQ từng bit (không mờ, không nhiễu)."""
    from earsr.degrade.pipelines import JPEGU_RANGE, needs_params
    rng = np.random.default_rng(2)
    hr = rng.integers(0, 256, (64, 48, 3), dtype=np.uint8)
    one = DegradeParams(jpeg_q=[[75, 1.0]])
    assert np.array_equal(degrade(hr, 4, "jpegmix", seed=3, key="a", params=one), degrade(hr, 4, "bicjpeg75"))
    two = DegradeParams(jpeg_q=[[75, 0.5], [93, 0.5]])
    refs = [degrade(hr, 4, "bicjpeg75"), degrade(hr, 4, "bicjpeg93")]
    hit = [next(i for i, r in enumerate(refs) if np.array_equal(degrade(hr, 4, "jpegmix", seed=0, key=str(k), params=two), r))
           for k in range(40)]
    assert set(hit) == {0, 1}                                  # cả hai mức đều được rút, và không có ảnh nào khác
    with pytest.raises(ValueError):
        degrade(hr, 4, "jpegmix")                              # thiếu phân bố mức nén
    lo, hi = JPEGU_RANGE
    pool = {q: degrade(hr, 4, f"bicjpeg{q}") for q in range(lo, hi + 1)}
    seen = set()
    for k in range(200):
        x = degrade(hr, 4, "jpegu", seed=0, key=str(k))
        seen.add(next(q for q, r in pool.items() if np.array_equal(x, r)))
    assert min(seen) <= lo + 3 and max(seen) >= hi - 3 and len(seen) > 20
    assert np.array_equal(degrade(hr, 4, "jpegu", seed=5, key="z"), degrade(hr, 4, "jpegu", seed=5, key="z"))
    assert needs_params("est") and needs_params("jpegmix") and not needs_params("jpegu") and not needs_params("bicjpeg75")


def test_est_output_unchanged_by_refactor():
    """Mã băm cố định của kiểu est: chặn việc sửa pipelines.py làm đổi ảnh LR của các lần chạy đã có."""
    import hashlib
    rng = np.random.default_rng(3)
    hr = rng.integers(0, 256, (64, 48, 3), dtype=np.uint8)
    p = DegradeParams(blur_sigma=(0.2, 0.6), noise_sigma=(0.85, 6.27), jpeg_q=[[75, 0.52], [93, 0.44], [81, 0.04]])
    h = hashlib.sha1(b"".join(degrade(hr, 4, "est", seed=1, key=str(k), params=p).tobytes() for k in range(8))).hexdigest()
    assert h == EST_SHA1


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
