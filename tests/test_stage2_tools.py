"""Công cụ của giai đoạn 2 và phần tùy chọn: điểm mốc, bảng kiểm tra gờ, bộ phân
loại "mô phỏng hay thật", cổng oracle, tiêu chí, bảng LaTeX, khảo sát người xem,
sổ ghi, hàng đợi, ảnh ngoài thực tế. Tất cả trên dữ liệu giả, CPU."""
import importlib.util
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest
import torch

from earsr import runlog
from earsr.data import wild
from earsr.degrade.pipelines import DegradeParams, degrade
from earsr.eval import oracle
from earsr.eval.metrics import fr_metrics
from earsr.eval.metrics_nr import NoReferenceMetrics, add_ntire_score, ntire_perceptual_score
from earsr.eval.realism import realism_test, simulate_small
from earsr.eval.ridge import ridge_check
from earsr.io import imread_rgb, imwrite_rgb
from earsr.landmarks.data import LandmarkDataset, apply_matrix, crop_matrix, split_items
from earsr.landmarks.metric import LandmarkScorer, box_to_pixels, load_detector, save_detector
from earsr.landmarks.nets import HeatmapNet, RegressNet, build_detector, soft_argmax
from earsr.landmarks.pts import find_pairs, read_pts, write_pts
from earsr.landmarks.train import train_detector
from earsr.report import criteria as C
from earsr.report import t4, viewer
from earsr.report.latex import booktabs, t2_table

ROOT = Path(__file__).resolve().parents[1]


def _script(name):
    sys.path.insert(0, str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _texture(rng, h, w, sigma=3.0):
    return cv2.GaussianBlur(rng.integers(0, 256, (h, w, 3), dtype=np.uint8), (0, 0), sigma)


# ----------------------------------------------------------------- điểm mốc

def test_pts_roundtrip_and_pairs(tmp_path):
    pts = np.array([[10.5, 20.25], [0.0, 3.0], [99.0, 7.5]], np.float32)
    write_pts(tmp_path / "a.pts", pts)
    assert "n_points: 3" in (tmp_path / "a.pts").read_text()
    assert np.allclose(read_pts(tmp_path / "a.pts"), pts, atol=1e-3)
    assert np.allclose(read_pts(tmp_path / "a.pts", one_based=False), pts + 1, atol=1e-3)
    cv2.imwrite(str(tmp_path / "a.jpg"), np.zeros((8, 8, 3), np.uint8))
    write_pts(tmp_path / "orphan.pts", pts)
    assert find_pairs(tmp_path) == [(tmp_path / "a.jpg", tmp_path / "a.pts")]
    (tmp_path / "bad.pts").write_text("version: 1\nn_points: 5\n{\n1 2\n}\n")
    with pytest.raises(ValueError):
        read_pts(tmp_path / "bad.pts")


def test_soft_argmax_and_coordinate_conventions():
    hm = torch.full((1, 2, 10, 12), -50.0)
    hm[0, 0, 3, 7] = 50.0
    hm[0, 1, 9, 0] = 50.0
    c = soft_argmax(hm)
    assert torch.allclose(c[0, 0], torch.tensor([7.0, 3.0]), atol=1e-3)
    assert torch.allclose(c[0, 1], torch.tensor([0.0, 9.0]), atol=1e-3)
    for arch in ("heatmap", "regress"):
        net = build_detector(arch, n_points=4, width=8, input_hw=(64, 48)).eval()
        assert net.predict(torch.rand(2, 3, 64, 48)).shape == (2, 4, 2)
    with pytest.raises(ValueError):
        HeatmapNet(input_hw=(60, 50))
    # RegressNet: đầu ra 0 ứng với tâm ảnh
    r = RegressNet(n_points=1, width=8, input_hw=(64, 48)).eval()
    for p in r.fc[-1].parameters():
        torch.nn.init.zeros_(p)
    assert torch.allclose(r.predict(torch.rand(1, 3, 64, 48))[0, 0], torch.tensor([23.5, 31.5]))


def test_crop_matrix_moves_points_with_the_image():
    img = np.zeros((300, 260, 3), np.uint8)
    pts = np.array([[80, 90], [150, 100], [120, 200], [95, 160]], np.float32)
    for x, y in pts:
        cv2.circle(img, (int(x), int(y)), 4, (255, 255, 255), -1)
    for angle, flip in [(0, False), (15, False), (-10, True)]:
        m = crop_matrix(pts, (136, 96), (0.2, 0.1, 0.3, 0.15), angle, flip)
        out = cv2.warpAffine(img, m, (96, 136), flags=cv2.INTER_LINEAR)
        q = apply_matrix(pts, m)
        assert (q[:, 0] > 0).all() and (q[:, 0] < 95).all() and (q[:, 1] > 0).all() and (q[:, 1] < 135).all()
        for x, y in q:
            assert out[int(round(y)), int(round(x))].mean() > 120, (angle, flip)
    # không méo: tỉ lệ co theo x và y bằng nhau
    m = crop_matrix(pts, (136, 96), (0.1,) * 4)
    assert abs(np.linalg.norm(m[:, 0]) - np.linalg.norm(m[:, 1])) < 1e-9


def _dots_dataset(d, n=24, k=4, seed=0):
    rng = np.random.default_rng(seed)
    cols = [(255, 60, 60), (60, 255, 60), (60, 60, 255), (255, 255, 60)]
    items = []
    for i in range(n):
        img = (rng.integers(0, 40, (200, 150, 3))).astype(np.uint8)
        base = np.array([[45, 50], [105, 60], [95, 150], [50, 140]], np.float32) + rng.uniform(-12, 12, (k, 2))
        for (x, y), c in zip(base, cols):
            cv2.circle(img, (int(round(x)), int(round(y))), 7, c, -1)
        p = d / f"im{i:02d}.png"
        cv2.imwrite(str(p), img[:, :, ::-1])
        write_pts(p.with_suffix(".pts"), base)
        items.append((p, base))
    return items


def test_detector_trains_and_scorer_measures(tmp_path):
    _dots_dataset(tmp_path, n=40)
    items = [(p, read_pts(q)) for p, q in find_pairs(tmp_path)]
    tr_a, va_a = split_items(items, "a")
    tr_b, va_b = split_items(items, "b")
    names = lambda its: {p.name for p, _ in its}
    assert not (names(tr_a + va_a) & names(tr_b + va_b)) and len(tr_a + va_a + tr_b + va_b) == len(items)
    ds = LandmarkDataset(items, (64, 48), train=True, seed=0)
    x, y = ds[0]
    assert x.shape == (3, 64, 48) and y.shape == (4, 2)
    tr, va = split_items(items, "all", val_frac=0.2)
    ck = tmp_path / "w" / "lm.pt"
    res = train_detector(tr, va, ck, arch="heatmap", width=8, input_hw=(64, 48), iters=300, batch_size=8,
                         lr=3e-3, val_every=100, workers=0, device="cpu", seed=0)
    # chấm tròn bán kính 7 px trên ảnh 150×200: sai số dưới 3% đường chéo hộp nghĩa là bộ dò
    # thật sự định vị được (đoán vị trí trung bình cho khoảng 10%)
    assert res["best_val_nme"] < 0.03, res["history"]
    net, info = load_detector(ck)
    assert info["arch"] == "heatmap" and info["n_points"] == 4 and tuple(info["input_hw"]) == (64, 48)
    sc = LandmarkScorer(ck)
    img = imread_rgb(items[0][0])
    p = sc.points(img)
    assert p.shape == (4, 2) and (p[:, 0] < 150).all() and (p[:, 1] < 200).all()
    assert sc.deviation(img, img) == 0.0
    assert np.abs(p - items[0][1]).max() < 6.0, "điểm mốc dò được phải gần nhãn, theo toạ độ ảnh gốc"
    blurred = cv2.GaussianBlur(img, (0, 0), 6.0)
    fl = sc.noise_floor(img)
    assert set(fl) == {"lm_floor_noise", "lm_floor_jpeg"} and fl["lm_floor_noise"] >= 0
    assert sc.deviation(blurred, img) >= 0
    t, l, b, r = sc.box(img)
    assert 0 <= t < b <= 1 and 0 <= l < r <= 1
    assert box_to_pixels((0.1, 0.2, 0.9, 0.8), 100, 50) == (10, 10, 90, 40)
    m = sc.metric_fn()(blurred, img, {"file": "x.png"})
    assert set(m) == {"lm_dev", "lm_floor_noise", "lm_floor_jpeg"}
    assert m == sc.metric_fn()(blurred, img, {"file": "x.png"}), "sàn nhiễu phải tái lập được"
    # bộ dò kiểu hồi quy: lưu và nạp lại
    reg = build_detector("regress", 4, 8, (64, 48))
    save_detector(tmp_path / "w" / "reg.pt", reg)
    assert load_detector(tmp_path / "w" / "reg.pt")[1]["arch"] == "regress"


# ----------------------------------------------------------------- gờ, chỉ số không tham chiếu

def test_ridge_check_counts_false_and_missed():
    hr = np.full((120, 100, 3), 60, np.uint8)
    cv2.ellipse(hr, (50, 60), (30, 45), 0, 0, 360, (200, 200, 200), 2)
    same = ridge_check(hr, hr)
    assert same["ridge_false"] == 0 and same["ridge_missed"] == 0 and same["ridge_f1"] == 1
    fake = hr.copy()
    cv2.line(fake, (10, 10), (90, 30), (220, 220, 220), 2)
    f = ridge_check(fake, hr)
    assert f["ridge_false"] > 0.1 and f["ridge_missed"] < 0.05
    gone = cv2.GaussianBlur(hr, (0, 0), 6.0)
    g = ridge_check(gone, hr)
    assert g["ridge_missed"] > 0.5 and g["ridge_f1"] < f["ridge_f1"]
    with pytest.raises(ValueError):
        ridge_check(hr[:50], hr)


def test_ntire_score_formula():
    assert ntire_perceptual_score(0.2, 0.1, 0.5, 0.4, 60, 5) == pytest.approx(0.8 + 0.9 + 0.5 + 0.4 + 0.6 + 0.5)
    assert ntire_perceptual_score(0, 0, 0, 0, 0, 12) == pytest.approx(2.0)    # NIQE > 10 không bị phạt âm
    assert "ntire_score" not in add_ntire_score({"lpips": 0.1, "dists": float("nan"), "clipiqa": 0, "maniqa": 0,
                                                 "musiq": 0, "niqe": 0})
    nr = NoReferenceMetrics()
    if not nr.available:
        assert nr(np.zeros((32, 32, 3), np.uint8)) == {} and nr.reason


# ----------------------------------------------------------------- N2: mô phỏng hay thật

def test_realism_classifier_separates_wrong_degradation_only():
    rng = np.random.default_rng(0)
    truth = DegradeParams(blur_sigma=(0.8, 1.2), noise_sigma=(8.0, 12.0), jpeg_q=[[60, 1.0]], source="truth")
    real, sim_same, sim_clean = {}, {}, {}
    for s in range(10):
        for j in range(6):
            a, b = _texture(rng, 220, 180, 2.0), _texture(rng, 220, 180, 2.0)
            real.setdefault(f"s{s}", []).append(simulate_small(a, 40, "est", 4, truth, seed=s * 100 + j, key="r"))
            sim_same.setdefault(f"s{s}", []).append(simulate_small(b, 40, "est", 4, truth, seed=s * 100 + j, key="x"))
            sim_clean.setdefault(f"s{s}", []).append(simulate_small(b, 40, "bic", 4))
    assert real["s0"][0].shape[1] == 40 and simulate_small(np.zeros((100, 100, 3), np.uint8), 40, "bic") is None
    res = realism_test(real, {"bic": sim_clean, "est": sim_same}, patch=16, steps=250, n_seeds=1, n_boot=300, width=16)
    k = res["kinds"]
    assert k["bic"]["balanced_acc"] > 0.85, k
    assert k["est"]["balanced_acc"] < 0.70, k
    p = res["pairs"]["bic - est"]
    assert p["diff_dist_from_chance"] > 0.15 and p["lo"] > 0
    with pytest.raises(ValueError):
        realism_test({"a": real["s0"]}, {"bic": {"a": sim_clean["s0"]}})


# ----------------------------------------------------------------- T4: oracle và đường trộn

def test_oracle_gate_and_blend_definitions():
    rng = np.random.default_rng(0)
    hr = _texture(rng, 48, 40, 1.5)
    f_s = cv2.GaussianBlur(hr, (0, 0), 1.2)
    f_t = np.clip(hr.astype(int) + rng.integers(-25, 26, hr.shape), 0, 255).astype(np.uint8)
    assert np.array_equal(oracle.blend(f_s, f_t, 1.0), f_s) and np.array_equal(oracle.blend(f_s, f_t, 0.0), f_t)
    out, frac = oracle.oracle_gate(f_s, f_t, hr, 0.0, "pixel")
    e = lambda x: np.abs(x.astype(float) - hr).mean(2)
    assert (e(out) <= np.minimum(e(f_s), e(f_t)) + 1e-9).all() and 0 < frac < 1
    assert oracle.oracle_gate(f_s, f_t, hr, 10.0, "pixel")[1] == 0.0          # biên rất lớn: luôn lấy f_T
    assert oracle.oracle_gate(f_s, f_t, hr, -10.0, "window")[1] == 1.0
    names = [(m, p) for m, p, _, _ in oracle.candidates(f_s, f_t, hr)]
    assert len(names) == 11 + 2 * 5 and ("blend", 1.0) in names and ("oracle_window", 0.0) in names


def _t4_frame(oracle_metric_scale):
    """Đường trộn tuyến tính giữa (PSNR 30, m 0,10) và (PSNR 28, m 0,05); oracle ở PSNR 29,5."""
    rng = np.random.default_rng(1)
    rows = []
    for s in range(30):
        for v in range(3):
            n = rng.normal(0, 0.002)
            for a in np.linspace(0, 1, 11):
                rows.append({"subject": f"{s:03d}", "key": f"{s}_{v}", "method": "blend", "param": round(a, 2),
                             "psnr_y": 28 + 2 * a + n, "lpips": 0.05 + 0.05 * a + n})
            for mode in ("oracle_pixel", "oracle_window"):
                rows.append({"subject": f"{s:03d}", "key": f"{s}_{v}", "method": mode, "param": 0.02,
                             "psnr_y": 29.5 + n, "lpips": 0.0875 * oracle_metric_scale + n})
    return pd.DataFrame(rows)


def test_t4_summary_decision():
    good = t4.summarize(_t4_frame(0.8), n_boot=200)       # thấp hơn đường trộn 20% ở cùng PSNR
    assert good["pass"] and all(abs(r["gain"] - 0.2) < 0.02 and r["within_psnr_budget"] for r in good["oracles"])
    assert abs(good["oracles"][0]["blend_lpips_same_psnr"] - 0.0875) < 1e-3
    weak = t4.summarize(_t4_frame(0.95), n_boot=200)      # chỉ hơn 5%: dưới ngưỡng 10%
    assert not weak["pass"] and "bỏ N3" in weak["decision"]
    d = _t4_frame(0.5)
    d.loc[d.method != "blend", "psnr_y"] -= 3.0           # ngoài ngân sách PSNR 0,5 dB
    assert not t4.summarize(d, n_boot=100)["pass"]
    assert t4.summarize(_t4_frame(0.8).drop(columns="lpips"))["pass"] is None


def test_t4_script_runs_with_stand_in_metric(tmp_path, monkeypatch):
    rt = _script("run_t4_oracle")
    rng = np.random.default_rng(0)
    bench = tmp_path / "bench"
    rows = []
    for s in range(6):
        for v in ("a", "b"):
            hr = _texture(rng, 64, 48, 1.5)
            imwrite_rgb(bench / "hr64" / f"{s:03d}_{v}.png", hr)
            imwrite_rgb(bench / "lr" / "hr64_x4_bic" / f"{s:03d}_{v}.png", degrade(hr, 4, "bic"))
            rows.append({"dataset": "t", "subject": f"{s:03d}", "view": v, "tier": 64, "file": f"hr64/{s:03d}_{v}.png",
                         "h": 64, "w": 48})
    pd.DataFrame(rows).to_csv(bench / "manifest.csv", index=False)

    class FakePerc:
        def __call__(self, sr, hr):
            g = lambda x: np.abs(np.diff(x.astype(float), axis=0)).mean()
            return {"lpips": abs(g(sr) - g(hr)) / 50 + 0.01}

    up = lambda lr: cv2.resize(lr, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    sharp = lambda lr: np.clip(up(lr).astype(int) + np.random.default_rng(1).integers(-20, 21, (64, 48, 3)), 0, 255).astype(np.uint8)
    df = rt.run(bench, 64, 4, "bic", up, sharp, {f"{s:03d}" for s in range(6)}, FakePerc())
    assert len(df) == 12 * 21 and {"psnr_y", "lpips", "frac_fs"} <= set(df.columns)
    s = t4.summarize(df, n_boot=100)
    assert s["n_subjects"] == 6 and len(s["blend_curve"]) == 11 and s["pass"] in (True, False)
    with pytest.raises(SystemExit):
        rt.main(["--bench", str(bench), "--use-folds", "1", "2"])


# ----------------------------------------------------------------- tiêu chí

def _csv(path, vals, fold_of=None, **cols):
    n = len(vals)
    d = pd.DataFrame({"key": [f"{i // 2:03d}_{i % 2}" for i in range(n)], "subject": [f"{i // 2:03d}" for i in range(n)],
                      "fold": [(fold_of or (lambda s: 2 + s % 3))(i // 2) for i in range(n)], "psnr_y": vals, **cols})
    d.to_csv(path, index=False)
    return str(path)


def test_criteria_functions(tmp_path):
    crit = C.load_criteria(ROOT / "configs" / "criteria.yaml")
    rng = np.random.default_rng(0)
    base = 36 + rng.normal(0, 1.0, 120)
    noise = lambda s=0.02: rng.normal(0, s, 120)
    ref = _csv(tmp_path / "ref.csv", base)
    plus15 = _csv(tmp_path / "p15.csv", base + 0.15 + noise())
    plus05 = _csv(tmp_path / "p05.csv", base + 0.05 + noise())
    assert C.n5b_variant_gain(plus15, ref, crit)["pass"] is True
    r = C.n5b_variant_gain(plus05, ref, crit)
    assert r["pass"] is False and r["lo"] > 0          # có ý nghĩa nhưng dưới ngưỡng 0,1 dB
    assert C.n5b_interaction(plus15, ref, plus05, ref)["pass"] is True
    assert C.n5b_interaction(plus05, ref, plus15, ref)["pass"] is False
    # nối nhiều file, và bắt lỗi khác tập ảnh
    d = pd.read_csv(ref)
    d[d.fold == 2].to_csv(tmp_path / "r2.csv", index=False)
    d[d.fold != 2].to_csv(tmp_path / "r34.csv", index=False)
    assert C.gain([tmp_path / "r2.csv", tmp_path / "r34.csv"], ref)["point"] == pytest.approx(0)
    with pytest.raises(ValueError):
        C.gain(tmp_path / "r2.csv", ref)
    with pytest.raises(ValueError):
        C.load([ref, ref])
    # N5a
    same = _csv(tmp_path / "same.csv", base + noise(0.005))
    minus2 = _csv(tmp_path / "m2.csv", base - 0.2 + noise())
    assert C.n5a_protocol({144: plus15}, {144: ref}, crit)["pass"] is True
    assert C.n5a_protocol({96: same, 144: same, 192: same}, {96: ref, 144: ref, 192: ref}, crit)["one_model_all_sizes"]["pass"]
    assert C.n5a_protocol({96: same, 144: minus2, 192: same}, {96: ref, 144: ref, 192: ref}, crit)["pass"] is False
    # điểm vận hành và thời gian thực
    assert C.fidelity_point(same, ref, crit)["pass"] and not C.fidelity_point(minus2, ref, crit)["pass"]
    assert C.realtime_class(20, 30, crit) == {"class": "realtime", "stable": True}
    assert C.realtime_class(20, 40, crit) == {"class": "realtime", "stable": False}
    assert C.realtime_class(50, 60, crit)["class"] == "near_realtime" and C.realtime_class(90, 99, crit)["class"] == "reference_only"
    assert C.matched_latency(10, 10.9, crit)["pass"] and not C.matched_latency(10, 11.5, crit)["pass"]
    assert C.pretrain_rank_stable({"a": 30, "b": 29}, {"a": 31, "b": 30})["pass"]
    assert not C.pretrain_rank_stable({"a": 30, "b": 29}, {"a": 30, "b": 31})["pass"]
    assert C.choose_n1_n3(0.06, 0.07, crit)["keep"] == "N3" and C.choose_n1_n3(0.08, 0.07, crit)["keep"] == "N1"
    assert C.choose_n1_n3(None, 0.04, crit)["keep"] == "N3" and C.choose_n1_n3(None, None, crit)["keep"] is None


def test_criteria_n1_and_n3(tmp_path):
    crit = C.load_criteria(ROOT / "configs" / "criteria.yaml")
    rng = np.random.default_rng(1)
    n = 120
    psnr = 36 + rng.normal(0, 1, n)
    dev = 0.05 + rng.uniform(0, 0.02, n)

    def mk(name, dev_scale, floor=0.01, lp=0.10, ridge=0.8, dp=0.0):
        return _csv(tmp_path / name, psnr + dp + rng.normal(0, 0.003, n), lm_dev=dev * dev_scale + rng.normal(0, 2e-4, n),
                    lm_floor_noise=np.full(n, floor), lm_floor_jpeg=np.full(n, floor * 0.8),
                    lpips=lp + rng.normal(0, 1e-4, n), ridge_f1=ridge + rng.normal(0, 1e-4, n))

    base, aux, ctrl = mk("base.csv", 1.0), mk("aux.csv", 0.90), mk("ctrl.csv", 0.99)
    r = C.n1_aux_head(aux, base, ctrl, crit)
    assert r["pass"] and r["1_landmark_reduction"]["improvement"] == pytest.approx(0.10, abs=0.01)
    assert r["3_dynamic_range"]["ratio"] > 2 and r["4_no_harm"]["lpips_checked"]
    assert not C.n1_aux_head(mk("aux2.csv", 0.98), base, ctrl, crit)["pass"]                 # giảm 2% < 5%
    assert not C.n1_aux_head(aux, base, mk("ctrl2.csv", 0.85), crit)["6_beats_unlabeled_control"]["pass"]
    assert not C.n1_aux_head(mk("aux3.csv", 0.9, lp=0.11), base, ctrl, crit)["4_no_harm"]["pass"]   # LPIPS xấu đi 10%
    assert not C.n1_aux_head(mk("aux4.csv", 0.9, dp=-0.1), base, ctrl, crit)["5_psnr_not_lower"]["pass"]
    lowdyn = C.n1_aux_head(mk("a5.csv", 0.9, floor=0.04), mk("b5.csv", 1.0, floor=0.04), ctrl, crit)
    assert not lowdyn["3_dynamic_range"]["pass"] and "note" in lowdyn
    # N3
    p_ok = mk("p.csv", 1, lp=0.080, dp=-0.3)
    blend, ldl = mk("blend.csv", 1, lp=0.090, dp=-0.3), mk("ldl.csv", 1, lp=0.085, dp=-0.2)
    assert C.n3_gate(p_ok, base, blend, ldl, crit)["pass"]
    assert not C.n3_gate(mk("p2.csv", 1, lp=0.080, dp=-0.8), base, blend, ldl, crit)["psnr_budget"]["pass"]
    assert not C.n3_gate(mk("p3.csv", 1, lp=0.089, dp=-0.3), base, blend, ldl, crit)["vs_blend"]["pass"]
    # script
    cc = _script("check_criteria")
    out = cc.main(["n1", "--a", aux, "--b", base, "--control", ctrl, "--out", str(tmp_path / "crit"),
                   "--criteria", str(ROOT / "configs" / "criteria.yaml")])
    assert out["pass"] and json.loads((tmp_path / "crit" / "n1.json").read_text())["criteria_version"] == 21
    out = cc.main(["n5a", "--tier", "144", "--a", aux, "--b", base, "--out", str(tmp_path / "crit"),
                   "--criteria", str(ROOT / "configs" / "criteria.yaml")])
    assert out["pass"] is False


# ----------------------------------------------------------------- bảng LaTeX

def test_latex_tables(tmp_path):
    df = pd.DataFrame({"model": ["a_b", "c"], "psnr": [30.126, 31.5], "lp": [0.2, 0.1], "n": [3, 4]})
    tex = booktabs(df, "Cap", "tab:x", {"psnr": 2, "lp": 3}, {"psnr": "max", "lp": "min"})
    assert r"a\_b & 30.13 & 0.200 & 3 \\" in tex and r"c & \textbf{31.50} & \textbf{0.100} & 4 \\" in tex
    assert tex.count(r"\toprule") == 1 and r"\label{tab:x}" in tex and tex.strip().endswith(r"\end{table}")
    q = pd.DataFrame([{"tier": 144, "scale": 4, "degrade": k, "model": m, "group": g, "psnr_y": p + d, "ssim_y": 0.9}
                      for k, d in (("bic", 0), ("bicjpeg75", -3)) for m, g, p in
                      (("bicubic", None, 36.7), ("span", "light", 39.0), ("swin", "mid", 39.1))])
    t = t2_table(q, 144, 4)
    assert t.index("bicubic") < t.index("span") < t.index("swin") and t.count(r"\midrule") == 3
    assert r"\textbf{39.10}" in t
    with pytest.raises(ValueError):
        t2_table(q, 96, 4)
    (tmp_path / "t2_summary").mkdir()
    q.to_csv(tmp_path / "t2_summary" / "quality.csv", index=False)
    made = _script("make_tables").main(["--results", str(tmp_path), "--out", str(tmp_path / "tex")])
    assert made == ["t2_hr144_x4.tex"]
    # trên kết quả thật đi kèm project
    made = _script("make_tables").main(["--results", str(ROOT / "results"), "--out", str(tmp_path / "tex2")])
    assert any(m.startswith("t2_hr144") for m in made)


# ----------------------------------------------------------------- khảo sát người xem

def test_viewer_study_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    items = []
    for i in range(12):
        hr = _texture(rng, 32, 24)
        items.append({"key": f"{i % 6:03d}_{i}", "subject": f"{i % 6:03d}", "part": "ref", "ref": hr,
                      "outputs": {"mOurs": hr, "mBase": cv2.GaussianBlur(hr, (0, 0), 2), "mBic": cv2.GaussianBlur(hr, (0, 0), 3)}})
    key = viewer.build_study(items, [("mOurs", "mBase"), ("mOurs", "mBic")], tmp_path / "s.html", tmp_path / "key.csv",
                             study="t", seed=3)
    html = (tmp_path / "s.html").read_text(encoding="utf-8")
    assert len(key) == 24 and key.trial.is_unique and 0.2 < (key.left == "mOurs").mean() < 0.8
    for word in ("mOurs", "mBase", "mBic", "000_0"):
        assert word not in html, "trang gửi người xem không được lộ tên mô hình hay mã ảnh"
    assert html.count("data:image/png;base64") == 24 * 3 and "image-rendering:pixelated" in html
    assert json.loads(html.split("const TRIALS = ")[1].split(";\nconst STUDY")[0])[0]["w"] == 48
    # 8 người xem: 80% chọn 'ours' ở cặp đầu, 50% ở cặp sau
    files = []
    for v in range(8):
        r = np.random.default_rng(v)
        ans = []
        for t in key.itertuples():
            p = 0.8 if t.pair == "mOurs|mBase" else 0.5
            want_a = r.random() < p
            a = t.pair.split("|")[0]
            ans.append({"trial": t.trial, "choice": "L" if (t.left == a) == want_a else "R", "ms": 900})
        f = tmp_path / f"answers_t_V{v}.json"
        f.write_text(json.dumps({"study": "t", "viewer": f"V{v}", "answers": ans}))
        files.append(f)
    m = viewer.load_answers(files, pd.read_csv(tmp_path / "key.csv", dtype=str))
    res = viewer.analyze(m, n_boot=500).set_index("model_b")
    assert res.loc["mBase", "pref_a"] > 0.7 and res.loc["mBase", "ci_excludes_half"] and res.loc["mBase", "n_viewers"] == 8
    assert 0.35 < res.loc["mBic", "pref_a"] < 0.65 and not res.loc["mBic", "ci_excludes_half"]
    mv = {(it["key"], n): v for it in items for n, v in (("mOurs", 0.1), ("mBase", 0.2), ("mBic", 0.3))}
    ag = viewer.metric_agreement(m[m.pair == "mOurs|mBase"], mv)
    assert ag["agreement"] > 0.8
    with pytest.raises(ValueError):
        viewer.load_answers(files + [files[0]], key)
    with pytest.raises(ValueError):
        viewer.build_study([{**items[0], "ref": None}], [("mOurs", "mBase")], tmp_path / "x.html", tmp_path / "x.csv")
    # script: tạo từ thư mục ảnh SR
    for it in items:
        imwrite_rgb(tmp_path / "hr" / f"{it['key']}.png", it["ref"])
        for n, im in it["outputs"].items():
            imwrite_rgb(tmp_path / "sr" / n / f"{it['key']}.png", im)
    vs = _script("viewer_study")
    vs.main(["make", "--part", "ref", "--hr-dir", str(tmp_path / "hr"), "--sr-dir", str(tmp_path / "sr"),
             "--pairs", "mOurs:mBase", "--n", "6", "--out", str(tmp_path / "study")])
    k2 = pd.read_csv(tmp_path / "study" / "key.csv", dtype=str)
    assert len(k2) == 6 and k2.subject.nunique() == 6      # mỗi người một ảnh trước


# ----------------------------------------------------------------- sổ ghi, hàng đợi, danh sách lệnh

def test_runlog_and_queue(tmp_path):
    f = tmp_path / "r" / "runs.csv"
    runlog.record(f, "A", "running", exp="T6")
    runlog.record(f, "B", "failed", reason="hết bộ nhớ")
    runlog.record(f, "A", "done", best_val_psnr_y=38.5)
    rows = runlog.read(f)
    assert [r["run_id"] for r in rows] == ["A", "B"] and rows[0]["status"] == "done" and rows[0]["exp"] == "T6"
    assert rows[0]["best_val_psnr_y"] == "38.5" and rows[1]["reason"] == "hết bộ nhớ"
    runlog.record(f, "B", "excluded", reason="sai cấu hình")
    assert runlog.status_of(f, "B") == "excluded" and runlog.status_of(f, "Z") is None and len(runlog.read(f)) == 2
    with pytest.raises(ValueError):
        runlog.record(f, "A", "ok")
    jobs = tmp_path / "jobs.txt"
    py = sys.executable
    jobs.write_text(f"# ghi chú\n{py} -c \"open(r'{tmp_path / 'one'}','a').write('x')\"\n{py} -c \"import sys; sys.exit(3)\"\n")
    rq = _script("run_queue")
    assert rq.main([str(jobs)]) == {"done": 1, "failed": 1, "skipped": 0}
    assert rq.main([str(jobs)]) == {"done": 0, "failed": 1, "skipped": 1}       # không chạy lại việc đã xong
    assert (tmp_path / "one").read_text() == "x"


def test_make_jobs_blocks(capsys):
    mj = _script("make_jobs")
    t6ii = mj.main(["t6ii", "--ami-raw", "/d/AMI"])
    assert len(t6ii) == 3 * (3 * 2 + 3 + 3 + 1) and all("--fold 1 " not in j and "--fold 5 " not in j for j in t6ii)
    assert sum("--protocol native" in j for j in t6ii) == 9
    lr = mj.main(["lr"])
    assert len(lr) == 9 and all("--fold 2" in j and "--no-test" in j and "--exp LR" in j for j in lr)
    pre = mj.main(["pre", "--ctl-variants", "zero-c56"])
    assert len(pre) == 7 and sum("--backbone rlfn" in j for j in pre) == 1 and sum("--variant zero-c56" in j for j in pre) == 1
    pt = _script("pretrain")
    t6iv = mj.main(["t6iv", "--lr-of", "span=1e-4", "--ctl-variants", "zero-c56"])
    assert len(t6iv) == 21 and all("--init-ckpt runs/PRE_" in j and "--pretrain ps20" in j for j in t6iv)
    assert sum("--backbone rlfn" in j and "PRE_rlfn-zero_ps20" in j for j in t6iv) == 3
    assert all("--lr 1e-4" in j for j in t6iv if "--backbone span" in j)
    full = mj.main(["pre", "--budget", "100", "--ctl-backbones"])
    assert len(full) == 5 and "pf" in mj.main(["t6iv", "--budget", "100"])[0]
    assert "PRE_span-zero-deep_ps20_na_x4_hrall_bic_f0" in " ".join(t6iv)
    assert len(mj.main(["s2"])) == 15 and len(mj.main(["t6iii"])) == 6
    n2 = mj.main(["n2", "--degrade-params", "configs/d.json"])
    assert len(n2) == 6 and sum("--degrade est --degrade-params configs/d.json" in j for j in n2) == 2
    assert sum("--degrade generic" in j and "--degrade-params" not in j for j in n2) == 2 and all("--fold 2" in j for j in n2)
    with pytest.raises(SystemExit):
        mj.main(["n2"])
    pad = mj.main(["pad"])
    assert len(pad) == 27 and sum("--variant reflect" in j and "--backbone errn26" in j for j in pad) == 3
    capsys.readouterr()
    # mã lần chạy do train.py tính ra khớp tên thư mục mà make_jobs giả định cho checkpoint tiền huấn luyện
    from earsr.runid import RunId
    assert str(RunId("PRE", "span", "zero-deep", "ps20", "na", 4, "all", "bic", 0)) == "PRE_span-zero-deep_ps20_na_x4_hrall_bic_f0"
    # mọi lệnh train.py sinh ra đều phân tích được bằng bộ đọc tham số của train.py
    tr = _script("train")
    import shlex
    ids = set()
    for j in n2 + pad:
        ids.add(tr.main(shlex.split(j)[2:] + ["--print-run-id"])["run_id"])
    assert len(ids) == len(n2) + len(pad), "mỗi lệnh phải cho một mã lần chạy riêng"
    for j in t6ii + lr + t6iv:
        tr.parse_args(shlex.split(j)[2:])


# ----------------------------------------------------------------- độ trễ ngang nhau

def test_match_latency_search():
    ml = _script("match_latency")
    assert ml.with_channels("zero", 56) == "zero-c56" and ml.with_channels("zero-deep", 40) == "zero-b11-c40"
    assert ml.with_channels("reflect-c48-b3+x", 64) == "reflect-b3-c64"
    fake = lambda v: 0.002 * int(v.split("-c")[-1]) ** 2          # độ trễ tăng theo bình phương số kênh
    r = ml.search(6.5, "zero", fake)
    assert r["variant"] == "zero-c56" and r["within_tol"] and abs(r["rel_diff"]) < 0.05
    assert not ml.search(1000.0, "zero", fake)["within_tol"]


# ----------------------------------------------------------------- ảnh ngoài thực tế và evaluate.py

def test_build_wild_and_evaluate_scripts(tmp_path):
    rng = np.random.default_rng(0)
    root = tmp_path / "wild"
    for s in range(8):
        d = root / f"p{s:02d}"
        d.mkdir(parents=True)
        for i, side in enumerate((240, 200, 150, 60)):
            cv2.imwrite(str(d / f"im {i}.jpg"), _texture(rng, side + 30, side))
        cv2.imwrite(str(d / "zz dup.jpg"), cv2.imread(str(d / "im 0.jpg")))
    bw = _script("build_wild")
    roles = tmp_path / "roles.json"
    rep = bw.main(["--root", str(root), "--name", "fake", "--out", str(tmp_path / "b"), "--tiers", "96", "--safety",
                   "2.0", "--roles", str(roles), "--role", "test"])
    r = json.loads(roles.read_text())
    n_test = sum(v == "test" for v in r.values())
    assert set(r.values()) <= set(wild.ROLE_FRACTIONS) and n_test >= 1
    assert rep[96] == 2 * n_test                          # 240 và 200 px đủ biên 2 lần; 150 và 60 không; ảnh trùng bị loại
    man = pd.read_csv(tmp_path / "b" / "manifest.csv", dtype=str)
    assert set(man.subject) == {k for k, v in r.items() if v == "test"} and (man.dataset == "fake").all()
    assert set(man.view) == {"im-0", "im-1"} and (tmp_path / "b" / "scan.csv").exists()
    drop = pd.read_csv(tmp_path / "b" / "dropped_hr96.csv")
    assert drop.why.str.contains("cạnh ngắn").sum() == 2 * n_test and drop.why.str.contains("trùng").sum() == n_test
    # chạy lại dùng scan.csv đã lưu, thêm tầng khác: manifest giữ cả hai tầng
    bw.main(["--root", str(root), "--name", "fake", "--out", str(tmp_path / "b"), "--tiers", "64", "--safety", "2.0",
             "--roles", str(roles), "--role", "test", "--max-per-subject", "1"])
    man = pd.read_csv(tmp_path / "b" / "manifest.csv", dtype=str)
    assert set(man.tier) == {"96", "64"} and (man.tier == "64").sum() == n_test
    # chấm trên bộ không chia fold, có bảng kiểm tra gờ
    ev = _script("evaluate")
    ev.main(["--bench", str(tmp_path / "b"), "--folds", "none", "--tiers", "96", "--kinds", "bic", "generic",
             "--models", "bicubic", "bicubic_sharp", "--out", str(tmp_path / "res"), "--no-perceptual", "--ridge",
             "--save-sr", str(tmp_path / "sr")])
    d = pd.read_csv(tmp_path / "res" / "bicubic__hr96_x4_generic.csv", dtype={"subject": str})
    assert len(d) == 2 * n_test and (d.fold == 0).all() and (d.dataset == "fake").all() and "ridge_f1" in d
    assert len(list((tmp_path / "sr" / "hr96_x4_bic" / "bicubic").glob("*.png"))) == 2 * n_test
    with pytest.raises(SystemExit):
        ev.main(["--bench", str(tmp_path / "b"), "--folds", "none", "--models", "--out", str(tmp_path / "res2")])


def test_boxes_restrict_metric_region(tmp_path):
    from earsr.eval.infer import evaluate_on_bench, load_boxes
    rng = np.random.default_rng(0)
    bench = tmp_path / "bench"
    hr = _texture(rng, 64, 48, 1.5)
    imwrite_rgb(bench / "hr64" / "000_a.png", hr)
    imwrite_rgb(bench / "lr" / "hr64_x4_bic" / "000_a.png", degrade(hr, 4, "bic"))
    pd.DataFrame([{"dataset": "t", "subject": "000", "view": "a", "tier": 64, "file": "hr64/000_a.png", "h": 64, "w": 48}]
                 ).to_csv(bench / "manifest.csv", index=False)
    pd.DataFrame([{"key": "000_a", "top": 0.25, "left": 0.25, "bottom": 0.75, "right": 0.75}]).to_csv(tmp_path / "boxes.csv", index=False)

    def pred(lr):          # đúng trong hộp, sai ngoài hộp
        out = np.zeros_like(hr)
        out[16:48, 12:36] = hr[16:48, 12:36]
        return out

    evaluate_on_bench("m", bench, 64, 4, "bic", None, tmp_path / "o.csv", predictor=pred, boxes=load_boxes(tmp_path / "boxes.csv"))
    d = pd.read_csv(tmp_path / "o.csv")
    assert np.isinf(d.psnr_y_box.iloc[0]) and d.psnr_y.iloc[0] < 20


def test_size_sweep(tmp_path):
    from earsr.data import ami
    from earsr.data.resize import imresize
    from earsr.data.splits import make_folds, save_folds
    rng = np.random.default_rng(0)
    raw = tmp_path / "raw"
    raw.mkdir()
    subs = [f"{i:03d}" for i in range(10)]
    for s in subs:
        cv2.imwrite(str(raw / f"{s}_front_ear.jpg"), _texture(rng, 351, 246, 5.0))
    folds = tmp_path / "f.json"
    save_folds(make_folds(subs, n_folds=5, n_val=2), folds)
    ss = _script("run_size_sweep")
    bic = lambda lr: imresize(lr, 4.0)
    df = ss.sweep(raw, folds, (2, 3), (16, 24), 4, {"bicubic": (bic, None), "only3": (bic, set(json.loads(folds.read_text())["folds"]["3"]["test"]))})
    assert set(df.lr_short) == {16, 24} and len(df[df.model == "bicubic"]) == 4 * 2 and len(df[df.model == "only3"]) == 2 * 2
    s = ss.summarize(df, n_boot=50)
    assert len(s) == 4 and (s.lo <= s.psnr_y).all() and (s.psnr_y <= s.hi).all()


# ----------------------------------------------------------------- hình và bảng so sánh (bản 24)

def test_figures_from_result_files(tmp_path):
    from earsr.report import figures as F
    rng = np.random.default_rng(0)
    # hình 2: phần hơn so với bicubic; thiếu mốc thì báo lỗi rõ
    ss = pd.DataFrame([{"model": m, "lr_short": s, "psnr_y": 30 + 0.1 * s + off, "lo": 30 + 0.1 * s + off - 0.1,
                        "hi": 30 + 0.1 * s + off + 0.1} for m, off in (("bicubic", 0), ("a", 2.0), ("b", 2.3))
                       for s in range(16, 65, 8)])
    p = F.fig_size_sweep(ss, tmp_path / "f2.pdf")
    assert p.exists() and p.stat().st_size > 2000 and p.with_suffix(".png").exists()
    assert F.fig_size_sweep(ss, tmp_path / "f2abs.pdf", relative_to=None).exists()
    with pytest.raises(ValueError):
        F.fig_size_sweep(ss[ss.model != "bicubic"], tmp_path / "x.pdf")
    with pytest.raises(ValueError):
        F.fig_size_sweep(ss, tmp_path / "x.pdf", models=["khong-co"])
    # hình 3: chất lượng theo chi phí
    q = pd.DataFrame([{"tier": 144, "scale": 4, "degrade": k, "model": m, "group": g, "psnr_y": v + d}
                      for k, d in (("bic", 0), ("bicjpeg75", -4)) for m, g, v in
                      (("bicubic", None, 36.7), ("s1", "light", 39.0), ("s2", "mid", 39.1), ("s3", "upper", 39.3))])
    cost = pd.DataFrame({"name": ["s1", "s2", "s3"], "median_ms": [5.0, 20.0, 90.0]})
    assert F.fig_quality_vs_cost(q, cost, tmp_path / "f3.pdf", threshold=33.0).exists()
    with pytest.raises(ValueError):
        F.fig_quality_vs_cost(q, cost, tmp_path / "x.pdf", tier=96)
    # hình 4: đường cong tính từ số đo theo ảnh
    rows = []
    for model, loss in (("net", 0.3), ("bicubic", 0.1)):
        for i in range(6):
            base = 38 + rng.normal(0, 0.5)
            for m, frac in ((0, 1.0), (4, 0.1), (8, 0.0), (24, 0.0)):
                rows.append({"model": model, "dataset": "ami", "config": "wide", "subject": f"{i:03d}", "key": f"{i}",
                             "cond": f"m{m}", "psnr_y": base - loss * frac})
    per = pd.DataFrame(rows)
    cur = F.context_curve(per)
    net0 = cur[(cur.model == "net") & (cur.context_px == 0)].loss_db.iloc[0]
    assert net0 == pytest.approx(0.3) and cur[(cur.model == "net") & (cur.context_px == 24)].loss_db.iloc[0] == 0
    assert F.fig_context(per, tmp_path / "f4.pdf").exists()
    # lưới định tính
    for name, val in (("lr", 60), ("m", 120), ("hr", 200)):
        size = (17, 12) if name == "lr" else (68, 48)
        for k in ("000_a", "001_a"):
            imwrite_rgb(tmp_path / name / f"{k}.png", np.full((*size, 3), val, np.uint8))
    g = F.fig_grid({"LR": tmp_path / "lr", "Model": tmp_path / "m", "HR": tmp_path / "hr"}, ["000_a", "001_a"],
                   tmp_path / "grid.pdf", zoom_lr=4)
    assert g.exists()
    with pytest.raises(FileNotFoundError):
        F.fig_grid({"LR": tmp_path / "lr"}, ["khong-co"], tmp_path / "x.pdf")


def test_make_figures_script_on_shipped_results(tmp_path):
    mf = _script("make_figures")
    made = mf.main(["--results", str(ROOT / "results"), "--out", str(tmp_path / "fig")])
    assert "fig4_context.pdf" in made                       # kết quả sơ bộ đi kèm project có T6
    assert mf.main(["--results", str(tmp_path / "empty"), "--out", str(tmp_path / "fig2")]) == []


def test_compare_table_script(tmp_path):
    rng = np.random.default_rng(0)
    base = 36 + rng.normal(0, 1, 60)

    def mk(name, dp, lp):
        d = pd.DataFrame({"key": [f"{i // 2:03d}_{i % 2}" for i in range(60)], "subject": [f"{i // 2:03d}" for i in range(60)],
                          "fold": [2 + (i // 2) % 3 for i in range(60)], "psnr_y": base + dp + rng.normal(0, 0.01, 60),
                          "lpips": lp + rng.normal(0, 1e-3, 60), "grad_psnr": [np.inf] + [30.0] * 59})
        for f in (2, 3, 4):
            d[d.fold == f].to_csv(tmp_path / f"{name}_f{f}.csv", index=False)

    mk("base", 0.0, 0.10)
    mk("ours", 0.2, 0.08)
    mk("same", 0.0, 0.10)
    pd.read_csv(tmp_path / "ours_f2.csv").to_csv(tmp_path / "partial.csv", index=False)
    pd.DataFrame({"name": ["span", "new"], "median_ms": [4.0, 4.2], "params": [150000, 160000]}).to_csv(tmp_path / "lat.csv", index=False)
    ct = _script("compare_table")
    t = ct.main(["--name", "main", "--row", f"Base={tmp_path}/base_f*.csv", "--row", f"Ours={tmp_path}/ours_f*.csv",
                 "--row", f"Same={tmp_path}/same_f*.csv", "--row", f"Partial={tmp_path}/partial.csv", "--ref", "Base",
                 "--metrics", "psnr_y", "lpips", "grad_psnr", "khong_co", "--cost", str(tmp_path / "lat.csv"),
                 "--cost-key", "Base=span", "--cost-key", "Ours=new", "--out-csv", str(tmp_path / "t"),
                 "--out-tex", str(tmp_path / "x"), "--n-boot", "300"]).set_index("method")
    assert t.loc["Ours", "psnr_y_diff"] == pytest.approx(0.2, abs=0.01) and t.loc["Ours", "psnr_y_sig"]
    assert t.loc["Ours", "lpips_diff"] < 0 and t.loc["Ours", "lpips_sig"] and not t.loc["Same", "psnr_y_sig"]
    assert np.isnan(t.loc["Partial", "psnr_y_diff"]) and t.loc["Partial", "n_images"] == 20
    assert t.loc["Base", "n_folds"] == 3 and np.isfinite(t.loc["Base", "grad_psnr"]) and "khong_co" not in t.columns
    assert t.loc["Ours", "median_ms"] == 4.2 and np.isnan(t.loc["Same", "median_ms"])
    tex = (tmp_path / "x" / "main.tex").read_text()
    ours = next(l for l in tex.split("\n") if l.startswith("Ours"))
    assert r"$^{\dagger}$" in ours and r"\textbf" in ours and "Latency (ms)" in tex
    assert r"$^{\dagger}$" not in next(l for l in tex.split("\n") if l.startswith("Same"))
    with pytest.raises(SystemExit):
        ct.main(["--name", "x", "--row", f"A={tmp_path}/khong-co*.csv"])
