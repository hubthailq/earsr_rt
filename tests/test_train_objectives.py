"""Các mục tiêu huấn luyện ngoài L1, dữ liệu trộn, điểm mốc trong bộ dữ liệu, và
``scripts/train.py`` chạy đầu cuối trên dữ liệu giả (vài bước, CPU)."""
import importlib.util
import cv2  # noqa: F811
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest
import torch

from earsr import runlog
from earsr.data import ami
from earsr.data.datasets import MixedDataset, SRTrainDataset, point_heatmaps, short_side_transform, resize_short_side
from earsr.data.splits import make_folds, save_folds
from earsr.models.optional.heads import TwoHeadGated
from earsr.models.span import SPAN
from earsr.models.variants import build_span_variant, is_published_shape, parse_variant, parse_variant_full
from earsr.train.finetune import build_n3, build_trainable, load_run_model, predictor_from_run, trainable_fraction
from earsr.train.losses import PerceptualLoss, UNetDiscriminatorSN, ldl_artifact_map, ldl_loss
from earsr.train.objectives import (AuxObjective, GanObjective, L1Objective, N3GateObjective, N3HeadsObjective,
                                    make_objective)

ROOT = Path(__file__).resolve().parents[1]


def _script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.fixture(scope="module")
def fake_ami(tmp_path_factory):
    d = tmp_path_factory.mktemp("fake")
    raw = d / "raw"
    raw.mkdir()
    rng = np.random.default_rng(0)
    subjects = [f"{i:03d}" for i in range(10)]
    for s in subjects:
        for v in ami.VIEWS:
            img = cv2.GaussianBlur(rng.integers(0, 256, (702, 492, 3), dtype=np.uint8), (0, 0), 6.0)
            cv2.imwrite(str(raw / f"{s}_{v}_ear.jpg"), img)
    bench = d / "bench"
    ami.build_ami_benchmark(raw, bench, tiers=(96,), strict=False)
    folds = d / "folds.json"
    save_folds(make_folds(subjects, n_folds=5, n_val=2), folds)
    extra = d / "extra"
    for s in ("p1", "p2"):
        (extra / s).mkdir(parents=True)
        for i, side in enumerate((200, 260, 120)):
            img = cv2.GaussianBlur(rng.integers(0, 256, (side + 40, side, 3), dtype=np.uint8), (0, 0), 4.0)
            cv2.imwrite(str(extra / s / f"{i}.jpg"), img)
    return {"dir": d, "raw": raw, "bench": bench, "folds": folds, "extra": extra}


# ----------------------------------------------------------------- variant

def test_variant_tokens():
    assert parse_variant_full("zero") == ("zeros", 48, 6)
    assert parse_variant_full("reflect-c40-b9+gan+lr1e-4") == ("reflect", 40, 9)
    assert parse_variant("zero-wide") == ("zeros", "wide") and parse_variant("zero-c56") == ("zeros", "custom")
    assert is_published_shape("zero+aux") and not is_published_shape("zero-c56") and not is_published_shape("reflect")
    m = build_span_variant("replicate-c20-b3")
    assert m.feature_channels == 20 and m.n_blocks == 3 and m.padding_mode == "replicate"
    with pytest.raises(ValueError):
        parse_variant_full("zero-huge")
    with pytest.raises(ValueError):
        build_trainable("span", "zero-c56", "pub", 4)
    with pytest.raises(ValueError):
        build_trainable("rlfn", "zero-c56", "none", 4)       # mô hình trong kho chỉ nhận kiểu đệm
    with pytest.raises(ValueError):
        build_trainable("rlfn", "reflect", "none", 4)        # RLFN có phép toán không cục bộ (ESA)


@pytest.mark.parametrize("backbone", ["span", "span26", "pds26", "pkdsr26", "dscf26", "disp26", "errn26"])
def test_padding_mode_on_any_backbone_changes_only_the_border(backbone):
    """Đổi kiểu đệm trên cùng một bộ trọng số: lòng ảnh giữ nguyên, chỉ vùng sát mép đổi;
    lưu rồi nạp lại cho đúng kết quả; mọi tham số vẫn học được."""
    torch.manual_seed(0)
    zero = build_trainable(backbone, "zero", "none", 4).eval()
    x = torch.rand(1, 3, 72, 72)
    with torch.no_grad():
        y0 = zero(x)
    for mode in ("replicate", "reflect"):
        m = build_trainable(backbone, mode, "none", 4)
        m.load_state_dict(zero.state_dict())
        m.eval()
        with torch.no_grad():
            y = m(x)
        d = (y - y0).abs().amax((0, 1))
        assert float(d.max()) > 1e-4, "kiểu đệm phải làm đổi đầu ra ở mép"
        r = 30 * 4                                          # lòng ảnh, cách mép 30 px ảnh vào
        inner = d[r:-r, r:-r]
        assert float(inner.max()) < 1e-5 * max(1.0, float(y0.abs().max())), (backbone, mode, float(inner.max()))
        assert trainable_fraction(m) == 1.0 or backbone == "span"


def test_deploy_form_baseline_is_trainable():
    """SPAN 28 kênh công bố ở dạng đã gộp; khi tinh chỉnh, mọi tham số phải có gradient."""
    m = build_trainable("span_ch28", "zero", "none", 4)
    assert trainable_fraction(m) == 1.0
    assert trainable_fraction(build_trainable("span", "zero", "none", 4)) == 1.0
    x = torch.rand(1, 3, 12, 12)
    m.train()
    m(x).mean().backward()
    assert m.net.block_1.c1_r.eval_conv.weight.grad.abs().sum() > 0


# ----------------------------------------------------------------- loss

def test_ldl_artifact_map_matches_definition():
    torch.manual_seed(0)
    hr, sr, ema = torch.rand(2, 3, 16, 16), torch.rand(2, 3, 16, 16), torch.rand(2, 3, 16, 16)
    w = ldl_artifact_map(hr, sr, ema, ksize=7)
    assert w.shape == (2, 1, 16, 16) and (w >= 0).all()
    r_sr, r_ema = (hr - sr).abs().sum(1, keepdim=True), (hr - ema).abs().sum(1, keepdim=True)
    assert (w[r_sr < r_ema] == 0).all() and (w[r_sr >= r_ema] > 0).all()
    # tính tay phương sai cục bộ tại một điểm trong lòng ảnh
    b, y, x = 1, 8, 9
    local = r_sr[b, 0, y - 3:y + 4, x - 3:x + 4].var(unbiased=True)
    glob = r_sr[b].var() ** 0.2
    if r_sr[b, 0, y, x] >= r_ema[b, 0, y, x]:
        assert torch.allclose(w[b, 0, y, x], local * glob, rtol=1e-4)
    # mô hình trùng EMA và trùng đáp án: không phạt
    assert float(ldl_loss(hr.clone(), hr, hr.clone())) == 0.0
    sr.requires_grad_(True)
    ldl_loss(sr, hr, ema).backward()
    assert torch.isfinite(sr.grad).all()


def test_perceptual_and_discriminator_shapes():
    p = PerceptualLoss(pretrained=False)
    a, b = torch.rand(1, 3, 48, 48), torch.rand(1, 3, 48, 48)
    assert float(p(a, a)) == 0.0 and float(p(a, b)) > 0
    assert not p.training and all(not q.requires_grad for q in p.parameters())
    assert set(p.features(a)) == {"conv1_2", "conv2_2", "conv3_4", "conv4_4", "conv5_4"}
    assert p.features(a)["conv5_4"].shape[1] == 512 and p.features(a)["conv1_2"].shape[1] == 64
    assert (p.features(a)["conv1_2"] < 0).any(), "đặc trưng phải lấy trước ReLU"
    d = UNetDiscriminatorSN(num_feat=8)
    assert d(torch.rand(2, 3, 96, 96)).shape == (2, 1, 96, 96)


# ----------------------------------------------------------------- mục tiêu

def _batch(n=2, p=8, s=4):
    torch.manual_seed(1)
    return [torch.rand(n, 3, p, p), torch.rand(n, 3, p * s, p * s)]


def _step(model, obj, batch, ema=None):
    obj.prepare(model)
    params = obj.parameters(model)
    opt = torch.optim.SGD(params, lr=1e-2)
    before = [q.detach().clone() for q in params]
    loss, logs = obj.loss(model.train(), batch, 0, ema)
    opt.zero_grad()
    loss.backward()
    opt.step()
    logs.update(obj.after_step(0))
    moved = sum(float((a - b.detach()).abs().sum()) for a, b in zip(before, params))
    return float(loss), logs, moved


def test_gan_and_ldl_objectives_update_generator_and_discriminator():
    torch.manual_seed(0)
    model = SPAN(feature_channels=8, n_blocks=2)
    obj = GanObjective(d_feat=8, percep_pretrained=False)
    d_before = [q.detach().clone() for q in obj.disc.parameters()]
    loss, logs, moved = _step(model, obj, _batch(p=8))
    assert np.isfinite(loss) and moved > 0 and {"pix", "percep", "g_gan", "d_real", "d_fake"} <= set(logs)
    assert sum(float((a - b).abs().sum()) for a, b in zip(d_before, obj.disc.parameters())) > 0
    assert all(q.requires_grad for q in obj.disc.parameters())
    # trạng thái lưu và nạp lại được
    obj2 = GanObjective(d_feat=8, percep_pretrained=False)
    obj2.load_state_dict(obj.state_dict())
    assert all(torch.equal(a, b) for a, b in zip(obj.disc.state_dict().values(), obj2.disc.state_dict().values()))
    ldl = make_objective("ldl", model, d_feat=8, percep_pretrained=False)
    assert ldl.name == "ldl" and ldl.w_ldl == 1.0
    with pytest.raises(RuntimeError):
        ldl.loss(model, _batch(), 0, None)
    import copy
    _, logs, _ = _step(model, ldl, _batch(), ema=copy.deepcopy(model).eval())
    assert "ldl" in logs


def test_aux_objective_masks_unlabeled_and_leaves_plain_backbone():
    torch.manual_seed(0)
    model = SPAN(feature_channels=8, n_blocks=2)
    obj = make_objective("aux", model, n_maps=5, w_aux=1.0)
    lr, hr = _batch()
    target = torch.rand(2, 5, 8, 8)
    _, logs, moved = _step(model, obj, [lr, hr, target, torch.tensor([1.0, 0.0])])
    assert moved > 0 and logs["labeled"] == 0.5
    # ảnh không nhãn không đóng góp vào loss phụ: đổi bản đồ của nó, loss phụ không đổi
    t2 = target.clone()
    t2[1] += 5.0
    l_a = obj.loss(model, [lr, hr, target, torch.tensor([1.0, 0.0])], 0, None)[1]["aux"]
    l_b = obj.loss(model, [lr, hr, t2, torch.tensor([1.0, 0.0])], 0, None)[1]["aux"]
    assert abs(l_a - l_b) < 1e-7
    assert obj.loss(model, [lr, hr, target, torch.zeros(2)], 0, None)[1]["aux"] == 0.0
    # checkpoint mô hình không chứa đầu phụ
    assert not any("head" in k for k in model.state_dict()) and "head" in obj.state_dict()
    with pytest.raises(TypeError):
        make_objective("aux", build_trainable("rlfn", "zero", "none", 4), n_maps=5)


def test_n3_two_stages_train_only_what_they_should():
    torch.manual_seed(0)
    back = SPAN(feature_channels=8, n_blocks=2)
    model = TwoHeadGated(back, hidden=4)
    snap = lambda mod: {k: v.clone() for k, v in mod.state_dict().items()}
    changed = lambda a, mod: {k for k, v in mod.state_dict().items() if not torch.equal(a[k], v)}
    # giai đoạn 1, cấu hình A: chỉ đầu kết cấu đổi
    s = snap(model)
    objA = N3HeadsObjective("A", d_feat=8, percep_pretrained=False)
    _step(model, objA, _batch())
    ch = changed(s, model)
    assert ch and all(k.startswith("head_t.") for k in ch)
    # cấu hình B: thân cũng đổi, cổng thì không
    s = snap(model)
    _, logs, _ = _step(model, N3HeadsObjective("B", d_feat=8, percep_pretrained=False), _batch())
    ch = changed(s, model)
    assert any(k.startswith("backbone.") for k in ch) and not any(k.startswith("gate.") for k in ch)
    assert "pix_s" in logs
    # giai đoạn 2: chỉ cổng đổi
    s = snap(model)
    _, logs, _ = _step(model, N3GateObjective(tau=0.0), _batch())
    ch = changed(s, model)
    assert ch and all(k.startswith("gate.") for k in ch) and 0 <= logs["label_keep"] <= 1


def test_build_n3_copies_fidelity_head():
    m = build_n3("zero-c8-b2", 4)
    x = torch.rand(1, 3, 10, 10)
    m.eval()
    _, f_s, f_t, _ = m(x, return_all=True)
    assert torch.allclose(f_s, f_t, atol=1e-6)


# ----------------------------------------------------------------- dữ liệu

def test_point_coordinates_follow_resize_crop_and_flip(tmp_path):
    """Vẽ một chấm sáng lên ảnh gốc; sau mọi phép biến đổi của bộ dữ liệu, đỉnh của
    bản đồ điểm mốc phải trùng vị trí chấm sáng trong patch LR."""
    h, w = 351, 246
    paths, lms = [], {}
    rng = np.random.default_rng(0)
    for i in range(4):
        img = np.zeros((h, w, 3), np.uint8)
        x0, y0 = int(rng.integers(60, w - 60)), int(rng.integers(80, h - 80))
        cv2.circle(img, (x0, y0), 9, (255, 255, 255), -1)
        p = tmp_path / f"{i}.png"
        cv2.imwrite(str(p), img)
        paths.append(p)
        lms[str(p)] = np.array([[x0, y0]], np.float32)
    for proto in ("rand", "fixed", "native"):
        ds = SRTrainDataset(paths, scale=4, protocol=proto, patch_lr=20, tier=96, rand_grid=(96, 120),
                            landmarks=lms, n_landmarks=1, aux_sigma=1.0, seed=3)
        hits = 0
        for idx in range(60):
            lr, hr, hm, has = ds[idx]
            assert has == 1 and hm.shape == (1, 20, 20)
            if hm.max() < 0.9:      # chấm nằm ngoài patch
                continue
            py, px = np.unravel_index(int(hm[0].argmax()), (20, 20))
            g = lr.mean(0).numpy()
            ys, xs = np.nonzero(g > 0.5 * g.max())
            if g.max() < 0.3 or len(ys) == 0:
                continue
            assert abs(ys.mean() - py) <= 1.0 and abs(xs.mean() - px) <= 1.0, (proto, idx)
            hits += 1
        assert hits >= 5, proto
    # ảnh không có nhãn: cờ bằng 0, bản đồ toàn 0
    ds = SRTrainDataset(paths, scale=4, protocol="fixed", patch_lr=16, tier=96, landmarks={}, n_landmarks=1)
    _, _, hm, has = ds[0]
    assert has == 0 and float(hm.abs().sum()) == 0


def test_short_side_transform_matches_resize():
    for (h, w, s) in [(702, 492, 144), (351, 246, 100), (300, 410, 96), (200, 200, 200)]:
        img = np.zeros((h, w, 3), np.uint8)
        out = resize_short_side(img, s, 4)
        fy, fx, oy, ox = short_side_transform(h, w, s, 4)
        assert abs((h * fy - 2 * oy) - out.shape[0]) <= 1 and abs((w * fx - 2 * ox) - out.shape[1]) <= 1
    assert point_heatmaps(np.array([[3.0, 5.0]]), 8, 1.0)[0].argmax() == 5 * 8 + 3


def test_min_downscale_never_upsamples(fake_ami):
    from earsr.data.datasets import list_images
    paths = list_images(fake_ami["extra"])
    ds = SRTrainDataset(paths, scale=4, protocol="rand", patch_lr=8, rand_grid=(96, 128, 160), min_downscale=2.0)
    # cạnh ngắn 200 -> chỉ 96; 260 -> 96, 128; 120 -> bị loại
    assert ds.dropped == 2 and sorted(set(ds.sizes)) == [(96,), (96, 128)]
    for i in range(20):
        lr, hr = ds[i]
        assert hr.shape == (3, 32, 32)
    with pytest.raises(ValueError):
        SRTrainDataset(paths, scale=4, protocol="fixed", tier=192, min_downscale=2.0)
    a = SRTrainDataset(paths, scale=4, protocol="rand", patch_lr=8, rand_grid=(96,), seed=1)
    mixed = MixedDataset([ds, a], probs=[0.25, 0.75], seed=5)
    assert torch.equal(mixed[7][0], MixedDataset([ds, a], probs=[0.25, 0.75], seed=5)[7][0])
    u = [np.random.default_rng([5, 7919, i]).random() < 0.25 for i in range(2000)]
    assert 0.2 < np.mean(u) < 0.3 and len(mixed.paths) == len(ds.paths) + len(a.paths)


# ----------------------------------------------------------------- train.py đầu cuối

def _args(f, exp, *more):
    return ["--exp", exp, "--fold", "2", "--ami-raw", str(f["raw"]), "--bench", str(f["bench"]), "--folds",
            str(f["folds"]), "--out", str(f["dir"] / "runs"), "--runs-csv", str(f["dir"] / "runs.csv"),
            "--tier", "96", "--protocol", "fixed", "--iters", "4", "--val-every", "2", "--batch-size", "2",
            "--patch-lr", "8", "--workers", "0", "--device", "cpu", "--no-perceptual", "--pretrain", "none",
            *more]


def test_train_script_end_to_end(fake_ami, monkeypatch):
    f = fake_ami
    tr = _script("train")
    monkeypatch.setattr(tr, "make_objective", lambda name, model=None, **kw: make_objective(
        name, model, **({**kw, "d_feat": 8, "percep_pretrained": False} if name in ("gan", "ldl", "n3s1A", "n3s1B") else kw)))
    # 1. mốc L1 với ảnh thêm và suy giảm ngẫu nhiên; có bảng kiểm tra gờ
    out = tr.main(_args(f, "S9", "--variant", "zero-c8-b2", "--degrade", "generic", "--extra-dir", str(f["extra"]),
                        "--extra-name", "fake", "--ridge", "--tag", "lr2e-4"))
    rid = "S9_span-zero-c8-b2+xfake+lr2e-4_none_fixed_x4_hr96_generic_f2"
    run = f["dir"] / "runs" / rid
    d = pd.read_csv(run / "test_hr96_x4_generic.csv", dtype={"subject": str})
    folds = json.loads(f["folds"].read_text())["folds"]["2"]
    assert set(d.subject) == set(folds["test"]) and len(d) == 14 and "ridge_f1" in d and (d.model == rid).all()
    assert "test_psnr_y_hr96" in out
    cfg = json.loads((run / "config.json").read_text())
    assert cfg["extra"]["variant"] == "zero-c8-b2" and cfg["extra"]["n_extra_images"] == 6
    assert runlog.status_of(f["dir"] / "runs.csv", rid) == "done"
    # nạp lại mô hình từ thư mục lần chạy: cho đúng số đo đã ghi
    model, _ = load_run_model(run)
    assert model.feature_channels == 8
    pred = predictor_from_run(run)
    from earsr.eval.infer import evaluate_on_bench
    evaluate_on_bench("again", f["bench"], 96, 4, "generic", f["folds"], f["dir"] / "again.csv", predictor=pred,
                      subjects=set(folds["test"]))
    d2 = pd.read_csv(f["dir"] / "again.csv")
    assert np.allclose(d.sort_values("key").psnr_y.values, d2.sort_values("key").psnr_y.values, atol=1e-6)

    # evaluate.py --runs: tự giới hạn ở người test của fold; export_for_device.py --runs: xuất ONNX khớp PyTorch
    import sys
    sys.path.insert(0, str(ROOT / "scripts"))
    _script("evaluate").main(["--bench", str(f["bench"]), "--folds", str(f["folds"]), "--tiers", "96", "--kinds",
                              "generic", "--runs", str(run), "--out", str(f["dir"] / "ev"), "--no-perceptual"])
    d3 = pd.read_csv(f["dir"] / "ev" / f"{rid}__hr96_x4_generic.csv", dtype={"subject": str})
    assert set(d3.subject) == set(folds["test"])
    assert np.allclose(d.sort_values("key").psnr_y.values, d3.sort_values("key").psnr_y.values, atol=1e-6)
    _script("export_for_device").main_argv = None
    old = sys.argv
    sys.argv = ["x", "--skip-zoo", "--variants", "--runs", str(run), "--out", str(f["dir"] / "onnx")]
    try:
        _script("export_for_device").main()
    finally:
        sys.argv = old
    idx = json.loads((f["dir"] / "onnx" / "index.json").read_text())
    assert len(idx) == 1 and idx[0]["kind"] == "trained" and idx[0]["rel_diff"] < 1e-4

    # 2. thân bản GAN khởi tạo từ thân L1
    best = str(run / "ckpt" / "best.pt")
    tr.main(_args(f, "S9", "--variant", "zero-c8-b2", "--objective", "gan", "--pretrain", "pf", "--init-ckpt", best,
                  "--no-test"))
    g = f["dir"] / "runs" / "S9_span-zero-c8-b2+gan_pf_fixed_x4_hr96_bic_f2"
    assert json.loads((g / "summary.json").read_text())["select"] == "last"
    assert "g_gan" in pd.read_csv(g / "log.csv").terms.iloc[-1]

    # 3. N3: giai đoạn 1 (cấu hình A) rồi giai đoạn 2, có hai điểm vận hành
    tr.main(_args(f, "S9", "--variant", "zero-c8-b2", "--objective", "n3s1A", "--pretrain", "pf", "--init-ckpt", best,
                  "--no-test"))
    s1 = f["dir"] / "runs" / "S9_span-zero-c8-b2+n3s1A_pf_fixed_x4_hr96_bic_f2" / "ckpt" / "best.pt"
    out = tr.main(_args(f, "S9", "--variant", "zero-c8-b2", "--objective", "n3s2", "--pretrain", "pf",
                        "--init-ckpt", str(s1), "--tau", "0.0"))
    s2 = f["dir"] / "runs" / "S9_span-zero-c8-b2+n3s2_pf_fixed_x4_hr96_bic_f2"
    ops = json.loads((s2 / "operating_points.json").read_text())
    assert len(ops["curve"]) == 11 and ops["F"] is not None
    assert (s2 / "test_hr96_x4_bic_headS.csv").exists() and (s2 / "test_hr96_x4_bic_headT.csv").exists()
    hs = pd.read_csv(s2 / "test_hr96_x4_bic_headS.csv")
    # cấu hình A: đầu trung thực giữ nguyên thân L1, nên PSNR bằng PSNR của thân trên ảnh bicubic
    m, _ = load_run_model(s2)
    assert isinstance(m, TwoHeadGated)
    assert np.isfinite(hs.psnr_y).all()

    # cùng mã lần chạy nhưng khác siêu tham số: bị chặn, không lặng lẽ dùng lại checkpoint
    with pytest.raises(SystemExit, match="siêu tham số khác"):
        tr.main(_args(f, "S9", "--variant", "zero-c8-b2", "--degrade", "generic", "--extra-dir", str(f["extra"]),
                      "--extra-name", "fake", "--ridge", "--tag", "lr2e-4", "--lr", "5e-5"))
    # 4. fold giữ kín bị chặn với phép thử quyết định; lần chạy lỗi được ghi sổ
    with pytest.raises(SystemExit):
        tr.main(["--exp", "T6", "--fold", "1", "--ami-raw", "x", "--bench", "x"])
    with pytest.raises(ValueError):
        tr.main(_args(f, "S9", "--variant", "zero-c56", "--pretrain", "pub", "--tag", "bad"))


def test_train_script_aux_with_landmarks(fake_ami):
    f = fake_ami
    tr = _script("train")
    from earsr.train.finetune import ami_train_paths
    paths = ami_train_paths(f["raw"], f["folds"], 2, "train")[:20]
    rng = np.random.default_rng(0)
    pts = np.stack([np.stack([rng.uniform(50, 440, 7), rng.uniform(50, 650, 7)], 1) for _ in paths]).astype(np.float32)
    npz = f["dir"] / "lm.npz"
    np.savez(npz, paths=np.array([str(p) for p in paths]), points=pts)
    out = tr.main(_args(f, "S9", "--variant", "zero-c8-b2", "--objective", "aux", "--landmarks", str(npz), "--no-test"))
    run = f["dir"] / "runs" / "S9_span-zero-c8-b2+aux_none_fixed_x4_hr96_bic_f2"
    terms = json.loads(pd.read_csv(run / "log.csv").terms.iloc[-1])
    assert "aux" in terms and out["objective"] == "aux"
    m, _ = load_run_model(run)       # checkpoint là thân thường, không có đầu phụ
    assert isinstance(m, SPAN)


def test_pretrain_script_controlled_track(tmp_path, monkeypatch):
    """Nhánh có kiểm soát: một mốc khác họ (RLFN) học lại từ đầu bằng đúng script tiền huấn luyện,
    rồi train.py tinh chỉnh từ checkpoint đó."""
    import sys
    rng = np.random.default_rng(0)
    d = tmp_path / "gen"
    d.mkdir()
    for i in range(3):
        cv2.imwrite(str(d / f"{i}.png"), cv2.GaussianBlur(rng.integers(0, 256, (160, 160, 3), dtype=np.uint8), (0, 0), 3))
    pt = _script("pretrain")
    for bb in ("rlfn", "span"):
        monkeypatch.setattr(sys, "argv", ["x", "--backbone", bb, "--variant", "zero", "--budget", "50", "--full-iters", "4",
                                          "--train-dir", str(d), "--val-dir", str(d), "--out", str(tmp_path / "runs"),
                                          "--batch-size", "2", "--patch-lr", "16", "--workers", "0", "--device", "cpu"])
        pt.main()
    run = tmp_path / "runs" / "PRE_rlfn-zero_ps50_na_x4_hrall_bic_f0"
    m, cfg = load_run_model(run)
    assert cfg["extra"]["backbone"] == "rlfn" and m(torch.rand(1, 3, 24, 24)).shape == (1, 3, 96, 96)
    # khởi tạo ngẫu nhiên: không trùng trọng số công bố
    pub = build_trainable("rlfn", "zero", "pub", 4)
    k = next(iter(pub.state_dict()))
    assert not torch.allclose(pub.state_dict()[k], m.state_dict()[k])
    m2 = build_trainable("rlfn", "zero", "ps50", 4, run / "ckpt" / "best.pt")
    assert all(torch.equal(a, b) for a, b in zip(m.state_dict().values(), m2.state_dict().values()))
    monkeypatch.setattr(sys, "argv", ["x", "--backbone", "rlfn", "--variant", "reflect", "--train-dir", str(d), "--val-dir", str(d)])
    with pytest.raises(SystemExit):
        pt.main()
