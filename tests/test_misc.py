import cv2
import numpy as np
import pytest
import torch

from earsr.data import wild
from earsr.degrade.fit_estimated import jpeg_quality_from_file, ks_distance, noise_sigma_immerkaer, sharpness
from earsr.eval.complexity import count_flops, count_params, ntire_efficiency_score
from earsr.models.optional.heads import AuxStructureHead, TwoHeadGated, gate_label
from earsr.models.span import SPAN
from earsr.models.variants import build_span_variant


def test_flops_probe_scaling_is_exact_for_conv_nets():
    m = SPAN(feature_channels=8, n_blocks=2, deploy=True)
    a = count_flops(m, (128, 128), probe_hw=(32, 32))
    b = count_flops(m, (128, 128), probe_hw=None)
    assert a["flops"] == b["flops"] and not a["flops_partial"]
    assert count_params(SPAN(deploy=True)) == 426288


def test_ntire_score_is_monotone():
    base = dict(base_runtime=10.0, base_flops=20.0, base_params=0.3)
    assert ntire_efficiency_score(5, 10, 0.2, **base) < ntire_efficiency_score(10, 20, 0.3, **base)
    assert abs(ntire_efficiency_score(10, 20, 0.3, **base) - np.e ** 2) < 1e-9


def test_gate_identities_and_aux_head_is_removable():
    """Cổng bằng 1 cho đúng đầu trung thực, bằng 0 cho đúng đầu kết cấu; đầu phụ
    không nằm trên đường tính ảnh ra."""
    torch.manual_seed(0)
    bb = build_span_variant("zero").eval()
    m = TwoHeadGated(bb).eval()
    x = torch.rand(1, 3, 12, 10)
    with torch.no_grad():
        out, f_s, f_t, g = m(x, return_all=True)
        assert out.shape == (1, 3, 48, 40) and g.shape == (1, 1, 48, 40)
        assert torch.equal(m(x, gate=1.0), f_s) and torch.equal(m(x, gate=0.0), f_t)
        assert torch.allclose(f_s, bb(x))  # đầu trung thực chính là thân gốc
        m.gate_bias = 50.0
        assert torch.allclose(m(x), f_s, atol=1e-5)  # dịch cổng về phía trung thực
        aux = AuxStructureHead(bb.feature_channels, n_maps=5)
        assert aux(bb.forward_features(x)).shape == (1, 5, 12, 10)
    hr = f_s.clone()
    lab = gate_label(f_s, f_s + 0.1, hr, tau=0.01)
    assert lab.mean() == 1.0  # đầu kết cấu sai hơn quá τ ở mọi nơi: giữ đầu trung thực
    assert gate_label(f_s, f_s, hr, tau=0.01).mean() == 0.0


def test_jpeg_quality_and_noise_estimation(tmp_path):
    rng = np.random.default_rng(0)
    img = cv2.GaussianBlur(rng.integers(0, 256, (96, 96, 3), dtype=np.uint8), (0, 0), 2.0)
    for q in (30, 75, 90):
        p = tmp_path / f"{q}.jpg"
        cv2.imwrite(str(p), img, [cv2.IMWRITE_JPEG_QUALITY, q])
        assert jpeg_quality_from_file(p) == q
    assert jpeg_quality_from_file(tmp_path / "missing.jpg") is None
    flat = np.full((120, 120), 128.0)
    for s in (2.0, 8.0):
        assert abs(noise_sigma_immerkaer(flat + rng.normal(0, s, flat.shape)) - s) < 0.15 * s
    g = img[..., 0]
    assert sharpness(cv2.GaussianBlur(g, (0, 0), 2.0)) < sharpness(g)
    assert ks_distance([1, 2, 3], [1, 2, 3]) == 0 and ks_distance([0, 0], [1, 1]) == 1


def test_wild_selection_and_roles(tmp_path):
    rng = np.random.default_rng(0)
    root = tmp_path / "raw"
    def save(subj, name, hw, seed):
        (root / subj).mkdir(parents=True, exist_ok=True)
        r = np.random.default_rng(seed)
        cv2.imwrite(str(root / subj / name), cv2.GaussianBlur(r.integers(0, 256, (*hw, 3), dtype=np.uint8), (0, 0), 3))
    save("s1", "a.png", (300, 220), 1)
    save("s1", "dup.png", (300, 220), 1)       # trùng hẳn
    save("s1", "small.png", (150, 120), 2)      # quá nhỏ cho biên an toàn 2 lần ở cỡ 96
    save("s2", "wide.png", (200, 900), 3)       # tỉ lệ khung bất thường
    save("s2", "b.png", (260, 200), 4)
    rows = wild.scan(root)
    keep, drop = wild.select(rows, tier=96, safety=2.0)
    assert sorted(r["path"] for r in keep) == ["s1/a.png", "s2/b.png"]
    why = {r["path"]: r["why"] for r in drop}
    assert "trùng" in why["s1/dup.png"] and "cạnh ngắn" in why["s1/small.png"] and "tỉ lệ" in why["s2/wide.png"]
    man = wild.build_hr(root, keep, tmp_path / "out", 96, "toy")
    assert man.exists() and len(man.read_text().splitlines()) == 3
    roles = wild.assign_roles([f"s{i}" for i in range(20)], {"train": 0.5, "test": 0.2, "fit": 0.1, "clf": 0.1, "viewer": 0.1})
    assert len(roles) == 20 and sorted(set(roles.values())) == ["clf", "fit", "test", "train", "viewer"]
    with pytest.raises(ValueError):
        wild.assign_roles(["a", "b"], {"train": 0.5})


def test_onnx_export_matches_pytorch(tmp_path):
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    from earsr.deploy.export_onnx import export_onnx

    for mode in ("zeros", "replicate", "reflect"):
        torch.manual_seed(0)
        m = SPAN(feature_channels=8, n_blocks=2, padding_mode=mode).eval()
        info = export_onnx(m, tmp_path / f"{mode}.onnx", (17, 13))
        assert info["checked"] and info["max_abs_diff"] < 1e-4
        assert ("Pad" in info["ops"]) == (mode != "zeros")
