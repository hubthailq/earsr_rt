import pytest
import torch

from earsr.eval.receptive_field import measure_receptive_field
from earsr.models.registry import SPECS, load_state
from earsr.models.span import PADDING_MODES, SPAB, SPAN, Conv3XC
from earsr.models.variants import SHAPES, all_variants, build_span_variant, parse_variant


def _randomize(m, std=0.2):
    g = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for p in m.parameters():
            p.copy_(torch.randn(p.shape, generator=g) * std)


def _rel(a, b):
    return float((a - b).abs().max() / b.abs().max().clamp_min(1e-12))


def test_spab_formula_matches_paper():
    """out = (H + x) * (sigmoid(H) - 0.5). Thứ tự ngược lại không gây lỗi chạy
    nhưng cho kết quả khác; kiểm thử này bắt đúng lỗi đó."""
    b = SPAB(4, deploy=True)
    with torch.no_grad():
        for c in (b.c1_r, b.c2_r, b.c3_r):
            c.eval_conv.weight.zero_()
        b.c1_r.eval_conv.bias.fill_(0.3)
        b.c2_r.eval_conv.bias.fill_(-0.2)
        b.c3_r.eval_conv.bias.copy_(torch.tensor([0.5, -1.0, 2.0, 0.0]))
    x = torch.randn(1, 4, 5, 6)
    out, inner = b.eval()(x)
    h = torch.tensor([0.5, -1.0, 2.0, 0.0]).view(1, 4, 1, 1).expand_as(x)
    right = (h + x) * (torch.sigmoid(h) - 0.5)
    wrong = x * (torch.sigmoid(h) - 0.5) + h
    assert torch.allclose(out, right, atol=1e-6)
    assert not torch.allclose(out, wrong, atol=1e-3)
    # giá trị thứ hai là c1(x) SAU kích hoạt SiLU (bản gốc dùng SiLU inplace)
    assert torch.allclose(inner, torch.nn.functional.silu(torch.full_like(x, 0.3)), atol=1e-6)


@pytest.mark.parametrize("mode", PADDING_MODES)
def test_reparam_branches_agree(mode):
    c = Conv3XC(5, 7, padding_mode=mode)
    _randomize(c)
    x = torch.randn(2, 5, 9, 11)
    y_train = c.train()(x)
    y_eval = c.eval()(x)
    assert (y_train - y_eval).abs().max() < 1e-4
    c.switch_to_deploy()
    assert not hasattr(c, "sk") and (c(x) - y_eval).abs().max() < 1e-6


@pytest.mark.parametrize("variant", all_variants())
def test_variants_train_eval_deploy_agree(variant):
    m = build_span_variant(variant)
    _randomize(m, 0.03)
    x = torch.rand(1, 3, 20, 14)
    with torch.no_grad():
        y_train = m.train()(x)
        y_eval = m.eval()(x)
        y_dep = m.switch_to_deploy()(x)
    assert y_eval.shape == (1, 3, 80, 56)
    assert y_eval.abs().max() > 0 and _rel(y_train, y_eval) < 1e-4 and _rel(y_dep, y_eval) < 1e-5
    pad, shape = parse_variant(variant)
    assert (m.feature_channels, m.n_blocks) == SHAPES[shape] and m.padding_mode == pad


def test_padding_changes_only_the_border():
    """Cùng trọng số, khác kiểu đệm: vùng giữa (xa mép hơn bán kính vùng nhìn)
    phải giống hệt, vùng sát mép phải khác."""
    torch.manual_seed(0)
    ref = SPAN(feature_channels=8, n_blocks=2, padding_mode="zeros", deploy=True)
    _randomize(ref)
    r = measure_receptive_field(ref, 4, size=41, n_inputs=1)["radius"]
    assert r == 9  # 1 (conv_1) + 2×3 (hai khối) + 1 (conv_2) + 1 (tầng phóng)
    x = torch.rand(1, 3, 40, 40)
    y0 = ref(x)
    for mode in ("replicate", "reflect"):
        m = SPAN(feature_channels=8, n_blocks=2, padding_mode=mode, deploy=True)
        m.load_state_dict(ref.state_dict())
        y = m(x)
        c = slice(r * 4, (40 - r) * 4)
        assert (y[..., c, c] - y0[..., c, c]).abs().max() < 1e-6
        assert (y - y0).abs().max() > 1e-3


def test_receptive_field_of_original_span():
    m = SPAN().eval().switch_to_deploy()  # khởi tạo mặc định, gộp nhánh
    info = measure_receptive_field(m, 4, size=61, n_inputs=1)
    assert info["radius"] == 21 and not info["saturated"]  # 1 + 6×3 + 1 + 1


def test_deploy_param_count():
    m = SPAN(deploy=True)
    assert sum(p.numel() for p in m.parameters()) == 426288


def test_loads_published_weights_strictly(weights_dir):
    for name, ch, deploy, rng in (("span_ch48", 48, False, 255.0), ("span_ch48_t44", 48, False, 1.0),
                                  ("span_ch28", 28, True, 255.0)):
        f = weights_dir / SPECS[name].weights
        if not f.is_file():
            pytest.skip(f"thiếu {f.name}")
        m = SPAN(feature_channels=ch, deploy=deploy, img_range=rng)
        sd = load_state(SPECS[name], weights_dir)
        assert "img_range" not in sd                      # trọng số công bố không mang dải giá trị
        m.load_state_dict(sd, strict=True)
        assert float(m.img_range) == rng                  # nên giá trị lúc dựng được giữ


def _smooth(h=24, w=20, seed=0):
    g = torch.Generator().manual_seed(seed)
    low = torch.rand(1, 3, h // 4, w // 4, generator=g)
    return torch.nn.functional.interpolate(low, size=(h, w), mode="bicubic", align_corners=False).clamp(0, 1)


def test_img_range_travels_with_the_weights():
    """Dải giá trị nằm trong state_dict: nạp vào mô hình dựng với dải khác vẫn cho cùng đầu ra."""
    a = SPAN(feature_channels=8, n_blocks=2, img_range=255.0)
    _randomize(a, 0.03)
    sd = a.state_dict()
    assert float(sd["img_range"]) == 255.0
    b = SPAN(feature_channels=8, n_blocks=2)             # dựng với dải mặc định 1
    b.load_state_dict(sd, strict=True)
    x = _smooth()
    with torch.no_grad():
        assert float(b.img_range) == 255.0 and _rel(b.eval()(x), a.eval()(x)) < 1e-6
    # dạng đã gộp cũng mang theo
    d = SPAN(feature_channels=8, n_blocks=2, deploy=True)
    d.load_state_dict(a.eval().switch_to_deploy().state_dict(), strict=True)
    with torch.no_grad():
        assert float(d.img_range) == 255.0 and _rel(d(x), a(x)) < 1e-5


def test_finetune_from_published_span_starts_from_the_published_model(weights_dir, tmp_path):
    """Thân SPAN dựng để tinh chỉnh từ trọng số công bố phải cho đúng đầu ra của mô hình công bố,
    kể cả sau khi lưu rồi nạp lại qua các đường dựng thân khác. Sai dải giá trị không gây lỗi chạy,
    chỉ làm PSNR Set5 tụt từ 32,20 xuống 12,70 dB; kiểm thử này bắt đúng lỗi đó."""
    from earsr.models.optional.heads import TwoHeadGated  # noqa: F401
    from earsr.models.registry import build_model
    from earsr.train.finetune import build_n3, build_trainable

    if not (weights_dir / SPECS["span_ch48"].weights).is_file():
        pytest.skip("thiếu trọng số SPAN chính thức")
    x = _smooth(36, 28)
    with torch.no_grad():
        ref = build_model("span_ch48", wdir=weights_dir)(x)
        assert 0.0 < float(ref.mean()) < 1.0 and float(ref.std()) < 0.5        # ảnh hợp lệ, không bão hòa
        for variant in ("zero", "replicate"):
            net = build_trainable("span", variant, "pub", 4, wdir=weights_dir)
            if variant == "zero":
                assert _rel(net.eval()(x), ref) < 1e-4
            # lưu như trainer rồi nạp lại theo ba đường: load_run_model, init_ckpt, build_n3
            p = tmp_path / f"{variant}.pt"
            torch.save({"model": net.state_dict()}, p)
            y = net.eval()(x)
            again = build_trainable("span", variant, "none", 4)
            again.load_state_dict(torch.load(p, weights_only=False)["model"], strict=True)
            assert _rel(again.eval()(x), y) < 1e-6
            assert _rel(build_trainable("span", variant, "pf", 4, init_ckpt=p).eval()(x), y) < 1e-6
            assert _rel(build_n3(variant, 4, backbone_ckpt=p).eval()(x, gate=1.0), y) < 1e-6
