import pytest
import numpy as np

from earsr.eval.metrics import fr_metrics, psnr, rgb_to_y, ssim


def test_y_channel_range_matches_matlab():
    white = np.full((2, 2, 3), 255, np.uint8)
    black = np.zeros((2, 2, 3), np.uint8)
    assert np.allclose(rgb_to_y(white), 235.0) and np.allclose(rgb_to_y(black), 16.0)
    red = np.zeros((1, 1, 3), np.uint8)
    red[..., 0] = 255
    assert np.allclose(rgb_to_y(red), 16 + 65.481)


def test_psnr_known_value():
    a = np.zeros((10, 10), np.float64)
    b = np.full((10, 10), 5.0)
    assert abs(psnr(a, b) - 10 * np.log10(255 ** 2 / 25.0)) < 1e-9
    assert psnr(a, a) == float("inf")


def test_ssim_identity_and_degradation():
    rng = np.random.default_rng(0)
    a = rng.integers(0, 256, (40, 40)).astype(np.float64)
    assert abs(ssim(a, a) - 1.0) < 1e-12
    assert ssim(a, np.clip(a + rng.normal(0, 20, a.shape), 0, 255)) < 0.99
    assert np.isnan(ssim(a[:8, :8], a[:8, :8]))  # nhỏ hơn cửa sổ 11 px


def test_scopes():
    rng = np.random.default_rng(0)
    hr = rng.integers(0, 256, (204, 144, 3), dtype=np.uint8)
    sr = hr.copy()
    sr[:10] = 0  # chỉ làm hỏng dải sát mép trên
    m = fr_metrics(sr, hr, scale=4, box=(60, 40, 160, 110))
    assert m["psnr_y"] < 40 and m["psnr_y_c"] == float("inf") and m["psnr_y_box"] == float("inf")
    assert set(m) >= {"psnr_y", "ssim_y", "psnr_rgb", "psnr_y_c", "ssim_y_c", "psnr_y_box"}


# ----------------------------------------------------------------- số đo bổ sung (bản 22)

def test_gmsd_ms_ssim_grad_and_lr_consistency():
    import cv2
    from earsr.data.resize import imresize
    from earsr.eval.metrics import gmsd, grad_psnr, lr_psnr, ms_ssim
    rng = np.random.default_rng(0)
    hr = cv2.GaussianBlur(rng.integers(0, 256, (204, 144, 3), dtype=np.uint8), (0, 0), 2.0)
    y = rgb_to_y(hr)
    # trùng nhau: giá trị tốt nhất
    assert gmsd(y, y) == 0.0 and np.isinf(grad_psnr(y, y)) and np.isinf(lr_psnr(hr, hr, 4))
    v, n = ms_ssim(y, y)
    assert v == pytest.approx(1.0) and n == 4          # 144 px: 4 tầng (144/8 = 18 ≥ 11, 144/16 = 9 < 11)
    assert ms_ssim(y[:20, :20], y[:20, :20])[1] == 1 and ms_ssim(y[:8, :8], y[:8, :8])[1] == 0
    assert ms_ssim(np.zeros((400, 400)) + y.mean(), np.zeros((400, 400)) + y.mean())[1] == 5
    # một tầng thì bằng SSIM thường
    a, b = y[:20, :20], rgb_to_y(cv2.GaussianBlur(hr, (0, 0), 1.0))[:20, :20]
    assert ms_ssim(a, b)[0] == pytest.approx(ssim(a, b), abs=1e-9)
    # làm mờ nhiều hơn thì xấu hơn, đơn điệu
    blur = [rgb_to_y(cv2.GaussianBlur(hr, (0, 0), s)) for s in (0.6, 1.2, 2.4)]
    g = [gmsd(x, y) for x in blur]
    m = [ms_ssim(x, y)[0] for x in blur]
    gp = [grad_psnr(x, y) for x in blur]
    assert g[0] < g[1] < g[2] and m[0] > m[1] > m[2] and gp[0] > gp[1] > gp[2]
    # GMSD theo định nghĩa, tính tay
    x = blur[1]
    d = lambda z: 0.25 * (z[0::2, 0::2] + z[1::2, 0::2] + z[0::2, 1::2] + z[1::2, 1::2])
    k = np.array([[1, 0, -1]] * 3) / 3.0
    mag = lambda z: np.hypot(cv2.filter2D(z, -1, k, borderType=cv2.BORDER_REFLECT),
                             cv2.filter2D(z, -1, k.T, borderType=cv2.BORDER_REFLECT))
    g1, g2 = mag(d(x)), mag(d(y))
    assert gmsd(x, y) == pytest.approx(((2 * g1 * g2 + 170) / (g1 ** 2 + g2 ** 2 + 170)).std(), rel=1e-9)
    # LR-PSNR: thêm chi tiết tần cao (bàn cờ biên độ nhỏ) gần như không đổi ảnh thu nhỏ, nên LR-PSNR vẫn cao
    # dù PSNR thường tụt; dịch độ sáng thì LR-PSNR tụt theo
    chk = (np.indices(hr.shape[:2]).sum(0) % 2 * 2 - 1)[..., None] * 6
    textured = np.clip(hr.astype(int) + chk, 0, 255).astype(np.uint8)
    shifted = np.clip(hr.astype(int) + 6, 0, 255).astype(np.uint8)
    assert psnr(rgb_to_y(textured), y) == pytest.approx(psnr(rgb_to_y(shifted), y), abs=1.0)
    assert lr_psnr(textured, hr, 4) > lr_psnr(shifted, hr, 4) + 10
    # có mặt trong bộ số đo của mỗi ảnh
    out = fr_metrics(textured, hr, 4)
    assert {"ms_ssim_y", "ms_ssim_scales", "gmsd", "grad_psnr", "lr_psnr_y"} <= set(out) and out["ms_ssim_scales"] == 4
