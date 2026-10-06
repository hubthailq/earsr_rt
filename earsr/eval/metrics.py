"""Số đo có ảnh tham chiếu: PSNR, SSIM (kênh Y và RGB), LPIPS, DISTS.

Quy ước:
- Ngoài PSNR, SSIM, LPIPS, DISTS còn có (mọi mô hình, mọi ảnh, không cần tải gì):
  MS-SSIM (số tầng giảm theo cỡ ảnh), GMSD, PSNR của gradient, và LR-PSNR (độ
  nhất quán với ảnh vào). Qua pyiqa (tùy chọn): ST-LPIPS, TOPIQ-FR, FSIM, VIF, PieAPP.
- PSNR-Y, SSIM-Y: kênh Y kiểu MATLAB ``rgb2ycbcr`` (BT.601, dải 16..235),
  bỏ viền bằng hệ số phóng. Đây là quy ước của NTIRE Image SR.
- PSNR-RGB: trên ảnh uint8 RGB, bỏ viền bằng hệ số phóng. Đây là quy ước của
  NTIRE Efficient SR (ngưỡng 26,90 dB tính theo cách này).
- Mọi số đo được tính trên ảnh uint8 đã làm tròn, như khi lưu ảnh ra đĩa.

LPIPS và DISTS cần trọng số ImageNet (AlexNet, VGG16). Nếu máy không tải được
trọng số, hàm trả về NaN kèm lý do, không dừng chương trình.
"""
from __future__ import annotations

import cv2
import numpy as np

__all__ = ["rgb_to_y", "psnr", "ssim", "ms_ssim", "gmsd", "grad_psnr", "lr_psnr", "crop_border", "crop_center",
           "PerceptualMetrics", "fr_metrics", "PYIQA_FR", "LR_PSNR_CAP"]

# số đo có tham chiếu lấy từ pyiqa (tên theo pyiqa; CHƯA CHẠY THẬT ở máy dựng project)
PYIQA_FR = ("dists", "stlpips", "topiq_fr", "fsim", "vif", "pieapp")
LR_PSNR_CAP = 100.0   # dB; xem lr_psnr


def rgb_to_y(img: np.ndarray) -> np.ndarray:
    """uint8 RGB -> kênh Y float64 trong dải 16..235 (không làm tròn), như MATLAB."""
    x = img.astype(np.float64) / 255.0
    return 16.0 + 65.481 * x[..., 0] + 128.553 * x[..., 1] + 24.966 * x[..., 2]


def crop_border(img: np.ndarray, b: int) -> np.ndarray:
    return img if b <= 0 else img[b:-b, b:-b]


def crop_center(img: np.ndarray, margin: int) -> np.ndarray:
    """Vùng giữa ảnh: bỏ ``margin`` điểm ảnh ở mỗi cạnh."""
    return crop_border(img, margin)


def psnr(a: np.ndarray, b: np.ndarray, peak: float = 255.0) -> float:
    if a.shape != b.shape:
        raise ValueError(f"khác kích thước: {a.shape} và {b.shape}")
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    if mse <= 1e-12:
        return float("inf")
    return float(10.0 * np.log10(peak * peak / mse))


def _ssim_single(a: np.ndarray, b: np.ndarray) -> float:
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    k = cv2.getGaussianKernel(11, 1.5)
    win = np.outer(k, k.T)
    mu1 = cv2.filter2D(a, -1, win)[5:-5, 5:-5]
    mu2 = cv2.filter2D(b, -1, win)[5:-5, 5:-5]
    mu1_sq, mu2_sq, mu12 = mu1 * mu1, mu2 * mu2, mu1 * mu2
    s1 = cv2.filter2D(a * a, -1, win)[5:-5, 5:-5] - mu1_sq
    s2 = cv2.filter2D(b * b, -1, win)[5:-5, 5:-5] - mu2_sq
    s12 = cv2.filter2D(a * b, -1, win)[5:-5, 5:-5] - mu12
    m = ((2 * mu12 + c1) * (2 * s12 + c2)) / ((mu1_sq + mu2_sq + c1) * (s1 + s2 + c2))
    return float(m.mean())


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    """SSIM của Wang và cộng sự (cửa sổ Gauss 11×11, σ = 1,5), dải 0..255.

    Ảnh 2 chiều: tính trực tiếp. Ảnh 3 kênh: trung bình theo kênh.
    Trả về NaN nếu ảnh nhỏ hơn 11 px ở một chiều.
    """
    if a.shape != b.shape:
        raise ValueError(f"khác kích thước: {a.shape} và {b.shape}")
    if min(a.shape[:2]) < 11:
        return float("nan")
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    if a.ndim == 2:
        return _ssim_single(a, b)
    return float(np.mean([_ssim_single(a[..., c], b[..., c]) for c in range(a.shape[2])]))


def _down2(x: np.ndarray) -> np.ndarray:
    """Giảm 2 lần bằng trung bình 2×2 (bỏ hàng, cột lẻ cuối)."""
    h, w = x.shape[0] // 2 * 2, x.shape[1] // 2 * 2
    x = x[:h, :w]
    return 0.25 * (x[0::2, 0::2] + x[1::2, 0::2] + x[0::2, 1::2] + x[1::2, 1::2])


def _ssim_cs(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """(SSIM, thành phần tương phản và cấu trúc) trung bình, cùng cửa sổ như ``ssim``."""
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    k = cv2.getGaussianKernel(11, 1.5)
    win = np.outer(k, k.T)
    f = lambda z: cv2.filter2D(z, -1, win)[5:-5, 5:-5]
    mu1, mu2 = f(a), f(b)
    s1, s2, s12 = f(a * a) - mu1 * mu1, f(b * b) - mu2 * mu2, f(a * b) - mu1 * mu2
    cs = (2 * s12 + c2) / (s1 + s2 + c2)
    lum = (2 * mu1 * mu2 + c1) / (mu1 * mu1 + mu2 * mu2 + c1)
    return float((lum * cs).mean()), float(cs.mean())


_MS_WEIGHTS = (0.0448, 0.2856, 0.3001, 0.2363, 0.1333)   # Wang, Simoncelli, Bovik 2003


def ms_ssim(a: np.ndarray, b: np.ndarray, max_scales: int = 5) -> tuple[float, int]:
    """MS-SSIM trên ảnh 2 chiều, dải 0..255. Trả về (giá trị, số tầng đã dùng).

    Bản chuẩn dùng 5 tầng và cần ảnh từ 176 px mỗi chiều. Ảnh tai của bài nhỏ
    hơn, nên số tầng được giảm cho vừa ảnh (mỗi tầng cần ít nhất 11 px sau khi
    giảm), và các trọng số chuẩn của những tầng được dùng được chuẩn hóa lại cho
    tổng bằng 1. Vì vậy giá trị KHÔNG so được với MS-SSIM 5 tầng trong tài liệu;
    chỉ so giữa các mô hình trên cùng cỡ ảnh. Bài phải ghi số tầng.
    """
    if a.shape != b.shape or a.ndim != 2:
        raise ValueError("ms_ssim cần hai ảnh 2 chiều cùng kích thước")
    n = 0
    while n < max_scales and min(a.shape) // (2 ** n) >= 11:
        n += 1
    if n == 0:
        return float("nan"), 0
    w = np.array(_MS_WEIGHTS[:n]) / sum(_MS_WEIGHTS[:n])
    a, b = a.astype(np.float64), b.astype(np.float64)
    val = 1.0
    for i in range(n):
        s, cs = _ssim_cs(a, b)
        val *= max(s if i == n - 1 else cs, 0.0) ** w[i]
        if i < n - 1:
            a, b = _down2(a), _down2(b)
    return float(val), n


def _prewitt_mag(x: np.ndarray) -> np.ndarray:
    kx = np.array([[1, 0, -1]] * 3, dtype=np.float64) / 3.0
    gx = cv2.filter2D(x, -1, kx, borderType=cv2.BORDER_REFLECT)
    gy = cv2.filter2D(x, -1, kx.T, borderType=cv2.BORDER_REFLECT)
    return np.sqrt(gx * gx + gy * gy)


def gmsd(a: np.ndarray, b: np.ndarray, c: float = 170.0) -> float:
    """Gradient Magnitude Similarity Deviation (Xue và cộng sự, 2014), ảnh 2 chiều dải 0..255.
    Càng THẤP càng tốt; 0 khi hai ảnh trùng nhau. Theo bài gốc: giảm 2 lần bằng
    trung bình 2×2, gradient Prewitt, hằng số c = 170."""
    if a.shape != b.shape or a.ndim != 2:
        raise ValueError("gmsd cần hai ảnh 2 chiều cùng kích thước")
    a, b = _down2(a.astype(np.float64)), _down2(b.astype(np.float64))
    if min(a.shape) < 3:
        return float("nan")
    g1, g2 = _prewitt_mag(a), _prewitt_mag(b)
    gms = (2 * g1 * g2 + c) / (g1 * g1 + g2 * g2 + c)
    return float(gms.std())


def grad_psnr(a: np.ndarray, b: np.ndarray) -> float:
    """PSNR giữa hai bản đồ độ lớn gradient (Sobel 3×3) của kênh Y: đo riêng độ đúng
    của cạnh và gờ, ít bị vùng phẳng chi phối như PSNR thường."""
    def mag(x):
        x = x.astype(np.float64)
        return np.hypot(cv2.Sobel(x, cv2.CV_64F, 1, 0, ksize=3), cv2.Sobel(x, cv2.CV_64F, 0, 1, ksize=3)) / 4.0
    return psnr(mag(a)[1:-1, 1:-1], mag(b)[1:-1, 1:-1])


def lr_psnr(sr: np.ndarray, hr: np.ndarray, scale: int) -> float:
    """Độ nhất quán với ảnh vào (LR-PSNR): thu nhỏ bicubic cả ảnh SR lẫn ảnh đáp án rồi
    tính PSNR-Y. Mô hình bịa chi tiết có thể đạt LPIPS tốt nhưng làm đổi nội dung
    tần thấp; số đo này bắt việc đó. Dùng ảnh đáp án thu nhỏ (không dùng ảnh LR đã
    suy giảm) để số đo có nghĩa với mọi kiểu suy giảm.

    Chặn trên ở ``LR_PSNR_CAP`` (100 dB): ảnh SR thu nhỏ lại trùng khít ảnh đáp án thu nhỏ cho PSNR vô cực,
    và một giá trị vô cực làm hỏng mọi trung bình."""
    from ..data.resize import imresize

    h, w = hr.shape[0] // scale, hr.shape[1] // scale
    return min(LR_PSNR_CAP, psnr(rgb_to_y(imresize(sr, out_size=(h, w))), rgb_to_y(imresize(hr, out_size=(h, w)))))


class PerceptualMetrics:
    """LPIPS (AlexNet) và DISTS, nạp một lần, dùng lại.

    ``available`` cho biết số đo nào dùng được; ``reason`` ghi lý do nếu không.
    Phiên bản thư viện và mạng nền được ghi vào ``info`` để lưu cùng kết quả.
    """

    def __init__(self, device: str = "cpu", want: tuple[str, ...] = ("lpips", "dists")):
        self.device = device
        self.fn: dict = {}        # tên -> GpuFirst giữ mạng của số đo (ưu tiên ``device``, hết bộ nhớ thì CPU)
        self._score: dict = {}    # tên -> hàm (mạng, x, y) -> float
        self.reason: dict = {}
        self.info: dict = {}
        for name in want:
            try:
                if name == "lpips":
                    import lpips  # type: ignore

                    self._add("lpips", lambda dev: lpips.LPIPS(net="alex", verbose=False).to(dev).eval(),
                              lambda net, x, y: float(net(x * 2 - 1, y * 2 - 1).item()))
                    self.info["lpips"] = f"lpips {getattr(lpips, '__version__', '?')}, net=alex, v0.1"
                elif name in PYIQA_FR:
                    import pyiqa  # type: ignore

                    net = self._add(name, lambda dev, name=name: pyiqa.create_metric(name, device=dev),
                                    lambda net, x, y: float(net(x, y).item()))
                    self.info[name] = (f"pyiqa {getattr(pyiqa, '__version__', '?')}, {name}, "
                                       f"lower_better={getattr(net, 'lower_better', '?')}")
                else:
                    raise ValueError(f"số đo không biết: {name}")
            except Exception as e:  # thiếu thư viện hoặc không tải được trọng số
                self.fn.pop(name, None)
                self.reason[name] = f"{type(e).__name__}: {str(e)[:200]}"

    def _add(self, name: str, factory, score):
        """Đăng ký một số đo: ``factory(device)`` dựng mạng, ``score(mạng, x, y)`` trả về một số.
        Mạng được dựng ngay (để lỗi nạp lộ ra ở đây); trả về bản vừa dựng."""
        from ..device import GpuFirst

        runner = GpuFirst(factory, self.device, name=name).warm()
        self.fn[name], self._score[name] = runner, score
        return runner._get(runner.last_device)

    @property
    def available(self) -> list[str]:
        return list(self.fn)

    def __call__(self, sr: np.ndarray, hr: np.ndarray) -> dict:
        import torch

        out = {}
        if not self.fn:
            return out
        from ..device import full_precision

        with torch.no_grad(), full_precision():
            x = torch.from_numpy(sr).permute(2, 0, 1).float().div(255).unsqueeze(0)
            y = torch.from_numpy(hr).permute(2, 0, 1).float().div(255).unsqueeze(0)
            for k, runner in self.fn.items():
                score = self._score[k]
                try:  # hết bộ nhớ GPU được GpuFirst đỡ (chạy lại trên CPU); lỗi khác thì ghi NaN
                    out[k] = runner.run(lambda net, dev: score(net, x.to(dev), y.to(dev)))
                except Exception:
                    out[k] = float("nan")
        return out


def fr_metrics(sr: np.ndarray, hr: np.ndarray, scale: int, center_margin_lr: int = 8,
               box: tuple[int, int, int, int] | None = None,
               perceptual: PerceptualMetrics | None = None) -> dict:
    """Bộ số đo cho một cặp ảnh uint8 RGB.

    Ba phạm vi (mục 1.9 của kế hoạch):
    - toàn ảnh, bỏ viền ``scale`` px (hậu tố rỗng);
    - vùng giữa, bỏ ``center_margin_lr * scale`` px mỗi cạnh (hậu tố ``_c``);
    - trong hộp bao vùng tai ``box = (top, left, bottom, right)`` theo toạ độ HR
      (hậu tố ``_box``), nếu có.
    """
    if sr.shape != hr.shape:
        raise ValueError(f"khác kích thước: {sr.shape} và {hr.shape}")
    out: dict = {}
    sy, hy = rgb_to_y(sr), rgb_to_y(hr)
    b = scale
    out["psnr_y"] = psnr(crop_border(sy, b), crop_border(hy, b))
    out["ssim_y"] = ssim(crop_border(sy, b), crop_border(hy, b))
    out["psnr_rgb"] = psnr(crop_border(sr, b), crop_border(hr, b))
    out["ms_ssim_y"], out["ms_ssim_scales"] = ms_ssim(crop_border(sy, b), crop_border(hy, b))
    out["gmsd"] = gmsd(crop_border(sy, b), crop_border(hy, b))
    out["grad_psnr"] = grad_psnr(crop_border(sy, b), crop_border(hy, b))
    out["lr_psnr_y"] = lr_psnr(sr, hr, scale)
    m = center_margin_lr * scale
    if min(hr.shape[:2]) - 2 * m >= 8:
        out["psnr_y_c"] = psnr(crop_center(sy, m), crop_center(hy, m))
        out["ssim_y_c"] = ssim(crop_center(sy, m), crop_center(hy, m))
    else:
        out["psnr_y_c"] = float("nan")
        out["ssim_y_c"] = float("nan")
    if box is not None:
        t, l, bo, r = box
        out["psnr_y_box"] = psnr(sy[t:bo, l:r], hy[t:bo, l:r])
    if perceptual is not None:
        out.update(perceptual(sr, hr))
    return out
