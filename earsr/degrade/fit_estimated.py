"""N2: ước lượng tham số suy giảm từ ảnh nhỏ thật (EarVN1.0).

File này chỉ dùng numpy, OpenCV và Pillow (không cần torch), để chạy được ở
nơi có dữ liệu thô.

Ba thứ được ước lượng:
1. Mức nén JPEG: đọc bảng lượng tử trong file, suy ra mức chất lượng theo công
   thức của libjpeg. Chắc chắn nhất trong ba thứ.
2. Mức nhiễu: phương pháp Immerkaer (1996) trên phần dư tần cao. Nén JPEG đã
   xóa bớt nhiễu, nên đây là cận dưới của nhiễu trước khi nén.
3. Độ mờ: không ước lượng được nhân mờ từ ảnh nhỏ đã nén. Thay vào đó, chọn
   dải sigma sao cho phân bố độ nét của ảnh mô phỏng (từ ảnh lớn của chính bộ
   dữ liệu) gần phân bố độ nét của ảnh nhỏ thật nhất (khoảng cách
   Kolmogorov-Smirnov). Bài phải nói rõ giới hạn này.

Tham số được chốt một lần, trước mọi lần huấn luyện, và chỉ từ nhóm người dành
cho việc khớp (mục 1.7 của kế hoạch).
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

# bảng lượng tử độ chói chuẩn của JPEG (Annex K), thứ tự hàng
_STD_LUMA = np.array([
    16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55,
    14, 13, 16, 24, 40, 57, 69, 56, 14, 17, 22, 29, 51, 87, 80, 62,
    18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92,
    49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99], dtype=np.float64)


def _table_for_quality(q: int) -> np.ndarray:
    s = 5000 / q if q < 50 else 200 - 2 * q
    return np.clip(np.floor((_STD_LUMA * s + 50) / 100), 1, 255)


def jpeg_quality_from_file(path: str | Path) -> int | None:
    """Mức chất lượng libjpeg (1..100) khớp nhất với bảng lượng tử độ chói của file.
    Trả về None nếu file không phải JPEG hoặc không có bảng."""
    from PIL import Image

    try:
        with Image.open(path) as im:
            q = getattr(im, "quantization", None)
            if not q:
                return None
            tab = np.sort(np.array(q[0], dtype=np.float64))
    except Exception:
        return None
    best, best_err = None, np.inf
    for cand in range(1, 101):
        err = np.abs(np.sort(_table_for_quality(cand)) - tab).mean()
        if err < best_err:
            best, best_err = cand, err
    return best


def noise_sigma_immerkaer(gray: np.ndarray) -> float:
    """Ước lượng độ lệch chuẩn nhiễu (thang 0..255) theo Immerkaer."""
    g = gray.astype(np.float64)
    if min(g.shape) < 3:
        return float("nan")
    k = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float64)
    r = cv2.filter2D(g, -1, k, borderType=cv2.BORDER_REFLECT_101)[1:-1, 1:-1]
    return float(np.sqrt(np.pi / 2) * np.abs(r).mean() / 6.0)


def sharpness(gray: np.ndarray) -> float:
    """Độ nét không phụ thuộc độ tương phản: năng lượng Laplace chia phương sai."""
    g = gray.astype(np.float64)
    v = g.var()
    if v < 1e-6:
        return 0.0
    return float(cv2.Laplacian(g, cv2.CV_64F).var() / v)


def ks_distance(a, b) -> float:
    a, b = np.sort(np.asarray(a, float)), np.sort(np.asarray(b, float))
    grid = np.concatenate([a, b])
    return float(np.abs(np.searchsorted(a, grid, side="right") / len(a)
                        - np.searchsorted(b, grid, side="right") / len(b)).max())


def fit(real_small: list, large: list, scale: int = 4, target_short: tuple[int, int] = (24, 48),
        blur_candidates=((0.0, 0.0), (0.0, 0.3), (0.0, 0.6), (0.2, 0.6), (0.2, 1.0), (0.2, 1.5), (0.5, 1.5),
                         (0.5, 2.0), (1.0, 2.5), (1.0, 3.0)),
        seed: int = 0, max_images: int = 2000) -> dict:
    """Trả về dict tham số theo dạng ``DegradeParams`` cùng số liệu chẩn đoán.

    real_small: đường dẫn ảnh nhỏ thật (cạnh ngắn trong ``target_short``).
    large     : đường dẫn ảnh lớn của cùng bộ dữ liệu, thuộc cùng nhóm người khớp.
    """
    from ..data.resize import imresize
    from . import ops

    rng = np.random.default_rng(seed)
    real_small = list(real_small)[:max_images]
    large = list(large)[:max_images]
    if len(real_small) < 20 or len(large) < 20:
        raise ValueError("cần ít nhất 20 ảnh nhỏ thật và 20 ảnh lớn")
    qs, sig, sharp_real = [], [], []
    for p in real_small:
        q = jpeg_quality_from_file(p)
        if q is not None:
            qs.append(q)
        g = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if g is None:
            continue
        sig.append(noise_sigma_immerkaer(g))
        sharp_real.append(sharpness(g))
    if not qs:
        raise ValueError("không đọc được bảng lượng tử JPEG nào")
    vals, counts = np.unique(qs, return_counts=True)
    jpeg_q = [[int(v), float(c / counts.sum())] for v, c in zip(vals, counts)]
    sig = np.array([s for s in sig if np.isfinite(s)])
    noise_rng = (float(np.quantile(sig, 0.05)), float(np.quantile(sig, 0.95)))

    def simulate(blur_rng):
        # Mỗi ứng viên dùng lại cùng một dòng số ngẫu nhiên (cùng cỡ ảnh, nhiễu, mức nén cho từng ảnh),
        # nên các ứng viên chỉ khác nhau ở độ mờ và kết quả không phụ thuộc thứ tự hay số ứng viên.
        rng = np.random.default_rng(seed)
        out = []
        for p in large:
            img = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if img is None:
                continue
            img = np.ascontiguousarray(img[:, :, ::-1])
            h, w = img.shape[:2]
            s_lr = int(rng.integers(target_short[0], target_short[1] + 1))
            s_hr = s_lr * scale
            if min(h, w) < s_hr:
                continue
            oh, ow = (int(round(h * s_hr / w)), s_hr) if h >= w else (s_hr, int(round(w * s_hr / h)))
            hr = imresize(img, out_size=(oh - oh % scale, ow - ow % scale))
            x = ops.gaussian_blur(hr, rng.uniform(*blur_rng) * scale / 2.0)
            x = ops.bicubic_down(x, scale)
            x = ops.gaussian_noise(x, rng.uniform(*noise_rng), rng)
            x = ops.jpeg(x, int(rng.choice(vals, p=counts / counts.sum())))
            out.append(sharpness(cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)))
        return out

    diag = []
    for cand in blur_candidates:
        sim = simulate(cand)
        if len(sim) < 20:
            continue
        diag.append({"blur_sigma": list(cand), "ks": ks_distance(sim, sharp_real), "n_sim": len(sim)})
    if not diag:
        raise ValueError("không mô phỏng được: ảnh lớn không đủ to cho cỡ ảnh nhỏ cần mô phỏng")
    best = min(diag, key=lambda d: d["ks"])
    # Tối ưu ở mức mờ lớn nhất của lưới nghĩa là lưới chưa đủ rộng. Tối ưu ở "không mờ" thì không phải lỗi
    # của lưới (không thể mờ ít hơn), nhưng nghĩa là suy giảm ước lượng gần như không thêm độ mờ nào.
    top = max(d["blur_sigma"][1] for d in diag)
    edge = {"best_is_no_blur": best["blur_sigma"][1] == 0.0, "best_at_upper_edge": best["blur_sigma"][1] == top}
    return {"params": {"blur_sigma": best["blur_sigma"], "blur_prob": 1.0, "noise_sigma": list(noise_rng),
                       "noise_prob": 1.0, "jpeg_q": jpeg_q, "jpeg_prob": 1.0,
                       "source": f"fit_estimated: {len(real_small)} ảnh nhỏ thật, {len(large)} ảnh lớn"},
            "diagnostics": {"blur_candidates": diag, **edge, "n_real_small": len(sharp_real), "n_jpeg_tables": len(qs),
                            "jpeg_q_median": float(np.median(qs)), "noise_sigma_median": float(np.median(sig)),
                            "limitation": "độ mờ chỉ khớp theo phân bố độ nét; nhiễu là cận dưới vì đã qua nén"}}


def save_fit(result: dict, params_path: str | Path, diag_path: str | Path | None = None) -> None:
    Path(params_path).parent.mkdir(parents=True, exist_ok=True)
    with open(params_path, "w") as f:
        json.dump(result["params"], f, indent=1)
    if diag_path:
        with open(diag_path, "w") as f:
            json.dump(result["diagnostics"], f, indent=1)
