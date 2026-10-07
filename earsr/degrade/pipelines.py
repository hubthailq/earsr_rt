"""Các kiểu suy giảm của bài (mục 1.7 của kế hoạch).

- ``bic``: bicubic, theo NTIRE.
- ``bicjpeg{q}``: bicubic rồi nén JPEG mức q. Dùng ở T2 để xem thứ hạng có đổi.
- ``generic``: mờ, thu nhỏ, nhiễu, nén với tham số rút ngẫu nhiên trong dải
  rộng kiểu Real-ESRGAN (bản một bậc, rút gọn).
- ``est``: cùng cấu trúc, tham số lấy từ một file do ``fit_estimated`` tạo ra
  từ ảnh EarVN1.0 thật (N2).
- ``jpegmix``: bicubic rồi JPEG, mức nén rút từ đúng phân bố ``jpeg_q`` của file tham số; không mờ, không
  nhiễu. Tách phần "phân bố mức nén" khỏi phần còn lại của ``est``.
- ``jpegu``: bicubic rồi JPEG, mức nén rút đều trong ``JPEGU_RANGE``; không cần file tham số. Mốc "có nén
  nhưng không đo gì từ dữ liệu".

Mỗi hàm nhận ảnh HR uint8 và trả về ảnh LR uint8. Với cùng ``seed`` và cùng
khóa ảnh, kết quả giống nhau từng bit.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import ops


def rng_for(seed: int, key: str) -> np.random.Generator:
    """Bộ sinh ngẫu nhiên riêng cho từng ảnh, không phụ thuộc thứ tự xử lý."""
    h = hashlib.sha1(f"{seed}|{key}".encode()).digest()
    return np.random.default_rng(int.from_bytes(h[:8], "little"))


@dataclass
class DegradeParams:
    """Dải tham số của suy giảm ngẫu nhiên. ``jpeg_q`` là danh sách (mức, xác suất)."""
    blur_sigma: tuple[float, float] = (0.2, 3.0)
    blur_prob: float = 1.0
    noise_sigma: tuple[float, float] = (1.0, 30.0)
    noise_prob: float = 1.0
    jpeg_q: list = field(default_factory=lambda: [[30, 0.25], [50, 0.25], [70, 0.25], [90, 0.25]])
    jpeg_prob: float = 1.0
    source: str = "generic-default"

    @staticmethod
    def load(path: str | Path) -> "DegradeParams":
        with open(path) as f:
            d = json.load(f)
        return DegradeParams(**d)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.__dict__, f, indent=1)


JPEGU_RANGE = (60, 95)             # dải mức nén của kiểu jpegu, gồm cả hai đầu
PARAM_KINDS = ("est", "jpegmix")   # các kiểu bắt buộc có file tham số ước lượng


def needs_params(kind: str) -> bool:
    return kind in PARAM_KINDS


def _draw_quality(p: DegradeParams, rng: np.random.Generator) -> int:
    qs = np.array([q for q, _ in p.jpeg_q], dtype=np.float64)
    pr = np.array([w for _, w in p.jpeg_q], dtype=np.float64)
    return int(rng.choice(qs, p=pr / pr.sum()))


def random_degrade(hr: np.ndarray, scale: int, p: DegradeParams, rng: np.random.Generator) -> np.ndarray:
    """mờ -> thu nhỏ bicubic -> nhiễu -> JPEG, tham số rút từ ``p``."""
    x = hr
    if rng.random() < p.blur_prob:
        # sigma tính theo điểm ảnh LR, nên nhân với hệ số phóng khi mờ trên HR
        s = rng.uniform(*p.blur_sigma)
        x = ops.gaussian_blur(x, s * scale / 2.0)
    x = ops.bicubic_down(x, scale)
    if rng.random() < p.noise_prob:
        x = ops.gaussian_noise(x, rng.uniform(*p.noise_sigma), rng)
    if rng.random() < p.jpeg_prob:
        x = ops.jpeg(x, _draw_quality(p, rng))
    return x


_BICJPEG = re.compile(r"^bicjpeg(\d{1,3})$")


def degrade(hr: np.ndarray, scale: int, kind: str, seed: int = 0, key: str = "",
            params: DegradeParams | None = None) -> np.ndarray:
    """Điểm vào duy nhất để tạo ảnh LR.

    kind: ``bic`` | ``bicjpegQ`` | ``generic`` | ``est`` | ``jpegmix`` | ``jpegu``.
    ``est`` và ``jpegmix`` bắt buộc có ``params`` (nạp từ file của ``fit_estimated``).
    """
    if kind == "bic":
        return ops.bicubic_down(hr, scale)
    m = _BICJPEG.match(kind)
    if m:
        return ops.jpeg(ops.bicubic_down(hr, scale), int(m.group(1)))
    if kind == "generic":
        return random_degrade(hr, scale, params or DegradeParams(), rng_for(seed, key))
    if kind == "est":
        if params is None:
            raise ValueError("kiểu 'est' cần params ước lượng từ ảnh thật")
        return random_degrade(hr, scale, params, rng_for(seed, key))
    if kind == "jpegmix":
        if params is None:
            raise ValueError("kiểu 'jpegmix' cần params ước lượng từ ảnh thật")
        return ops.jpeg(ops.bicubic_down(hr, scale), _draw_quality(params, rng_for(seed, key)))
    if kind == "jpegu":
        lo, hi = JPEGU_RANGE
        return ops.jpeg(ops.bicubic_down(hr, scale), int(rng_for(seed, key).integers(lo, hi + 1)))
    raise ValueError(f"kiểu suy giảm không biết: {kind}")
