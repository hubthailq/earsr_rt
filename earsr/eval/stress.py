"""Phép thử "ảnh sáng": mô hình có còn chạy đúng khi ảnh sáng hơn dữ liệu huấn luyện không.

Lý do có phép thử này (06/10/2026): mô hình tinh chỉnh chỉ trên AMI đạt PSNR validation bình thường
(39,3 dB) nhưng cho đầu ra hỏng hẳn trên ảnh có vùng sáng của EarVN1.0 (SPAN: 15,9 dB, mốc công bố 35,4 dB).
AMI không có ảnh nào như vậy, nên validation thường không thấy được. Phép thử nhân độ sáng của chính ảnh
validation lên rồi chấm lại, và so khoảng cách tới nội suy bicubic ở từng hệ số với khoảng cách ở ảnh gốc.
Mô hình ổn định giữ được khoảng cách đó; mô hình hỏng thì tụt hàng chục dB. So với chính nó ở ảnh gốc (chứ
không đòi hơn bicubic) vì với ảnh vào có nén, mô hình chưa tinh chỉnh vốn kém bicubic mà không hề mất ổn định.

Phép thử không dùng để chọn checkpoint (việc đó vẫn theo PSNR validation); nó chỉ báo động.
"""
from __future__ import annotations

import numpy as np

from ..data.resize import imresize
from ..degrade.pipelines import DegradeParams, degrade
from .metrics import crop_border, psnr, rgb_to_y

GAINS = (1.0, 1.4, 1.8)     # hệ số đầu là mốc so sánh (ảnh gốc)
MAX_GAP_DROP_DB = 1.5       # khoảng cách tới bicubic được tụt tối đa ngần này so với ở ảnh gốc


def brighten(img: np.ndarray, gain: float) -> np.ndarray:
    """Nhân độ sáng rồi cắt về 255 (vùng cháy sáng), như ảnh chụp thừa sáng."""
    return np.clip(np.round(img.astype(np.float64) * gain), 0, 255).astype(np.uint8)


def brightness_stress(predict, hrs: list, scale: int, kind: str = "bic", params: DegradeParams | None = None,
                      gains: tuple[float, ...] = GAINS, max_gap_drop_db: float = MAX_GAP_DROP_DB) -> dict:
    """Chấm ``predict`` (hàm lr uint8 -> sr uint8) trên các ảnh đáp án ``hrs`` đã nhân độ sáng.

    Ảnh vào được sinh lại từ ảnh đáp án đã đổi, bằng đúng kiểu suy giảm ``kind`` của lần chạy.
    Trả về PSNR-Y trung bình của mô hình và của bicubic ở từng hệ số, số ảnh mô hình kém bicubic
    quá 10 dB (``n_broken``), mức tụt lớn nhất của khoảng cách tới bicubic so với ở hệ số đầu
    (``gap_drop``), và ``stable``: không ảnh nào hỏng và mức tụt đó không quá ``max_gap_drop_db``.
    """
    f = lambda sr, hr: psnr(crop_border(rgb_to_y(sr), scale), crop_border(rgb_to_y(hr), scale))
    out: dict = {"gains": list(gains), "model": [], "bicubic": [], "n_broken": [], "n": len(hrs)}
    for g in gains:
        m, b = [], []
        for j, hr0 in enumerate(hrs):
            hr = brighten(hr0, g)
            lr = degrade(hr, scale, kind, seed=j, key=f"stress{g}", params=params)
            m.append(f(predict(lr), hr))
            b.append(f(imresize(lr, float(scale)), hr))
        m, b = np.array(m), np.array(b)
        out["model"].append(float(m.mean()))
        out["bicubic"].append(float(b.mean()))
        out["n_broken"].append(int((m < b - 10.0).sum()))
    gaps = [m - b for m, b in zip(out["model"], out["bicubic"])]
    out["gap_drop"] = float(max(gaps[0] - g for g in gaps))
    out["stable"] = bool(sum(out["n_broken"]) == 0 and out["gap_drop"] <= max_gap_drop_db)
    return out


def describe(res: dict) -> str:
    parts = [f"×{g:g}: {m:.2f} dB (bicubic {b:.2f}, hỏng {k}/{res['n']})"
             for g, m, b, k in zip(res["gains"], res["model"], res["bicubic"], res["n_broken"])]
    return (("ỔN ĐỊNH" if res["stable"] else "KHÔNG ỔN ĐỊNH") + f" trên ảnh sáng (khoảng cách tới bicubic tụt nhiều nhất "
            f"{res['gap_drop']:.2f} dB). " + "; ".join(parts))
