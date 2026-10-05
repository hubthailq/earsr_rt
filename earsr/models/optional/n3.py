"""N3: chọn hai điểm vận hành (F: thiên trung thực, P: thiên sắc nét) bằng độ
lệch cổng, trên tập validation. Không dùng người test."""
from __future__ import annotations

import numpy as np
import torch

from ...eval.metrics import crop_border, psnr, rgb_to_y
from ...io import to_tensor, to_uint8
from .heads import TwoHeadGated

BIAS_GRID = (-6.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 6.0)   # lưới ghi trước


@torch.no_grad()
def _val_psnr(model: TwoHeadGated, pairs, scale: int, gate=None) -> float:
    dev = next(model.parameters()).device
    v = []
    for lr, hr in pairs:
        sr = to_uint8(model(to_tensor(lr).to(dev), gate=gate))
        v.append(psnr(crop_border(rgb_to_y(sr), scale), crop_border(rgb_to_y(hr), scale)))
    return float(np.mean(v))


def choose_operating_points(model: TwoHeadGated, pairs: list, scale: int, fidelity_deficit_db: float = 0.05,
                            psnr_budget_db: float = 0.5, grid: tuple = BIAS_GRID) -> dict:
    """Quét độ lệch cổng trên lưới ghi trước.

    F: độ lệch NHỎ NHẤT mà PSNR không thấp hơn đầu trung thực quá ``fidelity_deficit_db``.
    P: độ lệch NHỎ NHẤT mà PSNR không thấp hơn đầu trung thực quá ``psnr_budget_db``
       (độ lệch nhỏ hơn = dùng đầu kết cấu nhiều hơn).
    Không có giá trị nào đạt thì điểm đó là None.
    """
    model.eval()
    old = model.gate_bias
    ref = _val_psnr(model, pairs, scale, gate=1.0)
    curve = []
    for b in sorted(grid):
        model.gate_bias = float(b)
        curve.append({"gate_bias": float(b), "psnr_y": _val_psnr(model, pairs, scale)})
    model.gate_bias = old
    pick = lambda d: next((c["gate_bias"] for c in curve if c["psnr_y"] >= ref - d), None)
    return {"ref_psnr_y_fidelity_head": ref, "psnr_y_texture_head": _val_psnr(model, pairs, scale, gate=0.0),
            "F": pick(fidelity_deficit_db), "P": pick(psnr_budget_db), "curve": curve,
            "fidelity_deficit_db": fidelity_deficit_db, "psnr_budget_db": psnr_budget_db}
