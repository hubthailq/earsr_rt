"""Số tham số và số phép tính, đếm trên bản deploy (đã gộp nhánh).

FLOPs ở đây là số phép nhân cộng của Conv2d và Linear cho một ảnh vào, không
tính bias, giống cách fvcore đếm tích chập. Phép nhân ma trận trong attention
của transformer không được đếm; với SwinIR con số vì vậy thấp hơn thực tế và
được đánh dấu ``flops_partial``.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def count_flops(model: nn.Module, input_hw: tuple[int, int] = (256, 256),
                probe_hw: tuple[int, int] | None = (64, 64)) -> dict:
    """Đếm ở ``probe_hw`` rồi nhân theo số điểm ảnh lên ``input_hw``. Với mạng
    thuần tích chập, kết quả đúng bằng đếm trực tiếp (kiểm thử có so), và nhanh
    hơn nhiều với mạng lớn. ``probe_hw=None``: đếm trực tiếp ở ``input_hw``."""
    if probe_hw is not None and tuple(probe_hw) != tuple(input_hw):
        r = _count_flops_direct(model, probe_hw)
        k = (input_hw[0] * input_hw[1]) / (probe_hw[0] * probe_hw[1])
        return {"flops": int(round(r["flops"] * k)), "flops_g": r["flops"] * k / 1e9,
                "flops_partial": r["flops_partial"], "input_hw": tuple(input_hw), "probe_hw": tuple(probe_hw)}
    return _count_flops_direct(model, input_hw)


@torch.no_grad()
def _count_flops_direct(model: nn.Module, input_hw: tuple[int, int]) -> dict:
    total = [0]
    partial = [False]
    hooks = []

    def conv_hook(m: nn.Conv2d, inp, out):
        k = m.kernel_size[0] * m.kernel_size[1] * (m.in_channels // m.groups)
        total[0] += int(out.numel() / out.shape[0]) * k

    def lin_hook(m: nn.Linear, inp, out):
        total[0] += int(out.numel() / out.shape[0]) * m.in_features

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            hooks.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(lin_hook))
        elif isinstance(m, nn.MultiheadAttention) or type(m).__name__ in ("WindowAttention",):
            partial[0] = True
    was = model.training
    model.eval()
    try:
        model(torch.zeros(1, 3, *input_hw))
    finally:
        for h in hooks:
            h.remove()
        model.train(was)
    return {"flops": total[0], "flops_g": total[0] / 1e9, "flops_partial": partial[0], "input_hw": input_hw}


def ntire_efficiency_score(runtime: float, flops: float, params: float, base_runtime: float,
                           base_flops: float, base_params: float,
                           weights: tuple[float, float, float] = (0.7, 0.15, 0.15)) -> float:
    """Điểm hiệu quả kiểu NTIRE Efficient SR: tổng có trọng số của exp(2 × giá trị / giá trị của mốc).

    Thấp hơn là tốt hơn. Ba giá trị của mốc phải đo trong cùng môi trường với mô
    hình đang chấm; hàm này không có hằng số mặc định nào của NTIRE.
    """
    import math

    w1, w2, w3 = weights
    return (w1 * math.exp(2 * runtime / base_runtime) + w2 * math.exp(2 * flops / base_flops)
            + w3 * math.exp(2 * params / base_params))
